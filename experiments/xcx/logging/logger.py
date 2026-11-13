"""Interval-gated metrics using Megatron callbacks and tensor observations."""

import re
from functools import partial

import torch
import torch.distributed as dist

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_observation import capture_tensor_observations, is_observing_tensor
from megatron.core.transformer.identity_op import IdentityOp
from megatron.training import global_vars
from megatron.training.callbacks import Callback

from .metrics import ROUTER_METRICS, merge_packets, merge_summary, router_statistics, summarize

HIDDEN_MODULES = {
    "embeddings": "embedding",
    "input_layernorm": "input_layernorm",
    "attention::linear_qkv": "self_attention.linear_qkv",
    "attention::linear_q": "self_attention.linear_q",
    "attention::linear_kv": "self_attention.linear_kv",
    "attention::core_attention": "self_attention.core_attention",
    "attention::o_proj": "self_attention.linear_proj",
    "pre_mlp_layernorm": "pre_mlp_layernorm",
    "mlp": "mlp",
    "moe::shared_experts": "mlp.shared_experts",
    "moe::router": "mlp.router",
}
PARAM_MODULES = {
    "embedding": "embedding.word_embeddings",
    "lm_head": "output_layer",
    "input_layernorm": "input_layernorm",
    "pre_mlp_layernorm": "pre_mlp_layernorm",
    "attention::linear_qkv": "self_attention.linear_qkv",
    "attention::linear_q": "self_attention.linear_q",
    "attention::linear_kv": "self_attention.linear_kv",
    "attention::o_proj": "self_attention.linear_proj",
    "mlp::linear_fc1": "mlp.linear_fc1",
    "mlp::linear_fc2": "mlp.linear_fc2",
    "moe::router::weight": "mlp.router",
    "moe::router::bias": "mlp.router",
    "moe::router::probe_vector": "mlp.router",
    "moe::router::channel_beta": "mlp.router",
    "moe::experts": "mlp.experts",
}


def add_metric_args(parser):
    """Register the supported YuLan selectors without modifying upstream arguments."""
    group = parser.add_argument_group("XCX basic metrics")

    def positive(value):
        result = int(value)
        if result < 1:
            raise ValueError("interval must be positive")
        return result

    group.add_argument(
        "--log-metrics-interval", type=positive, default=None,
        help="Collect metrics every N steps; defaults to tensorboard-log-interval.",
    )
    group.add_argument("--log-hidden-states", nargs="+", choices=tuple(HIDDEN_MODULES), default=[])
    group.add_argument("--log-params", nargs="+", choices=tuple(PARAM_MODULES), default=[])
    group.add_argument(
        "--log-moe-metrics", nargs="+",
        choices=("load_balance", *ROUTER_METRICS), default=[],
    )
    return parser


def _rank(group):
    return group.rank() if group is not None else 0


def _sites(model):
    """Use global layer numbers so PP/VPP chunks never overwrite each other's tags."""
    while hasattr(model, "module"):
        model = model.module
    modules = list(model.named_modules())
    layers = {name: module.layer_number for name, module in modules
              if re.search(r"(?:^|\.)layers\.[0-9]+$", name) and hasattr(module, "layer_number")}
    for name, module in modules:
        prefix = next((key for key in reversed(layers) if name.startswith(key + ".")), None)
        if prefix is not None:
            yield module, layers[prefix], name[len(prefix) + 1:]
        elif name == "embedding" or name.startswith("embedding."):
            yield module, 0, name
        elif name == "output_layer":
            yield module, model.config.num_layers + 1, name


def _parameter_matches(selector, relative, name):
    target = PARAM_MODULES[selector]
    if selector.startswith("moe::router::"):
        return relative == target and name == selector.rsplit("::", 1)[-1]
    if selector == "input_layernorm" and relative == "self_attention.linear_qkv":
        return name == "layer_norm_weight"
    if selector == "pre_mlp_layernorm" and relative == "mlp.linear_fc1":
        return name == "layer_norm_weight"
    matches = relative == target or (
        selector == "moe::experts" and relative.startswith(target + ".")
    )
    return matches and name.startswith("weight")


class BasicMetricsLogger(Callback):
    """Collect only compact statistics during due training steps, then write one metric batch."""

    def __init__(self, args, *, groups=None) -> None:
        self.hidden = tuple(dict.fromkeys(args.log_hidden_states))
        self.params = tuple(dict.fromkeys(args.log_params))
        self.moe = frozenset(args.log_moe_metrics)
        self.interval = args.log_metrics_interval or args.tensorboard_log_interval
        self.enabled = bool(self.hidden or self.params or self.moe)
        if self.interval < 1:
            raise ValueError("metric interval must be positive")
        if self.enabled and getattr(args, "cuda_graph_impl", None) not in (None, "none"):
            raise ValueError("XCX basic metrics currently require CUDA graphs to be disabled")
        if self.params and any(getattr(args, key, False) for key in (
            "use_torch_fsdp2", "use_torch_fsdp", "use_megatron_fsdp",
        )):
            raise ValueError("XCX parameter metrics currently require ordinary Megatron DDP")
        if self.enabled and any(getattr(args, key, 1) > 1 for key in (
            "gtp_weight_remat_size", "expert_gtp_weight_remat_size",
        )):
            raise ValueError("XCX basic metrics do not yet support GTP weight rematerialization")
        self.groups = groups
        self._handles = []
        self._capture = None
        self._summaries, self._loads, self._scalars, self._logits = {}, {}, {}, {}
        self._seen = set()
        self._due = False
        self.step = 0
        self._step_offset = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._end_capture()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._clear()

    def _clear(self):
        self._summaries.clear()
        self._loads.clear()
        self._scalars.clear()
        self._logits.clear()
        self._seen.clear()

    def _end_capture(self):
        if self._capture is not None:
            self._capture.__exit__(None, None, None)
            self._capture = None
        self._due = False

    def on_train_start(self, context):
        if not self.enabled:
            return
        self._step_offset = (
            global_vars.get_args().iteration - global_vars.get_train_state().iteration
        )
        if self.groups is None:
            self.groups = ProcessGroupCollection.use_mpu_process_groups(
                ["tp", "dp_cp", "expt_tp", "expt_dp"]
            )
        for chunk in context.model:
            for module, layer, relative in _sites(chunk):
                for selector in self.hidden:
                    if selector != "moe::router" and HIDDEN_MODULES[selector] == relative:
                        if isinstance(module, IdentityOp):
                            raise ValueError(
                                f"{selector} is fused into another module; its activation "
                                "is not exposed here. Use a visible site or the local backend."
                            )
                        self._handles.append(module.register_forward_hook(
                            partial(self._hidden_hook, selector, layer)
                        ))
                selected = [(selector, name) for selector in self.params
                            for name, _ in module.named_parameters(recurse=False)
                            if _parameter_matches(selector, relative, name)]
                if selected:
                    self._handles.append(module.register_forward_pre_hook(
                        partial(self._param_hook, selected, layer)
                    ))
                if relative == "mlp.router" and self.moe:
                    self._handles.append(module.register_forward_hook(
                        partial(self._router_hook, layer), with_kwargs=True
                    ))

    def on_train_step_start(self, context):
        self._end_capture()
        self._clear()
        self.step = global_vars.get_train_state().iteration + self._step_offset + 1
        self._due = self.step % self.interval == 0
        if self._due:
            kinds = {"xcx_hidden", "xcx_param", "xcx_router"}
            if "moe::router" in self.hidden or any(
                name.startswith("router_logits") for name in self.moe
            ):
                kinds.add("router_logits")
            self._capture = capture_tensor_observations(self._observe, frozenset(kinds))
            self._capture.__enter__()

    def _record(self, key, value):
        if value is not None:
            self._summaries[key] = (
                merge_summary(self._summaries[key], value)
                if key in self._summaries else value
            )

    @torch.no_grad()
    def _hidden_hook(self, selector, layer, module, inputs, output):
        if not is_observing_tensor("xcx_hidden"):
            return
        value = output[0] if isinstance(output, tuple) else output
        sharded = selector in (
            "attention::linear_qkv", "attention::linear_q", "attention::linear_kv",
            "attention::core_attention",
        )
        group = self.groups.tp if sharded and not getattr(module, "gather_output", False) else None
        summary = summarize(value, hidden=True, shard_group=group)
        if group is None or _rank(group) == 0:
            self._record(("hidden_states", selector, layer), summary)

    @torch.no_grad()
    def _param_hook(self, selected, layer, module, inputs):
        if not is_observing_tensor("xcx_param"):
            return
        for selector, name in selected:
            key = selector, id(module), name
            if key in self._seen:
                continue
            self._seen.add(key)
            value = getattr(module, name)
            expert = not getattr(value, "allreduce", True)
            tp = self.groups.expt_tp if expert else self.groups.tp
            dp = self.groups.expt_dp if expert else self.groups.dp_cp
            sharded = getattr(value, "tensor_model_parallel", False)
            summary = summarize(value, shard_group=tp if sharded else None)
            if _rank(tp) == 0 and _rank(dp) == 0:
                self._record(("param", selector, layer), summary)

    @torch.no_grad()
    def _observe(self, owner, name, kind, tensor, tp_shard_dim, sequence_dim, batch_dim):
        if kind != "router_logits" or getattr(owner, "layer_number", None) is None:
            return
        if "moe::router" in self.hidden:
            self._record(
                ("hidden_states", "moe::router", owner.layer_number),
                summarize(tensor, hidden=True),
            )
        if any(metric.startswith("router_logits") for metric in self.moe):
            self._logits[id(owner)] = tensor.detach()

    @torch.no_grad()
    def _router_hook(self, layer, module, inputs, kwargs, output):
        if not is_observing_tensor("xcx_router"):
            return
        logits = self._logits.pop(id(module), None)
        if not module.config.sequence_parallel and _rank(self.groups.tp) != 0:
            return
        probs, routing_map = output
        padding_mask = kwargs.get("padding_mask", inputs[1] if len(inputs) > 1 else None)
        if padding_mask is not None:
            valid = ~padding_mask.flatten().bool()
            probs, routing_map = probs[valid], routing_map[valid]
        if "load_balance" in self.moe:
            value = torch.stack((routing_map.float().sum(0), probs.detach().float().sum(0)))
            self._loads[layer] = self._loads[layer] + value if layer in self._loads else value
        diagnostics = router_statistics(logits, probs, routing_map, self.moe, module.topk)
        for name, values in diagnostics.items():
            key = name, layer
            value = torch.stack((values.sum(), values.new_tensor(values.numel())))
            self._scalars[key] = self._scalars[key] + value if key in self._scalars else value

    def on_train_step_end(self, context):
        due = self._due
        self._end_capture()
        self._logits.clear()
        if not due:
            return
        packet = {
            name: {key: value.detach().cpu().tolist() for key, value in records.items()}
            for name, records in (
                ("summaries", self._summaries), ("loads", self._loads),
                ("scalars", self._scalars),
            )
        }
        world = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        packets = [None] * world if rank == world - 1 else None
        if world > 1:
            dist.gather_object(packet, packets, dst=world - 1)
        else:
            packets = [packet]
        self._clear()
        if packets is not None:
            values = merge_packets(packets)
            tb, wandb = global_vars.get_tensorboard_writer(), global_vars.get_wandb_writer()
            if tb is not None:
                for name, value in values.items():
                    tb.add_scalar(name, value, self.step)
            if wandb is not None and values:
                wandb.log(values, self.step)

    def on_train_end(self, context):
        self.close()
