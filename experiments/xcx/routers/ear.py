"""Linear-probe EAR/EARH with native TopK routing and optional KL Aux."""

import math
import weakref
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F

from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    compute_routing_scores_for_aux_loss,
)

from ..losses.orthogonal import router_orthogonal_loss
from ..models.moe import expert_fc1_weights, expert_gate_weights
from .kl_aux import KLAuxRouter

REPS_MODES = ("reference", "local_pack", "local_fused", "buffered", "prefetch",
              "batched", "compiled_stack", "compiled_stack_async")
SCHEDULED_MODES = frozenset(("buffered", "prefetch", "batched", "compiled_stack",
                             "compiled_stack_async"))


@dataclass(frozen=True)
class EARConfig:
    add_static: bool = False
    grad_mode: str = "end2end"
    per_expert_probe: bool = False
    probe_init: str = "mean"
    normalize: bool = False
    normalize_tau: float = 1.
    channel_beta: bool = False
    lb_static_only: bool = False
    reps_mode: str = "reference"
    gate_up_interaction: bool = False
    gate_up_chunk_size: int = 2048
    w_gate_init: str = "default"
    w_gate_init_seed: int = 12345
    orthogonal_coeff: float = 0.
    orthogonal_update_mode: str = "coupled"
    orthogonal_preprocess: str = "none"
    orthogonal_pca_rank: int = 1

    def __post_init__(self):
        if self.w_gate_init not in ("default", "none", "mean_orth") or self.w_gate_init_seed < 0:
            raise ValueError("Invalid EAR gate initialization")
        if self.gate_up_chunk_size < 1 or self.orthogonal_pca_rank < 0:
            raise ValueError("EAR chunk size must be positive and PCA rank nonnegative")
        if not math.isfinite(self.orthogonal_coeff) or self.orthogonal_coeff < 0:
            raise ValueError("Orthogonal coefficient must be finite and nonnegative")
        if self.orthogonal_update_mode not in ("coupled", "decoupled"):
            raise ValueError("Unknown orthogonal update mode")
        if self.orthogonal_preprocess not in ("none", "mean_center", "pca"):
            raise ValueError("Unknown orthogonal preprocessing")
        if self.gate_up_interaction and (self.normalize or self.channel_beta
                                        or self.reps_mode != "reference" or self.orthogonal_coeff):
            raise ValueError("Gate-up interaction requires reference reps, without row "
                             "normalization, channel beta or orthogonal regularization")
        if (self.orthogonal_coeff and self.orthogonal_update_mode == "decoupled"
                and self.reps_mode != "reference"):
            raise ValueError("Decoupled orthogonal updates currently require reference reps")
        if self.reps_mode not in REPS_MODES:
            raise ValueError("Unknown EAR reps mode")
        if self.reps_mode in SCHEDULED_MODES and self.grad_mode != "end2end":
            raise ValueError("Scheduled EAR reps currently require end2end gradients")
        if self.grad_mode not in ("end2end", "probe_only", "gate_only", "stop_w_gate"):
            raise ValueError("Unknown EAR gradient mode")
        if self.probe_init not in ("mean", "ones", "zeros"):
            raise ValueError("EAR probe_init must be mean, ones or zeros")
        if not math.isfinite(self.normalize_tau) or self.normalize_tau <= 0:
            raise ValueError("EAR normalization tau must be finite and positive")
        if self.lb_static_only and not self.add_static:
            raise ValueError("Static-only balancing requires EARH (--moe-router-add-static-router)")


def add_ear_args(parser):
    group = parser.add_argument_group("XCX EAR/EARH")
    group.add_argument("--use-weight-aware-router", action="store_true")
    for flag in ("add-static-router", "per-expert-probe", "ear-normalize", "use-channel-beta",
                 "lb-static-only"):
        group.add_argument("--moe-router-" + flag, action="store_true")
    group.add_argument("--moe-router-grad-mode", default="end2end",
                       choices=("end2end", "probe_only", "gate_only", "stop_w_gate"))
    group.add_argument("--moe-router-probe-init", default="mean", choices=("mean", "ones", "zeros"))
    group.add_argument("--moe-router-ear-normalize-tau", type=float, default=1.)
    group.add_argument("--ear-reps-mode", choices=REPS_MODES, default="reference")
    group.add_argument("--moe-router-gate-up-interaction", action="store_true")
    group.add_argument("--moe-router-gate-up-interaction-chunk-size", type=int, default=2048)
    group.add_argument("--moe-router-w-gate-init", choices=("default", "none", "mean_orth"), default="default")
    group.add_argument("--moe-router-w-gate-init-seed", type=int, default=12345)
    group.add_argument("--moe-router-orthogonal-loss-coeff", type=float, default=0.)
    group.add_argument("--moe-router-orthogonal-update-mode", choices=("coupled", "decoupled"), default="coupled")
    group.add_argument("--moe-router-orthogonal-preprocess", choices=("none", "mean_center", "pca"), default="none")
    group.add_argument("--moe-router-orthogonal-pca-rank", type=int, default=1)
    return parser


def ear_config_from_args(args):
    if not args.use_weight_aware_router:
        if (args.moe_router_add_static_router or args.moe_router_lb_static_only
                or args.moe_router_per_expert_probe or args.moe_router_ear_normalize
                or args.moe_router_use_channel_beta or args.moe_router_grad_mode != "end2end"
                or args.moe_router_probe_init != "mean" or args.moe_router_ear_normalize_tau != 1.
                or args.ear_reps_mode != "reference" or args.moe_router_gate_up_interaction
                or args.moe_router_w_gate_init not in ("default", "none")
                or args.moe_router_orthogonal_loss_coeff
                or args.moe_router_orthogonal_update_mode != "coupled"
                or args.moe_router_orthogonal_preprocess != "none"
                or args.moe_router_gate_up_interaction_chunk_size != 2048
                or args.moe_router_w_gate_init_seed != 12345
                or args.moe_router_orthogonal_pca_rank != 1):
            raise ValueError("EAR options require --use-weight-aware-router")
        return None
    if (args.overlap_param_gather or args.use_torch_fsdp2 or args.use_megatron_fsdp):
        raise ValueError("EAR currently requires materialized expert weights: disable "
                         "overlap-param-gather and use ordinary Megatron DDP")
    if args.ear_reps_mode in SCHEDULED_MODES and args.overlap_grad_reduce:
        raise ValueError("Scheduled EAR reps currently require disabling overlap-grad-reduce")
    if args.moe_router_orthogonal_loss_coeff and args.moe_router_orthogonal_update_mode == "decoupled":
        if (args.use_distributed_optimizer or getattr(args, "use_precision_aware_optimizer", False)
                or getattr(args, "use_layer_wise_distributed_optimizer", False)):
            raise ValueError("Decoupled EAR updates currently require replicated FP32/BF16 "
                             "optimizer parameters without precision-aware/distributed optimizer")
    return EARConfig(
        add_static=args.moe_router_add_static_router, grad_mode=args.moe_router_grad_mode,
        per_expert_probe=args.moe_router_per_expert_probe, probe_init=args.moe_router_probe_init,
        normalize=args.moe_router_ear_normalize, normalize_tau=args.moe_router_ear_normalize_tau,
        channel_beta=args.moe_router_use_channel_beta, lb_static_only=args.moe_router_lb_static_only,
        reps_mode=args.ear_reps_mode,
        gate_up_interaction=args.moe_router_gate_up_interaction,
        gate_up_chunk_size=args.moe_router_gate_up_interaction_chunk_size,
        w_gate_init=args.moe_router_w_gate_init, w_gate_init_seed=args.moe_router_w_gate_init_seed,
        orthogonal_coeff=args.moe_router_orthogonal_loss_coeff,
        orthogonal_update_mode=args.moe_router_orthogonal_update_mode,
        orthogonal_preprocess=args.moe_router_orthogonal_preprocess,
        orthogonal_pca_rank=args.moe_router_orthogonal_pca_rank,
    )


class EARRouter(KLAuxRouter):
    """Compute weight-aware logits; inherit native dispatch, observers and auxiliary losses."""

    def __init__(self, *, ear, kl=None, **kwargs) -> None:
        super().__init__(kl=kl, **kwargs)
        self.ear = ear
        self.expert_group = kwargs["pg_collection"].ep
        self._experts_ref = None
        self._expert_weights_override = None
        self._static_aux_logits = None
        self._pending_rows = None
        self._reps_gather = None
        f = self.config.moe_ffn_hidden_size
        shape = (self.num_experts, f) if ear.per_expert_probe else (f,)
        fill = {"mean": 1. / f, "ones": 1., "zeros": 0.}[ear.probe_init]
        self.probe_vector = torch.nn.Parameter(self.weight.new_full(shape, fill))
        self.probe_vector.sequence_parallel = self.config.sequence_parallel
        self.probe_vector.requires_grad_(ear.grad_mode not in ("stop_w_gate", "gate_only"))
        self.channel_beta = None
        if ear.channel_beta:
            self.channel_beta = torch.nn.Parameter(self.weight.new_ones(self.config.hidden_size))
            self.channel_beta.sequence_parallel = self.config.sequence_parallel
            # Rows are detached in stop_w_gate; leaving this trainable produces an unused param.
            self.channel_beta.requires_grad_(ear.grad_mode not in ("stop_w_gate", "gate_only"))
        if not ear.add_static:
            del self.weight

    def bind_experts(self, experts, num_local_experts):
        self._experts_ref = weakref.ref(experts)
        self._num_local_experts = num_local_experts

    def _gate_weights(self):
        if self._expert_weights_override is not None:
            return self._expert_weights_override
        experts = self._experts_ref() if self._experts_ref is not None else None
        if experts is None:
            raise RuntimeError("EAR requires an expert weight provider or expert_weights argument")
        return expert_gate_weights(experts, self._num_local_experts,
                                   self.config.moe_ffn_hidden_size, self.config.hidden_size)

    def _probe(self, count, source):
        probe = self.probe_vector.to(dtype=source.dtype, device=source.device)
        if self.ear.per_expert_probe:
            offset = self.expert_group.rank() * count
            probe = probe[offset:offset + count]
        return probe

    def _sources_and_probe(self):
        if self._expert_weights_override is not None:
            sources = self._expert_weights_override.unbind(0)
        else:
            experts = self._experts_ref() if self._experts_ref is not None else None
            if experts is None:
                raise RuntimeError("EAR requires a bound expert provider")
            sources = expert_fc1_weights(experts, self._num_local_experts,
                                        self.config.moe_ffn_hidden_size, self.config.hidden_size)
        if self.ear.grad_mode == "probe_only":
            sources = tuple(weight.detach() for weight in sources)
        return sources, self._probe(len(sources), sources[0])

    def local_rows(self):
        if self.ear.reps_mode in ("local_pack", "local_fused", "compiled_stack",
                                   "compiled_stack_async"):
            from .ear_compiled_reps import compiled_local_rows, pack_gate_weights

            sources, probe = self._sources_and_probe()
            if self.ear.reps_mode != "local_pack":
                return compiled_local_rows(probe, sources, self.config.moe_ffn_hidden_size,
                                            self.ear.per_expert_probe)
            weights = pack_gate_weights(sources, self.config.moe_ffn_hidden_size)
        else:
            weights = self._gate_weights()
            probe = self._probe(weights.shape[0], weights)
        if self.ear.grad_mode == "probe_only":
            weights = weights.detach()
        if self.ear.per_expert_probe:
            probe = probe.unsqueeze(1)
        else:
            probe = probe.view(1, 1, -1).expand(weights.shape[0], -1, -1)
        return torch.bmm(probe, weights).squeeze(1)

    def fresh_orthogonal_rows(self):
        rows = self.local_rows()
        if self.expert_group.size() > 1:
            rows = torch.cat(dist_nn.all_gather(rows, group=self.expert_group))
        if self.channel_beta is not None:
            rows = rows * self.channel_beta.to(rows)
        if self.ear.grad_mode == "stop_w_gate":
            rows = rows.detach()
        return rows

    def orthogonal_loss(self, rows):
        return router_orthogonal_loss(rows, preprocess=self.ear.orthogonal_preprocess,
                                      pca_rank=self.ear.orthogonal_pca_rank)

    def _interaction_logits(self, input, dtype):
        sources, _ = self._sources_and_probe()
        # Unlike linear EAR stop_w_gate, interaction mode detaches weights but keeps x live.
        if self.ear.grad_mode == "stop_w_gate":
            sources = tuple(weight.detach() for weight in sources)
        full = torch.stack(sources).to(device=input.device, dtype=dtype)
        probe = self._probe(len(sources), full).to(dtype)
        x = input.reshape(-1, input.shape[-1]).to(dtype)
        local_tokens, padded_tokens = x.shape[0], x.shape[0]
        if self.expert_group.size() > 1:
            # Nonlinear scores depend on x: exchanging only local logits would mix inputs.
            # Pad to support unequal token populations, then evaluate all inputs locally.
            count = torch.tensor([local_tokens], dtype=torch.int64, device=x.device)
            counts = [torch.empty_like(count) for _ in range(self.expert_group.size())]
            dist.all_gather(counts, count, group=self.expert_group)
            padded_tokens = int(torch.stack(counts).max())
            padded = F.pad(x, (0, 0, 0, padded_tokens - local_tokens))
            x = torch.cat(dist_nn.all_gather(padded, group=self.expert_group))
        parts = []
        for start in range(0, x.shape[0], self.ear.gate_up_chunk_size):
            part = F.linear(x[start:start + self.ear.gate_up_chunk_size], full.reshape(-1, full.shape[-1]))
            part = part.view(-1, len(sources), 2, self.config.moe_ffn_hidden_size)
            interactions = F.silu(part[:, :, 0]) * part[:, :, 1]
            parts.append((interactions * probe.unsqueeze(0)).sum(-1)
                         if self.ear.per_expert_probe else torch.matmul(interactions, probe))
        local = torch.cat(parts)
        if self.expert_group.size() > 1:
            local = torch.cat(dist_nn.all_gather(local, group=self.expert_group), dim=-1)
            start = self.expert_group.rank() * padded_tokens
            local = local[start:start + local_tokens]
        logits = local.view(*input.shape[:-1], self.num_experts)
        if self.ear.add_static:
            logits = F.linear(input.to(dtype), self.weight.to(dtype),
                              self.bias.to(dtype) if self.bias is not None else None) + logits
        elif self.bias is not None:
            logits = logits + self.bias.to(dtype)
        return logits

    def gating(self, input):
        dtype = {"fp32": torch.float32, "fp64": torch.float64}.get(
            self.config.moe_router_dtype, input.dtype,
        )
        if self.ear.gate_up_interaction:
            logits = self._interaction_logits(input, dtype)
            self._set_static_aux_logits(input.to(dtype), logits, dtype)
            return logits
        pending, self._pending_rows = self._pending_rows, None
        if pending is not None:
            rows = pending.consume()
        elif self._reps_gather is not None:
            rows = self._reps_gather(self.local_rows())
        else:
            if self.ear.reps_mode in SCHEDULED_MODES:
                raise RuntimeError("Scheduled EAR reps require the model post-wrap hook; "
                                   "use build_distributed_models or EARRepsRuntime.install")
            rows = self.local_rows()
            if self.expert_group.size() > 1:
                rows = torch.cat(dist_nn.all_gather(rows, group=self.expert_group), dim=0)
        if self.channel_beta is not None:
            rows = rows * self.channel_beta.to(rows)
        if self.ear.grad_mode == "stop_w_gate":
            rows = rows.detach()
        orth_rows = rows
        x, rows = input.to(dtype), rows.to(dtype)
        if self.ear.normalize:
            rows = rows / (rows.norm(dim=-1, keepdim=True) + 1e-6) * self.ear.normalize_tau
        if self.ear.add_static:
            rows = rows + self.weight.to(dtype)
        bias = self.bias.to(dtype) if self.bias is not None else None
        logits = F.linear(x, rows, bias)
        if self.ear.orthogonal_coeff and self.training and torch.is_grad_enabled():
            metric = self.orthogonal_loss(orth_rows)
            get_moe_metrics_tracker().record("router_orthogonal_loss", metric, self.layer_number,
                                              self.config.num_layers)
            if self.ear.orthogonal_update_mode == "coupled":
                loss = metric * (self.ear.orthogonal_coeff / self.tp_cp_group.size())
                if self.calculate_per_token_loss:
                    loss = loss * logits.shape[0]  # Preserve YuLan's sequence-length scaling.
                logits = MoEAuxLossAutoScaler.apply(logits, loss)
        self._set_static_aux_logits(x, logits, dtype)
        return logits

    def _set_static_aux_logits(self, x, logits, dtype):
        if (self.ear.lb_static_only and self.training and torch.is_grad_enabled()
                and self.is_aux_loss_enabled()):
            static = F.linear(x.detach(), self.weight.to(dtype),
                              self.bias.to(dtype) if self.bias is not None else None)
            self._static_aux_logits = logits.detach() + (static - static.detach())

    def forward(self, input, padding_mask=None, input_ids=None, packed_seq_params=None,
                *, expert_weights=None):
        self._expert_weights_override = expert_weights
        try:
            return super().forward(input, padding_mask, input_ids, packed_seq_params)
        finally:
            self._expert_weights_override = None
            self._static_aux_logits = None

    def _balancing_scores(self, scores, routing_map):
        if self._static_aux_logits is None:
            return scores
        _, static_scores = compute_routing_scores_for_aux_loss(
            self._static_aux_logits.reshape_as(scores), self.topk, self.score_function,
            fused=self.config.moe_router_aux_loss_fusion,
            padding_mask=~routing_map.any(-1),
        )
        # Keep the native scores bit-exact; only the balancing gradient uses the static path.
        return scores.detach() + (static_scores - static_scores.detach())

    def _apply_aux_loss(self, probs, scores_for_aux_loss, routing_map, *args, **kwargs):
        if self.get_aux_loss_coeff("aux_loss") != 0:
            scores_for_aux_loss = self._balancing_scores(scores_for_aux_loss, routing_map)
        return super()._apply_aux_loss(
            probs, scores_for_aux_loss, routing_map,
            *args, **kwargs,
        )

    def _apply_seq_aux_loss(self, probs, scores_for_aux_loss, routing_map, *args, **kwargs):
        if self.get_aux_loss_coeff("seq_aux_loss") != 0:
            scores_for_aux_loss = self._balancing_scores(scores_for_aux_loss, routing_map)
        return super()._apply_seq_aux_loss(
            probs, scores_for_aux_loss, routing_map,
            *args, **kwargs,
        )

    def _apply_global_aux_loss(self, probs, scores_for_aux_loss, routing_map, *args, **kwargs):
        if self.get_aux_loss_coeff("global_aux_loss") != 0:
            scores_for_aux_loss = self._balancing_scores(scores_for_aux_loss, routing_map)
        return super()._apply_global_aux_loss(
            probs, scores_for_aux_loss, routing_map,
            *args, **kwargs,
        )
