"""Numerical and lifecycle regressions for the experiment-side metrics logger."""

import math
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.xcx.logging.logger import BasicMetricsLogger, _sites, add_metric_args
from experiments.xcx.logging.metrics import (
    ROUTER_METRICS,
    merge_packets,
    merge_summary,
    router_statistics,
    summarize,
    summary_values,
)
from megatron.core.tensor_observation import (
    is_observing_tensor,
    observe_tensor,
    suspend_tensor_observations,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.training import global_vars
from megatron.training.callbacks import CallbackContext


def args(**kwargs):
    return SimpleNamespace(**{
        "log_hidden_states": [], "log_params": [], "log_moe_metrics": [],
        "log_metrics_interval": None, "tensorboard_log_interval": 1,
        **kwargs,
    })


class Layer(nn.Module):
    def __init__(self, number):
        super().__init__()
        self.layer_number = number
        self.input_layernorm = nn.LayerNorm(2)
        self.self_attention = nn.Module()
        self.self_attention.linear_qkv = nn.Linear(2, 2, bias=False)
        self.mlp = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        return self.mlp(self.self_attention.linear_qkv(self.input_layernorm(x)))


class Model(nn.Module):
    def __init__(self, numbers=(1,)):
        super().__init__()
        self.config = SimpleNamespace(num_layers=max(numbers))
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([Layer(number) for number in numbers])

    def forward(self, x):
        for layer in self.decoder.layers:
            x = layer(x)
        return x


class Writer:
    def __init__(self):
        self.values = {}

    def add_scalar(self, name, value, step):
        self.values[name, step] = value


@pytest.fixture
def runtime(monkeypatch):
    writer = Writer()
    state = SimpleNamespace(iteration=0)
    monkeypatch.setattr(global_vars, "_GLOBAL_TENSORBOARD_WRITER", writer)
    monkeypatch.setattr(global_vars, "_GLOBAL_WANDB_WRITER", None)
    monkeypatch.setattr(global_vars, "_GLOBAL_TRAIN_STATE", state)
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", SimpleNamespace(iteration=0))
    return writer, state


def groups(**kwargs):
    return SimpleNamespace(**{"tp": None, "dp_cp": None, "expt_tp": None, "expt_dp": None,
                             **kwargs})


def test_activation_definitions_match_yulan_row_statistics():
    x = torch.tensor([[1., 3.], [2., 8.]], requires_grad=True)
    summary = summarize(x, hidden=True)
    values = summary_values(summary)
    assert not summary.requires_grad
    assert values["mean"] == pytest.approx(x.mean().item())
    assert values["std"] == pytest.approx(x.std(-1).mean().item())
    assert values["rms"] == pytest.approx(x.square().mean(-1).sqrt().mean().item())
    assert values["l2norm"] == pytest.approx(x.norm(dim=-1).mean().item())
    assert values["l1"] == pytest.approx(x.abs().mean().item())
    assert values["max"] == 8 and values["min"] == 1


def test_parameters_and_edge_cases():
    x = torch.tensor([-4., -2., -1.])
    values = summary_values(summarize(x))
    assert values["std"] == pytest.approx(x.std().item())
    assert values["rms"] == pytest.approx(x.square().mean().sqrt().item())
    assert values["max"] == -1 and values["min"] == -4 and values["absmax"] == 4
    assert summary_values(summarize(torch.tensor(3.)))["std"] == 0
    assert summarize(torch.empty(0)) is None
    near_constant = torch.tensor([1., 1.00001, 0.99999])
    assert summary_values(summarize(near_constant))["std"] == pytest.approx(
        near_constant.std().item(), rel=1e-5
    )


def test_unequal_microbatches_are_weighted_by_population():
    first = torch.tensor([[1., 2.]])
    second = torch.tensor([[4., 5.], [7., 9.], [3., 6.]])
    merged = merge_summary(summarize(first, hidden=True), summarize(second, hidden=True))
    torch.testing.assert_close(merged, summarize(torch.cat((first, second)), hidden=True))


def test_tp_shards_are_combined_before_nonlinear_statistics(monkeypatch):
    group = SimpleNamespace(size=lambda: 2)
    remote = torch.tensor([[5., 7.]])

    def all_reduce(value, op=None, group=None):
        if value.ndim == 2:
            value.add_(torch.stack((torch.tensor([2.]), remote.sum(-1),
                                    remote.square().sum(-1), remote.abs().sum(-1))))
        elif op == torch.distributed.ReduceOp.MAX:
            value.copy_(torch.maximum(value, torch.tensor([7., -5.])))
        else:
            # Remote centered sum about the merged mean 4: (5-4)^2 + (7-4)^2.
            value.add_(10)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    actual = summarize(torch.tensor([[1., 3.]]), hidden=True, shard_group=group)
    expected = summarize(torch.tensor([[1., 3., 5., 7.]]), hidden=True)
    torch.testing.assert_close(actual, expected)


def test_router_entropy_margin_and_dropped_rows():
    logits = torch.zeros(2, 3, requires_grad=True)
    probs = torch.tensor([[0.5, 0.5, 0.], [0., 0., 0.]])
    mask = probs.bool()
    values = router_statistics(logits, probs, mask, ROUTER_METRICS, 2)
    assert values["router_logits_entropy"].mean().item() == pytest.approx(math.log(3))
    assert values["router_logits_entropy_normalized"].mean().item() == pytest.approx(1)
    for prefix in ("router_logits_topk", "router_probs_topk"):
        assert values[prefix + "_entropy"].tolist() == pytest.approx([math.log(2)])
        assert values[prefix + "_entropy_normalized"].tolist() == pytest.approx([1])
    assert values["router_logits_topk_margin"].tolist() == [0, 0]
    assert values["router_probs_top1_margin"].tolist() == [0]
    assert all(not value.requires_grad for value in values.values())


def test_global_merge_includes_empty_pp_stages_and_raw_step_loads():
    empty = {"summaries": {}, "loads": {}, "scalars": {}}
    packets = [empty, {**empty, "loads": {3: [[0, 2, 2, 4], [1, 1, 1, 1]]},
                       "scalars": {("entropy", 3): [2, 1]}},
               {**empty, "loads": {3: [[0, 0, 2, 2], [1, 1, 1, 1]]},
                "scalars": {("entropy", 3): [9, 3]}}]
    result = merge_packets(packets)
    assert result["load_balance/max_vio"] == pytest.approx(1)
    assert result["load_balance/min_vio"] == pytest.approx(-1)
    assert result["load_balance/dead_expert_count"] == 1
    assert result["load_balance/dead_expert_fraction"] == 0.25
    assert result["load_balance/max_vio_mass"] == 0
    assert result["max_vio_per_layer/layer_2"] == 1
    assert result["load_balance/entropy"] == pytest.approx(11 / 4)


def test_logger_interval_resume_recompute_and_cleanup(runtime):
    writer, state = runtime
    model = Model((3, 5))
    context = CallbackContext(model=[model])
    options = args(log_hidden_states=["mlp"], log_params=["mlp::linear_fc1"],
                   log_metrics_interval=2)
    # Use a selector available on the tiny model to exercise parameter pre-hooks.
    options.log_params = ["attention::linear_qkv"]
    with BasicMetricsLogger(options, groups=groups()) as logger:
        logger.on_train_start(context)
        state.iteration = 4  # Resume: step 5 must remain off, step 6 is due.
        logger.on_train_step_start(context)
        model(torch.ones(1, 2))
        logger.on_train_step_end(context)
        assert not writer.values
        state.iteration = 5
        logger.on_train_step_start(context)
        model(torch.tensor([[1., 2.]]))
        model(torch.tensor([[3., 4.], [5., 6.]]))
        before = {key: value.clone() for key, value in logger._summaries.items()}
        with suspend_tensor_observations():
            model(torch.tensor([[100., -100.]]))
        for key, value in before.items():
            torch.testing.assert_close(logger._summaries[key], value)
        assert before["hidden_states", "mlp", 3][0] == 3
        assert before["param", "attention::linear_qkv", 3][0] == 1
        logger.on_train_step_end(context)
        assert not is_observing_tensor("xcx_hidden")
        assert ("hidden_states_mean/_layer_03_mlp", 6) in writer.values
        assert ("hidden_states_mean/_layer_05_mlp", 6) in writer.values
        assert ("param_mean/attention::linear_qkv/layer_03", 6) in writer.values
        assert not logger._summaries and not logger._logits
    assert all(not module._forward_hooks and not module._forward_pre_hooks
               for module in model.modules())


def test_logger_exception_restores_observation_scope(runtime):
    model = Model()
    context = CallbackContext(model=[model])
    with pytest.raises(RuntimeError, match="training failure"):
        with BasicMetricsLogger(args(log_hidden_states=["mlp"]), groups=groups()) as logger:
            logger.on_train_start(context)
            logger.on_train_step_start(context)
            model(torch.ones(1, 2))
            raise RuntimeError("training failure")
    assert not is_observing_tensor("xcx_hidden")
    assert all(not module._forward_hooks for module in model.modules())


def test_resume_step_offset_uses_checkpoint_iteration(runtime):
    writer, state = runtime
    global_vars.get_args().iteration = 4
    model = Model()
    context = CallbackContext(model=[model])
    with BasicMetricsLogger(args(log_hidden_states=["mlp"], log_metrics_interval=2),
                            groups=groups()) as logger:
        logger.on_train_start(context)
        for index in range(2):
            state.iteration = index
            logger.on_train_step_start(context)
            model(torch.ones(1, 2))
            logger.on_train_step_end(context)
    assert ("hidden_states_mean/mlp", 6) in writer.values
    assert {step for _, step in writer.values} == {6}


def test_logging_preserves_outputs_and_gradients(runtime):
    model = Model()
    reference = deepcopy(model)
    context = CallbackContext(model=[model])
    x = torch.tensor([[1., 2.], [3., -1.]])
    options = args(log_hidden_states=["mlp", "input_layernorm"],
                   log_params=["attention::linear_qkv"])
    with BasicMetricsLogger(options, groups=groups()) as logger:
        logger.on_train_start(context)
        logger.on_train_step_start(context)
        actual, expected = model(x), reference(x)
        actual.square().sum().backward()
        expected.square().sum().backward()
        logger.on_train_step_end(context)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for param, ref in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(param.grad, ref.grad, rtol=0, atol=0)


def test_fused_identity_site_is_rejected_instead_of_logging_wrong_activation(runtime):
    model = Model()
    model.decoder.layers[0].input_layernorm = IdentityOp()
    with pytest.raises(ValueError, match="fused"):
        with BasicMetricsLogger(args(log_hidden_states=["input_layernorm"]),
                                groups=groups()) as logger:
            logger.on_train_start(CallbackContext(model=[model]))


def test_tp_parameter_replicas_do_not_change_weighting(runtime):
    writer, _ = runtime
    group = SimpleNamespace(rank=lambda: 1, size=lambda: 2)
    model = Model()
    context = CallbackContext(model=[model])
    with BasicMetricsLogger(args(log_params=["attention::linear_qkv"]),
                            groups=groups(tp=group)) as logger:
        logger.on_train_start(context)
        logger.on_train_step_start(context)
        model(torch.ones(1, 2))
        logger.on_train_step_end(context)
    assert not writer.values


def test_router_padding_and_off_interval(runtime):
    writer, _ = runtime

    class Router(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_number, self.topk = 1, 2
            self.config = SimpleNamespace(sequence_parallel=False)

        def forward(self, logits, padding_mask=None):
            observed = logits if padding_mask is None else logits[~padding_mask]
            observe_tensor(self, "router_logits", "router_logits", observed)
            probs = logits.reshape(-1, 3).softmax(-1)
            routing_map = torch.tensor([[True, True, False]]).expand_as(probs)
            return probs * routing_map, routing_map

    model = Model()
    model.decoder.layers[0].mlp.router = Router()
    router = model.decoder.layers[0].mlp.router
    context = CallbackContext(model=[model])
    options = args(log_hidden_states=["moe::router"],
                   log_moe_metrics=["load_balance", "router_logits_entropy"])
    with BasicMetricsLogger(options, groups=groups()) as logger:
        logger.on_train_start(context)
        logger.on_train_step_start(context)
        router(torch.zeros(2, 1, 3), padding_mask=torch.tensor([[True], [False]]))
        logger.on_train_step_end(context)
    assert writer.values["load_balance/max_tokens_per_expert", 1] == 1
    assert writer.values["load_balance/router_logits_entropy", 1] == pytest.approx(math.log(3))


def test_disabled_logger_installs_nothing():
    model = Model()
    logger = BasicMetricsLogger(args())
    logger.on_train_start(CallbackContext(model=[model]))
    assert not logger.enabled and not logger._handles and logger.groups is None
    assert BasicMetricsLogger(args(log_hidden_states=["mlp"], cuda_graph_impl="none")).enabled


def test_unsupported_modes_fail_explicitly():
    with pytest.raises(ValueError, match="CUDA graphs"):
        BasicMetricsLogger(args(log_hidden_states=["mlp"], cuda_graph_impl="local"))
    with pytest.raises(ValueError, match="DDP"):
        BasicMetricsLogger(args(log_params=["embedding"], use_megatron_fsdp=True))
    for key in ("gtp_weight_remat_size", "expert_gtp_weight_remat_size"):
        with pytest.raises(ValueError, match="rematerialization"):
            BasicMetricsLogger(args(log_hidden_states=["mlp"], **{key: 2}))


def test_args_and_global_layer_names():
    from argparse import ArgumentParser

    parser = add_metric_args(ArgumentParser())
    parsed = parser.parse_args(["--log-metrics-interval", "7", "--log-params", "embedding"])
    assert parsed.log_metrics_interval == 7 and parsed.log_params == ["embedding"]
    with pytest.raises(SystemExit):
        parser.parse_args(["--log-metrics-interval", "0"])
    assert [(layer, name) for _, layer, name in _sites(Model((3, 5))) if name == "mlp"] == [
        (3, "mlp"), (5, "mlp"),
    ]
