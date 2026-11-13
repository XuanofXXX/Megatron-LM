"""Packing, schedules and forward lifetime; CUDA benchmark checks actual Inductor/NCCL."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from experiments.xcx.models.ear_reps import EARRepsRuntime
from experiments.xcx.models.gpt import XCXModelConfig, configure_ear_model
from experiments.xcx.models.moe import EARMoELayer
from experiments.xcx.routers import ear_compiled_reps as compiled
from experiments.xcx.routers.ear import EARConfig
from experiments.xcx.tests.test_ear import config, cpu_group, router
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
from megatron.core.transformer.moe.router import TopKRouter


@pytest.fixture(autouse=True)
def aot_cpu(monkeypatch):
    method = TopKRouter._apply_expert_bias
    monkeypatch.setattr(TopKRouter, "_apply_expert_bias",
                        getattr(method, "_torchdynamo_orig_callable", method))
    # Exercise AOTAutograd without paying for CPU Inductor compilation in every test.
    monkeypatch.setattr(compiled, "pack_gate_weights", torch.compile(
        compiled._pack_gate_weights, backend="aot_eager", fullgraph=True))
    monkeypatch.setattr(compiled, "compiled_local_rows", torch.compile(
        compiled._local_rows, backend="aot_eager", fullgraph=True))
    monkeypatch.setattr(compiled, "compiled_stack_rows", torch.compile(
        compiled._stack_rows, backend="aot_eager", fullgraph=True))


@pytest.mark.parametrize("mode", ["local_pack", "local_fused"])
@pytest.mark.parametrize("grad_mode", ["end2end", "probe_only", "gate_only", "stop_w_gate"])
@pytest.mark.parametrize("separate", [False, True])
def test_local_mode_logits_and_every_gradient_match(mode, grad_mode, separate):
    torch.manual_seed(23)
    ear = EARConfig(add_static=True, per_expert_probe=separate, channel_beta=True,
                    normalize=True, grad_mode=grad_mode)
    a, b = router(ear), router(replace(ear, reps_mode=mode))
    b.load_state_dict(a.state_dict())
    w = torch.randn(4, 6, 8, requires_grad=True)
    x = torch.randn(7, 1, 8, requires_grad=True)
    a._expert_weights_override = b._expert_weights_override = w
    out_a, out_b = a.gating(x), b.gating(x)
    torch.testing.assert_close(out_a, out_b, atol=0, rtol=0)
    upstream = torch.randn_like(out_a)
    ga = torch.autograd.grad(out_a, [x, w] + [p for p in a.parameters() if p.requires_grad],
                             upstream, allow_unused=True)
    gb = torch.autograd.grad(out_b, [x, w] + [p for p in b.parameters() if p.requires_grad],
                             upstream, allow_unused=True)
    for aa, bb in zip(ga, gb):
        assert (aa is None) == (bb is None)
        if aa is not None:
            torch.testing.assert_close(aa, bb, atol=0, rtol=0)


class _MoE(EARMoELayer):
    def __init__(self):
        nn.Module.__init__(self)
        self.router = router(EARConfig(add_static=True, per_expert_probe=True))
        self.experts = nn.Module()
        self.experts.local_experts = nn.ModuleList()
        for _ in range(4):
            expert = nn.Module()
            expert.linear_fc1 = nn.Linear(8, 12, bias=False)
            expert.linear_fc2 = nn.Linear(6, 8, bias=False)
            self.experts.local_experts.append(expert)
        self.router.bind_experts(self.experts, 4)

    def forward(self, x):
        probs, _ = self.router(x)
        outputs = []
        for expert in self.experts.local_experts:
            gate, up = expert.linear_fc1(x).chunk(2, -1)
            outputs.append(expert.linear_fc2(F.silu(gate) * up))
        return (torch.stack(outputs, -2) * probs.reshape(*x.shape[:-1], 4, 1)).sum(-2)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _MoE()

    def forward(self, x):
        return x + .1 * self.mlp(x)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(), _Layer(), _Layer()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _snapshot(block, x, repetitions=1):
    block.zero_grad(set_to_none=True)
    inp = x.detach().clone().requires_grad_()
    for _ in range(repetitions):
        out = block(inp)
        out.square().mean().backward()
    return [out.detach(), inp.grad] + [p.grad for p in block.parameters()]


@pytest.mark.parametrize("mode", ["buffered", "batched", "compiled_stack"])
def test_schedules_accumulation_parameter_changes_and_cleanup(mode):
    torch.manual_seed(27)
    block = _Block()
    x = torch.randn(7, 1, 8)
    MoEAuxLossAutoScaler.set_loss_scale(torch.tensor(1.))
    for repetitions in (1, 3):
        reference = _snapshot(block, x, repetitions)
        with EARRepsRuntime(mode=mode) as runtime:
            runtime.install([SimpleNamespace(decoder=block)])
            result = _snapshot(block, x, repetitions)
            for a, b in zip(reference, result):
                assert (a is None) == (b is None)
                if a is not None:
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
            assert all(r._pending_rows is None for r in runtime.schedules[0].routers)
            assert runtime.schedules[0].token is None
        assert not block._forward_hooks and not block._forward_pre_hooks
        assert all(layer.mlp.router._reps_gather is None for layer in block.layers)
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.mul_(1.01)


def test_exception_clears_unconsumed_rows_and_hooks():
    block = _Block()
    def fail(module, args):
        raise RuntimeError("injected")

    handle = block.layers[0].register_forward_pre_hook(fail)
    with EARRepsRuntime(mode="batched") as runtime:
        runtime.install([SimpleNamespace(decoder=block)])
        with pytest.raises(RuntimeError, match="injected"):
            block(torch.randn(7, 1, 8))
        schedule = runtime.schedules[0]
        assert not schedule.active and schedule.token is None
        assert all(r._pending_rows is None for r in schedule.routers)
    handle.remove()
    assert not block._forward_pre_hooks and not block._forward_hooks


def test_config_installs_official_hook_and_rejects_unsafe_schedule():
    cfg = XCXModelConfig(transformer=config(), vocab_size=32)
    with configure_ear_model(cfg, EARConfig(reps_mode="batched")):
        assert len(cfg.post_wrap_hooks) == 1
        assert "post_wrap_hooks" not in cfg.as_dict()
    for field, value in (("tensor_model_parallel_size", 2), ("pipeline_model_parallel_size", 2),
                         ("recompute_granularity", "full")):
        cfg = XCXModelConfig(transformer=config(), vocab_size=32)
        setattr(cfg.transformer, field, value)
        with pytest.raises(ValueError, match="TP=PP=1"):
            configure_ear_model(cfg, EARConfig(reps_mode="batched"))
    with pytest.raises(ValueError, match="end2end"):
        EARConfig(reps_mode="prefetch", grad_mode="probe_only")


def test_schedule_requires_install_and_a_separate_communicator():
    result = router(EARConfig(reps_mode="batched"))
    with pytest.raises(RuntimeError, match="post-wrap hook"):
        result.gating(torch.randn(7, 1, 8))
    block = _Block()
    ep = block.layers[0].mlp.router.expert_group
    with EARRepsRuntime(mode="batched", group=ep) as runtime:
        with pytest.raises(ValueError, match="dedicated EAR reps group"):
            runtime.install([SimpleNamespace(decoder=block)])
    assert all(layer.mlp.router._reps_gather is None for layer in block.layers)
