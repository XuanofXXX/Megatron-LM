"""EP nonlinear scores and independent updates against a full expert-matrix oracle."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from experiments.xcx.losses.orthogonal import router_orthogonal_loss
from experiments.xcx.routers.ear import EARConfig, EARRouter
from experiments.xcx.tests.test_ear_distributed import groups
from experiments.xcx.training.orthogonal import EAROrthogonalUpdate
from megatron.core.transformer.transformer_config import TransformerConfig

pytestmark = pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2,
                                reason="requires torchrun with two ranks")


def make_router(groups, ear):
    cfg = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=2,
                            num_moe_experts=4, moe_ffn_hidden_size=6, moe_router_topk=2,
                            gated_linear_unit=True, activation_func=F.silu,
                            use_cpu_initialization=True, add_bias_linear=False,
                            gradient_accumulation_fusion=False)
    result = EARRouter(ear=ear, config=cfg, pg_collection=groups)
    result.set_layer_number(1)
    return result


@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("mode", ["end2end", "probe_only", "gate_only", "stop_w_gate"])
def test_gate_up_unequal_rank_inputs_and_all_gradients(groups, separate, mode):
    torch.manual_seed(71)
    result = make_router(groups, EARConfig(gate_up_interaction=True, gate_up_chunk_size=2,
                                           per_expert_probe=separate, grad_mode=mode, add_static=True))
    result.double()  # Separate the distributed algebra oracle from FP32 GEMM tiling effects.
    rank = dist.get_rank()
    all_weights = torch.randn(4, 12, 8, dtype=torch.float64, requires_grad=True)
    local = all_weights.detach()[rank * 2:rank * 2 + 2].clone().requires_grad_()
    all_x = torch.randn(8, 1, 8, dtype=torch.float64, requires_grad=True)
    coeff = torch.randn(8, 1, 4, dtype=torch.float64)
    selection = slice(0, 3) if rank == 0 else slice(3, 8)
    x = all_x.detach()[selection].clone().requires_grad_()
    result._expert_weights_override = local
    actual = result.gating(x)
    (actual * coeff[selection]).sum().backward()
    p = result.probe_vector.detach().clone().requires_grad_(result.probe_vector.requires_grad)
    static = result.weight.detach().clone().requires_grad_()
    w = all_weights.detach() if mode in ("stop_w_gate", "probe_only") else all_weights
    projection = F.linear(all_x, w.reshape(-1, 8)).reshape(8, 1, 4, 2, 6)
    interactions = F.silu(projection[..., 0, :]) * projection[..., 1, :]
    reference = (interactions * (p if separate else p.expand(4, -1))).sum(-1)
    reference = reference + F.linear(all_x, static)
    (reference * coeff).sum().backward()
    for param in (result.probe_vector, result.weight):
        if param.requires_grad:
            dist.all_reduce(param.grad)
    torch.testing.assert_close(actual, reference[selection], rtol=2e-5, atol=2e-7)
    torch.testing.assert_close(x.grad, all_x.grad[selection], rtol=2e-5, atol=2e-7)
    if mode in ("end2end", "gate_only"):
        torch.testing.assert_close(local.grad, all_weights.grad[rank * 2:rank * 2 + 2],
                                   rtol=2e-5, atol=2e-7)
    else:
        assert local.grad is None
    for param, ref in ((result.probe_vector, p), (result.weight, static)):
        if param.requires_grad:
            torch.testing.assert_close(param.grad, ref.grad, rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("mode", ["end2end", "probe_only", "gate_only", "stop_w_gate"])
def test_decoupled_ep_update_matches_global_loss_once(groups, separate, mode):
    torch.manual_seed(54)
    result = make_router(groups, EARConfig(per_expert_probe=separate, grad_mode=mode,
                                           channel_beta=True, orthogonal_coeff=.2,
                                           orthogonal_update_mode="decoupled"))
    rank = dist.get_rank()
    original = torch.randn(4, 12, 8)
    experts = nn.Module()
    experts.local_experts = nn.ModuleList()
    for value in original[rank * 2:rank * 2 + 2]:
        expert = nn.Module()
        expert.linear_fc1 = nn.Linear(8, 12, bias=False)
        expert.linear_fc1.weight.data.copy_(value)
        experts.local_experts.append(expert)
    result.bind_experts(experts, 2)
    model = nn.ModuleList([result, experts])
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.07)
    context = SimpleNamespace(model=[model], optimizer=optimizer, skipped_iter=False)
    callback = EAROrthogonalUpdate()
    callback.on_train_start(context)
    callback.on_train_step_start(context)
    probe = result.probe_vector.detach().clone().requires_grad_(result.probe_vector.requires_grad)
    beta = result.channel_beta.detach().clone().requires_grad_(result.channel_beta.requires_grad)
    weights = original.clone().requires_grad_(mode in ("end2end", "gate_only"))
    if mode != "stop_w_gate":
        rows = torch.einsum("ef,efh->eh", probe if separate else probe.expand(4, -1), weights[:, :6])
        loss = router_orthogonal_loss(rows * beta) * .2
        loss.backward()
    callback.on_train_step_end(context)
    for actual, initial in ((result.probe_vector, probe), (result.channel_beta, beta)):
        wanted = initial.detach() if initial.grad is None else initial.detach() - .07 * initial.grad
        torch.testing.assert_close(actual, wanted, rtol=2e-5, atol=2e-7)
    expected = original if weights.grad is None else original - .07 * weights.grad
    for index, expert in enumerate(experts.local_experts):
        torch.testing.assert_close(expert.linear_fc1.weight, expected[rank * 2 + index],
                                   rtol=2e-5, atol=2e-7)
        assert expert.linear_fc1.weight.grad is None
