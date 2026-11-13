"""Two-rank EP gather/backward oracle against a full expert matrix."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from experiments.xcx.losses.kl_aux import KLAuxConfig, symmetric_kl_load_balancing_loss_func
from experiments.xcx.models.ear_reps import _GatherRows
from experiments.xcx.routers.ear import EARConfig, EARRouter
from megatron.core.distributed.finalize_model_grads import (
    _update_router_expert_bias,
    reset_model_temporary_tensors,
)
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig

pytestmark = pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2,
                                reason="requires torch.distributed.run with two ranks")


@pytest.mark.parametrize("mode", ["end2end", "probe_only", "gate_only", "stop_w_gate"])
def test_earh_aux_free_bias_uses_global_counts_with_unequal_rank_tokens(groups, monkeypatch, mode):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    method = TopKRouter._apply_expert_bias
    monkeypatch.setattr(TopKRouter, "_apply_expert_bias",
                        getattr(method, "_torchdynamo_orig_callable", method))
    cfg = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=2,
                            num_moe_experts=4, moe_ffn_hidden_size=6, moe_router_topk=2,
                            gated_linear_unit=True, use_cpu_initialization=True,
                            add_bias_linear=False, gradient_accumulation_fusion=False,
                            moe_router_load_balancing_type="none", moe_aux_loss_coeff=0.,
                            moe_router_score_function="sigmoid", moe_router_enable_expert_bias=True,
                            moe_router_bias_update_rate=.01)
    result = EARRouter(ear=EARConfig(add_static=True, grad_mode=mode), config=cfg,
                       pg_collection=groups)
    result.weight.data.zero_()
    rank = dist.get_rank()
    weights = torch.zeros(2, 6, 8)
    weights[:, :, 0] = 2. if rank == 0 else -2.
    x = torch.zeros(1 + 2 * rank, 1, 8)
    x[:, :, 0] = 1. if rank == 0 else -1.
    result(x, expert_weights=weights)
    _update_router_expert_bias([result], cfg, tp_dp_cp_group=groups.tp_dp_cp)
    biases = [torch.empty_like(result.expert_bias) for _ in range(2)]
    dist.all_gather(biases, result.expert_bias, group=groups.tp_dp_cp)
    for bias in biases:
        torch.testing.assert_close(bias, torch.tensor([.01, .01, -.01, -.01]))
    reset_model_temporary_tensors(cfg, [result])
    assert result.local_tokens_per_expert.sum() == 0


@pytest.mark.parametrize("unused", [False, True])
def test_buffered_gather_ordered_backward_and_unused_rows(groups, unused):
    torch.manual_seed(71)
    rank = dist.get_rank()
    values = torch.randn(2, 2, 3, 8)
    coefficients = torch.randn(2, 2, 6, 8)
    a, b = (values[rank, i].clone().requires_grad_() for i in range(2))
    first, token = _GatherRows.apply(a, groups.ep, None)
    second, _ = _GatherRows.apply(b, groups.ep, token)
    # A zero-element dependency can be the only consumer of an otherwise unused gather.
    loss = (second * coefficients[rank, 1]).sum()
    if not unused:
        loss = loss + (first * coefficients[rank, 0]).sum()
    loss.backward()
    expected = coefficients.sum(0).reshape(2, 2, 3, 8)
    torch.testing.assert_close(a.grad, torch.zeros_like(a) if unused else expected[0, rank])
    torch.testing.assert_close(b.grad, expected[1, rank])


@pytest.fixture(scope="module")
def groups():
    dist.init_process_group("gloo")
    singletons = [dist.new_group([rank]) for rank in range(2)]
    group = singletons[dist.get_rank()]
    yield SimpleNamespace(tp=group, expt_tp=group, cp=group, tp_cp=group,
                          tp_dp_cp=dist.group.WORLD, ep=dist.group.WORLD)
    dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["end2end", "probe_only", "gate_only", "stop_w_gate"])
@pytest.mark.parametrize("per_expert", [False, True])
def test_ep_rows_and_gradients_match_full_expert_reference(groups, mode, per_expert):
    torch.manual_seed(91)
    cfg = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=2,
                            num_moe_experts=4, moe_ffn_hidden_size=6, moe_router_topk=2,
                            gated_linear_unit=True, use_cpu_initialization=True,
                            add_bias_linear=False, gradient_accumulation_fusion=False)
    ear = EARConfig(add_static=True, grad_mode=mode, per_expert_probe=per_expert,
                    channel_beta=True, normalize=True, normalize_tau=1.3)
    result = EARRouter(ear=ear, config=cfg, pg_collection=groups)
    weights = torch.randn(4, 6, 8, requires_grad=True)
    rank = dist.get_rank()
    local = weights.detach()[rank * 2:rank * 2 + 2].clone().requires_grad_()
    all_x = torch.randn(2, 5, 1, 8, requires_grad=True)
    coefficients = torch.randn(2, 5, 1, 4)
    x = all_x.detach()[rank].clone().requires_grad_()
    result._expert_weights_override = local
    actual = result.gating(x)
    (actual * coefficients[rank]).sum().backward()

    probe = result.probe_vector.detach().clone().requires_grad_(result.probe_vector.requires_grad)
    beta = result.channel_beta.detach().clone().requires_grad_(result.channel_beta.requires_grad)
    static = result.weight.detach().clone().requires_grad_()
    w = weights.detach() if mode == "probe_only" else weights
    rows = torch.einsum("ef,efh->eh", probe if per_expert else probe.expand(4, -1), w) * beta
    if mode == "stop_w_gate":
        rows = rows.detach()
    rows = rows / (rows.norm(dim=-1, keepdim=True) + 1e-6) * 1.3 + static
    reference = F.linear(all_x, rows)
    (reference * coefficients).sum().backward()
    torch.testing.assert_close(actual, reference[rank])
    torch.testing.assert_close(x.grad, all_x.grad[rank])
    if mode in ("end2end", "gate_only"):
        torch.testing.assert_close(local.grad, weights.grad[rank * 2:rank * 2 + 2])
    else:
        assert local.grad is None
    for param, ref in ((result.probe_vector, probe), (result.channel_beta, beta),
                       (result.weight, static)):
        if param.requires_grad:
            dist.all_reduce(param.grad)
            torch.testing.assert_close(param.grad, ref.grad)
        else:
            assert param.grad is None


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("static_only", [False, True])
def test_ep_global_kl_normalization_matches_full_batch(groups, monkeypatch, per_token, static_only):
    method = TopKRouter._apply_expert_bias
    monkeypatch.setattr(TopKRouter, "_apply_expert_bias",
                        getattr(method, "_torchdynamo_orig_callable", method))
    torch.manual_seed(43)
    cfg = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=2,
                            num_moe_experts=4, moe_ffn_hidden_size=6, moe_router_topk=2,
                            gated_linear_unit=True, use_cpu_initialization=True,
                            add_bias_linear=False, gradient_accumulation_fusion=False,
                            calculate_per_token_loss=per_token, moe_aux_loss_coeff=.1)
    result = EARRouter(ear=EARConfig(add_static=True, per_expert_probe=True, channel_beta=True,
                                    lb_static_only=static_only),
                       kl=KLAuxConfig(mode="global", beta=.7), config=cfg, pg_collection=groups)
    result.set_layer_number(1)
    all_weights = torch.randn(4, 6, 8, requires_grad=True)
    rank = dist.get_rank()
    local_weights = all_weights.detach()[rank * 2:rank * 2 + 2].clone().requires_grad_()
    all_x = torch.randn(8, 1, 8)
    x = all_x[:3] if rank == 0 else all_x[3:]
    x = torch.cat((x, torch.full_like(x[:1], 10.))).requires_grad_()
    mask = torch.zeros(x.shape[:2], dtype=torch.bool)
    mask[-1] = True
    MoEAuxLossAutoScaler.set_loss_scale(torch.tensor(1.))
    probs, _ = result(x, padding_mask=mask, expert_weights=local_weights)
    (probs * 0).sum().backward()

    probe = result.probe_vector.detach().clone().requires_grad_()
    beta = result.channel_beta.detach().clone().requires_grad_()
    static = result.weight.detach().clone().requires_grad_()
    rows = torch.einsum("ef,efh->eh", probe, all_weights) * beta
    logits = F.linear(all_x, rows + static).reshape(8, 4)
    if static_only:
        static_logits = F.linear(all_x, static).reshape_as(logits)
        logits = logits.detach() + (static_logits - static_logits.detach())
    scores = logits.softmax(-1)
    route = torch.zeros_like(scores, dtype=torch.bool).scatter(1, scores.topk(2).indices, True)
    loss, _ = symmetric_kl_load_balancing_loss_func(
        scores, route.sum(0), 8, 2, 4, .1, beta=.7,
    )
    loss.backward()
    normalization = 8 if per_token else 2
    # Native DDP scales all expert grads by dense DP size; EP shards are not all-reduced.
    local_weights.grad.div_(normalization)
    expected = all_weights.grad
    expected = torch.zeros_like(all_weights) if expected is None else expected
    torch.testing.assert_close(local_weights.grad, expected[rank * 2:rank * 2 + 2],
                               rtol=1e-5, atol=1e-7)
    for param, ref in ((result.probe_vector, probe), (result.channel_beta, beta),
                       (result.weight, static)):
        dist.all_reduce(param.grad)
        param.grad.div_(normalization)
        expected = torch.zeros_like(ref) if ref.grad is None else ref.grad
        torch.testing.assert_close(param.grad, expected, rtol=1e-5, atol=1e-7)
