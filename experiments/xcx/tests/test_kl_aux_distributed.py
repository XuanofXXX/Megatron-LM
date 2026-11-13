"""Two-rank CPU oracle for KL DP normalization, unequal tokens and padding.

Run with python -m torch.distributed.run --standalone --nproc-per-node 2 -m pytest ...
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from experiments.xcx.losses.kl_aux import KLAuxConfig, symmetric_kl_load_balancing_loss_func
from experiments.xcx.routers.kl_aux import KLAuxRouter
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig

pytestmark = pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2,
                                reason="requires torch.distributed.run with two ranks")


@pytest.fixture(scope="module")
def groups():
    dist.init_process_group("gloo")
    singletons = [dist.new_group([rank]) for rank in range(2)]
    group = singletons[dist.get_rank()]
    yield SimpleNamespace(tp=group, expt_tp=group, cp=group, tp_cp=group,
                          tp_dp_cp=dist.group.WORLD)
    dist.destroy_process_group()


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("padding", [False, True])
def test_dp_gradient_matches_single_batch_reference(groups, monkeypatch, per_token, padding):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    method = TopKRouter._apply_expert_bias
    monkeypatch.setattr(TopKRouter, "_apply_expert_bias",
                        getattr(method, "_torchdynamo_orig_callable", method))
    config = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=2,
                               num_moe_experts=4, moe_router_topk=2, moe_aux_loss_coeff=.1,
                               add_bias_linear=False, use_cpu_initialization=True,
                               calculate_per_token_loss=per_token)
    result = KLAuxRouter(kl=KLAuxConfig(mode="global", beta=.7), config=config,
                         pg_collection=groups)
    result.set_layer_number(1)
    weight = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 50 - .3
    with torch.no_grad():
        result.weight.copy_(weight)
    torch.manual_seed(123)
    all_x = torch.randn(8, 1, 8)
    x = all_x[:3] if dist.get_rank() == 0 else all_x[3:]
    mask = None
    if padding:
        x = torch.cat((x, torch.full_like(x[:1], 10.)))
        mask = torch.zeros(x.shape[:2], dtype=torch.bool)
        mask[-1] = True
    probs, _ = result(x, padding_mask=mask)
    MoEAuxLossAutoScaler.set_loss_scale(torch.tensor(1.))
    (probs * 0).sum().backward()
    dist.all_reduce(result.weight.grad)
    result.weight.grad.div_(8 if per_token else 2)

    reference_weight = weight.clone().requires_grad_()
    scores = torch.nn.functional.linear(all_x, reference_weight).reshape(8, 4).softmax(-1)
    route = torch.zeros_like(scores, dtype=torch.bool).scatter(1, scores.topk(2).indices, True)
    reference, _ = symmetric_kl_load_balancing_loss_func(
        scores, route.sum(0), 8, 2, 4, .1, beta=.7,
    )
    reference.backward()
    torch.testing.assert_close(result.weight.grad, reference_weight.grad, rtol=1e-5, atol=1e-7)
