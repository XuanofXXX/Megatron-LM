"""Independent formula/gradient and optimizer-state oracles for EAR extensions."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from experiments.xcx.losses.kl_aux import KLAuxConfig
from experiments.xcx.losses.orthogonal import router_orthogonal_loss
from experiments.xcx.models.ear_init import init_ear_gate, mean_orth_gate
from experiments.xcx.models.moe import EARMoELayer, expert_fc1_parameters
from experiments.xcx.routers.ear import EARConfig
from experiments.xcx.tests.test_ear import cpu_group, native_cpu, router
from experiments.xcx.training.orthogonal import EAROrthogonalUpdate
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler


def bind(result):
    experts = nn.Module()
    experts.local_experts = nn.ModuleList()
    for _ in range(4):
        expert = nn.Module()
        expert.linear_fc1 = nn.Linear(8, 12, bias=False)
        experts.local_experts.append(expert)
    result.bind_experts(experts, 4)
    return experts


@pytest.mark.parametrize("mode", ["end2end", "probe_only", "gate_only", "stop_w_gate"])
@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("chunk", [2, 2048])
def test_interaction_logits_and_gradients(mode, separate, static, chunk):
    torch.manual_seed(91)
    result = router(EARConfig(gate_up_interaction=True, gate_up_chunk_size=chunk,
                             grad_mode=mode, per_expert_probe=separate, add_static=static),
                     add_bias_linear=True)
    experts = bind(result)
    x = torch.randn(7, 1, 8, requires_grad=True)
    actual = result.gating(x)
    expected = []
    for i, expert in enumerate(experts.local_experts):
        w = expert.linear_fc1.weight
        if mode in ("probe_only", "stop_w_gate"):
            w = w.detach()
        gate, up = F.linear(x, w).chunk(2, -1)
        probe = result.probe_vector[i] if separate else result.probe_vector
        expected.append((F.silu(gate) * up * probe).sum(-1))
    expected = torch.stack(expected, -1) + result.bias
    if static:
        expected = expected + F.linear(x, result.weight)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    params = [x] + [p for p in result.parameters() if p.requires_grad]
    params += list(expert_fc1_parameters(experts))
    upstream = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, params, upstream, retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(expected, params, upstream, allow_unused=True)
    for a, b in zip(ga, gb):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
    assert ga[0].norm() > 0
    assert (ga[-1] is not None) == (mode in ("end2end", "gate_only"))


@pytest.mark.parametrize("preprocess", ["none", "mean_center", "pca"])
def test_orthogonal_formula_and_gradient(preprocess):
    torch.manual_seed(13)
    rows = torch.randn(4, 8, requires_grad=True)
    loss = router_orthogonal_loss(rows, preprocess=preprocess)
    w = rows.float()
    if preprocess != "none":
        w = w - w.mean(0, keepdim=True)
    if preprocess == "pca":
        _, _, vh = torch.linalg.svd(w, full_matrices=False)
        w = w - (w @ vh[:1].transpose(0, 1)) @ vh[:1]
    w = F.normalize(w, p=2, dim=-1, eps=1e-8)
    reference = ((w @ w.transpose(0, 1) - torch.eye(4)) ** 2).mean()
    torch.testing.assert_close(loss, reference, atol=0, rtol=0)
    a, = torch.autograd.grad(loss, rows, retain_graph=True)
    b, = torch.autograd.grad(reference, rows)
    torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("mode", ["end2end", "probe_only", "gate_only", "stop_w_gate"])
def test_coupled_attachment_matches_explicit_orthogonal_gradient(per_token, mode):
    torch.manual_seed(22)
    result = router(EARConfig(orthogonal_coeff=.3, grad_mode=mode, add_static=True,
                             channel_beta=True), calculate_per_token_loss=per_token)
    experts = bind(result)
    x = torch.randn(7, 2, 8, requires_grad=True)
    MoEAuxLossAutoScaler.set_loss_scale(torch.tensor(1.))
    logits = result.gating(x)
    params = [x] + [p for p in result.parameters() if p.requires_grad]
    params += list(expert_fc1_parameters(experts))
    actual = torch.autograd.grad(logits.sum() * 0, params, allow_unused=True)
    loss = result.orthogonal_loss(result.fresh_orthogonal_rows()) * .3 * (7 if per_token else 1)
    expected = (torch.autograd.grad(loss, params, allow_unused=True) if loss.requires_grad
                else [None] * len(params))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(torch.zeros(()) if a is None else a,
                                   torch.zeros_like(a) if b is None and a is not None else
                                   torch.zeros(()) if b is None else b)
    assert torch.count_nonzero(actual[0]) == 0  # Orthogonal loss is data-independent.


def test_mean_orth_initialization_is_deterministic_local_and_gate_only():
    torch.manual_seed(81)
    before = torch.random.get_rng_state().clone()
    full = mean_orth_gate(4, 0, 4, 6, 8, torch.device("cpu"), .02, 12345, 3)
    local = mean_orth_gate(4, 2, 2, 6, 8, torch.device("cpu"), .02, 12345, 3)
    # Noise is deliberately seeded by EP offset; only the orthogonal mean rows agree.
    torch.testing.assert_close(local.mean(1), full[2:].mean(1), atol=1e-9, rtol=1e-5)
    gram = F.normalize(full.mean(1), dim=-1) @ F.normalize(full.mean(1), dim=-1).T
    torch.testing.assert_close(gram, torch.eye(4), atol=2e-6, rtol=1e-5)
    assert torch.equal(before, torch.random.get_rng_state())
    assert not torch.equal(full, mean_orth_gate(4, 0, 4, 6, 8, torch.device("cpu"), .02, 12345, 4))
    result = router(EARConfig(w_gate_init="mean_orth", probe_init="zeros"))
    experts = bind(result)
    originals = [p.detach().clone() for p in expert_fc1_parameters(experts)]
    layer = SimpleNamespace(config=result.config, router=result, experts=experts,
                            num_local_experts=4, layer_number=3)
    init_ear_gate(layer)
    for param, old, wanted in zip(expert_fc1_parameters(experts), originals, full):
        torch.testing.assert_close(param[:6], wanted, atol=0, rtol=0)
        torch.testing.assert_close(param[6:], old[6:], atol=0, rtol=0)
    torch.testing.assert_close(result.probe_vector, torch.full((6,), 1/6))
    layer.config.perform_initialization = False
    stored = [p.clone() for p in expert_fc1_parameters(experts)]
    init_ear_gate(layer)
    assert all(torch.equal(p, q) for p, q in zip(expert_fc1_parameters(experts), stored))


def test_mean_orth_waits_for_global_layer_number_and_runs_once():
    result = router(EARConfig(w_gate_init="mean_orth"))
    experts = bind(result)
    layer = EARMoELayer.__new__(EARMoELayer)
    nn.Module.__init__(layer)
    layer.config, layer.router, layer.experts = result.config, result, experts
    layer.num_local_experts, layer.layer_number, layer._ear_init_done = 4, None, False
    before = experts.local_experts[0].linear_fc1.weight.detach().clone()
    layer._maybe_init_ear()
    assert torch.equal(experts.local_experts[0].linear_fc1.weight, before)
    layer.set_layer_number(3)
    first = experts.local_experts[0].linear_fc1.weight.detach().clone()
    assert layer._ear_init_done and not torch.equal(first, before)
    with torch.no_grad():
        experts.local_experts[0].linear_fc1.weight.add_(1.)
    layer.set_layer_number(3)
    torch.testing.assert_close(experts.local_experts[0].linear_fc1.weight, first + 1.)


@pytest.mark.parametrize("mode", ["end2end", "probe_only", "gate_only", "stop_w_gate"])
@pytest.mark.parametrize("master_copy", [False, True])
def test_decoupled_update_lr_master_and_adam_moments(mode, master_copy):
    torch.manual_seed(43)
    result = router(EARConfig(orthogonal_coeff=.2, orthogonal_update_mode="decoupled",
                             grad_mode=mode, add_static=True, channel_beta=True))
    experts = bind(result)
    model = nn.ModuleList([result, experts])
    params = [p for p in model.parameters() if p.requires_grad]
    masters = [nn.Parameter(p.detach().float().clone()) for p in params] if master_copy else params
    if master_copy:
        model.bfloat16()
        for param, main in zip(params, masters):
            param.main_param = main
    optimizer = torch.optim.Adam([{"params": [p], "lr": .01 * (i + 1)} for i, p in enumerate(masters)])
    callback = EAROrthogonalUpdate()
    context = SimpleNamespace(model=[model], optimizer=optimizer, skipped_iter=False)
    callback.on_train_start(context)
    callback.on_train_step_start(context)
    for p in masters:
        p.grad = torch.randn_like(p)
    optimizer.step()
    if master_copy:
        with torch.no_grad():
            for p, main in zip(params, masters):
                p.copy_(main)
    state = copy.deepcopy(optimizer.state_dict())
    initial = [p.detach().clone() for p in masters]
    with torch.enable_grad():
        loss = result.orthogonal_loss(result.fresh_orthogonal_rows()) * .2
        grads = (torch.autograd.grad(loss, params, allow_unused=True) if loss.requires_grad
                 else [None] * len(params))
    # Simulate the scheduler changing group LR before on_train_step_end.
    for group in optimizer.param_groups:
        group["lr"] = 9.
    callback.on_train_step_end(context)
    for i, (p, main, old, grad) in enumerate(zip(params, masters, initial, grads)):
        expected = old if grad is None else old.add(grad.to(old), alpha=-.01 * (i + 1))
        torch.testing.assert_close(main, expected, rtol=0, atol=0)
        torch.testing.assert_close(p, main.to(p), rtol=0, atol=0)
    for actual, expected in zip(optimizer.state_dict()["state"].values(), state["state"].values()):
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    context.skipped_iter = True
    old = [p.clone() for p in masters]
    callback.on_train_step_end(context)
    assert all(torch.equal(p, q) for p, q in zip(masters, old))


@pytest.mark.parametrize("kwargs", [
    {"gate_up_interaction": True, "channel_beta": True},
    {"gate_up_interaction": True, "normalize": True},
    {"gate_up_interaction": True, "orthogonal_coeff": .1},
    {"orthogonal_coeff": .1, "orthogonal_update_mode": "decoupled", "reps_mode": "local_fused"},
    {"orthogonal_coeff": -1}, {"gate_up_chunk_size": 0}, {"w_gate_init_seed": -1},
])
def test_invalid_advanced_combinations(kwargs):
    with pytest.raises(ValueError):
        EARConfig(**kwargs)


def test_gate_up_static_only_kl_keeps_main_gradients():
    result = router(EARConfig(gate_up_interaction=True, add_static=True, lb_static_only=True),
                     KLAuxConfig(mode="global"))
    experts = bind(result)
    x = torch.randn(7, 1, 8, requires_grad=True)
    MoEAuxLossAutoScaler.set_loss_scale(torch.tensor(1.))
    probs, _ = result(x)
    params = [x, result.probe_vector, *expert_fc1_parameters(experts), result.weight]
    grads = torch.autograd.grad(probs.sum() * 0, params, retain_graph=True, allow_unused=True)
    assert all(g is None or torch.count_nonzero(g) == 0 for g in grads[:-1])
    assert grads[-1].norm() > 0
    main_grads = torch.autograd.grad((probs * torch.randn_like(probs)).sum(), params, allow_unused=True)
    assert all(g is not None and g.norm() > 0 for g in (main_grads[0], main_grads[1], main_grads[-1]))
    assert any(g is not None and g.norm() > 0 for g in main_grads[2:-1])


def test_pca_zero_rank_preserves_source_mean_center_and_unsafe_optimizer_is_rejected():
    rows = torch.randn(4, 8)
    torch.testing.assert_close(router_orthogonal_loss(rows, preprocess="pca", pca_rank=0),
                               router_orthogonal_loss(rows, preprocess="mean_center"), rtol=0, atol=0)
    EARConfig(orthogonal_pca_rank=0)
    callback = EAROrthogonalUpdate()
    context = SimpleNamespace(optimizer=SimpleNamespace(
        config=SimpleNamespace(use_precision_aware_optimizer=True)))
    with pytest.raises(ValueError, match="ordinary replicated"):
        callback.on_train_start(context)
