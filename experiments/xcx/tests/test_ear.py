"""EAR/EARH logits, gradient paths, expert views and native spec composition."""

from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from experiments.xcx.logging.logger import _parameter_matches
from experiments.xcx.losses.kl_aux import KLAuxConfig
from experiments.xcx.models.gpt import XCXModelConfig, configure_ear_model, ear_layer_spec
from experiments.xcx.models.moe import EARMoELayer, expert_fc1_weights, expert_gate_weights
from experiments.xcx.routers.ear import EARConfig, EARRouter, add_ear_args, ear_config_from_args
from experiments.xcx.training.kl_aux import kl_routers
from megatron.core.distributed.finalize_model_grads import (
    _update_router_expert_bias,
    reset_model_temporary_tensors,
)
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.fixture(scope="module", autouse=True)
def cpu_group(tmp_path_factory):
    path = tmp_path_factory.mktemp("ear-group") / "store"
    dist.init_process_group("gloo", init_method=f"file://{path}", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


@pytest.fixture(autouse=True)
def native_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    method = TopKRouter._apply_expert_bias
    monkeypatch.setattr(TopKRouter, "_apply_expert_bias",
                        getattr(method, "_torchdynamo_orig_callable", method))


def config(**kwargs):
    return TransformerConfig(**{
        "num_layers": 2, "hidden_size": 8, "num_attention_heads": 2,
        "num_moe_experts": 4, "moe_router_topk": 2, "moe_aux_loss_coeff": .1,
        "moe_ffn_hidden_size": 6, "gated_linear_unit": True,
        "use_cpu_initialization": True, "add_bias_linear": False,
        "gradient_accumulation_fusion": False, **kwargs,
    })


def router(ear=None, kl=None, **kwargs):
    group = dist.group.WORLD
    groups = SimpleNamespace(**{k: group for k in (
        "tp", "expt_tp", "cp", "tp_cp", "tp_dp_cp", "ep",
    )})
    result = EARRouter(ear=ear or EARConfig(), kl=kl, config=config(**kwargs),
                       pg_collection=groups)
    result.set_layer_number(1)
    return result


@pytest.mark.parametrize("mode", ["end2end", "probe_only", "gate_only", "stop_w_gate"])
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("per_expert", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
def test_logits_and_gradient_modes(mode, static, per_expert, normalized):
    torch.manual_seed(17)
    result = router(EARConfig(add_static=static, grad_mode=mode,
                             per_expert_probe=per_expert, normalize=normalized,
                             normalize_tau=1.7, channel_beta=True), add_bias_linear=True)
    weights = torch.randn(4, 6, 8, requires_grad=True)
    x = torch.randn(7, 1, 8, requires_grad=True)
    result._expert_weights_override = weights
    actual = result.gating(x)
    p = result.probe_vector
    # Independent einsum reference; the routing path uses batched matrix multiplication.
    w = weights.detach() if mode == "probe_only" else weights
    rows = torch.einsum("ef,efh->eh", p if per_expert else p.expand(4, -1), w)
    rows = rows * result.channel_beta
    if mode == "stop_w_gate":
        rows = rows.detach()
    if normalized:
        rows = rows / (torch.linalg.vector_norm(rows, dim=-1, keepdim=True) + 1e-6) * 1.7
    if static:
        rows = rows + result.weight
    expected = F.linear(x, rows, result.bias)
    torch.testing.assert_close(actual, expected)
    multiplier = torch.randn_like(actual)
    trainable = [x, weights] + [p for p in result.parameters() if p.requires_grad]
    got = torch.autograd.grad((actual * multiplier).sum(), trainable, allow_unused=True,
                              retain_graph=True)
    want = torch.autograd.grad((expected * multiplier).sum(), trainable, allow_unused=True)
    for a, b in zip(got, want):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b)
    assert got[0].norm() > 0  # Input remains differentiable in every mode.
    assert (got[1] is not None) == (mode in ("end2end", "gate_only"))
    assert result.probe_vector.requires_grad == (mode in ("end2end", "probe_only"))
    assert result.channel_beta.requires_grad == result.probe_vector.requires_grad
    assert hasattr(result, "weight") == static


@pytest.mark.parametrize("score", ["softmax", "sigmoid"])
def test_forward_keeps_native_routing(score):
    torch.manual_seed(3)
    result = router(EARConfig(add_static=True, normalize=True), moe_router_score_function=score)
    weights = torch.randn(4, 6, 8)
    x = torch.randn(9, 1, 8)
    result._expert_weights_override = weights
    rows = weights.mean(1)
    rows = rows / (rows.norm(dim=-1, keepdim=True) + 1e-6) + result.weight
    reference = TopKRouter(config=result.config, pg_collection=SimpleNamespace(
        tp=dist.group.WORLD, expt_tp=dist.group.WORLD, cp=dist.group.WORLD,
        tp_cp=dist.group.WORLD, tp_dp_cp=dist.group.WORLD,
    ))
    reference.set_layer_number(1)
    reference.weight.data.copy_(rows)
    with torch.no_grad():
        a, amap = result(x, expert_weights=weights)
        b, bmap = reference(x)
    torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)
    assert torch.equal(amap, bmap)
    assert result._expert_weights_override is None and result._static_aux_logits is None


@pytest.mark.parametrize("kind", ["aux_loss", "seq_aux_loss", "global_aux_loss", "kl", "global_kl"])
@pytest.mark.parametrize("per_token", [False, True])
def test_static_only_isolates_only_balancing_gradient(kind, per_token):
    torch.manual_seed(5)
    kl = None
    if kind in ("kl", "global_kl"):
        kl = KLAuxConfig(mode="global" if kind == "global_kl" else "local",
                         global_mode="previous_step" if kind == "global_kl" else "microbatch",
                         dynamic_beta=True)
    balance = "aux_loss" if kl is not None else kind
    ear = EARConfig(add_static=True, channel_beta=True)
    full = router(ear, kl, moe_router_load_balancing_type=balance,
                  calculate_per_token_loss=per_token)
    isolated = router(replace(ear, lb_static_only=True), kl,
                      moe_router_load_balancing_type=balance, calculate_per_token_loss=per_token)
    isolated.load_state_dict(full.state_dict())
    weights = torch.randn(4, 6, 8, requires_grad=True)
    x = torch.randn(7, 2, 8, requires_grad=True)
    mask = torch.zeros(7, 2, dtype=torch.bool)
    mask[-1, 1] = True
    MoEAuxLossAutoScaler.set_loss_scale(torch.tensor(1.))
    a, amap = full(x, padding_mask=mask, expert_weights=weights)
    b, bmap = isolated(x, padding_mask=mask, expert_weights=weights)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert torch.equal(amap, bmap)
    params = [x, weights, isolated.probe_vector, isolated.channel_beta, isolated.weight]
    grads = torch.autograd.grad((b * 0).sum(), params, retain_graph=True, allow_unused=True)
    for grad in grads[:-1]:
        assert grad is None or torch.count_nonzero(grad) == 0
    assert grads[-1].norm() > 0
    # Static router's balancing gradient equals the unisolated implementation exactly.
    expected, = torch.autograd.grad((a * 0).sum(), [full.weight])
    torch.testing.assert_close(grads[-1], expected)
    # Main-task gradients must continue to reach inputs, expert gates and the probe.
    main = (b * torch.randn_like(b)).sum()
    main_grads = torch.autograd.grad(main, params, allow_unused=True)
    for grad in main_grads:
        assert grad is not None and grad.norm() > 0
    if kl is not None:
        isolated.finish_kl_step()
        assert isolated.previous_global_aux_valid == (kl.mode == "global")
        assert isolated._get_sym_kl_beta() is not None


@pytest.mark.parametrize("mode", ["end2end", "probe_only", "gate_only", "stop_w_gate"])
@pytest.mark.parametrize("balancing", ["aux", "static_only", "aux_free"])
def test_earh_gradient_flow_keeps_expert_main_task_training(mode, balancing):
    """Separate main-task expert gradients, router gradients and attached Aux gradients."""
    torch.manual_seed(42)
    result = router(EARConfig(add_static=True, grad_mode=mode,
                             lb_static_only=balancing == "static_only"),
                    moe_router_load_balancing_type="none" if balancing == "aux_free" else "aux_loss",
                    moe_aux_loss_coeff=0. if balancing == "aux_free" else .1,
                    moe_router_score_function="sigmoid", moe_router_enable_expert_bias=balancing == "aux_free")
    gate = (.2 * torch.randn(4, 6, 8)).requires_grad_()
    up = (.2 * torch.randn(4, 6, 8)).requires_grad_()
    down = (.2 * torch.randn(4, 8, 6)).requires_grad_()
    x = torch.randn(9, 1, 8, requires_grad=True)
    MoEAuxLossAutoScaler.set_loss_scale(torch.tensor(1.))
    probs, _ = result(x, expert_weights=gate)
    flat = x.flatten(0, 1)
    # A small dense oracle computes all experts and then applies native sparse route weights.
    activation = F.silu(torch.einsum("th,efh->tef", flat, gate))
    activation = activation * torch.einsum("th,efh->tef", flat, up)
    outputs = torch.einsum("tef,ehf->teh", activation, down)
    values = (outputs * torch.randn(9, 1, 8)).sum(-1)
    params = [x, gate, up, down, result.weight]
    if result.probe_vector.requires_grad:
        params.append(result.probe_vector)

    def grads(loss):
        return [torch.zeros_like(p) if g is None else g for p, g in zip(
            params, torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True),
        )]

    aux = grads((probs * 0).sum())
    routed = grads((probs * values.detach()).sum())
    main_router = [a - b for a, b in zip(routed, aux)]
    main_expert = grads((probs.detach() * values).sum())
    total = grads((probs * values).sum())
    for got, a, b, c in zip(total, main_expert, main_router, aux):
        torch.testing.assert_close(got, a + b + c, atol=1e-7, rtol=1e-5)
    assert main_expert[1].norm() > 0  # Expert gate still learns from its own FFN in every mode.
    assert main_expert[2].norm() > 0 and main_expert[3].norm() > 0
    assert main_router[4].norm() > 0  # Static router always learns from the main task.
    assert (main_router[1].norm() > 1e-9) == (mode in ("end2end", "gate_only"))
    if result.probe_vector.requires_grad:
        assert main_router[-1].norm() > 0
    if balancing == "aux_free":
        assert all(torch.count_nonzero(g) == 0 for g in aux)
        assert not result.expert_bias.requires_grad
    elif balancing == "static_only":
        assert aux[4].norm() > 0
        assert all(torch.count_nonzero(g) == 0 for i, g in enumerate(aux) if i != 4)


def test_earh_native_aux_free_bias_selection_update_and_reset():
    torch.manual_seed(71)
    result = router(EARConfig(add_static=True), moe_router_load_balancing_type="none",
                    moe_aux_loss_coeff=0., moe_router_score_function="sigmoid",
                    moe_router_enable_expert_bias=True, moe_router_bias_update_rate=.01)
    result.expert_bias.copy_(torch.tensor([.8, .4, -.4, -.8]))
    gate = .1 * torch.randn(4, 6, 8)
    x = torch.randn(7, 1, 8, requires_grad=True)
    padding = torch.tensor([[False]] * 6 + [[True]])
    probs, mapping = result(x, padding_mask=padding, expert_weights=gate)
    logits = F.linear(x, gate.mean(1) + result.weight).reshape(-1, 4)
    scores = logits.sigmoid()
    selected = (scores + result.expert_bias).topk(2, dim=-1).indices
    raw = scores.gather(1, selected)
    expected = torch.zeros_like(scores).scatter(1, selected, raw / raw.sum(-1, keepdim=True))
    torch.testing.assert_close(probs, expected)
    assert mapping[:, :2].all() and not mapping[:, 2:].any()
    assert result.local_tokens_per_expert.tolist() == [6, 6, 0, 0]
    # No-grad recomputation/evaluation must not count an extra microbatch.
    with torch.no_grad():
        result(x, padding_mask=padding, expert_weights=gate)
    assert result.local_tokens_per_expert.tolist() == [6, 6, 0, 0]
    before = result.expert_bias.clone()
    _update_router_expert_bias([result], result.config, tp_dp_cp_group=dist.group.WORLD)
    torch.testing.assert_close(result.expert_bias, before + torch.tensor([-.01, -.01, .01, .01]))
    reset_model_temporary_tensors(result.config, [result])
    assert result.local_tokens_per_expert.sum() == 0
    assert result.expert_bias.dtype == torch.float32
    assert "expert_bias" in result.state_dict() and "local_tokens_per_expert" not in result.state_dict()


@pytest.mark.parametrize("layout", ["sequential", "te_list", "te_3d", "te_2d", "legacy"])
def test_expert_views_are_differentiable_without_up_gradients(layout):
    values = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4)
    if layout == "sequential":
        params = [nn.Parameter(x.clone()) for x in values]
        experts = SimpleNamespace(local_experts=[SimpleNamespace(
            linear_fc1=SimpleNamespace(weight=p)) for p in params])
    elif layout == "te_list":
        params = [nn.Parameter(x.clone()) for x in values]
        experts = SimpleNamespace(linear_fc1=SimpleNamespace(weight0=params[0], weight1=params[1]))
    else:
        data = (values.permute(2, 0, 1).reshape(4, 12) if layout == "legacy" else
                values.reshape(12, 4) if layout == "te_2d" else values)
        params = [nn.Parameter(data.clone())]
        experts = (SimpleNamespace(weight1=params[0]) if layout == "legacy" else
                   SimpleNamespace(linear_fc1=SimpleNamespace(weight=params[0])))
    gates = expert_gate_weights(experts, 2, 3, 4)
    sources = expert_fc1_weights(experts, 2, 3, 4)
    torch.testing.assert_close(torch.stack(sources), values)
    packed = torch.stack([w[:3] for w in sources])
    actual = torch.autograd.grad(packed.sum(), params, retain_graph=True)
    expected = torch.autograd.grad(gates.sum(), params, retain_graph=True)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(gates, values[:, :3])
    gates.sum().backward()
    gradients = (params[0].grad.reshape(4, 2, 6).permute(1, 2, 0) if layout == "legacy"
                 else torch.stack([p.grad for p in params]) if len(params) == 2
                 else params[0].grad.reshape(2, 6, 4))
    torch.testing.assert_close(gradients[:, :3], torch.ones(2, 3, 4))
    assert torch.count_nonzero(gradients[:, 3:]) == 0


def test_binding_keeps_experts_out_of_router_state_and_kl_finalize():
    result = router()
    experts = nn.Module()
    experts.local_experts = nn.ModuleList([nn.Module() for _ in range(4)])
    for expert in experts.local_experts:
        expert.linear_fc1 = nn.Linear(8, 12, bias=False)
    result.bind_experts(experts, 4)
    assert list(result.state_dict()) == ["probe_vector"]
    result(torch.randn(7, 1, 8))[0].sum().backward()
    assert experts.local_experts[0].linear_fc1.weight.grad is not None
    assert list(kl_routers([result])) == []
    assert _parameter_matches("moe::router::probe_vector", "mlp.router", "probe_vector")
    assert not _parameter_matches("moe::router::weight", "mlp.router", "probe_vector")


def test_spec_composes_ear_kl_without_touching_original(monkeypatch):
    from experiments.xcx.models import gpt

    original = SimpleNamespace(layer_specs=[SimpleNamespace(submodules=SimpleNamespace(
        mlp=partial(MoELayer, submodules=MoESubmodules(experts=nn.Identity)),
    ))])
    monkeypatch.setattr(gpt, "default_layer_spec", lambda config, vp_stage: original)
    cfg = XCXModelConfig(transformer=config(), vocab_size=32)
    kl = KLAuxConfig(mode="global", dynamic_beta=True)
    configure_ear_model(cfg, EARConfig(add_static=True), kl)
    spec = ear_layer_spec(cfg)
    mlp = spec.layer_specs[0].submodules.mlp
    assert mlp.func is EARMoELayer
    assert mlp.keywords["submodules"].router.func is EARRouter
    assert mlp.keywords["submodules"].router.keywords["kl"] == kl
    assert original.layer_specs[0].submodules.mlp.func is MoELayer
    saved = cfg.as_dict()
    assert saved["ear"]["add_static"] and saved["kl_aux"]["dynamic_beta"]
    assert saved["transformer_layer_spec"]["_target_"] == "experiments.xcx.models.gpt.ear_layer_spec"


@pytest.mark.parametrize("field,value", [("gated_linear_unit", False),
                                        ("expert_tensor_parallel_size", 2),
                                        ("gradient_accumulation_fusion", True),
                                        ("fp8", "hybrid"), ("moe_latent_size", 4)])
def test_unsupported_weight_paths_fail_early(field, value):
    cfg = XCXModelConfig(transformer=config(), vocab_size=32)
    setattr(cfg.transformer, field, value)
    with pytest.raises(ValueError):
        configure_ear_model(cfg, EARConfig())


@pytest.mark.parametrize("kwargs", [{"grad_mode": "bad"}, {"probe_init": "bad"},
                                    {"normalize_tau": 0}, {"lb_static_only": True}])
def test_invalid_ear_config(kwargs):
    with pytest.raises(ValueError):
        EARConfig(**kwargs)


def test_cli_defaults_and_explicit_enable():
    from argparse import ArgumentParser

    parser = add_ear_args(ArgumentParser())
    assert ear_config_from_args(parser.parse_args([])) is None
    with pytest.raises(ValueError, match="require --use-weight-aware-router"):
        ear_config_from_args(parser.parse_args(["--moe-router-per-expert-probe"]))
    args = parser.parse_args(["--use-weight-aware-router", "--moe-router-add-static-router",
                              "--moe-router-grad-mode", "gate_only"])
    args.overlap_param_gather = args.use_torch_fsdp2 = args.use_megatron_fsdp = False
    assert ear_config_from_args(args) == EARConfig(add_static=True, grad_mode="gate_only")
    args.overlap_param_gather = True
    with pytest.raises(ValueError, match="materialized expert weights"):
        ear_config_from_args(args)
