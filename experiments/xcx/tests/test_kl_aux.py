"""KL mathematics, complete-step feedback, beta state and stock-builder integration."""

import math
from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn

from experiments.xcx.losses.kl_aux import (
    KLAuxConfig,
    add_kl_args,
    kl_config_from_args,
    kl_model_args,
    symmetric_kl_load_balancing_loss_func,
)
from experiments.xcx.models.gpt import KLModelConfig, configure_kl_model, kl_layer_spec
from experiments.xcx.routers.kl_aux import KLAuxRouter
from experiments.xcx.training.kl_aux import KLMetricsLogger, finalize_kl_model_grads
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import global_vars
from megatron.training.callbacks import CallbackContext


@pytest.fixture(scope="module", autouse=True)
def cpu_group(tmp_path_factory):
    # The official reduction helper requires initialized distributed state, even at size 1.
    path = tmp_path_factory.mktemp("kl-group") / "store"
    dist.init_process_group("gloo", init_method=f"file://{path}", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


@pytest.fixture(autouse=True)
def keep_native_router_on_cpu(monkeypatch):
    # Native Router.gating lazily moves weights to current_device on first use.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    # Exercise the native body without its GPU-oriented torch.compile device discovery.
    method = TopKRouter._apply_expert_bias
    monkeypatch.setattr(TopKRouter, "_apply_expert_bias",
                        getattr(method, "_torchdynamo_orig_callable", method))


def config(**kwargs):
    return TransformerConfig(**{
        "num_layers": 4, "hidden_size": 8, "num_attention_heads": 2,
        "num_moe_experts": 4, "moe_router_topk": 2, "moe_aux_loss_coeff": .1,
        "use_cpu_initialization": True, "add_bias_linear": False,
        **kwargs,
    })


def router(options=None, **kwargs):
    group = dist.group.WORLD
    groups = SimpleNamespace(tp=group, expt_tp=group, cp=group, tp_cp=group, tp_dp_cp=group)
    result = KLAuxRouter(kl=options or KLAuxConfig(), config=config(**kwargs), pg_collection=groups)
    result.set_layer_number(1)
    return result


def maps(indices, experts=4):
    indices = torch.tensor(indices)
    return torch.zeros(len(indices), experts, dtype=torch.bool).scatter(1, indices, True)


@pytest.mark.parametrize("beta", [0., .5, 1.])
@pytest.mark.parametrize("rho", [0., .05])
def test_fixed_beta_metric_and_gradient_against_analytic_kl(beta, rho):
    torch.manual_seed(13)
    logits = torch.randn(10, 4, requires_grad=True)
    scores = logits.softmax(-1)
    counts = torch.tensor([2., 4., 6., 8.])
    loss, metric = symmetric_kl_load_balancing_loss_func(
        scores, counts, 10, 2, 4, .3, min_load_rho=rho, coeff_clip=0, beta=beta,
    )
    loss.backward()
    # Differentiate the two KL definitions at the smoothed hard load, independently of scores.
    hard = counts / 20
    point = ((1 - rho) * hard + rho / 4).requires_grad_()
    reference = beta * (point * (point / .25).log()).sum()
    reference += (1 - beta) * (.25 * (.25 / point).log()).sum()
    coefficient, = torch.autograd.grad(reference, point)
    p = scores.detach()
    expected = .3 / 10 * p * (coefficient - (p * coefficient).sum(-1, keepdim=True))
    torch.testing.assert_close(metric, reference.detach())
    torch.testing.assert_close(logits.grad, expected, atol=1e-8, rtol=1e-5)
    assert not metric.requires_grad


def test_tolerance_clip_and_detached_multiplicative_ste():
    scores = torch.full((10, 4), .25, requires_grad=True)
    counts = torch.tensor([2., 4., 6., 8.])
    loss, _ = symmetric_kl_load_balancing_loss_func(
        scores, counts, 10, 2, 4, 1, beta=.5, tolerance=(.8, 1.2), coeff_clip=.2,
    )
    gradient, = torch.autograd.grad(loss, scores)
    assert gradient[0, 0] < 0 and gradient[0, 3] > 0
    torch.testing.assert_close(gradient[:, 1:3], torch.zeros(10, 2))
    assert gradient.abs().max() <= .020001
    args = (counts, 10, 2, 4, 1)
    first, _ = symmetric_kl_load_balancing_loss_func(scores, *args)
    second, _ = symmetric_kl_load_balancing_loss_func(scores, *args, multiplicative_ste=True)
    a, = torch.autograd.grad(first, scores)
    b, = torch.autograd.grad(second, scores)
    ratio = (counts / 20) / (scores.detach().mean(0) + torch.finfo(torch.float32).eps)
    torch.testing.assert_close(b, a * ratio)


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
def test_kl_changes_only_auxiliary_gradient_not_routing(per_token, score_function):
    result = router(calculate_per_token_loss=per_token, moe_router_score_function=score_function)
    reference = TopKRouter(config=result.config, pg_collection=SimpleNamespace(
        tp=dist.group.WORLD, expt_tp=dist.group.WORLD, cp=dist.group.WORLD,
        tp_cp=dist.group.WORLD, tp_dp_cp=dist.group.WORLD,
    ))
    reference.load_state_dict(result.state_dict())
    reference.set_layer_number(1)
    x = torch.randn(7, 1, 8, requires_grad=True)
    actual, mapping = result(x)
    with torch.no_grad():
        expected, expected_map = reference(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(mapping, expected_map)
    MoEAuxLossAutoScaler.set_loss_scale(torch.tensor(1.))
    (actual * 0).sum().backward()
    assert result.weight.grad.norm() > 0
    assert torch.isfinite(result.weight.grad).all()


def test_previous_feedback_is_frozen_until_complete_step_and_token_weighted():
    result = router(KLAuxConfig(mode="global", global_mode="previous_step"))
    a = maps([[0, 1]])
    b = maps([[2, 3], [2, 3], [2, 3]])
    result._apply_aux_loss(a.float(), torch.full_like(a, .25, dtype=torch.float32), a)
    assert not result.previous_global_aux_valid
    result._apply_aux_loss(b.float(), torch.full_like(b, .25, dtype=torch.float32), b)
    result.finish_kl_step()
    expected = torch.tensor([1., 1., 3., 3.]) / 8
    torch.testing.assert_close(result.previous_global_aux_load, expected)
    assert result.previous_global_aux_valid
    for _ in range(2):
        result._apply_aux_loss(a.float(), torch.full_like(a, .25, dtype=torch.float32), a)
        torch.testing.assert_close(result.previous_global_aux_load, expected)
    _, expected_metric = symmetric_kl_load_balancing_loss_func(
        torch.full_like(a, .25, dtype=torch.float32), expected * 2, 1, 2, 4, .1,
    )
    torch.testing.assert_close(result._kl_metric_sum / result._kl_metric_tokens, expected_metric)
    # A reset must not commit an incomplete step or erase the last complete distribution.
    result.reset_kl_step()
    torch.testing.assert_close(result.previous_global_aux_load, expected)
    result.finish_kl_step()  # An empty step keeps the previous distribution.
    torch.testing.assert_close(result.previous_global_aux_load, expected)


def test_global_microbatch_and_legacy_modes_match_for_equal_microbatch_sizes():
    a = maps([[0, 1], [0, 1]])
    scores = torch.full_like(a, .25, dtype=torch.float32)
    current = router(KLAuxConfig(mode="global"))
    legacy = router(KLAuxConfig(mode="global", global_mode="legacy_prefix"))
    for _ in range(2):
        current._apply_aux_loss(a.float(), scores, a)
        legacy._apply_aux_loss(a.float(), scores, a)
    torch.testing.assert_close(current._kl_metric_sum, legacy._kl_metric_sum)


def test_zero_valid_tokens_are_finite_and_do_not_publish_feedback():
    result = router(KLAuxConfig(mode="global", global_mode="previous_step"))
    mapping = torch.zeros(3, 4, dtype=torch.bool)
    scores = torch.zeros(3, 4, requires_grad=True)
    output = result._apply_aux_loss(torch.zeros_like(scores), scores, mapping, True)
    MoEAuxLossAutoScaler.set_loss_scale(torch.tensor(1.))
    output.sum().backward()
    assert torch.isfinite(scores.grad).all() and torch.count_nonzero(scores.grad) == 0
    result.finish_kl_step()
    assert not result.previous_global_aux_valid


@pytest.mark.parametrize("counts,expected", [
    ([0, 0, 0, 12], .6), ([0, 4, 4, 4], .4), ([0, 0, 0, 2], .5), ([0, 0, 0, 0], .5),
])
def test_beta_uses_raw_per_layer_loads_and_deadband(counts, expected):
    result = router(KLAuxConfig(dynamic_beta=True, beta_update_rate=.1, beta_range=(.1, .9)))
    result._kl_actual_counts.copy_(torch.tensor(counts))
    result.finish_kl_step()
    assert result._get_sym_kl_beta() == pytest.approx(expected)
    assert result._kl_actual_counts.sum() == 0


def test_beta_clamps_and_updates_layers_independently():
    options = KLAuxConfig(dynamic_beta=True, beta_update_rate=.1, beta_range=(.1, .9), beta_init=.89)
    first = router(options)
    second = router(replace(options, beta_init=.11))
    first._kl_actual_counts.copy_(torch.tensor([0., 0., 0., 12.]))
    second._kl_actual_counts.copy_(torch.tensor([0., 4., 4., 4.]))
    first.finish_kl_step()
    second.finish_kl_step()
    assert first._get_sym_kl_beta() == pytest.approx(.9)
    assert second._get_sym_kl_beta() == pytest.approx(.1)


@pytest.mark.parametrize("restored,expected", [(2., .75), (-1., .25), (float("nan"), .5)])
def test_checkpoint_repairs_beta_and_rebootstraps_feedback(restored, expected):
    options = KLAuxConfig(mode="global", dynamic_beta=True, global_mode="previous_step")
    source = router(options)
    source.sym_kl_beta.fill_(restored)
    result = router(options)
    result._get_sym_kl_beta()  # Populate the CPU cache before loading.
    result.previous_global_aux_valid.fill_(True)
    result.load_state_dict(source.state_dict())
    assert result._get_sym_kl_beta() == pytest.approx(expected)
    assert not result.previous_global_aux_valid
    assert "sym_kl_beta" in result.state_dict()
    assert "previous_global_aux_load" not in result.state_dict()


def test_bfloat16_conversion_preserves_fp32_algorithm_state():
    result = router(KLAuxConfig(dynamic_beta=True, beta_init=.37123))
    expected = result.sym_kl_beta.clone()
    result.bfloat16()
    assert result.weight.dtype == torch.bfloat16
    assert result.sym_kl_beta.dtype == torch.float32
    torch.testing.assert_close(result.sym_kl_beta, expected, rtol=0, atol=0)


def test_finalize_preserves_official_arguments_return_and_order():
    result = router(KLAuxConfig(mode="global"))
    chunk = nn.Module()
    chunk.router = result
    calls = []

    def base(model, num_tokens, **kwargs):
        calls.append(kwargs)
        assert result.previous_global_aux_valid == False
        result._kl_counts.copy_(torch.tensor([1., 1., 3., 3.]))
        result._kl_tokens.fill_(4)
        return "official-result"

    answer = finalize_kl_model_grads([chunk], 4, pg_collection="groups", base=base,
                                     force_all_reduce=True)
    assert answer == "official-result"
    assert calls == [{"pg_collection": "groups", "force_all_reduce": True}]
    assert result.previous_global_aux_valid


def test_spec_injection_and_config_serialization_are_experiment_local(monkeypatch):
    from experiments.xcx.models import gpt

    original = SimpleNamespace(layer_specs=[SimpleNamespace(submodules=SimpleNamespace(
        mlp=partial(MoELayer, submodules=MoESubmodules(experts=nn.Identity)),
    ))])
    monkeypatch.setattr(gpt, "default_layer_spec", lambda config, vp_stage: original)
    cfg = KLModelConfig(transformer=config(), vocab_size=32)
    options = KLAuxConfig(mode="global", global_mode="previous_step", dynamic_beta=True)
    configure_kl_model(cfg, options)
    spec = kl_layer_spec(cfg)
    factory = spec.layer_specs[0].submodules.mlp.keywords["submodules"].router
    assert factory.func is KLAuxRouter and factory.keywords["kl"] == options
    assert original.layer_specs[0].submodules.mlp.keywords["submodules"].router is TopKRouter
    saved = cfg.as_dict()
    assert saved["kl_aux"]["global_mode"] == "previous_step"
    assert saved["transformer_layer_spec"]["_target_"] == "experiments.xcx.models.gpt.kl_layer_spec"
    assert "finalize_model_grads_func" not in saved["transformer"]


@pytest.mark.parametrize("kwargs", [{"beta": 2}, {"rho": 1}, {"beta_range": (.8, .9)},
                                    {"tolerance": (1.1, 2)}, {"beta_update_rate": -1}])
def test_invalid_options_fail_early(kwargs):
    with pytest.raises(ValueError):
        KLAuxConfig(**kwargs)


@pytest.fixture
def kl_parser():
    from argparse import ArgumentParser

    from megatron.training.arguments import add_megatron_arguments

    return add_kl_args(add_megatron_arguments(ArgumentParser(allow_abbrev=False)))


def test_cli_preserves_beta_options_and_requires_explicit_kl_mode(kl_parser):
    args = kl_parser.parse_args([
        "--moe-router-load-balancing-type", "global_sym_kl_aux_loss",
        "--moe-router-global-aux-mode", "previous_step",
        "--moe-router-sym-kl-loss-beta", "0.7",
        "--moe-router-sym-kl-loss-beta-range", "0.1,0.9",
        "--moe-router-sym-kl-loss-dynamic-beta",
    ])
    options = kl_config_from_args(args)
    assert options.mode == "global"
    assert options.beta == .7 and options.beta_range == (.1, .9)
    assert options.dynamic_beta and options.global_mode == "previous_step"
    with pytest.raises(ValueError, match="requires"):
        kl_config_from_args(kl_parser.parse_args(["--moe-router-sym-kl-loss-dynamic-beta"]))


@pytest.mark.parametrize("kind,mode", [("sym_kl_aux_loss", "local"),
                                      ("global_sym_kl_aux_loss", "global")])
@pytest.mark.parametrize("validated", [False, True])
def test_cli_selector_installs_kl_through_native_aux_hook(kl_parser, kind, mode, validated):
    args = kl_parser.parse_args(["--moe-router-load-balancing-type", kind])
    if validated:
        args.moe_router_load_balancing_type = kind  # Official validate_args unwraps nargs=1.
    original = args.moe_router_load_balancing_type
    options = kl_config_from_args(args)
    assert options.mode == mode
    native_args = kl_model_args(args, options)
    assert native_args is not args
    assert args.moe_router_load_balancing_type == original
    cfg = KLModelConfig(transformer=config(
        moe_router_load_balancing_type=native_args.moe_router_load_balancing_type,
        moe_expert_capacity_factor=1.,
    ), vocab_size=32)
    configure_kl_model(cfg, options)
    saved = cfg.as_dict()
    assert saved["kl_aux"]["mode"] == mode
    assert cfg.transformer.moe_router_load_balancing_type == "aux_loss"


@pytest.mark.parametrize("kind", ["aux_loss", "seq_aux_loss", "global_aux_loss", "none"])
def test_cli_native_balancing_does_not_enable_kl(kl_parser, kind):
    args = kl_parser.parse_args(["--moe-router-load-balancing-type", kind])
    assert kl_config_from_args(args) is None
    assert kl_model_args(args, None) is args


@pytest.mark.parametrize("kinds", [
    ["sym_kl_aux_loss", "aux_loss"],
    ["sym_kl_aux_loss", "global_sym_kl_aux_loss"],
])
def test_cli_rejects_multiple_losses_with_kl(kl_parser, kinds):
    args = kl_parser.parse_args(["--moe-router-load-balancing-type", *kinds])
    with pytest.raises(ValueError, match="requires one"):
        kl_config_from_args(args)


def test_cli_removed_kl_switch_is_rejected(kl_parser, capsys):
    with pytest.raises(SystemExit):
        kl_parser.parse_args(["--kl-aux", "global"])
    assert "unrecognized arguments: --kl-aux global" in capsys.readouterr().err


def test_cli_extension_does_not_change_official_entrypoint():
    from argparse import ArgumentParser

    from megatron.training.arguments import add_megatron_arguments

    native = add_megatron_arguments(ArgumentParser())
    action = next(a for a in native._actions if a.dest == "moe_router_load_balancing_type")
    assert "sym_kl_aux_loss" not in action.choices


def test_startup_helpers_reuse_parsed_experiment_args(monkeypatch):
    from megatron.training import arguments, ft_integration, inprocess_restart

    def unexpected_parse(*args, **kwargs):
        pytest.fail("startup helper reparsed the CLI without experiment choices")

    monkeypatch.setattr(arguments, "parse_args", unexpected_parse)
    args = SimpleNamespace(inprocess_restart=False, enable_ft_package=False,
                           moe_router_load_balancing_type="global_sym_kl_aux_loss")
    train = object()
    assert inprocess_restart.maybe_wrap_for_inprocess_restart(train, args=args) == (train, None)
    ft_integration.setup(args=args)


def test_kl_logger_resumes_global_step_numbers(monkeypatch):
    class Writer:
        def __init__(self):
            self.steps = []

        def add_scalar(self, name, value, step):
            self.steps.append(step)

    writer = Writer()
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", SimpleNamespace(iteration=4))
    monkeypatch.setattr(global_vars, "_GLOBAL_TRAIN_STATE", SimpleNamespace(iteration=0))
    monkeypatch.setattr(global_vars, "_GLOBAL_TENSORBOARD_WRITER", writer)
    monkeypatch.setattr(global_vars, "_GLOBAL_WANDB_WRITER", None)
    chunk = nn.Module()
    chunk.router = router()
    chunk.router._kl_last_metric.copy_(torch.tensor([.4, 1.]))
    context = CallbackContext(model=[chunk])
    logger = KLMetricsLogger(KLAuxConfig(), interval=1)
    logger.on_train_start(context)
    logger.on_train_step_end(context)
    assert writer.steps and set(writer.steps) == {5}
