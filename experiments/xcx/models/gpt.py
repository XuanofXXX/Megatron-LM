"""Select experiment routers through the stock GPT builder and MoE spec."""

import math
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import partial

import torch.nn.functional as F

from megatron.core.transformer.moe.moe_layer import MoESubmodules
from megatron.training.config.instantiate_utils import target_allowlist
from megatron.training.models.gpt import GPTModelConfig, default_layer_spec

from ..losses.kl_aux import KLAuxConfig
from ..routers.ear import SCHEDULED_MODES, EARConfig, EARRouter
from ..routers.kl_aux import KLAuxRouter
from ..training.kl_aux import finalize_kl_model_grads
from .moe import EARMoELayer


@dataclass(kw_only=True)
class XCXModelConfig(GPTModelConfig):
    """Stock GPT config with optional, serializable algorithm settings."""

    ear: EARConfig | None = None
    kl_aux: KLAuxConfig | None = None


@dataclass(kw_only=True)
class KLModelConfig(XCXModelConfig):
    """Retain the config symbol/default used by existing KL checkpoints."""

    kl_aux: KLAuxConfig = field(default_factory=KLAuxConfig)


def _layer_spec(model_config, vp_stage=None):
    spec = deepcopy(default_layer_spec(model_config, vp_stage))
    for layer in getattr(spec, "layer_specs", [spec]):
        mlp = layer.submodules.mlp
        if isinstance(mlp, partial) and isinstance(mlp.keywords.get("submodules"), MoESubmodules):
            ear = getattr(model_config, "ear", None)
            router = (partial(EARRouter, ear=ear, kl=model_config.kl_aux) if ear is not None
                      else partial(KLAuxRouter, kl=model_config.kl_aux))
            mlp.keywords["submodules"] = replace(
                mlp.keywords["submodules"],
                router=router,
            )
            if ear is not None:
                layer.submodules.mlp = partial(EARMoELayer, **mlp.keywords)
    return spec


def kl_layer_spec(model_config, vp_stage=None):
    return _layer_spec(model_config, vp_stage)


def ear_layer_spec(model_config, vp_stage=None):
    return _layer_spec(model_config, vp_stage)


def configure_ear_model(model_config, options, kl=None):
    """Bind materialized expert gate weights using a thin native MoELayer subclass."""
    config = model_config.transformer
    if config.num_moe_experts is None or config.num_moe_experts < 2:
        raise ValueError("EAR requires a MoE model with at least two experts")
    if not config.gated_linear_unit or config.moe_latent_size is not None:
        raise ValueError("EAR requires gated experts without latent projections")
    if options.gate_up_interaction and config.activation_func is not F.silu:
        raise ValueError("Gate-up interaction requires --swiglu (activation_func=SiLU)")
    if options.w_gate_init == "mean_orth" and config.num_moe_experts > config.hidden_size:
        raise ValueError("mean_orth requires num_experts <= hidden_size")
    if options.orthogonal_coeff and options.orthogonal_update_mode == "decoupled":
        if config.tensor_model_parallel_size != 1 or config.context_parallel_size != 1:
            raise ValueError("Decoupled EAR updates currently require TP=CP=1")
    if config.expert_tensor_parallel_size != 1:
        raise ValueError("EAR currently requires expert-tensor-parallel-size 1")
    if config.gradient_accumulation_fusion:
        raise ValueError("EAR routing gradients require --no-gradient-accumulation-fusion")
    if config.fp8 or config.fp4:
        raise ValueError("EAR currently requires ordinary FP32/BF16/FP16 expert weights")
    if model_config.transformer_layer_spec is not None or model_config.restore_modelopt_state:
        raise ValueError("EAR currently uses the official GPT layer spec without ModelOpt")
    if (config.cuda_graph_impl not in (None, "none") or config.moe_router_fusion
            or config.mtp_num_layers or config.transformer_impl == "inference_optimized"
            or getattr(config, "moe_num_hash_layers", 0)
            or config.gtp_weight_remat_size > 1 or config.expert_gtp_weight_remat_size > 1
            or config.overlap_moe_expert_parallel_comm):
        raise ValueError("EAR currently requires eager GPT MoE without MTP, hash routing, "
                         "router fusion, GTP weight rematerialization or MoE overlap")
    if options.lb_static_only and (config.moe_router_force_load_balancing
                                   or config.moe_router_force_biased is not None):
        raise ValueError("Static-only balancing requires unmodified learned router logits")
    if options.reps_mode in SCHEDULED_MODES and (
        config.tensor_model_parallel_size != 1 or config.pipeline_model_parallel_size != 1
        or config.recompute_granularity is not None
    ):
        raise ValueError("Scheduled EAR reps require TP=PP=1 without activation recomputation")
    if kl is not None:
        configure_kl_model(model_config, kl)
    model_config.ear = options
    model_config.transformer_layer_spec = ear_layer_spec
    for symbol in (XCXModelConfig, EARConfig, ear_layer_spec):
        target_allowlist.add_exact(f"{symbol.__module__}.{symbol.__qualname__}")
    if options.reps_mode in SCHEDULED_MODES:
        from .ear_reps import EARRepsRuntime

        runtime = EARRepsRuntime()
        model_config.post_wrap_hooks.append(runtime.install)
        return runtime
    return nullcontext()


def configure_kl_model(model_config, options):
    """Install only experiment-side spec/finalize functions; core settings stay native."""
    config = model_config.transformer
    if config.num_moe_experts is None or config.num_moe_experts < 2:
        raise ValueError("KL Aux requires a MoE model with at least two experts")
    if config.moe_router_load_balancing_type != "aux_loss":
        raise ValueError("KL model construction requires the native aux_loss hook; "
                         "adapt CLI inputs with kl_model_args before building the model config")
    if not isinstance(config.moe_aux_loss_coeff, (int, float)) or not (
        math.isfinite(config.moe_aux_loss_coeff) and config.moe_aux_loss_coeff >= 0
    ):
        raise ValueError("KL Aux requires one finite nonnegative --moe-aux-loss-coeff")
    if config.moe_router_score_function not in ("softmax", "sigmoid"):
        raise ValueError("KL Aux supports softmax and sigmoid router scores")
    if model_config.transformer_layer_spec is not None or model_config.restore_modelopt_state:
        raise ValueError("KL Aux currently uses the official GPT layer spec without ModelOpt")
    if (config.cuda_graph_impl not in (None, "none") or config.moe_router_fusion
            or config.mtp_num_layers or config.transformer_impl == "inference_optimized"
            or getattr(config, "moe_num_hash_layers", 0)
            or config.gtp_weight_remat_size > 1 or config.expert_gtp_weight_remat_size > 1):
        raise ValueError("KL Aux currently requires ordinary eager GPT MoE training without "
                         "MTP, hash routing, router fusion or GTP weight rematerialization")
    model_config.kl_aux = options
    model_config.transformer_layer_spec = kl_layer_spec
    config.finalize_model_grads_func = partial(
        finalize_kl_model_grads, base=config.finalize_model_grads_func,
    )
    # Keep the official config deserializer's allowlist enabled and register only our symbols.
    for symbol in (KLModelConfig, KLAuxConfig, kl_layer_spec):
        target_allowlist.add_exact(f"{symbol.__module__}.{symbol.__qualname__}")
