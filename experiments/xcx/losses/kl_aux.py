"""YuLan's beta-weighted KL gradient proxy and its experiment options."""

import math
from copy import copy
from dataclasses import dataclass

import torch

KL_BALANCING_TYPES = {"sym_kl_aux_loss": "local", "global_sym_kl_aux_loss": "global"}


@dataclass(frozen=True)
class KLAuxConfig:
    mode: str = "local"
    beta: float = 0.5
    rho: float = 0.05
    coeff_clip: float = 20.0
    multiplicative_ste: bool = False
    tolerance: tuple[float, float] | None = None
    global_mode: str = "microbatch"
    dynamic_beta: bool = False
    beta_init: float = 0.5
    beta_range: tuple[float, float] = (0.25, 0.75)
    beta_update_rate: float = 1e-3
    beta_deadband: float = 1.0

    def __post_init__(self):
        if self.mode not in ("local", "global"):
            raise ValueError("KL mode must be local or global")
        if self.global_mode not in ("microbatch", "previous_step", "legacy_prefix"):
            raise ValueError("Unknown global feedback mode")
        if self.mode == "local" and self.global_mode != "microbatch":
            raise ValueError("Global feedback requires --moe-router-load-balancing-type "
                             "global_sym_kl_aux_loss")
        if not 0 <= self.beta <= 1 or not 0 <= self.rho < 1:
            raise ValueError("KL beta must be in [0,1] and rho in [0,1)")
        lo, hi = self.beta_range
        if not 0 <= lo <= self.beta_init <= hi <= 1:
            raise ValueError("Dynamic beta requires 0 <= lower <= init <= upper <= 1")
        for value in (self.coeff_clip, self.beta_update_rate, self.beta_deadband):
            if not math.isfinite(value) or value < 0:
                raise ValueError("KL clip, update rate and deadband must be finite and nonnegative")
        if self.tolerance is not None:
            lower, upper = self.tolerance
            if not 0 < lower <= 1 <= upper or not math.isfinite(upper):
                raise ValueError("KL load tolerance must satisfy 0 < lower <= 1 <= upper < inf")


def symmetric_kl_load_balancing_loss_func(
    probs, tokens_per_expert, total_num_tokens, topk, num_experts, moe_aux_loss_coeff,
    min_load_rho=0.05, coeff_clip=20.0, beta=0.5, multiplicative_ste=False, tolerance=None,
):
    """Return a differentiable gradient proxy and a separate, detached KL diagnostic.

    The proxy differentiates soft scores against coefficients evaluated at smoothed hard
    loads. It is deliberately not the reported KL's autograd derivative.
    """
    if topk <= 0 or num_experts <= 0:
        raise ValueError("topk and num_experts must be positive")
    if not torch.is_tensor(total_num_tokens) and total_num_tokens <= 0:
        raise ValueError("total_num_tokens must be positive")
    if not 0 <= beta <= 1 or not 0 <= min_load_rho < 1:
        raise ValueError("beta must be in [0,1] and rho in [0,1)")
    scores = probs.float()
    total = torch.as_tensor(total_num_tokens, device=scores.device).clamp_min(1)
    hard_load = tokens_per_expert.detach().to(scores) / (total * topk)
    smoothed = (hard_load * (1 - min_load_rho) + min_load_rho / num_experts).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    relative = num_experts * smoothed
    coeff = beta * (relative.log() + 1) - (1 - beta) / relative
    if tolerance is not None:
        lower, upper = tolerance
        if not 0 < lower <= 1 <= upper or not math.isfinite(upper):
            raise ValueError("KL load tolerance must satisfy 0 < lower <= 1 <= upper < inf")
        raw = num_experts * hard_load
        boundary = (1 - min_load_rho) * raw.clamp(lower, upper) + min_load_rho
        boundary_coeff = beta * (boundary.log() + 1) - (1 - beta) / boundary
        coeff = torch.where((raw >= lower) & (raw <= upper), 0., coeff - boundary_coeff)
    if coeff_clip is not None and coeff_clip > 0:
        coeff = coeff.clamp(-coeff_clip, coeff_clip)
    soft_load = scores.sum(0) / total
    proxy_coeff = coeff.detach()
    if multiplicative_ste:
        ratio = hard_load / (soft_load + torch.finfo(scores.dtype).eps)
        proxy_coeff = proxy_coeff * ratio.detach()
    loss = (proxy_coeff * soft_load).sum() * moe_aux_loss_coeff
    metric = beta * (smoothed * relative.log()).sum() - (1 - beta) * relative.log().mean()
    return loss, metric.detach()


def add_kl_args(parser):
    # Extend the official selector only in the experiment entry point.
    action = next((action for action in parser._actions
                   if action.dest == "moe_router_load_balancing_type"), None)
    if action is not None:
        action.choices = list(dict.fromkeys([*action.choices, *KL_BALANCING_TYPES]))
        action.help += (' "sym_kl_aux_loss" selects local KL Aux; '
                        '"global_sym_kl_aux_loss" selects global KL Aux.')
    group = parser.add_argument_group("XCX KL Aux")
    prefix = "--moe-router-sym-kl-loss-"
    for name, default in (("beta", .5), ("rho", .05), ("coeff-clip", 20.),
                          ("beta-init", .5), ("beta-update-rate", 1e-3), ("beta-deadband", 1.)):
        group.add_argument(prefix + name, type=float, default=default)
    for name in ("dynamic-beta", "multiplicative-ste"):
        group.add_argument(prefix + name, action="store_true")
    group.add_argument(prefix + "beta-range", type=lambda s: tuple(map(float, s.split(","))),
                       default=(.25, .75), metavar="LOWER,UPPER")
    group.add_argument(prefix + "tolerance", type=float, nargs=2, default=None)
    group.add_argument("--moe-router-global-aux-mode",
                       choices=("microbatch", "previous_step", "legacy_prefix"),
                       default="microbatch")
    return parser


def kl_config_from_args(args):
    balancing = getattr(args, "moe_router_load_balancing_type", "aux_loss")
    if isinstance(balancing, list):
        if any(value in KL_BALANCING_TYPES for value in balancing) and len(balancing) != 1:
            raise ValueError("KL Aux currently requires one --moe-router-load-balancing-type")
        balancing = balancing[0] if len(balancing) == 1 else balancing
    mode = KL_BALANCING_TYPES.get(balancing) if isinstance(balancing, str) else None
    if mode is None:
        if args.moe_router_sym_kl_loss_dynamic_beta:
            raise ValueError("Dynamic KL beta requires --moe-router-load-balancing-type "
                             "sym_kl_aux_loss or global_sym_kl_aux_loss")
        return None
    fields = ("beta", "rho", "coeff_clip", "multiplicative_ste", "tolerance", "dynamic_beta",
              "beta_init", "beta_range", "beta_update_rate", "beta_deadband")
    values = {key: getattr(args, "moe_router_sym_kl_loss_" + key) for key in fields}
    return KLAuxConfig(mode=mode, global_mode=args.moe_router_global_aux_mode, **values)


def kl_model_args(args, options):
    """Adapt a private copy for native model/data helpers; leave the input namespace intact.

    The experiment spec replaces the ordinary Aux callback with KL. The native
    TransformerConfig therefore uses its supported aux_loss hook, while kl_aux in
    the serialized model config records the actual algorithm and feedback mode.
    """
    if options is None:
        return args
    result = copy(args)
    result.moe_router_load_balancing_type = "aux_loss"
    return result
