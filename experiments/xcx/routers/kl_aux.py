"""TopKRouter with KL Aux, complete-step feedback and per-layer beta state."""

import math

import torch
import torch.distributed as dist

from megatron.core.transformer.moe.moe_utils import get_tokens_per_expert_and_token_count
from megatron.core.transformer.moe.router import TopKRouter

from ..losses.kl_aux import symmetric_kl_load_balancing_loss_func


class KLAuxRouter(TopKRouter):
    _float_buffers = (
        "sym_kl_beta", "previous_global_aux_load", "_kl_counts", "_kl_tokens",
        "_kl_microbatches", "_kl_actual_counts", "_kl_metric_sum", "_kl_metric_tokens",
        "_kl_last_metric",
    )

    def __init__(self, *, kl=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.kl = kl
        self._beta_value = None
        if kl is None:
            return
        device = self.weight.device
        scalar = lambda: torch.zeros((), dtype=torch.float32, device=device)
        experts = lambda: torch.zeros(self.num_experts, dtype=torch.float32, device=device)
        self.register_buffer(
            "sym_kl_beta",
            torch.tensor(kl.beta_init, device=device, dtype=torch.float32)
            if kl.dynamic_beta else None,
        )
        self.register_buffer("previous_global_aux_load", experts(), persistent=False)
        self.register_buffer("previous_global_aux_valid", torch.tensor(False, device=device),
                             persistent=False)
        for name in ("_kl_counts", "_kl_actual_counts"):
            self.register_buffer(name, experts(), persistent=False)
        for name in ("_kl_tokens", "_kl_microbatches", "_kl_metric_sum", "_kl_metric_tokens"):
            self.register_buffer(name, scalar(), persistent=False)
        self.register_buffer("_kl_last_metric", torch.zeros(2, device=device, dtype=torch.float32),
                             persistent=False)

    def _apply(self, fn, recurse=True):
        # Module.half()/bfloat16() must not round algorithm state before moving it to GPU.
        saved = {name: self._buffers.get(name) for name in self._float_buffers}
        result = super()._apply(fn, recurse=recurse)
        for name, value in saved.items():
            if value is not None:
                self._buffers[name] = value.to(
                    device=self._buffers[name].device, dtype=torch.float32,
                )
        return result

    def _get_sym_kl_beta(self):
        if self.kl is None:
            return None
        if self.sym_kl_beta is None:
            return self.kl.beta
        if self._beta_value is None:
            value = float(self.sym_kl_beta)
            value = value if math.isfinite(value) else self.kl.beta_init
            lo, hi = self.kl.beta_range
            self._beta_value = min(max(value, lo), hi)
            self.sym_kl_beta.fill_(self._beta_value)
        return self._beta_value

    def _apply_aux_loss(self, probs, scores_for_aux_loss, routing_map,
                        with_padding_mask=False, packed_seq_params=None):
        if self.kl is None:
            return super()._apply_aux_loss(
                probs, scores_for_aux_loss, routing_map, with_padding_mask, packed_seq_params,
            )
        coeff = self.get_aux_loss_coeff("aux_loss")
        if coeff == 0:
            return probs
        local_groups = self._get_aux_loss_groups(packed_seq_params)
        global_mode = self.kl.mode == "global"
        if (global_mode and packed_seq_params is not None
                and packed_seq_params.local_cp_size is not None):
            raise ValueError("Global KL Aux does not yet support runtime Dynamic CP")
        groups = self.tp_dp_cp_group if global_mode else local_groups.loss_reduce_groups
        counts, local_tokens, total = get_tokens_per_expert_and_token_count(
            routing_map, groups, topk=self.topk,
            # Count actual valid assignments, including unequal/padded token populations.
            with_padding_mask=True,
        )
        total = torch.as_tensor(total, device=counts.device, dtype=torch.float32)
        feedback = counts
        if global_mode:
            with torch.no_grad():
                self._kl_counts.add_(counts)
                self._kl_tokens.add_(total)
                self._kl_microbatches.add_(1)
                if self.kl.global_mode == "previous_step":
                    current = counts / (total.clamp_min(1) * self.topk)
                    load = torch.where(self.previous_global_aux_valid,
                                       self.previous_global_aux_load, current)
                    feedback = load * total * self.topk
                elif self.kl.global_mode == "legacy_prefix":
                    feedback = self._kl_counts / self._kl_microbatches
        loss, metric = symmetric_kl_load_balancing_loss_func(
            scores_for_aux_loss, feedback, total, self.topk, self.num_experts, coeff,
            min_load_rho=self.kl.rho, coeff_clip=self.kl.coeff_clip,
            beta=self._get_sym_kl_beta(), multiplicative_ste=self.kl.multiplicative_ste,
            tolerance=self.kl.tolerance,
        )
        metric = torch.where(total > 0, metric, 0.)
        loss = torch.where(total > 0, loss, 0.)
        with torch.no_grad():
            self._kl_metric_sum.add_(metric * local_tokens)
            self._kl_metric_tokens.add_(local_tokens)

        local_size = math.prod(group.size() for group in local_groups.loss_reduce_groups)
        domain_size = self.tp_dp_cp_group.size() if global_mode else local_size
        # DP averages gradients in ordinary loss mode. Per-token mode instead sums valid
        # token losses and normalizes in official finalize; do not compensate DP twice.
        scale = (
            domain_size / local_size
            if global_mode and not self.calculate_per_token_loss else 1.
        )
        proxy = loss * scale
        logged_forward = coeff * scale * metric / domain_size
        attached = (proxy - proxy.detach()) + logged_forward
        return self.attach_and_log_load_balancing_loss(
            probs, coeff * scale, attached, "load_balancing_loss",
            self.tp_dp_cp_group if global_mode else local_groups.metric_reduce_group,
            avg_group=None if global_mode else local_groups.metric_avg_group,
            needs_dp_avg=False if global_mode else local_groups.metric_needs_dp_avg,
            aux_loss_logging_reduce_groups=(
                None if global_mode else local_groups.metric_pre_reduce_groups
            ),
            aux_loss_scale_num_tokens=total,
        )

    def _apply_expert_bias(self, routing_map, padding_mask=None):
        super()._apply_expert_bias(routing_map, padding_mask=padding_mask)
        if (self.kl is not None and self.kl.dynamic_beta and self.training and torch.is_grad_enabled()
                and self.get_aux_loss_coeff("aux_loss") > 0):
            with torch.no_grad():
                if padding_mask is not None:
                    routing_map = routing_map & (~padding_mask.flatten()).unsqueeze(-1)
                self._kl_actual_counts.add_(routing_map.sum(0))

    @torch.no_grad()
    def finish_kl_step(self):
        """Publish only complete steps; reset never commits partially collected counts."""
        if self.kl is None:
            return
        if self.kl.mode == "global":
            valid = self._kl_tokens > 0
            load = self._kl_counts / (self._kl_tokens.clamp_min(1) * self.topk)
            self.previous_global_aux_load.copy_(torch.where(
                valid, load, self.previous_global_aux_load
            ))
            self.previous_global_aux_valid.logical_or_(valid)
        if self.kl.dynamic_beta:
            counts = self._kl_actual_counts.clone()
            if self.tp_dp_cp_group.size() > 1:
                dist.all_reduce(counts, group=self.tp_dp_cp_group)
            population, gap = torch.stack((
                counts.sum(), counts.max() + counts.min() - 2 * counts.mean(),
            )).cpu().tolist()
            current = self._get_sym_kl_beta()
            if population > 0:
                delta = (self.kl.beta_update_rate if gap > self.kl.beta_deadband else
                         -self.kl.beta_update_rate if gap < -self.kl.beta_deadband else 0.)
                lo, hi = self.kl.beta_range
                current = min(max(current + delta, lo), hi)
            self.sym_kl_beta.fill_(current)
            self._beta_value = current
        self._kl_last_metric.copy_(torch.stack((self._kl_metric_sum, self._kl_metric_tokens)))
        self.reset_kl_step()

    @torch.no_grad()
    def reset_kl_step(self):
        if self.kl is None:
            return
        for name in ("_kl_counts", "_kl_tokens", "_kl_microbatches", "_kl_actual_counts",
                     "_kl_metric_sum", "_kl_metric_tokens"):
            getattr(self, name).zero_()

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._beta_value = None
        if self.kl is None:
            return
        self.reset_kl_step()
        self.previous_global_aux_load.zero_()
        self.previous_global_aux_valid.fill_(False)
        self._kl_last_metric.zero_()
