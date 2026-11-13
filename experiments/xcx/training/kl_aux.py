"""Step-finalization and logging for KL Aux; the stock training loop remains in charge."""

import torch.distributed as dist

from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.utils import get_attr_wrapped_model
from megatron.training import global_vars
from megatron.training.callbacks import Callback

from ..routers.kl_aux import KLAuxRouter


def kl_routers(model):
    return [module for chunk in model
            for module in get_attr_wrapped_model(chunk, "modules")()
            if isinstance(module, KLAuxRouter) and module.kl is not None]


def finalize_kl_model_grads(model, num_tokens=None, pg_collection=None, *, base=None, **kwargs):
    """Run official gradient synchronization/normalization before committing KL state."""
    result = (base or finalize_model_grads)(
        model, num_tokens, pg_collection=pg_collection, **kwargs,
    )
    for router in kl_routers(model):
        router.finish_kl_step()
    return result


class KLMetricsLogger(Callback):
    """Expose KL diagnostics and post-update per-layer beta through existing writers."""

    def __init__(self, options, interval: int) -> None:
        self.options = options
        self.interval = interval
        self._step_offset = 0

    def on_train_start(self, context):
        self._step_offset = (
            global_vars.get_args().iteration - global_vars.get_train_state().iteration
        )

    def on_train_step_end(self, context):
        step = global_vars.get_train_state().iteration + self._step_offset + 1
        if step % self.interval:
            return
        local = {
            router.layer_number: (*router._kl_last_metric.detach().cpu().tolist(),
                                  router._get_sym_kl_beta())
            for router in kl_routers(context.model)
        }
        world = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        packets = [None] * world if rank == world - 1 else None
        if world > 1:
            # Dense PP stages participate with an empty packet as well.
            dist.gather_object(local, packets, dst=world - 1)
        else:
            packets = [local]
        if packets is None:
            return
        layers = {}
        for packet in packets:
            for layer, (metric_sum, tokens, beta) in packet.items():
                item = layers.setdefault(layer, [0., 0., 0., 0])
                for index, value in enumerate((metric_sum, tokens, beta, 1)):
                    item[index] += value
        name = ("global_sym_kl_load_balancing_loss" if self.options.mode == "global"
                else "sym_kl_load_balancing_loss")
        values, divergences, betas = {}, [], []
        for layer, (metric_sum, tokens, beta_sum, replicas) in layers.items():
            beta = beta_sum / replicas
            betas.append(beta)
            values[f"sym_kl_beta/layer_{layer - 1}"] = beta
            if tokens > 0:
                divergence = metric_sum / tokens
                divergences.append(divergence)
                values[f"{name}/layer_{layer - 1}"] = divergence
        if betas:
            values["sym_kl_beta"] = sum(betas) / len(betas)
        if divergences:
            values[name] = sum(divergences) / len(divergences)
        tb, wandb = global_vars.get_tensorboard_writer(), global_vars.get_wandb_writer()
        if tb is not None:
            for key, value in values.items():
                tb.add_scalar(key, value, step)
        if wandb is not None and values:
            wandb.log(values, step)
