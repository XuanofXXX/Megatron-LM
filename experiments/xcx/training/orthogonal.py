"""Post-optimizer EAR regularization using official callbacks and master parameters."""

import torch
import torch.distributed as dist

from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker
from megatron.core.utils import get_attr_wrapped_model
from megatron.training import global_vars
from megatron.training.callbacks import Callback

from ..models.moe import expert_fc1_parameters
from ..routers.ear import EARRouter


def ear_routers(model):
    return [module for chunk in model
            for module in get_attr_wrapped_model(chunk, "modules")()
            if isinstance(module, EARRouter)]


class EAROrthogonalUpdate(Callback):
    """One SGD-style regularization update after successful ordinary optimizer steps.

    The normal optimizer owns moments, master weights and LR scheduling. This callback
    only updates its existing parameter tensors; it owns no checkpoint state.
    """

    def on_train_start(self, context):
        config = getattr(context.optimizer, "config", None)
        if any(getattr(config, name, False) for name in (
            "use_distributed_optimizer", "use_precision_aware_optimizer",
            "use_layer_wise_distributed_optimizer",
        )):
            raise ValueError("Decoupled EAR updates require ordinary replicated optimizer params")
        owned = {id(p) for group in context.optimizer.param_groups for p in group["params"]}
        for router in ear_routers(context.model):
            if router.ear.orthogonal_coeff and router.ear.orthogonal_update_mode == "decoupled":
                params = list(router.parameters()) + list(expert_fc1_parameters(router._experts_ref()))
                for parameter in params:
                    if not parameter.requires_grad:
                        continue
                    if getattr(parameter, "main_param_sharded", False):
                        raise ValueError("Decoupled EAR updates require replicated optimizer params")
                    master = getattr(parameter, "main_param", parameter)
                    if id(master) not in owned or master.shape != parameter.shape:
                        raise ValueError("EAR requires full parameters in the optimizer groups")

    def on_train_step_start(self, context):
        # on_train_step_end runs after the scheduler; retain the LR actually used by Adam.
        self._groups = {id(param): float(group["lr"])
                        for group in context.optimizer.param_groups for param in group["params"]}

    @torch.no_grad()
    def on_train_step_end(self, context):
        if context.skipped_iter:
            return
        for router in ear_routers(context.model):
            options = router.ear
            if (not options.orthogonal_coeff or options.orthogonal_update_mode != "decoupled"
                    or options.grad_mode == "stop_w_gate"):
                continue
            experts = router._experts_ref()
            if experts is None:
                raise RuntimeError("Decoupled EAR requires a live expert provider")
            params = []
            if router.probe_vector.requires_grad:
                params.append(router.probe_vector)
            if options.grad_mode != "probe_only":
                params.extend(p for p in expert_fc1_parameters(experts) if p.requires_grad)
            if router.channel_beta is not None and router.channel_beta.requires_grad:
                params.append(router.channel_beta)
            if not params:
                continue
            entries = []
            for param in params:
                master = getattr(param, "main_param", param)
                if getattr(param, "main_param_sharded", False) or master.shape != param.shape:
                    raise ValueError("Decoupled EAR requires full optimizer master parameters")
                lr = self._groups.get(id(master))
                if lr is None:
                    raise ValueError("EAR parameter is absent from the optimizer parameter groups")
                entries.append((master, lr))
            with torch.enable_grad():
                loss = router.orthogonal_loss(router.fresh_orthogonal_rows())
                loss = loss * (options.orthogonal_coeff / router.tp_cp_group.size())
                grads = torch.autograd.grad(loss, params, allow_unused=True)
            scale = router.expert_group.size()
            for param, grad, (master, lr) in zip(params, grads, entries):
                if grad is None:
                    continue
                # Gather backward sums replicated objectives across EP; shared probe/beta
                # also need the contributions of all disjoint local expert sets.
                if param is router.probe_vector or param is router.channel_beta:
                    if scale > 1:
                        dist.all_reduce(grad, group=router.expert_group)
                grad = grad / scale
                master.add_(grad.to(master), alpha=-lr)
                if master is not param:
                    param.copy_(master)


class EAROrthogonalLogger(Callback):
    """Read the native tracker before its regular report clears unselected metrics."""

    def __init__(self, interval):
        self.interval = interval
        self._step_offset = 0

    def on_train_start(self, context):
        self._step_offset = global_vars.get_args().iteration - global_vars.get_train_state().iteration

    def on_train_step_end(self, context):
        step = global_vars.get_train_state().iteration + self._step_offset + 1
        if step % self.interval:
            return
        entry = get_moe_metrics_tracker().metrics.get("router_orthogonal_loss")
        local = {}
        if entry is not None:
            for router in ear_routers(context.model):
                if router.ear.orthogonal_coeff:
                    local[router.layer_number] = float(entry.values[router.layer_number - 1]) / get_num_microbatches()
        world = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        packets = [None] * world if rank == world - 1 else None
        if world > 1:
            dist.gather_object(local, packets, dst=world - 1)
        else:
            packets = [local]
        if packets is None:
            return
        layers = {}
        for packet in packets:
            for layer, value in packet.items():
                layers.setdefault(layer, []).append(value)
        values = {f"router_orthogonal_loss/layer_{layer - 1}": sum(v) / len(v)
                  for layer, v in layers.items()}
        if values:
            values["router_orthogonal_loss"] = sum(values.values()) / len(values)
        tb, wandb = global_vars.get_tensorboard_writer(), global_vars.get_wandb_writer()
        if tb is not None:
            for tag, value in values.items():
                tb.add_scalar(tag, value, step)
        if wandb is not None and values:
            wandb.log(values, step)
