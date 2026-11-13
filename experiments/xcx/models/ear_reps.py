"""One-forward EAR schedules installed through official post-wrap and module hooks."""

from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch.profiler import record_function

from .moe import EARMoELayer


class _GatherRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local, group, previous):
        ctx.group, ctx.shape = group, local.shape
        ctx.has_previous = previous is not None
        ctx.dtype, ctx.device = local.dtype, local.device
        if group.size() == 1:
            rows = local.clone()
        else:
            rows = local.new_empty((local.shape[0] * group.size(), *local.shape[1:]))
            with record_function("ear/reps_all_gather"):
                work = dist.all_gather_into_tensor(rows, local.contiguous(), group=group,
                                                   async_op=True)
                work.wait()  # Orders the current stream, including side-stream preparation.
        ctx.set_materialize_grads(False)
        # This dependency enforces reverse layer order for backward collectives on all ranks.
        return rows, local.new_empty((0,))

    @staticmethod
    def backward(ctx, grad_rows, grad_token):
        if grad_rows is None:
            grad_rows = torch.zeros((ctx.shape[0] * ctx.group.size(), *ctx.shape[1:]),
                                     dtype=ctx.dtype, device=ctx.device)
        if ctx.group.size() == 1:
            local = grad_rows
        else:
            local = grad_rows.new_empty(ctx.shape)
            with record_function("ear/reps_reduce_scatter"):
                work = dist.reduce_scatter_tensor(local, grad_rows.contiguous(), group=ctx.group,
                                                   async_op=True)
                work.wait()
        return local, None, local.new_empty((0,)) if ctx.has_previous else None


@dataclass
class _PendingRows:
    rows: torch.Tensor
    ready: torch.cuda.Event | None = None

    def consume(self):
        if self.ready is not None:
            stream = torch.cuda.current_stream(self.rows.device)
            stream.wait_event(self.ready)
            self.rows.record_stream(stream)
        return self.rows


class _EARRepsSchedule:
    def __init__(self, block, layers, mode, group):
        self.layers = layers
        self.routers = [layer.mlp.router for layer in layers]
        self.mode, self.group = mode, group
        self.active, self.token = False, None
        self.handles = []
        self.stream = None
        if mode in ("prefetch", "compiled_stack_async"):
            device = self.routers[0].probe_vector.device
            if device.type != "cuda":
                raise ValueError("EAR prefetch requires CUDA")
            self.stream = torch.cuda.Stream(device=device)
        shapes = {(r._num_local_experts, r.config.moe_ffn_hidden_size, r.config.hidden_size)
                  for r in self.routers}
        if len(shapes) != 1:
            raise ValueError("Scheduled EAR requires uniform expert shapes within a block")
        for router in self.routers:
            if router._reps_gather is not None:
                raise RuntimeError("EAR schedule already installed")
            if mode != "reference" and (
                router.ear.grad_mode != "end2end" or router.config.tensor_model_parallel_size != 1
                or router.config.pipeline_model_parallel_size != 1
                or router.config.recompute_granularity is not None
            ):
                raise ValueError("Scheduled EAR reps require end2end, TP=PP=1 and no recompute")
            if dist.get_process_group_ranks(group) != dist.get_process_group_ranks(router.expert_group):
                raise ValueError("EAR reps communicator must have the same ordered EP ranks")
            if mode != "reference" and group is router.expert_group:
                raise ValueError("Use a dedicated EAR reps group, separate from token dispatch")
        for router in self.routers:
            router._reps_gather = self.gather
        self.handles.append(block.register_forward_pre_hook(self.begin))
        self.handles.append(block.register_forward_hook(self.finish, always_call=True))
        if mode == "prefetch":
            for i, layer in enumerate(layers[:-1]):
                self.handles.append(layer.register_forward_pre_hook(
                    lambda module, args, index=i + 1: self.prepare(index)))

    def gather(self, local):
        if self.mode == "reference":
            return (torch.cat(dist_nn.all_gather(local, group=self.group))
                    if self.group.size() > 1 else local)
        rows, self.token = _GatherRows.apply(local, self.group, self.token)
        return rows

    def begin(self, module, args):
        if self.active:
            raise RuntimeError("Nested EAR reps scopes are unsupported")
        self.active, self.token = True, None
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream(self.stream.device))
        try:
            if self.mode == "prefetch":
                self.prepare(0)
            elif self.mode in ("batched", "compiled_stack", "compiled_stack_async"):
                with torch.cuda.stream(self.stream) if self.stream is not None else nullcontext():
                    self.prepare_batch()
        except BaseException:
            self.finish()
            raise

    def _publish(self, routers, rows):
        ready = None
        if self.stream is not None:
            ready = torch.cuda.Event()
            ready.record(self.stream)
        for router, row in zip(routers, rows):
            if router._pending_rows is not None:
                raise RuntimeError("EAR rows already pending")
            router._pending_rows = _PendingRows(row, ready)

    def prepare(self, index):
        if not self.active:
            raise RuntimeError("EAR preparation outside its forward scope")
        router = self.routers[index]
        with torch.cuda.stream(self.stream), record_function("ear/prepare_next_reps"):
            self._publish([router], [self.gather(router.local_rows())])

    def prepare_batch(self):
        with record_function("ear/prepare_batch_reps"):
            if self.mode in ("compiled_stack", "compiled_stack_async"):
                from ..routers.ear_compiled_reps import compiled_stack_rows

                pairs = [r._sources_and_probe() for r in self.routers]
                sources, probes = zip(*pairs)
                local = compiled_stack_rows(tuple(probes), tuple(sources),
                                             self.routers[0].config.moe_ffn_hidden_size,
                                             tuple(r.ear.per_expert_probe for r in self.routers))
            else:
                local = torch.cat([r.local_rows() for r in self.routers])
            gathered = self.gather(local)
            count = len(self.routers)
            n = self.routers[0]._num_local_experts
            # Collective is rank-major, while native layers consume global expert rows.
            rows = gathered.view(self.group.size(), count, n, -1).transpose(0, 1)
            rows = rows.reshape(count, n * self.group.size(), -1)
            self._publish(self.routers, rows.unbind(0))

    def finish(self, *args):
        for router in self.routers:
            router._pending_rows = None
        self.active, self.token = False, None

    def close(self):
        self.finish()
        if self.stream is not None:
            self.stream.synchronize()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for router in self.routers:
            router._reps_gather = None


class EARRepsRuntime:
    """Own hooks/streams/groups; returned by configure_ear_model for scoped cleanup."""

    def __init__(self, *, mode=None, group=None):
        self.mode, self.group = mode, group
        self.schedules, self.owned_groups = [], []

    def install(self, models):
        try:
            for model in models:
                while hasattr(model, "module"):
                    ddp = getattr(model, "ddp_config", None)
                    if ddp is not None and (ddp.overlap_grad_reduce or ddp.overlap_param_gather):
                        raise ValueError("Scheduled EAR reps require disabling DDP overlap")
                    model = model.module
                block = model.decoder
                layers = [layer for layer in block.layers if isinstance(layer.mlp, EARMoELayer)]
                if not layers:
                    continue
                ep = layers[0].mlp.router.expert_group
                group = self.group
                if group is None:
                    # EP groups partition the world; only their members create the duplicate.
                    group = dist.new_group(dist.get_process_group_ranks(ep),
                                           backend=dist.get_backend(ep),
                                           use_local_synchronization=True)
                    self.owned_groups.append(group)
                mode = self.mode or layers[0].mlp.router.ear.reps_mode
                self.schedules.append(_EARRepsSchedule(block, layers, mode, group))
        except BaseException:
            self.close()
            raise
        return models

    def close(self):
        for schedule in reversed(self.schedules):
            schedule.close()
        self.schedules.clear()
        for group in reversed(self.owned_groups):
            dist.destroy_process_group(group)
        self.owned_groups.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
