"""Full native GPT forward/backward EAR comparison, optionally including Megatron DDP.

Run with torchrun --module experiments.xcx.tests.benchmark_ear_reps. Compilation and
strict output/input/parameter-gradient checks precede timing; optimizer/data I/O are excluded.
"""

import argparse
import gc
import json
import os
import statistics
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from experiments.xcx.losses.kl_aux import KL_BALANCING_TYPES, KLAuxConfig
from experiments.xcx.models.ear_reps import EARRepsRuntime
from experiments.xcx.models.gpt import XCXModelConfig, configure_ear_model
from experiments.xcx.routers.ear import REPS_MODES, EARConfig
from experiments.xcx.training.kl_aux import kl_routers
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.models.gpt import GPTModelBuilder


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for key, default in (("layers", 8), ("hidden", 256), ("ffn", 256), ("experts", 8),
                         ("tokens", 256), ("batch", 1), ("warmup", 5), ("iterations", 15),
                         ("rounds", 4), ("accumulation", 2)):
        p.add_argument("--" + key, type=int, default=default)
    p.add_argument("--modes", nargs="+", choices=REPS_MODES, default=list(REPS_MODES))
    p.add_argument("--hybrid", action="store_true")
    p.add_argument("--per-expert-probe", action="store_true")
    p.add_argument("--channel-beta", action="store_true")
    p.add_argument("--static-only", action="store_true")
    p.add_argument("--moe-router-load-balancing-type", choices=("aux_loss", *KL_BALANCING_TYPES),
                   default="aux_loss")
    p.add_argument("--orthogonal-coeff", type=float, default=0.)
    p.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
    p.add_argument("--ddp", action="store_true")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--output", type=Path, default=Path("/tmp/xcx-ear-reps-benchmark.json"))
    args = p.parse_args()
    if any(value < 1 for value in vars(args).values() if type(value) is int):
        p.error("numeric arguments must be positive")
    return args


def main(args):
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=600))
    parallel_state.initialize_model_parallel(expert_model_parallel_size=world,
                                              expert_tensor_parallel_size=1)
    groups = ProcessGroupCollection.use_mpu_process_groups()
    reps_group = dist.new_group(list(range(world)), backend="nccl")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    torch.manual_seed(1234)
    model_parallel_cuda_manual_seed(1234)
    cfg = TransformerConfig(
        num_layers=args.layers, hidden_size=args.hidden, num_attention_heads=4,
        ffn_hidden_size=args.ffn, moe_ffn_hidden_size=args.ffn,
        num_moe_experts=args.experts, moe_router_topk=2, moe_aux_loss_coeff=.01,
        expert_model_parallel_size=world, expert_tensor_parallel_size=1,
        moe_token_dispatcher_type="alltoall", gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        add_bias_linear=False, gradient_accumulation_fusion=False,
        masked_softmax_fusion=False, bias_activation_fusion=False,
        persist_layer_norm=False, attention_dropout=0., hidden_dropout=0.,
        params_dtype=dtype, bf16=args.dtype == "bf16", use_cpu_initialization=True,
        transformer_impl="local",
    )
    model_cfg = XCXModelConfig(transformer=cfg, vocab_size=256, seq_length=args.tokens,
                                share_embeddings_and_output_weights=False)
    ear = EARConfig(add_static=args.hybrid, per_expert_probe=args.per_expert_probe,
                    channel_beta=args.channel_beta, lb_static_only=args.static_only,
                    orthogonal_coeff=args.orthogonal_coeff)
    kl_mode = KL_BALANCING_TYPES.get(args.moe_router_load_balancing_type)
    configure_ear_model(model_cfg, ear, KLAuxConfig(mode=kl_mode) if kl_mode else None)
    native = GPTModelBuilder(model_cfg).build_model(groups, pre_process=False, post_process=True)
    native = native.to(device=device, dtype=dtype).train()
    model = (DistributedDataParallel(cfg, DistributedDataParallelConfig(
        grad_reduce_in_fp32=True, overlap_grad_reduce=False), native, pg_collection=groups)
             if args.ddp else native)
    routers = [layer.mlp.router for layer in native.decoder.layers]
    input_generator = torch.Generator(device=device).manual_seed(4321 + rank)
    x = torch.randn(args.tokens, args.batch, args.hidden, device=device, dtype=dtype,
                     generator=input_generator,
                     requires_grad=True)
    mask = torch.ones(1, 1, args.tokens, args.tokens, dtype=torch.bool, device=device).triu(1)
    MoEAuxLossAutoScaler.set_loss_scale(torch.ones((), device=device))
    initial = {name: param.detach().cpu().clone() for name, param in native.named_parameters()}

    def step(accumulation=1, snapshot=False):
        model.zero_grad(set_to_none=True)
        if args.ddp:
            model.zero_grad_buffer()
        x.grad = None
        get_moe_metrics_tracker().clear()
        for router in routers:
            router.reset_global_aux_loss_tracker()
        for _ in range(accumulation):
            native.set_input_tensor(x)
            output = model(None, None, mask, decoder_input=x)
            output.float().square().mean().backward()
        if args.ddp:
            finalize_model_grads([model], pg_collection=groups)
        for router in kl_routers([model]):
            router.finish_kl_step()
        if snapshot:
            torch.cuda.synchronize()
            values = {"output": output.detach().cpu(), "input_grad": x.grad.detach().cpu()}
            for name, param in native.named_parameters():
                grad = param.main_grad if args.ddp else param.grad
                if grad is not None:
                    values[name] = grad.detach().cpu().clone()
            assert all(torch.isfinite(value).all() for value in values.values())
            return values

    def install(mode):
        for router in routers:
            router.ear = replace(ear, reps_mode=mode)
        runtime_mode = "reference" if mode in ("reference", "local_pack", "local_fused") else mode
        runtime = EARRepsRuntime(mode=runtime_mode, group=reps_group)
        runtime.install([model])
        return runtime

    def compare(reference, actual, mode, phase):
        error = None
        try:
            assert reference.keys() == actual.keys()
            for key in reference:
                torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0,
                                            msg=lambda msg: f"{mode}/{phase}/{key}: {msg}")
        except (AssertionError, RuntimeError) as exc:
            error = str(exc)
        errors = [None] * world
        dist.all_gather_object(errors, error)
        if any(errors):
            raise AssertionError("\n".join(str(e) for e in errors if e))

    checks, samples = [], []
    try:
        for phase, accumulation in (("initial", 1), ("repeat", 1), ("accumulation", 3),
                                     ("after_parameter_update", 1)):
            if phase == "after_parameter_update":
                with torch.no_grad():
                    for param in native.parameters():
                        param.mul_(1.01)
            with install("reference"):
                reference = step(accumulation, snapshot=True)
            for mode in args.modes:
                with install(mode):
                    actual = step(accumulation, snapshot=True)
                compare(reference, actual, mode, phase)
                checks.append({"mode": mode, "phase": phase, "tensors": len(actual), "exact": True})
                if rank == 0:
                    print(f"CHECK {mode} {phase}: {len(actual)} tensors exact", flush=True)
            del reference, actual
        with torch.no_grad():
            for name, param in native.named_parameters():
                param.copy_(initial[name])
        if not args.check_only:
            for round_id in range(args.rounds):
                order = args.modes if round_id % 2 == 0 else list(reversed(args.modes))
                for mode in order:
                    with install(mode):
                        for _ in range(args.warmup):
                            step(args.accumulation)
                        torch.cuda.synchronize()
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        baseline = torch.cuda.memory_allocated()
                        for iteration in range(args.iterations):
                            dist.barrier()
                            torch.cuda.synchronize()
                            start = time.perf_counter()
                            step(args.accumulation)
                            torch.cuda.synchronize()
                            elapsed = 1000 * (time.perf_counter() - start)
                            local = [elapsed, torch.cuda.max_memory_allocated(),
                                     torch.cuda.max_memory_reserved(), baseline]
                            ranks = [None] * world
                            dist.all_gather_object(ranks, local)
                            samples.append({"round": round_id, "mode": mode,
                                            "iteration": iteration, "ranks": ranks,
                                            "max_ms": max(row[0] for row in ranks)})
                    if rank == 0:
                        print(f"TIMED round={round_id} mode={mode}", flush=True)
        if rank == 0:
            summary = {}
            for mode in args.modes:
                rows = [row for row in samples if row["mode"] == mode]
                if not rows:
                    continue
                times = [row["max_ms"] for row in rows]
                median = statistics.median(times)
                summary[mode] = {
                    "median_ms": median, "mean_ms": statistics.mean(times),
                    "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0.,
                    "tokens_per_second": args.tokens * args.batch * world * args.accumulation * 1000 / median,
                    "max_peak_allocated_mib": max(r[1] for row in rows for r in row["ranks"]) / 2**20,
                    "max_peak_reserved_mib": max(r[2] for row in rows for r in row["ranks"]) / 2**20,
                }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "world_size": world,
                "cpu_affinity": sorted(os.sched_getaffinity(0)), "checks": checks,
                "samples": samples, "summary": summary,
            }, indent=2) + "\n")
            print(json.dumps(summary, indent=2), flush=True)
    finally:
        dist.destroy_process_group(reps_group)
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main(parse_args())
