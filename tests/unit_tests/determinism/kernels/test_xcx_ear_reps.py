"""Replay the opt-in EAR packing/projection graphs, including every source gradient."""

import pytest
import torch

from experiments.xcx.routers.ear_compiled_reps import (
    compiled_local_rows,
    compiled_stack_rows,
    pack_gate_weights,
)
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("separate", [False, True])
def test_local_pack_and_reps_replay(dtype, separate):
    seeded()
    weights = tuple(torch.randn(256, 192, device="cuda", dtype=dtype, requires_grad=True)
                    for _ in range(4))
    probe = torch.randn((4, 128) if separate else (128,), device="cuda", dtype=dtype,
                         requires_grad=True)
    assert_replays_bit_exact(lambda *w: pack_gate_weights(w, 128), weights,
                             contention=True, what="XCX EAR gate packing")
    assert_replays_bit_exact(lambda p, *w: compiled_local_rows(p, w, 128, separate),
                             (probe, *weights), contention=True, what="XCX EAR local reps")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_stack_reps_replay(dtype):
    seeded()
    weights = tuple(torch.randn(256, 192, device="cuda", dtype=dtype, requires_grad=True)
                    for _ in range(8))
    probes = (torch.randn(128, device="cuda", dtype=dtype, requires_grad=True),
              torch.randn(4, 128, device="cuda", dtype=dtype, requires_grad=True))
    assert_replays_bit_exact(
        lambda p, q, *w: compiled_stack_rows((p, q), (w[:4], w[4:]), 128, (False, True)),
        (*probes, *weights), contention=True, what="XCX EAR stack reps",
    )
