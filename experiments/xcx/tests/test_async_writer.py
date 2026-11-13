"""Check observable writer behavior without contacting external logging services."""

import threading

import numpy as np
import pytest
import torch

from experiments.xcx.logging.async_writer import AsyncSummaryWriter, AsyncWandbWriter
from experiments.xcx.logging.writers import async_log_writers
from megatron.training import global_vars


class RecordingWriter:
    def __init__(self):
        self.calls = []
        self.config = {}

    def add_scalar(self, *args, **kwargs):
        self.calls.append(("scalar", args, kwargs))

    def add_text(self, *args, **kwargs):
        self.calls.append(("text", args, kwargs))

    def add_histogram(self, *args, **kwargs):
        self.calls.append(("histogram", args, kwargs))

    def log(self, *args, **kwargs):
        self.calls.append(("log", args, kwargs))

    def flush(self):
        self.calls.append(("flush", (), {}))

    def close(self):
        self.calls.append(("close", (), {}))

    def finish(self):
        self.calls.append(("finish", (), {}))


def test_enqueue_returns_while_writer_is_busy_and_close_drains(monkeypatch):
    backend = RecordingWriter()
    started, release = threading.Event(), threading.Event()

    def blocking_text(*args, **kwargs):
        started.set()
        assert release.wait(5), "test did not release the writer"
        backend.calls.append(("text", args, kwargs))

    monkeypatch.setattr(backend, "add_text", blocking_text)
    writer = AsyncSummaryWriter(backend, max_queue=1)
    try:
        writer.add_text("config", "test")
        assert started.wait(5)
        value = torch.tensor(3.0, requires_grad=True)
        writer.add_scalar("loss", value, 7)
        with torch.no_grad():
            value.fill_(99)
        assert not backend.calls
    finally:
        release.set()
        writer.close()
    assert [call[0] for call in backend.calls] == ["text", "scalar", "close"]
    captured = backend.calls[1][1][1]
    assert captured.item() == 3
    assert captured.device.type == "cpu" and not captured.requires_grad


def test_wandb_snapshots_nested_mutable_data():
    backend = RecordingWriter()
    writer = AsyncWandbWriter(backend)
    value = torch.tensor(3.0, requires_grad=True)
    data = {"loss": value, "array": np.array([1, 2]), "nested": [1, {"a": 2}]}
    writer.log(data, step=7, commit=False)
    with torch.no_grad():
        value.fill_(99)
    data["array"][0] = 99
    data["nested"][1]["a"] = 99
    data["new"] = 99
    writer.finish()
    writer.finish()
    captured = backend.calls[0][1][0]
    assert captured["loss"].item() == 3
    assert not captured["loss"].requires_grad
    np.testing.assert_array_equal(captured["array"], [1, 2])
    assert captured["nested"] == [1, {"a": 2}]
    assert "new" not in captured
    assert backend.calls[0][2] == {"step": 7, "commit": False}
    assert [call[0] for call in backend.calls] == ["log", "finish"]
    assert writer.config is backend.config


def test_flush_and_other_methods_preserve_order():
    backend = RecordingWriter()
    writer = AsyncSummaryWriter(backend)
    try:
        writer.add_scalar("a", 1, 1)
        writer.flush()
        assert [call[0] for call in backend.calls] == ["scalar", "flush"]
        writer.add_scalar("b", 2, 2)
        writer.add_histogram("hist", [1, 2], 2)
        assert [call[0] for call in backend.calls] == [
            "scalar", "flush", "scalar", "histogram"
        ]
    finally:
        writer.close()
    writer.close()
    assert [call[0] for call in backend.calls].count("close") == 1
    with pytest.raises(ValueError, match="closed"):
        writer.add_scalar("late", 0, 3)


def test_background_failure_is_raised_and_backend_still_closes(monkeypatch):
    backend = RecordingWriter()

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(backend, "add_scalar", fail)
    writer = AsyncSummaryWriter(backend)
    writer.add_scalar("loss", 1, 1)
    with pytest.raises(RuntimeError, match="Async log write failed") as error:
        writer.flush()
    assert isinstance(error.value.__cause__, OSError)
    with pytest.raises(RuntimeError, match="Async log write failed"):
        writer.close()
    assert [call[0] for call in backend.calls] == ["close"]
    writer.close()


@pytest.mark.parametrize("max_queue", [0, -1])
def test_queue_must_be_bounded(max_queue):
    with pytest.raises(ValueError, match="positive"):
        AsyncSummaryWriter(RecordingWriter(), max_queue=max_queue)


@pytest.mark.parametrize("failure", [None, RuntimeError, SystemExit])
def test_training_scope_restores_writers_even_on_exit(monkeypatch, failure):
    tb, wandb = RecordingWriter(), RecordingWriter()
    monkeypatch.setattr(global_vars, "_GLOBAL_TENSORBOARD_WRITER", tb)
    monkeypatch.setattr(global_vars, "_GLOBAL_WANDB_WRITER", wandb)

    def run():
        with async_log_writers(True):
            wrapped_tb = global_vars.get_tensorboard_writer()
            wrapped_wandb = global_vars.get_wandb_writer()
            assert isinstance(wrapped_tb, AsyncSummaryWriter)
            assert isinstance(wrapped_wandb, AsyncWandbWriter)
            wrapped_tb.add_scalar("loss", 1, 1)
            wrapped_wandb.log({"loss": 1}, 1)
            wrapped_wandb.finish()  # Upstream finishes WandB before returning.
            if failure is not None:
                raise failure("training exit")

    if failure is None:
        run()
    else:
        with pytest.raises(failure, match="training exit"):
            run()
    assert global_vars.get_tensorboard_writer() is tb
    assert global_vars.get_wandb_writer() is wandb
    assert [call[0] for call in tb.calls] == ["scalar", "close"]
    assert [call[0] for call in wandb.calls] == ["log", "finish"]


def test_scope_cleanup_continues_after_writer_failure(monkeypatch):
    tb, wandb = RecordingWriter(), RecordingWriter()
    monkeypatch.setattr(global_vars, "_GLOBAL_TENSORBOARD_WRITER", tb)
    monkeypatch.setattr(global_vars, "_GLOBAL_WANDB_WRITER", wandb)

    def fail(*args, **kwargs):
        raise OSError("wandb write failed")

    monkeypatch.setattr(wandb, "log", fail)
    with pytest.raises(RuntimeError, match="Async log write failed"):
        with async_log_writers(True):
            global_vars.get_tensorboard_writer().add_scalar("loss", 1, 1)
            global_vars.get_wandb_writer().log({"loss": 1}, 1)
    assert global_vars.get_tensorboard_writer() is tb
    assert global_vars.get_wandb_writer() is wandb
    assert tb.calls[-1][0] == "close"
    assert wandb.calls[-1][0] == "finish"


@pytest.mark.parametrize("enabled,has_writer", [(False, True), (True, False)])
def test_disabled_or_missing_writers_are_unchanged(monkeypatch, enabled, has_writer):
    backend = RecordingWriter() if has_writer else None
    monkeypatch.setattr(global_vars, "_GLOBAL_TENSORBOARD_WRITER", backend)
    monkeypatch.setattr(global_vars, "_GLOBAL_WANDB_WRITER", None)
    with async_log_writers(enabled):
        assert global_vars.get_tensorboard_writer() is backend
        assert global_vars.get_wandb_writer() is None
    if has_writer:
        assert not backend.calls


def test_real_tensorboard_events(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter

    writer = AsyncSummaryWriter(SummaryWriter(log_dir=tmp_path))
    try:
        for step in range(4):
            writer.add_scalar("loss", torch.tensor(step / 2), step)
        writer.add_text("config", "test", 0)
        writer.flush()
    finally:
        writer.close()
    events = EventAccumulator(str(tmp_path)).Reload()
    assert [(event.step, event.value) for event in events.Scalars("loss")] == [
        (0, 0), (1, 0.5), (2, 1), (3, 1.5)
    ]
    assert "config/text_summary" in events.Tags()["tensors"]
