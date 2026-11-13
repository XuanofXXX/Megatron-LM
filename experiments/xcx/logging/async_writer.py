"""Ordered background writes with CPU snapshots and explicit shutdown."""

import copy
import logging
import queue
import threading
from functools import partial

import torch

logger = logging.getLogger(__name__)


def _snapshot(value):
    """Detach tensors and copy mutable payloads before the producer reuses them."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    return copy.deepcopy(value)


class _AsyncWriter:
    """Serialize calls through one worker and bound the number of pending writes."""

    def __init__(self, writer, max_queue: int = 1024) -> None:
        if max_queue < 1:
            raise ValueError("max_queue must be positive")
        self._writer = writer
        self._queue = queue.Queue(maxsize=max_queue)
        self._lock = threading.Lock()
        self._closed = False
        self._error = None
        self._thread = threading.Thread(
            target=self._worker, name=f"xcx-{type(self).__name__}", daemon=True
        )
        self._thread.start()

    def _worker(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                func, args, kwargs = item
                try:
                    func(*args, **kwargs)
                except Exception as exc:
                    if self._error is None:
                        self._error = exc
                    logger.exception("Async log write failed")
            finally:
                self._queue.task_done()

    def _check_error(self):
        if self._error is not None:
            raise RuntimeError("Async log write failed") from self._error

    def _check_open(self):
        if self._closed:
            raise ValueError("Async writer is closed")
        self._check_error()

    def _enqueue(self, func, *args, **kwargs):
        with self._lock:
            self._check_open()
            self._queue.put((func, _snapshot(args), _snapshot(kwargs)))

    def _call(self, func, *args, **kwargs):
        # Less common writer methods stay synchronous, after earlier queued calls.
        with self._lock:
            self._check_open()
            self._queue.join()
            self._check_error()
            return func(*args, **kwargs)

    def _close(self, finalizer):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(None)
            self._thread.join()
            try:
                finalizer()
            finally:
                self._check_error()

    def __getattr__(self, name):
        value = getattr(self._writer, name)
        return partial(self._call, value) if callable(value) else value


class AsyncSummaryWriter(_AsyncWriter):
    """Queue TensorBoard scalars/text; flush and close wait for pending writes."""

    def add_scalar(self, *args, **kwargs):
        self._enqueue(self._writer.add_scalar, *args, **kwargs)

    def add_text(self, *args, **kwargs):
        self._enqueue(self._writer.add_text, *args, **kwargs)

    def flush(self):
        self._call(self._writer.flush)

    def close(self):
        self._close(self._writer.close)


class AsyncWandbWriter(_AsyncWriter):
    """Queue WandB log calls; finish drains the queue and finishes the run once."""

    def log(self, *args, **kwargs):
        self._enqueue(self._writer.log, *args, **kwargs)

    def finish(self):
        self._close(self._writer.finish)
