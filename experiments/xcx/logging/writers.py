"""Install async writer adapters for the lifetime of the experiment entry point."""

from contextlib import ExitStack, contextmanager

from megatron.training import global_vars

from .async_writer import AsyncSummaryWriter, AsyncWandbWriter


@contextmanager
def async_log_writers(enabled: bool):
    """Wrap initialized writers and always close them and restore the originals."""
    with ExitStack() as stack:
        if enabled:
            for get_writer, set_writer, wrapper, close_method in (
                (
                    global_vars.get_tensorboard_writer,
                    global_vars.set_tensorboard_writer,
                    AsyncSummaryWriter,
                    "close",
                ),
                (
                    global_vars.get_wandb_writer,
                    global_vars.set_wandb_writer,
                    AsyncWandbWriter,
                    "finish",
                ),
            ):
                writer = get_writer()
                if writer is not None:
                    adapted = wrapper(writer)
                    stack.callback(set_writer, writer)
                    stack.callback(getattr(adapted, close_method))
                    set_writer(adapted)
        yield
