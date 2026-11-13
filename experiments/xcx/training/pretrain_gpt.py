"""Run XCX GPT experiments with Megatron's data pipeline and training loop."""

import time
from contextlib import nullcontext

import pretrain_gpt as gpt
from experiments.xcx.logging.logger import BasicMetricsLogger, add_metric_args
from experiments.xcx.logging.writers import async_log_writers
from experiments.xcx.losses.kl_aux import add_kl_args, kl_config_from_args, kl_model_args
from experiments.xcx.models.gpt import (
    KLModelConfig,
    XCXModelConfig,
    configure_ear_model,
    configure_kl_model,
)
from experiments.xcx.routers.ear import add_ear_args, ear_config_from_args
from experiments.xcx.training.kl_aux import KLMetricsLogger
from experiments.xcx.training.orthogonal import EAROrthogonalLogger, EAROrthogonalUpdate
from megatron.training.global_vars import set_args


def add_args(parser):
    """Keep upstream optional arguments and register experiment-only switches."""
    if gpt.has_nvidia_modelopt:
        parser = gpt.add_modelopt_args(parser)
    group = parser.add_argument_group("XCX logging")
    group.add_argument(
        "--async-log-writer",
        action="store_true",
        help="Write TensorBoard scalars/text and WandB logs in background threads.",
    )
    return add_ear_args(add_kl_args(add_metric_args(parser)))


def main() -> None:
    """Prepare the upstream GPT configuration and start training."""
    gpt.set_startup_timestamps(main_entry=time.time())
    gpt.print_rank_0(f"> PyTorch version ................ {gpt.get_torch_version()}")
    gpt.print_rank_0(f"> Megatron-Core version .......... {gpt.mcore_version}")
    gpt.print_rank_0(f"> Transformer Engine version ... {gpt.get_te_version()}")

    gpt.train_valid_test_datasets_provider.is_distributed = True
    args = gpt.parse_and_validate_args(
        extra_args_provider=add_args,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
    pretrain, store = gpt.inprocess_restart.maybe_wrap_for_inprocess_restart(gpt.pretrain, args=args)
    if gpt.has_nvidia_modelopt:
        gpt.maybe_enable_modelopt(args)
    metrics = BasicMetricsLogger(args)
    kl = kl_config_from_args(args)
    ear = ear_config_from_args(args)
    # Native data helpers also reconstruct TransformerConfig from the runtime args.
    # Keep their supported Aux hook consistent with model construction.
    args = kl_model_args(args, kl)
    set_args(args)
    if gpt.has_nvidia_modelopt and getattr(args, "modelopt_enabled", False):
        if kl is not None or ear is not None:
            raise ValueError("EAR/KL Aux currently use the stock GPT builder without ModelOpt")
        model_config = gpt.gpt_config_from_args(
            args, model_config_cls=gpt.ModelOptModelConfig, vocab_size_from_tokenizer=True
        )
    else:
        model_config = gpt.gpt_config_from_args(
            args, vocab_size_from_tokenizer=True,
            **({"model_config_cls": XCXModelConfig} if ear is not None else
               {"model_config_cls": KLModelConfig} if kl is not None else {}),
        )
    callbacks = [metrics] if metrics.enabled else []
    ear_runtime = nullcontext()
    if ear is not None:
        ear_runtime = configure_ear_model(model_config, ear, kl)
        if ear.orthogonal_coeff and ear.orthogonal_update_mode == "decoupled":
            callbacks.append(EAROrthogonalUpdate())
        if ear.orthogonal_coeff:
            callbacks.append(EAROrthogonalLogger(metrics.interval))
    elif kl is not None:
        configure_kl_model(model_config, kl)
    if kl is not None:
        callbacks.append(KLMetricsLogger(kl, metrics.interval))

    config = gpt.pretrain_cfg_container_from_args(args, model_config)
    gpt.set_run_config(config)
    gpt.initialize_runtime_services(args, training=True)
    with async_log_writers(args.async_log_writer), metrics, ear_runtime:
        gpt.resolve_tokenizer_vocab_size(config, args.padded_vocab_size)
        pretrain(
            config,
            gpt.train_valid_test_datasets_provider,
            gpt.ModelType.encoder_or_decoder,
            gpt.forward_step,
            store=store,
            get_embedding_ranks=gpt.get_embedding_ranks,
            callbacks=callbacks or None,
        )


if __name__ == "__main__":
    main()
