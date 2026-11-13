"""Copied YuLan recipes retain their launch interface on the native argument parser."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.xcx.losses.kl_aux import kl_config_from_args
from experiments.xcx.routers.ear import ear_config_from_args
from experiments.xcx.training.pretrain_gpt import add_args
from megatron.training.arguments import add_megatron_arguments

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def launch(tmp_path):
    capture = tmp_path / "argv.json"
    python = tmp_path / "capture-python"
    python.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "if sys.argv[1:3] != ['-m', 'torch.distributed.run']:\n"
        f"    os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
        "with open(os.environ['CAPTURE_ARGS'], 'w') as f: json.dump(sys.argv[1:], f)\n"
        "sys.exit(int(os.environ.get('CAPTURE_EXIT', '0')))\n"
    )
    python.chmod(0o755)

    def run(script="pretrain_moe.sh", *args, **updates):
        env = dict(os.environ, PYTHON=str(python), CAPTURE_ARGS=str(capture),
                   DATA_PATH=str(tmp_path / "data with spaces"),
                   TOKENIZER_MODEL_PATH=str(tmp_path / "tokenizer with spaces"),
                   OUTPUT_CHECKPOINT_PATH=str(tmp_path / "output with spaces"),
                   GPUS_PER_NODE="2", NNODES="1", TASK_UUID="recipe-test",
                   SEQ_LEN="32", BATCH_SIZE="1", GLOBAL_BATCH_SIZE="2",
                   TRAIN_TOKENS="1_024", LR_WARMUP_TOKENS="64", LR_DECAY_TOKENS="1_024",
                   SAVE_TOKENS="128", EVAL_TOKENS="256")
        for key, value in updates.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        capture.unlink(missing_ok=True)
        result = subprocess.run(["bash", str(CONFIGS / script), *args], cwd=tmp_path,
                                env=env, capture_output=True, text=True)
        return result, json.loads(capture.read_text()) if capture.exists() else None

    return run


def parse_training_args(argv):
    assert argv[:2] == ["-m", "torch.distributed.run"]
    module = argv.index("--module")
    assert argv[module + 1] == "experiments.xcx.training.pretrain_gpt"
    parser = add_args(add_megatron_arguments(argparse.ArgumentParser(allow_abbrev=False)))
    return parser.parse_args(argv[module + 2:])


@pytest.mark.parametrize("script,ear,kl", [
    ("pretrain_moe.sh", False, False),
    ("pretrain_earh.sh", True, False),
    ("pretrain_kl_aux.sh", False, True),
])
def test_presets_use_native_parser_and_original_paths(launch, tmp_path, script, ear, kl):
    result, argv = launch(script)
    assert result.returncode == 0, result.stderr
    args = parse_training_args(argv)
    assert args.bf16 and args.fp8 is None
    assert args.data_path == [str(tmp_path / "data with spaces")]
    assert args.tokenizer_model == str(tmp_path / "tokenizer with spaces")
    assert Path(args.save).parent == tmp_path / "output with spaces" / "checkpoint"
    assert Path(args.tensorboard_dir).parent == tmp_path / "output with spaces" / "log"
    assert (Path(args.save) / script).is_file()
    assert (ear_config_from_args(args) is not None) == ear
    options = kl_config_from_args(args)
    assert (options is not None) == kl
    if kl:
        assert options.mode == "global" and options.beta == .7 and options.rho == 0
    assert args.train_samples == 32 and args.lr_decay_samples == 32
    assert args.lr_warmup_samples == 2 and args.save_interval == 2 and args.eval_interval == 4
    assert args.micro_batch_size == 1 and not args.use_distributed_optimizer


def test_original_environment_enables_earh_aux_free(launch):
    result, argv = launch("pretrain_earh.sh", LOAD_BALANCING="dsv3",
                          MOE_ROUTER_GRAD_MODE="probe_only", MP_SIZE="2")
    assert result.returncode == 0, result.stderr
    args = parse_training_args(argv)
    assert args.moe_router_load_balancing_type == ["none"] and args.moe_aux_loss_coeff == [0.]
    assert args.moe_router_enable_expert_bias and args.moe_router_score_function == "sigmoid"
    assert args.tensor_model_parallel_size == 2
    assert ear_config_from_args(args).grad_mode == "probe_only"
    assert kl_config_from_args(args) is None


def test_extra_args_preserve_quoting_and_cli_has_last_word(launch):
    extra = shlex.join(["--data-path", "0.7", "/first prefix", "0.3", "/second prefix",
                       "--moe-router-grad-mode", "gate_only"])
    result, argv = launch("pretrain_earh.sh", "--moe-router-grad-mode", "probe_only", EXTRA_ARGS=extra)
    assert result.returncode == 0, result.stderr
    args = parse_training_args(argv)
    assert args.data_path == ["0.7", "/first prefix", "0.3", "/second prefix"]
    assert ear_config_from_args(args).grad_mode == "probe_only"


def test_original_mock_tokenizer_setting(launch):
    result, argv = launch(TOKENIZER_TYPE="mock", DATA_PATH=None, TOKENIZER_MODEL_PATH=None)
    assert result.returncode == 0, result.stderr
    args = parse_training_args(argv)
    assert args.mock_data and args.tokenizer_type == "NullTokenizer" and args.vocab_size == 256


def test_original_per_split_paths(launch):
    result, argv = launch(TRAIN_DATA_PATH="/train prefix", VALID_DATA_PATH="/valid prefix", DATA_PATH=None)
    assert result.returncode == 0, result.stderr
    args = parse_training_args(argv)
    assert args.split is None and args.data_path is None
    assert args.train_data_path == ["/train prefix"] and args.valid_data_path == ["/valid prefix"]


def test_multinode_uses_original_variables(launch):
    result, argv = launch(NNODES="2", NODE_RANK="1", MASTER_ADDR="10.0.0.1", MASTER_PORT="6123")
    assert result.returncode == 0, result.stderr
    assert argv[argv.index("--nproc_per_node") + 1] == "2"
    assert argv[argv.index("--node_rank") + 1] == "1"
    assert argv[argv.index("--master_addr") + 1] == "10.0.0.1"
    assert argv[argv.index("--master_port") + 1] == "6123"


@pytest.mark.parametrize("missing", ["MASTER_ADDR", "NODE_RANK", "TASK_UUID"])
def test_multinode_requires_shared_launch_settings(launch, missing):
    settings = dict(NNODES="2", NODE_RANK="1", MASTER_ADDR="10.0.0.1")
    settings[missing] = None
    result, argv = launch(**settings)
    assert result.returncode != 0 and missing in result.stderr
    assert argv is None


def test_auto_resume_keeps_name_when_token_budget_changes(launch):
    result, argv = launch()
    assert result.returncode == 0, result.stderr
    checkpoint = Path(parse_training_args(argv).save)
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("2\n")
    result, argv = launch(TRAIN_TOKENS="2_048", TASK_UUID="resumed")
    assert result.returncode == 0, result.stderr
    args = parse_training_args(argv)
    assert args.save == args.load == str(checkpoint)
    assert args.train_samples == 64 and args.exit_on_missing_checkpoint and not args.perform_initialization


def test_training_failure_propagates_through_tee(launch):
    result, _ = launch(CAPTURE_EXIT="7")
    assert result.returncode == 7


@pytest.mark.parametrize("settings", [
    {"LOSS_MASK_ENABLE": "true"}, {"MOE_ROUTER_AUX_LOSS_PROXY": "logits"},
    {"MOE_USE_ROUTING_FREE_ROUTER": "true"},
    {"LOAD_BALANCING": "global_aux_loss", "MOE_ROUTER_GLOBAL_AUX_MODE": "previous_step"},
])
def test_unmigrated_behavior_is_not_silently_ignored(launch, settings):
    result, argv = launch(**settings)
    assert result.returncode != 0 and argv is None
    assert "ERROR" in result.stderr
