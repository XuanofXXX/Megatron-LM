"""Launch recipes preserve native arguments, paths and distributed launch settings."""

import argparse
import json
import os
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
        "with open(os.environ['CAPTURE_ARGS'], 'w') as f: json.dump(sys.argv[1:], f)\n"
    )
    python.chmod(0o755)

    def run(script, *args, **updates):
        env = dict(os.environ, PYTHON=str(python), CAPTURE_ARGS=str(capture),
                   DATA_PATH=str(tmp_path / "data with spaces"),
                   TOKENIZER_MODEL_PATH=str(tmp_path / "tokenizer with spaces"),
                   OUTPUT_DIR=str(tmp_path / "output with spaces"), NPROC_PER_NODE="2", NNODES="1")
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
def test_presets_use_native_parser_and_preserve_paths(launch, tmp_path, script, ear, kl):
    result, argv = launch(script)
    assert result.returncode == 0, result.stderr
    args = parse_training_args(argv)
    assert args.bf16 and args.fp8 is None
    assert args.data_path == [str(tmp_path / "data with spaces")]
    assert args.tokenizer_model == str(tmp_path / "tokenizer with spaces")
    assert args.save == str(tmp_path / "output with spaces" / "checkpoints")
    assert (ear_config_from_args(args) is not None) == ear
    options = kl_config_from_args(args)
    assert (options is not None) == kl
    if kl:
        assert options.mode == "global" and options.beta == .7 and options.rho == 0


def test_native_cli_overrides_defaults_and_enables_aux_free(launch):
    result, argv = launch(
        "pretrain_earh.sh", "--moe-router-grad-mode", "probe_only",
        "--moe-router-load-balancing-type", "none", "--moe-aux-loss-coeff", "0",
        "--moe-router-score-function", "sigmoid", "--moe-router-enable-expert-bias",
        "--data-path", "0.7", "/first prefix", "0.3", "/second prefix",
    )
    assert result.returncode == 0, result.stderr
    args = parse_training_args(argv)
    assert args.data_path == ["0.7", "/first prefix", "0.3", "/second prefix"]
    assert args.moe_router_load_balancing_type == ["none"] and args.moe_aux_loss_coeff == [0.]
    assert args.moe_router_enable_expert_bias and args.moe_router_score_function == "sigmoid"
    assert ear_config_from_args(args).grad_mode == "probe_only"
    assert kl_config_from_args(args) is None


def test_native_mock_data_and_tokenizer_cli_need_no_path_environment(launch):
    result, argv = launch("pretrain_moe.sh", "--mock-data", "--tokenizer-type", "NullTokenizer",
                          "--vocab-size", "256", DATA_PATH=None, TOKENIZER_MODEL_PATH=None)
    assert result.returncode == 0, result.stderr
    args = parse_training_args(argv)
    assert args.mock_data and args.tokenizer_type == "NullTokenizer" and args.tokenizer_model is None


def test_per_split_data_paths_do_not_receive_an_incompatible_split(launch):
    result, argv = launch("pretrain_moe.sh", "--train-data-path", "/train prefix",
                          "--valid-data-path", "/valid prefix", DATA_PATH=None)
    assert result.returncode == 0, result.stderr
    args = parse_training_args(argv)
    assert args.split is None and args.data_path is None
    assert args.train_data_path == ["/train prefix"] and args.valid_data_path == ["/valid prefix"]


def test_multinode_launcher_uses_shared_rendezvous(launch):
    result, argv = launch("pretrain_moe.sh", NNODES="2", NODE_RANK="1",
                          MASTER_ADDR="10.0.0.1", MASTER_PORT="6123")
    assert result.returncode == 0, result.stderr
    assert "--standalone" not in argv
    assert argv[argv.index("--node-rank") + 1] == "1"
    assert argv[argv.index("--master-addr") + 1] == "10.0.0.1"
    assert argv[argv.index("--master-port") + 1] == "6123"


@pytest.mark.parametrize("missing", ["MASTER_ADDR", "NODE_RANK"])
def test_multinode_requires_rendezvous_before_launch(launch, missing):
    settings = dict(NNODES="2", NODE_RANK="1", MASTER_ADDR="10.0.0.1")
    settings[missing] = None
    result, argv = launch("pretrain_moe.sh", **settings)
    assert result.returncode != 0 and missing in result.stderr
    assert argv is None


def test_training_template_requires_explicit_data(launch):
    result, argv = launch("pretrain_moe.sh", DATA_PATH=None)
    assert result.returncode == 2 and "DATA_PATH" in result.stderr
    assert argv is None
