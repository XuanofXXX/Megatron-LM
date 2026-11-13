#!/bin/bash
# Copied from YuLan-Pretrain/scripts/pretrain/moe_pretrain_template.sh.
# Keep the original environment variables, token budgets, naming and resume flow.
# Adapted for Megatron Core 0.20 and the migrated XCX GPT features.
# This script processes parameters and builds training command
# All parameters should be exported from the launch script

set -o pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export TOKENIZERS_PARALLELISM=false
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1

get_script_from_pid() {
    local pid="$1"
    local arg
    local cwd

    cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)

    [[ -r "/proc/$pid/cmdline" ]] || return 1

    while IFS= read -r -d '' arg; do
        [[ "$arg" == *.sh ]] || continue

        if [[ "$arg" = /* ]]; then
            if [[ -f "$arg" ]]; then
                readlink -f "$arg"
                return 0
            fi
        else
            if [[ -n "$cwd" && -f "$cwd/$arg" ]]; then
                readlink -f "$cwd/$arg"
                return 0
            fi
        fi
    done < "/proc/$pid/cmdline"

    return 1
}

backup_caller_scripts() {
    local backup_root="$1"
    local backup_dir="$backup_root/script_backup_$(date +%Y%m%d_%H%M%S)"
    local chain_file="$backup_dir/script_chain.txt"
    local pid
    local ppid
    local script
    local src
    local dst
    local scripts=()

    mkdir -p "$backup_dir"

    pid=$$
    while [[ "$pid" -ne 1 ]]; do
        script=$(get_script_from_pid "$pid" 2>/dev/null || true)
        if [[ -n "${script:-}" ]]; then
            scripts+=("$script")
        fi

        ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
        [[ -z "$ppid" ]] && break
        pid="$ppid"
    done

    {
        echo "script chain:"
        for script in "${scripts[@]}"; do
            echo "$script"
        done
        echo
        echo "backup caller scripts:"
    } > "$chain_file"

    for i in "${!scripts[@]}"; do
        # scripts[0] is this template script and is copied separately below.
        [[ "$i" -eq 0 ]] && continue

        src="${scripts[$i]}"
        dst="$backup_dir/${i}_$(basename "$src")"

        cp -- "$src" "$dst"
        echo "$src -> $dst" >> "$chain_file"
    done

    echo -e "\033[32mCaller script backup: ${backup_dir}\033[0m"
}

# ============================================================================
# Parameter Processing
# ============================================================================

# Path arguments (required from launch script)
DATA_PATH=${DATA_PATH}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-""}
VALID_DATA_PATH=${VALID_DATA_PATH:-""}
OUTPUT_CHECKPOINT_PATH=${OUTPUT_CHECKPOINT_PATH:-"${repo_root}/runs/xcx"}
TOKENIZER_MODEL_PATH=${TOKENIZER_MODEL_PATH}
DATA_PATH_CACHE=${DATA_PATH_CACHE:-""}
PRETRAIN_CHECKPOINT_PATH=${PRETRAIN_CHECKPOINT_PATH:-"none"}

# Training arguments (required from launch script)
SEQ_LEN=${SEQ_LEN:-2048}
BATCH_SIZE=${BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
MP_SIZE=${MP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}
ACTIVATION_CHECKPOINT=${ACTIVATION_CHECKPOINT:-false}
LOG_INTERVAL=${LOG_INTERVAL:-1}
TENSORBOARD_LOG_INTERVAL=${TENSORBOARD_LOG_INTERVAL:-1}
SEED=${SEED:-24}
OPTIMIZER=${OPTIMIZER:-"adam"}

# Learning rate arguments (required from launch script)
LR=${LR:-3e-4}
MIN_LR=${MIN_LR:-3e-5}
LR_DECAY_STYLE=${LR_DECAY_STYLE:-cosine}
TRAIN_TOKENS=${TRAIN_TOKENS:-655_360_000}
LR_WARMUP_TOKENS=${LR_WARMUP_TOKENS:-6_553_600}
LR_DECAY_TOKENS=${LR_DECAY_TOKENS:-${TRAIN_TOKENS}}
SAVE_TOKENS=${SAVE_TOKENS:-65_536_000}

# Model arguments (required from launch script)
MODEL_SIZE=${MODEL_SIZE:-moe}
NUM_LAYERS=${NUM_LAYERS:-8}
HIDDEN_SIZE=${HIDDEN_SIZE:-512}
NUM_ATTN_HEADS=${NUM_ATTN_HEADS:-8}
NUM_QUERY_GROUPS=${NUM_QUERY_GROUPS:-${NUM_ATTN_HEADS}}
ROTARY_BASE=${ROTARY_BASE:-10000}
TIE_EMBEDDING=${TIE_EMBEDDING:-false}
INIT_STD=${INIT_STD:-0.02}
NORM_EPSILON=${NORM_EPSILON:-1e-06}

# Linear attention arguments (required from launch script)
LINEAR_ATTN_TYPE=${LINEAR_ATTN_TYPE}
LINEAR_CONV_KERNEL_DIM=${LINEAR_CONV_KERNEL_DIM}
LINEAR_KEY_HEAD_DIM=${LINEAR_KEY_HEAD_DIM}
LINEAR_VALUE_HEAD_DIM=${LINEAR_VALUE_HEAD_DIM}
LINEAR_NUM_KEY_HEADS=${LINEAR_NUM_KEY_HEADS}
LINEAR_NUM_VALUE_HEADS=${LINEAR_NUM_VALUE_HEADS}
LINEAR_ATTN_FREQ=${LINEAR_ATTN_FREQ}

# Hybrid model arguments (required from launch script)
HYBRID_ATTN=${HYBRID_ATTN}
HYBRID_MLP_RATIO=${HYBRID_MLP_RATIO}

# MoE arguments (optional, defaults for non-MoE)
NUM_EXPERTS=${NUM_EXPERTS:-8}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-2048}
MOE_FFN_HIDDEN_SIZE=${MOE_FFN_HIDDEN_SIZE:-512}
MOE_TOPK=${MOE_TOPK:-2}
NUM_SHARED_EXPERTS=${NUM_SHARED_EXPERTS:-1}
SHARED_EXPERTS_HIDDEN_SIZE=${SHARED_EXPERTS_HIDDEN_SIZE:-0}
LOAD_BALANCING=${LOAD_BALANCING:-"aux_loss"}
MOE_AUX_LOSS_COEFF=${MOE_AUX_LOSS_COEFF:-0.005}
MOE_Z_LOSS_COEFF=${MOE_Z_LOSS_COEFF:-""}
MOE_ROUTER_GLOBAL_AUX_MODE=${MOE_ROUTER_GLOBAL_AUX_MODE:-microbatch}
# Shared KL/Global Aux proxy: scores (default), logits, or topk_boundary (KL only).
MOE_ROUTER_AUX_LOSS_PROXY=${MOE_ROUTER_AUX_LOSS_PROXY:-scores}
MOE_ROUTER_SYM_KL_LOSS_TEMPERATURE=${MOE_ROUTER_SYM_KL_LOSS_TEMPERATURE:-1.0}
# Optional pair, e.g. "0.8 1.2". Empty disables the tolerance interval.
MOE_ROUTER_SYM_KL_LOSS_TOLERANCE=${MOE_ROUTER_SYM_KL_LOSS_TOLERANCE:-}
MOE_ROUTER_SYM_KL_LOSS_RHO=${MOE_ROUTER_SYM_KL_LOSS_RHO:-0.05}
MOE_ROUTER_SYM_KL_LOSS_COEFF_CLIP=${MOE_ROUTER_SYM_KL_LOSS_COEFF_CLIP:-20}
MOE_ROUTER_SYM_KL_LOSS_BETA=${MOE_ROUTER_SYM_KL_LOSS_BETA:-0.5}
MOE_ROUTER_SYM_KL_LOSS_MULTIPLICATIVE_STE=${MOE_ROUTER_SYM_KL_LOSS_MULTIPLICATIVE_STE:-false}
MOE_ROUTER_SYM_KL_LOSS_DYNAMIC_BETA=${MOE_ROUTER_SYM_KL_LOSS_DYNAMIC_BETA:-false}
MOE_ROUTER_SYM_KL_LOSS_BETA_INIT=${MOE_ROUTER_SYM_KL_LOSS_BETA_INIT:-0.5}
MOE_ROUTER_SYM_KL_LOSS_BETA_RANGE=${MOE_ROUTER_SYM_KL_LOSS_BETA_RANGE:-"0.25,0.75"}
MOE_ROUTER_SYM_KL_LOSS_BETA_UPDATE_RATE=${MOE_ROUTER_SYM_KL_LOSS_BETA_UPDATE_RATE:-1e-3}
MOE_ROUTER_SYM_KL_LOSS_BETA_DEADBAND=${MOE_ROUTER_SYM_KL_LOSS_BETA_DEADBAND:-1.0}
MOE_ROUTER_SCORE_FUNCTION=${MOE_ROUTER_SCORE_FUNCTION:-"softmax"}
MOE_EXPERT_CAPACITY_FACTOR=${MOE_EXPERT_CAPACITY_FACTOR:-0}
MOE_ROUTER_BIAS_UPDATE_RATE=${MOE_ROUTER_BIAS_UPDATE_RATE:-1e-3}
MOE_ROUTER_NORM_TOPK_PROB=${MOE_ROUTER_NORM_TOPK_PROB:-true}
MOE_ROUTER_TOPK_WEIGHT_NORMALIZATION=${MOE_ROUTER_TOPK_WEIGHT_NORMALIZATION:-"auto"}
MOE_TOKEN_DISPATCHER_TYPE=${MOE_TOKEN_DISPATCHER_TYPE:-""}
ADD_QKV_BIAS=${ADD_QKV_BIAS:-true}
CLIP_GRAD=${CLIP_GRAD:-0.5}
NUM_WORKERS=${NUM_WORKERS:-${GPUS_PER_NODE:-8}}
DDP_BUCKET_SIZE=${DDP_BUCKET_SIZE:-16777216}
MUON_NUM_NS_STEPS=${MUON_NUM_NS_STEPS:-8}
MUON_SCALE_MODE=${MUON_SCALE_MODE:-"spectral"}
MUON_TP_MODE=${MUON_TP_MODE:-"duplicated"}

# xcx additional MoE features
MOE_EXPERT_OUTPUT_NORM=${MOE_EXPERT_OUTPUT_NORM:-false}
MOE_ROUTER_PER_EXPERT_PROBE=${MOE_ROUTER_PER_EXPERT_PROBE:-false}
MOE_ROUTER_GATE_UP_INTERACTION=${MOE_ROUTER_GATE_UP_INTERACTION:-false}
MOE_ROUTER_EAR_NORMALIZE=${MOE_ROUTER_EAR_NORMALIZE:-false}
MOE_ROUTER_EAR_NORMALIZE_TAU=${MOE_ROUTER_EAR_NORMALIZE_TAU:-1.0}
MOE_ROUTER_GRAD_MODE=${MOE_ROUTER_GRAD_MODE:-"end2end"}

# Dead-neuron monitor: every N iters count neurons with mean |silu(gate)*up| <= eps
MOE_DEAD_NEURON_LOG_INTERVAL=${MOE_DEAD_NEURON_LOG_INTERVAL:-0}
MOE_DEAD_NEURON_EPS=${MOE_DEAD_NEURON_EPS:-1e-6}
MOE_DEAD_NEURON_PER_EXPERT_LOGGING=${MOE_DEAD_NEURON_PER_EXPERT_LOGGING:-false}

MOE_ROUTER_USE_TOPK_SCORES_AS_PROBS=${MOE_ROUTER_USE_TOPK_SCORES_AS_PROBS:-false}
MOE_USE_WEIGHT_AWARE_ROUTER=${MOE_USE_WEIGHT_AWARE_ROUTER:-false}
MOE_USE_EAR_HYBRID=${MOE_USE_EAR_HYBRID:-false}
MOE_USE_SIGMOID_ABS_ROUTER=${MOE_USE_SIGMOID_ABS_ROUTER:-false}
MOE_USE_ROUTERLESS_AOE_ROUTER=${MOE_USE_ROUTERLESS_AOE_ROUTER:-false}
MOE_AOE_FULL_RANK=${MOE_AOE_FULL_RANK:-false}
MOE_AOE_D_LOW=${MOE_AOE_D_LOW:-""}
MOE_USE_ROUTING_FREE_ROUTER=${MOE_USE_ROUTING_FREE_ROUTER:-false}
MOE_ROUTING_FREE_FULL_RANK=${MOE_ROUTING_FREE_FULL_RANK:-false}
MOE_ROUTING_FREE_D_LOW=${MOE_ROUTING_FREE_D_LOW:-""}
MOE_ROUTING_FREE_TARGET_DENSITY=${MOE_ROUTING_FREE_TARGET_DENSITY:-0.25}
MOE_ROUTING_FREE_CALIBRATE_INIT=${MOE_ROUTING_FREE_CALIBRATE_INIT:-false}
MOE_ROUTING_FREE_MAX_ACTIVE_EXPERTS=${MOE_ROUTING_FREE_MAX_ACTIVE_EXPERTS:-""}
MOE_ROUTING_FREE_GLOBAL_DENSITY=${MOE_ROUTING_FREE_GLOBAL_DENSITY:-true}
MOE_ROUTING_FREE_LAMBDA_INIT=${MOE_ROUTING_FREE_LAMBDA_INIT:-1e-10}
MOE_ROUTING_FREE_LAMBDA_STEP=${MOE_ROUTING_FREE_LAMBDA_STEP:-0.02}
MOE_ROUTING_FREE_MU=${MOE_ROUTING_FREE_MU:-0.5}
MOE_ROUTING_FREE_BIAS_INIT=${MOE_ROUTING_FREE_BIAS_INIT:-1e-6}
MOE_ROUTING_FREE_INFERENCE_THRESHOLD=${MOE_ROUTING_FREE_INFERENCE_THRESHOLD:-0.0}
MOE_ROUTER_AOE_CHUNK_SIZE=${MOE_ROUTER_AOE_CHUNK_SIZE:-2048}
MOE_AOE_ENABLE_BIAS=${MOE_AOE_ENABLE_BIAS:-false}
MOE_AOE_BIAS_INIT=${MOE_AOE_BIAS_INIT:-0.0}
MOE_USE_LATENT_MOE=${MOE_USE_LATENT_MOE:-false}
MOE_ROUTER_ORTHOGONAL_LOSS_COEFF=${MOE_ROUTER_ORTHOGONAL_LOSS_COEFF:-""}
MOE_ROUTER_ORTHOGONAL_PREPROCESS=${MOE_ROUTER_ORTHOGONAL_PREPROCESS:-"none"}
MOE_ROUTER_ORTHOGONAL_PCA_RANK=${MOE_ROUTER_ORTHOGONAL_PCA_RANK:-1}
MOE_ERC_LOSS_COEFF=${MOE_ERC_LOSS_COEFF:-""}
MOE_ERC_LOSS_ALPHA=${MOE_ERC_LOSS_ALPHA:-1.0}
MOE_ERC_LOSS_USE_NOISE=${MOE_ERC_LOSS_USE_NOISE:-true}

append_tag () {
  if [ -z "${WANDB_TAGS:-}" ]; then
    WANDB_TAGS="$1"
  else
    WANDB_TAGS="$WANDB_TAGS,$1"
  fi
}
compact_decimal_token () {
  local raw="$1"
  "${PYTHON:-python}" - "$raw" <<'PY'
from decimal import Decimal, InvalidOperation
import sys

raw = sys.argv[1].strip()
if raw == "":
    print("")
    raise SystemExit(0)
try:
    value = Decimal(raw)
except InvalidOperation:
    print(raw)
    raise SystemExit(0)

if value == 0:
    print("0")
    raise SystemExit(0)

sign = "n" if value < 0 else ""
value = abs(value)
if value >= 1:
    plain = format(value.normalize(), "f")
    if "." in plain:
        plain = plain.rstrip("0").rstrip(".")
    plain_text = plain or "0"
    print(f"{sign}{plain_text}")
    raise SystemExit(0)

normalized = value.normalize()
digits = "".join(str(d) for d in normalized.as_tuple().digits) or "0"
exp = normalized.adjusted() - (len(digits) - 1)
digits = digits.rstrip("0") or "0"

if exp < 0:
    print(f"{sign}{digits}m{-exp}")
else:
    plain = format(value.normalize(), "f")
    if "." in plain:
        plain = plain.rstrip("0").rstrip(".")
    plain_text = plain or "0"
    print(f"{sign}{plain_text}")
PY
}

MOE_AUX_LOSS_COEFF_TOKEN=""
[ -n "${MOE_AUX_LOSS_COEFF:-}" ] && MOE_AUX_LOSS_COEFF_TOKEN=$(compact_decimal_token "${MOE_AUX_LOSS_COEFF}")
MOE_Z_LOSS_COEFF_TOKEN=""
[ -n "${MOE_Z_LOSS_COEFF}" ] && MOE_Z_LOSS_COEFF_TOKEN=$(compact_decimal_token "${MOE_Z_LOSS_COEFF}")
# MoE
append_tag "ep${NUM_EXPERTS}topk${MOE_TOPK}ffn${MOE_FFN_HIDDEN_SIZE}"
[[ "${LOAD_BALANCING}" == *global* ]] && append_tag "gstat_${MOE_ROUTER_GLOBAL_AUX_MODE}"


# 其他
[ "$MOE_EXPERT_OUTPUT_NORM" = "true" ] && append_tag expert_output_norm

# Scores
[ "${MOE_ROUTER_USE_TOPK_SCORES_AS_PROBS}" = "true" ] && append_tag raw_topk_probs

# EAR
[ "$MOE_ROUTER_PER_EXPERT_PROBE" = "true" ] && append_tag router_per_expert_probe
[ "${MOE_ROUTER_GATE_UP_INTERACTION}" = "true" ] && append_tag router_gate_up_interaction
[ "${MOE_ROUTER_EAR_NORMALIZE}" = "true" ] && append_tag "router_ear_norm_tau_${MOE_ROUTER_EAR_NORMALIZE_TAU}"
[ "${MOE_USE_WEIGHT_AWARE_ROUTER}" = "true" ] && append_tag "router_grad_${MOE_ROUTER_GRAD_MODE}"
[ "${MOE_USE_WEIGHT_AWARE_ROUTER}" = "true" ] && [ "${MOE_ROUTER_USE_TOPK_SCORES_AS_PROBS}" = "true" ] && append_tag "router_raw_topk_probs"

# AoE
[ "${MOE_USE_ROUTERLESS_AOE_ROUTER}" = "true" ] && append_tag routerless_aoe
[ "${MOE_AOE_FULL_RANK}" = "true" ] && append_tag aoe_full_rank
[ "${MOE_USE_ROUTERLESS_AOE_ROUTER}" = "true" ] && [ -n "${MOE_AOE_D_LOW}" ] && append_tag "aoe_dlow_${MOE_AOE_D_LOW}"
[ "${MOE_USE_ROUTERLESS_AOE_ROUTER}" = "true" ] && [ "${MOE_AOE_ENABLE_BIAS}" = "true" ] && append_tag "aoe_bias_${MOE_AOE_BIAS_INIT}"

# Orthogonal Loss
[ -n "${MOE_ROUTER_ORTHOGONAL_LOSS_COEFF}" ] && append_tag "router_orth_${MOE_ROUTER_ORTHOGONAL_LOSS_COEFF}"
[ -n "${MOE_ROUTER_ORTHOGONAL_LOSS_COEFF}" ] && [ "${MOE_ROUTER_ORTHOGONAL_PREPROCESS}" != "none" ] && append_tag "router_orth_pre_${MOE_ROUTER_ORTHOGONAL_PREPROCESS}"
[ -n "${MOE_ROUTER_ORTHOGONAL_LOSS_COEFF}" ] && [ "${MOE_ROUTER_ORTHOGONAL_PREPROCESS}" = "pca" ] && append_tag "router_orth_pca_${MOE_ROUTER_ORTHOGONAL_PCA_RANK}"

# ERC Loss
[ -n "${MOE_ERC_LOSS_COEFF}" ] && append_tag "erc_${MOE_ERC_LOSS_COEFF}"
[ -n "${MOE_ERC_LOSS_COEFF}" ] && [ "${MOE_ERC_LOSS_ALPHA}" != "1.0" ] && append_tag "erc_alpha_${MOE_ERC_LOSS_ALPHA}"
[ -n "${MOE_ERC_LOSS_COEFF}" ] && [ "${MOE_ERC_LOSS_USE_NOISE}" = "false" ] && append_tag "erc_no_noise"

# Load balancing
[ "${MOE_ROUTER_AUX_LOSS_PROXY}" != "scores" ] && append_tag "auxproxy_${MOE_ROUTER_AUX_LOSS_PROXY}"
[ "${LOAD_BALANCING}" != "none" ] && append_tag "load_balance_${LOAD_BALANCING}"
[ -n "${MOE_Z_LOSS_COEFF_TOKEN}" ] && append_tag "z_loss_${MOE_Z_LOSS_COEFF_TOKEN}"
if [[ " ${LOAD_BALANCING} " == *" sym_kl_aux_loss "* || " ${LOAD_BALANCING} " == *" global_sym_kl_aux_loss "* ]]; then
    [ "${MOE_ROUTER_AUX_LOSS_PROXY}" = "topk_boundary" ] && append_tag "klproxy_t${MOE_ROUTER_SYM_KL_LOSS_TEMPERATURE}"
    [ -n "${MOE_ROUTER_SYM_KL_LOSS_TOLERANCE}" ] && append_tag "kltol_${MOE_ROUTER_SYM_KL_LOSS_TOLERANCE// /to}"
    append_tag "symkl_rho_${MOE_ROUTER_SYM_KL_LOSS_RHO}"
    append_tag "symkl_clip_${MOE_ROUTER_SYM_KL_LOSS_COEFF_CLIP}"
    append_tag "symkl_beta_${MOE_ROUTER_SYM_KL_LOSS_BETA}"
    [ "${MOE_ROUTER_SYM_KL_LOSS_MULTIPLICATIVE_STE}" = "true" ] && append_tag "symkl_mste"
    if [ "${MOE_ROUTER_SYM_KL_LOSS_DYNAMIC_BETA}" = "true" ]; then
        append_tag "symkl_dynamic_beta"
        append_tag "symkl_beta_init_${MOE_ROUTER_SYM_KL_LOSS_BETA_INIT}"
        append_tag "symkl_beta_range_${MOE_ROUTER_SYM_KL_LOSS_BETA_RANGE//,/to}"
        append_tag "symkl_beta_step_${MOE_ROUTER_SYM_KL_LOSS_BETA_UPDATE_RATE}"
        append_tag "symkl_beta_deadband_${MOE_ROUTER_SYM_KL_LOSS_BETA_DEADBAND}"
    fi
fi
[ -n "${MOE_ROUTER_SCORE_FUNCTION}" ] && append_tag "score_${MOE_ROUTER_SCORE_FUNCTION}"
[ "${MOE_ROUTER_TOPK_WEIGHT_NORMALIZATION}" != "auto" ] && append_tag "topknorm_${MOE_ROUTER_TOPK_WEIGHT_NORMALIZATION}"


export WANDB_TAGS

# Loss mask arguments (required from launch script)
LOSS_MASK_ENABLE=${LOSS_MASK_ENABLE:-"false"}

# Ensure EP_SIZE=1 when NUM_EXPERTS=0 (GDN doesn't use MoE)
if [ "${NUM_EXPERTS}" = "0" ]; then
    EP_SIZE=1
fi

# Multi-node arguments (defaults for single node)
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
if [ "${PET_NNODES:-${NNODES:-1}}" -gt 1 ]; then
    : "${MASTER_ADDR:?Set MASTER_ADDR for multi-node training}"
    : "${PET_NODE_RANK:=${NODE_RANK:?Set NODE_RANK on each node}}"
fi
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${PET_NNODES:-${NNODES:-1}}
NODE_RANK=${PET_NODE_RANK:-${NODE_RANK:-0}}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# Tokenizer and naming
TOKENIZER_TYPE=${TOKENIZER_TYPE:-LLamaTokenizer}
NAME_PREFIX=${NAME_PREFIX:-""}
EVAL_ONLY=${EVAL_ONLY:-0}
EXTRA_ARGS=${EXTRA_ARGS:-""}
USE_DDP_OVERLAP=${USE_DDP_OVERLAP:-"false"}

# Unsupported YuLan features must not silently turn into different experiments.
unsupported() {
    echo "ERROR: $1 is not migrated to experiments/xcx yet." >&2
    exit 2
}
for option in LOSS_MASK_ENABLE MOE_EXPERT_OUTPUT_NORM MOE_ROUTER_USE_TOPK_SCORES_AS_PROBS \
    MOE_USE_SIGMOID_ABS_ROUTER MOE_USE_ROUTERLESS_AOE_ROUTER MOE_USE_ROUTING_FREE_ROUTER \
    MOE_USE_LATENT_MOE MOE_SHARED_EXPERTS_GATE; do
    [ "${!option}" = true ] && unsupported "$option=true"
done
for option in HYBRID_ATTN LINEAR_ATTN_TYPE LINEAR_ATTN_FREQ MTP_NUM_LAYERS \
    UPCYCLING_ALGORITHM UPCYCLING_GRANULARITY MOE_ERC_LOSS_COEFF; do
    [ -n "${!option}" ] && unsupported "$option"
done
[ "${MOE_DEAD_NEURON_LOG_INTERVAL}" != 0 ] && unsupported MOE_DEAD_NEURON_LOG_INTERVAL
[ "${MOE_ROUTER_TOPK_WEIGHT_NORMALIZATION}" != auto ] && unsupported MOE_ROUTER_TOPK_WEIGHT_NORMALIZATION
[ "${MOE_ROUTER_AUX_LOSS_PROXY}" != scores ] && unsupported MOE_ROUTER_AUX_LOSS_PROXY
[ "${MOE_ROUTER_SYM_KL_LOSS_TEMPERATURE}" != 1.0 ] && unsupported MOE_ROUTER_SYM_KL_LOSS_TEMPERATURE
[ "${HF_EOD_TOKEN_ID:-auto}" != auto ] && unsupported HF_EOD_TOKEN_ID
[ "${EVAL_ONLY}" != 0 ] && unsupported EVAL_ONLY
# Native global_aux_loss uses prefix feedback. Only KL has the migrated modes.
if [[ " ${LOAD_BALANCING} " == *" global_aux_loss "* ]] && \
   [ "${MOE_ROUTER_GLOBAL_AUX_MODE}" != legacy_prefix ]; then
    echo "ERROR: global_aux_loss currently requires MOE_ROUTER_GLOBAL_AUX_MODE=legacy_prefix; use global_sym_kl_aux_loss for microbatch/previous_step feedback." >&2
    exit 2
fi
TRANSFORMER_IMPL=${TRANSFORMER_IMPL:-local}
EVAL_TOKENS=${EVAL_TOKENS:-${SAVE_TOKENS}}
# Compute sample-based training parameters
TRAIN_SAMPLES=$(( ${TRAIN_TOKENS//_/}  / ${SEQ_LEN} ))
LR_DECAY_SAMPLES=$(( ${LR_DECAY_TOKENS//_/}  / ${SEQ_LEN} ))
LR_WARMUP_SAMPLES=$(( ${LR_WARMUP_TOKENS//_/}  / ${SEQ_LEN} ))
SAVE_INTERVAL=$(( ${SAVE_TOKENS//_/} / ${SEQ_LEN} / ${GLOBAL_BATCH_SIZE} ))
EVAL_INTERVAL=$(( ${EVAL_TOKENS//_/} / ${SEQ_LEN} / ${GLOBAL_BATCH_SIZE} ))
if (( TRAIN_SAMPLES < GLOBAL_BATCH_SIZE || SAVE_INTERVAL < 1 || EVAL_INTERVAL < 1 )); then
    echo "ERROR: token budgets must cover at least one GLOBAL_BATCH_SIZE * SEQ_LEN batch." >&2
    exit 2
fi

# ============================================================================
# Experiment Naming and Checkpoint Setup
# ============================================================================

# Resolve one job ID before creating any training or logging directories.
# An explicit TASK_UUID must be identical on every node of the launch.
if [ -z "${TASK_UUID:-}" ]; then
    if [ "${NNODES}" -eq 1 ]; then
        TASK_UUID=$(date "+%Y.%m.%d-%H.%M.%S") || exit 1
    else
        echo "ERROR: Set the same TASK_UUID on every node of a multi-node launch." >&2
        exit 2
    fi
fi
export TASK_UUID
JOB_ID=${TASK_UUID}

# Build concise experiment name with abbreviations (2-3 letters per parameter)
# Format: gdn{msz}-nl{nl}-hs{hs}-la{ck,kh,vh,nkh,nvh}-ah{ah}qg{qg}-mp{mp}pp{pp}-sl{sl}bs{bs}lr{lr}g{g}
# Abbreviations: nl=num_layers, hs=hidden_size, la=linear_attn, ck=conv_kernel, kh=key_head_dim,
# vh=value_head_dim, nkh=num_key_heads, nvh=num_value_heads, ah=attn_heads, qg=query_groups,
# mp=tensor_parallel, pp=pipeline_parallel, sl=seq_len, bs=batch_size, lr=learning_rate, g=gpus

# Simplify LR values for readability
LR_SHORT=$(echo "${LR}" | sed 's/^0\.//' | sed 's/e-0*/e/g' | sed 's/\.0*e/e/g')
MIN_LR_SHORT=$(echo "${MIN_LR}" | sed 's/^0\.//' | sed 's/e-0*/e/g' | sed 's/\.0*e/e/g')

# Base model architecture
NAME_BASE="${MODEL_SIZE}-nl${NUM_LAYERS}-hs${HIDDEN_SIZE}-sd${SEED}"

# Linear attention architecture (compact format)
if [ -n "${HYBRID_ATTN:-}" ]; then
    NAME_LA="${LINEAR_CONV_KERNEL_DIM}-kh${LINEAR_KEY_HEAD_DIM}-vh${LINEAR_VALUE_HEAD_DIM}-nkh${LINEAR_NUM_KEY_HEADS}-nvh${LINEAR_NUM_VALUE_HEADS}"
else
    NAME_LA=""
fi

# Attention config (compact)
NAME_ATTN="ah${NUM_ATTN_HEADS}qg${NUM_QUERY_GROUPS}${ATTN_PATTERN}"

# mtp config
MTP_NAME="mtp${MTP_NUM_LAYERS}"

# Parallelism config (compact)
NAME_PARALLEL="mp${MP_SIZE}pp${PP_SIZE}"
[ "${NUM_EXPERTS}" != "0" ] && NAME_PARALLEL="${NAME_PARALLEL}ep${EP_SIZE}"
[ "${CP_SIZE}" != "1" ] && NAME_PARALLEL="${NAME_PARALLEL}cp${CP_SIZE}"

# Training config (compact)
# Optimizer取第一个字母就好
NAME_TRAIN="sl${SEQ_LEN}bs${GLOBAL_BATCH_SIZE}lr${LR_SHORT}mlr${MIN_LR_SHORT}"
if [ "$OPTIMIZER" != "adam" ]; then
    NAME_TRAIN="opt${OPTIMIZER:0:1}${NAME_TRAIN}"
fi

# Build final name
if [ "${NUM_EXPERTS}" != "0" ]; then
    # Upcycling
    [ -n "${UPCYCLING_ALGORITHM}" ] && NAME_BASE="${NAME_BASE}-ua${UPCYCLING_ALGORITHM}"
    [ -n "${UPCYCLING_GRANULARITY}" ] && NAME_BASE="${NAME_BASE}-ug${UPCYCLING_GRANULARITY}"

    # Expert output norm
    [ "${MOE_EXPERT_OUTPUT_NORM}" = "true" ] && NAME_BASE="${NAME_BASE}-epn"

    # AoE
    [ "${MOE_USE_ROUTERLESS_AOE_ROUTER}" == "true" ] && NAME_BASE="${NAME_BASE}-d${MOE_AOE_D_LOW}"
    [ "${MOE_USE_ROUTERLESS_AOE_ROUTER}" == "true" ] && [ "${MOE_AOE_ENABLE_BIAS}" = "true" ] && NAME_BASE="${NAME_BASE}-ab${MOE_AOE_BIAS_INIT}"

    # EAR
    [ "${MOE_USE_WEIGHT_AWARE_ROUTER}" == "true" ] && [ "${MOE_ROUTER_PER_EXPERT_PROBE}" = "true" ] && NAME_BASE="${NAME_BASE}-epp"
    [ "${MOE_USE_WEIGHT_AWARE_ROUTER}" == "true" ] && [ "${MOE_ROUTER_GATE_UP_INTERACTION}" = "true" ] && NAME_BASE="${NAME_BASE}-gui"
    [ "${MOE_USE_WEIGHT_AWARE_ROUTER}" == "true" ] && [ "${MOE_ROUTER_EAR_NORMALIZE}" = "true" ] && NAME_BASE="${NAME_BASE}-earn${MOE_ROUTER_EAR_NORMALIZE_TAU}"
    [ "${MOE_USE_WEIGHT_AWARE_ROUTER}" == "true" ] && [ "${MOE_ROUTER_GRAD_MODE}" != "end2end" ] && NAME_BASE="${NAME_BASE}-rg${MOE_ROUTER_GRAD_MODE}"

    # Load Balancing
    [ "${LOAD_BALANCING}" != "none" ] && NAME_BASE="${NAME_BASE}-lb${LOAD_BALANCING:0:1}"
    [ "${MOE_ROUTER_SCORE_FUNCTION}" == "softmax" ] && NAME_BASE="${NAME_BASE}-sm"
    [ "${MOE_ROUTER_SCORE_FUNCTION}" == "sigmoid" ] && NAME_BASE="${NAME_BASE}-sg"
    [ "${MOE_ROUTER_AUX_LOSS_PROXY}" != "scores" ] && NAME_BASE="${NAME_BASE}-logits"
    [ "${LOAD_BALANCING}" == "global_aux_loss" ] && [ -n "$MOE_AUX_LOSS_COEFF_TOKEN" ] && NAME_BASE="${NAME_BASE}-aux${MOE_AUX_LOSS_COEFF_TOKEN}"
    [ "${LOAD_BALANCING}" == "aux_loss" ] && [ -n "$MOE_AUX_LOSS_COEFF_TOKEN" ] && NAME_BASE="${NAME_BASE}-aux${MOE_AUX_LOSS_COEFF_TOKEN}"
    if [[ " ${LOAD_BALANCING} " == *" sym_kl_aux_loss "* || " ${LOAD_BALANCING} " == *" global_sym_kl_aux_loss "* ]]; then
        [ -n "$MOE_AUX_LOSS_COEFF_TOKEN" ] && NAME_BASE="${NAME_BASE}-symkl${MOE_AUX_LOSS_COEFF_TOKEN}"
        NAME_BASE="${NAME_BASE}-r${MOE_ROUTER_SYM_KL_LOSS_RHO}c${MOE_ROUTER_SYM_KL_LOSS_COEFF_CLIP}b${MOE_ROUTER_SYM_KL_LOSS_BETA}"
        # [ "${MOE_ROUTER_AUX_LOSS_PROXY}" = "topk_boundary" ] && NAME_BASE="${NAME_BASE}-t${MOE_ROUTER_SYM_KL_LOSS_TEMPERATURE}"
        # if [ -n "${MOE_ROUTER_SYM_KL_LOSS_TOLERANCE}" ]; then
        #     NAME_BASE="${NAME_BASE}-tol${MOE_ROUTER_SYM_KL_LOSS_TOLERANCE// /to}"
        # fi
        if [ "${MOE_ROUTER_SYM_KL_LOSS_DYNAMIC_BETA}" = "true" ]; then
            NAME_BASE="${NAME_BASE}-d"
            # NAME_BASE="${NAME_BASE}-dynb-bi${MOE_ROUTER_SYM_KL_LOSS_BETA_INIT}"
            # NAME_BASE="${NAME_BASE}br${MOE_ROUTER_SYM_KL_LOSS_BETA_RANGE//,/to}"
            # NAME_BASE="${NAME_BASE}bu${MOE_ROUTER_SYM_KL_LOSS_BETA_UPDATE_RATE}bd${MOE_ROUTER_SYM_KL_LOSS_BETA_DEADBAND}"
        fi
    fi
    [ "${LOAD_BALANCING}" == "global_centered_aux_loss" ] && [ -n "$MOE_AUX_LOSS_COEFF_TOKEN" ] && NAME_BASE="${NAME_BASE}-cen${MOE_AUX_LOSS_COEFF_TOKEN}"
    [ -n "${MOE_Z_LOSS_COEFF_TOKEN}" ] && NAME_BASE="${NAME_BASE}-z${MOE_Z_LOSS_COEFF_TOKEN}"

    # Orthogonal Loss
    [ -n "${MOE_ROUTER_ORTHOGONAL_LOSS_COEFF}" ] && NAME_BASE="${NAME_BASE}-orth${MOE_ROUTER_ORTHOGONAL_LOSS_COEFF}"
    [ -n "${MOE_ROUTER_ORTHOGONAL_LOSS_COEFF}" ] && [ "${MOE_ROUTER_ORTHOGONAL_PREPROCESS}" = "mean_center" ] && NAME_BASE="${NAME_BASE}-orthmc"
    [ -n "${MOE_ROUTER_ORTHOGONAL_LOSS_COEFF}" ] && [ "${MOE_ROUTER_ORTHOGONAL_PREPROCESS}" = "pca" ] && NAME_BASE="${NAME_BASE}-orthpca${MOE_ROUTER_ORTHOGONAL_PCA_RANK}"

    # ERC Loss
    [ -n "${MOE_ERC_LOSS_COEFF}" ] && NAME_BASE="${NAME_BASE}-erc${MOE_ERC_LOSS_COEFF}"
    [ -n "${MOE_ERC_LOSS_COEFF}" ] && [ "${MOE_ERC_LOSS_ALPHA}" != "1.0" ] && NAME_BASE="${NAME_BASE}-erca${MOE_ERC_LOSS_ALPHA}"

    # if [[ "${LOAD_BALANCING}" == *global* ]]; then
    #     NAME_BASE="${NAME_BASE}-gstat${MOE_ROUTER_GLOBAL_AUX_MODE}"
    # fi
    # MoE Basic Config
    NAME="${NAME_PREFIX}${NAME_BASE}-ep${NUM_EXPERTS}t${MOE_TOPK}hd${MOE_FFN_HIDDEN_SIZE}-${NAME_PARALLEL}-${NAME_TRAIN}"
else
    NAME="${NAME_PREFIX}${NAME_BASE}-dff${FFN_HIDDEN_SIZE}-${NAME_PARALLEL}-${NAME_TRAIN}"
fi

WANDB_RUN_GROUP_RAW="$(echo "$NAME" | sed 's/-sd[^-]*//')"
if [ ${#WANDB_RUN_GROUP_RAW} -gt 120 ]; then
    WANDB_RUN_GROUP_HASH=$(printf '%s' "$WANDB_RUN_GROUP_RAW" | cksum | awk '{print $1}')
    export WANDB_RUN_GROUP="${WANDB_RUN_GROUP_RAW:0:108}-g${WANDB_RUN_GROUP_HASH}"
else
    export WANDB_RUN_GROUP="$WANDB_RUN_GROUP_RAW"
fi

CHECKPOINT_PATH="${OUTPUT_CHECKPOINT_PATH}/checkpoint/${NAME}"

LOG_DIR="${OUTPUT_CHECKPOINT_PATH}/log/${JOB_ID}_${NAME}g${WORLD_SIZE}"
RANK_LOG=${LOG_DIR}/LOG_NODE_RANK_${NODE_RANK}.log
mkdir -p "${CHECKPOINT_PATH}"
mkdir -p "${LOG_DIR}"
[ ! -d "$LOG_DIR/checkpoint" ] && ln -s "../../checkpoint/${NAME}" "$LOG_DIR/checkpoint"
echo "$JOB_ID" >> "$CHECKPOINT_PATH/linked_runs.txt"
cp "$0" "${LOG_DIR}/"
backup_caller_scripts "${LOG_DIR}"

# Check continual-pretrain or from-scratch
LATEST_CHECKPOINT_ITERATION_FILE="$CHECKPOINT_PATH/latest_checkpointed_iteration.txt"
if [ -f "$LATEST_CHECKPOINT_ITERATION_FILE" ]; then
    LATEST_CHECKPOINT_ITERATION=$(tr -d '[:space:]' < "$LATEST_CHECKPOINT_ITERATION_FILE")
    # if [ "$LATEST_CHECKPOINT_ITERATION" = "1550" ]; then
    #     echo -e "\033[33mSkip run because latest checkpoint iteration is 1550: ${LATEST_CHECKPOINT_ITERATION_FILE}\033[0m" | tee -a "${RANK_LOG}"
    #     exit 0
    # fi

    if [ "$LATEST_CHECKPOINT_ITERATION" = "1" ]; then
        # `1` is treated as an initial checkpoint marker rather than a
        # resumable continual-pretraining checkpoint.  Do not pass
        # --no-initialization, so the model is initialized and trained anew.
        LOAD_CHECKPOINT_PATH="none"
        CONTINUE_TRAIN=${CONTINUE_TRAIN:-'false'}
        echo -e "\033[33mLatest checkpoint iteration is 1; training from scratch instead of loading $CHECKPOINT_PATH\033[0m" | tee -a "${RANK_LOG}"
    else
        LOAD_CHECKPOINT_PATH="${CHECKPOINT_PATH}"
        EXTRA_ARGS="${EXTRA_ARGS} --exit-on-missing-checkpoint --no-initialization"
        CONTINUE_TRAIN=${CONTINUE_TRAIN:-'true'}
        echo -e "\033[32mFind existing checkpoint $CHECKPOINT_PATH\033[0m" | tee -a "${RANK_LOG}"
    fi
elif [ "${PRETRAIN_CHECKPOINT_PATH}" != "none" ]; then
    if [ -f "${PRETRAIN_CHECKPOINT_PATH}/latest_checkpointed_iteration.txt" ]; then
        LOAD_CHECKPOINT_PATH="${PRETRAIN_CHECKPOINT_PATH}"
        EXTRA_ARGS="${EXTRA_ARGS} --exit-on-missing-checkpoint"
        CONTINUE_TRAIN=${CONTINUE_TRAIN:-'false'}
        echo -e "\033[32mCheckpoint '$CHECKPOINT_PATH' does not exists. Load from pretrained checkpoint '$LOAD_CHECKPOINT_PATH'\033[0m" | tee -a "${RANK_LOG}"
    else
        echo "ERROR: PRETRAIN_CHECKPOINT_PATH != none but $PRETRAIN_CHECKPOINT_PATH does not exist" | tee -a "${RANK_LOG}"
        exit 1
    fi
else
    LOAD_CHECKPOINT_PATH="none"
    CONTINUE_TRAIN=${CONTINUE_TRAIN:-'false'}
    echo -e "\033[32mNo checkpoint specified, training from scratch\033[0m" | tee -a "${RANK_LOG}"
fi

# ============================================================================
# Parameter Validation and Setup
# ============================================================================

# Validate required parameters
if [[ "${TOKENIZER_TYPE}" != mock && -z "${DATA_PATH}" && -z "${TRAIN_DATA_PATH}" && -z "${VALID_DATA_PATH}" ]]; then
    echo "ERROR: DATA_PATH or TRAIN_DATA_PATH and VALID_DATA_PATH must be set" | tee -a "${RANK_LOG}"
    exit 1
fi
if [[ -z "${OUTPUT_CHECKPOINT_PATH}" ]]; then
    echo "ERROR: OUTPUT_CHECKPOINT_PATH is not set" | tee -a "${RANK_LOG}"
    exit 1
fi

# Setup tokenizer (keep LLamaTokenizer as the launch-script spelling).
TOKENIZER_ARGS=()
if [[ "${TOKENIZER_TYPE}" == mock && -z "${TOKENIZER_MODEL_PATH}" ]]; then
    TOKENIZER_ARGS=(--tokenizer-type NullTokenizer --vocab-size "${VOCAB_SIZE:-256}")
elif [[ "${TOKENIZER_TYPE}" == LLamaTokenizer || "${TOKENIZER_TYPE}" == HuggingFaceTokenizer || "${TOKENIZER_TYPE}" == mock ]]; then
    : "${TOKENIZER_MODEL_PATH:?Set TOKENIZER_MODEL_PATH to a HuggingFace tokenizer directory}"
    TOKENIZER_ARGS=(--tokenizer-type HuggingFaceTokenizer --tokenizer-model "${TOKENIZER_MODEL_PATH}")
else
    echo "ERROR: Unknown TOKENIZER_TYPE=${TOKENIZER_TYPE}" >&2
    exit 2
fi

# Setup data path arguments. Keep each environment path as one argument.
# Weighted blends can still be passed in EXTRA_ARGS or as ordinary CLI arguments.
DATA_PATH_ARGS=()
if [[ "${TOKENIZER_TYPE}" == mock ]]; then
    DATA_PATH_ARGS=(--mock-data)
elif [[ -n "${TRAIN_DATA_PATH}" || -n "${VALID_DATA_PATH}" ]]; then
    [ -n "${TRAIN_DATA_PATH}" ] && DATA_PATH_ARGS+=(--train-data-path "${TRAIN_DATA_PATH}")
    [ -n "${VALID_DATA_PATH}" ] && DATA_PATH_ARGS+=(--valid-data-path "${VALID_DATA_PATH}")
else
    DATA_PATH_ARGS=(--data-path "${DATA_PATH}")
fi
if [[ -n "${DATA_PATH_CACHE}" ]]; then
    DATA_PATH_ARGS+=(--data-cache-path "${DATA_PATH_CACHE}")
fi

# Setup embedding tying
if [[ "1${TIE_EMBEDDING}" == "1false" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} \
        --untie-embeddings-and-output-weights
    "
fi

# MoE configuration - only apply when NUM_EXPERTS > 0
if [ "${NUM_EXPERTS}" -gt "0" ]; then
    if [[ " ${LOAD_BALANCING} " == *" sym_kl_aux_loss "* || " ${LOAD_BALANCING} " == *" global_sym_kl_aux_loss "* ]]; then
        if [[ ${MOE_ROUTER_SCORE_FUNCTION} != "softmax" && ${MOE_ROUTER_SCORE_FUNCTION} != "sigmoid" ]]; then
            echo "ERROR: sym_kl_aux_loss/global_sym_kl_aux_loss only supports MOE_ROUTER_SCORE_FUNCTION=softmax or sigmoid" | tee -a "${RANK_LOG}"
            exit 1
        fi
        if [[ ${MOE_USE_SIGMOID_ABS_ROUTER} == "true" ]]; then
            echo "ERROR: sym_kl_aux_loss/global_sym_kl_aux_loss does not support MOE_USE_SIGMOID_ABS_ROUTER=true" | tee -a "${RANK_LOG}"
            exit 1
        fi
    fi
    if [[ ${LOAD_BALANCING} == "dsv3" ]]; then
        EXTRA_ARGS="${EXTRA_ARGS} \
            --moe-router-enable-expert-bias
            --moe-router-bias-update-rate ${MOE_ROUTER_BIAS_UPDATE_RATE}
            --moe-aux-loss-coeff 0
        "
        MOE_ROUTER_SCORE_FUNCTION="sigmoid"
        LOAD_BALANCING=none
    elif [[ ${LOAD_BALANCING} == "none" ]]; then
        EXTRA_ARGS="${EXTRA_ARGS} --moe-aux-loss-coeff 0"
    else
        EXTRA_ARGS="${EXTRA_ARGS} \
            --moe-aux-loss-coeff ${MOE_AUX_LOSS_COEFF}
        "
    fi

fi

# ============================================================================
# Build Training Arguments
# ============================================================================

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --disable-bias-linear
    --seq-length ${SEQ_LEN}
    --max-position-embeddings ${SEQ_LEN}
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN_SIZE}
    --ffn-hidden-size ${FFN_HIDDEN_SIZE}
    --num-attention-heads ${NUM_ATTN_HEADS}
    --init-method-std ${INIT_STD}
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --group-query-attention
    --num-query-groups ${NUM_QUERY_GROUPS}
    --rotary-base ${ROTARY_BASE}
    --norm-epsilon ${NORM_EPSILON}
    --transformer-impl "${TRANSFORMER_IMPL}"
    --no-gradient-accumulation-fusion
)
if [ "${TRANSFORMER_IMPL}" = local ]; then
    MODEL_ARGS+=(--no-persist-layer-norm --no-masked-softmax-fusion --no-bias-gelu-fusion --no-rope-fusion)
else
    MODEL_ARGS+=(--cross-entropy-loss-fusion --cross-entropy-fusion-impl te --attention-backend flash)
fi
if [ "${ADD_QKV_BIAS}" = "true" ]; then
    MODEL_ARGS+=(--add-qkv-bias)
fi
    # --no-masked-softmax-fusion

# MoE Args - only set when NUM_EXPERTS > 0
if [ -n "${NUM_EXPERTS}" ] && [ "${NUM_EXPERTS}" -gt "0" ]; then
    if [ -z "${MOE_TOKEN_DISPATCHER_TYPE}" ]; then
        if [ "${EP_SIZE}" -ge "1" ]; then
            MOE_TOKEN_DISPATCHER_TYPE="alltoall"
        else
            MOE_TOKEN_DISPATCHER_TYPE="allgather"
        fi
    fi
    MOE_ARGS=(
        --num-experts ${NUM_EXPERTS}
        --moe-router-topk ${MOE_TOPK}
        --moe-ffn-hidden-size ${MOE_FFN_HIDDEN_SIZE}
        --moe-router-load-balancing-type ${LOAD_BALANCING}
        --moe-router-score-function ${MOE_ROUTER_SCORE_FUNCTION}
        --moe-token-dispatcher-type ${MOE_TOKEN_DISPATCHER_TYPE}
        # --moe-permute-fusion
    )
    if [ "${TRANSFORMER_IMPL}" = transformer_engine ]; then
        MOE_ARGS+=(--moe-grouped-gemm)
    fi
    # "none" omits the dtype override: routing projections follow model dtype.
    # Keep existing launchers on FP32 unless explicitly overridden.
    if [ "${MOE_ROUTER_DTYPE:-fp32}" != "none" ]; then
        MOE_ARGS+=(--moe-router-dtype "${MOE_ROUTER_DTYPE:-fp32}")
    fi
    if [ -n "${MOE_Z_LOSS_COEFF}" ]; then
        MOE_ARGS+=(--moe-z-loss-coeff ${MOE_Z_LOSS_COEFF})
    fi
    if [ -n "${MOE_ROUTER_TOPK_SCALING_FACTOR:-}" ]; then
        MOE_ARGS+=(--moe-router-topk-scaling-factor ${MOE_ROUTER_TOPK_SCALING_FACTOR})
    fi
    # if [[ "${LOAD_BALANCING}" == "quantile_balancing" ]]; then
    #     echo "关闭 --moe-router-fusion"
    # else
        # MOE_ARGS+=(--moe-router-fusion)
    # fi

    MOE_ARGS+=(--moe-router-global-aux-mode "${MOE_ROUTER_GLOBAL_AUX_MODE}")
    if [[ " ${LOAD_BALANCING} " == *" sym_kl_aux_loss "* || " ${LOAD_BALANCING} " == *" global_sym_kl_aux_loss "* ]]; then
        if [ -n "${MOE_ROUTER_SYM_KL_LOSS_TOLERANCE}" ]; then
            read -r -a kl_tolerance_bounds <<< "${MOE_ROUTER_SYM_KL_LOSS_TOLERANCE}"
            if [ "${#kl_tolerance_bounds[@]}" -ne 2 ]; then
                echo "MOE_ROUTER_SYM_KL_LOSS_TOLERANCE requires two bounds" >&2
                exit 1
            fi
            MOE_ARGS+=(--moe-router-sym-kl-loss-tolerance "${kl_tolerance_bounds[@]}")
        fi
        MOE_ARGS+=(
            --moe-router-sym-kl-loss-rho ${MOE_ROUTER_SYM_KL_LOSS_RHO}
            --moe-router-sym-kl-loss-coeff-clip ${MOE_ROUTER_SYM_KL_LOSS_COEFF_CLIP}
            --moe-router-sym-kl-loss-beta ${MOE_ROUTER_SYM_KL_LOSS_BETA}
        )
        if [ "${MOE_ROUTER_SYM_KL_LOSS_MULTIPLICATIVE_STE}" = "true" ]; then
            MOE_ARGS+=(--moe-router-sym-kl-loss-multiplicative-ste)
        fi
        if [ "${MOE_ROUTER_SYM_KL_LOSS_DYNAMIC_BETA}" = "true" ]; then
            MOE_ARGS+=(
                --moe-router-sym-kl-loss-dynamic-beta
                --moe-router-sym-kl-loss-beta-init ${MOE_ROUTER_SYM_KL_LOSS_BETA_INIT}
                --moe-router-sym-kl-loss-beta-range ${MOE_ROUTER_SYM_KL_LOSS_BETA_RANGE}
                --moe-router-sym-kl-loss-beta-update-rate ${MOE_ROUTER_SYM_KL_LOSS_BETA_UPDATE_RATE}
                --moe-router-sym-kl-loss-beta-deadband ${MOE_ROUTER_SYM_KL_LOSS_BETA_DEADBAND}
            )
        fi
    fi
    if [ "${MOE_TOKEN_DISPATCHER_TYPE}" = "flex" ]; then
        MOE_ARGS+=(
            --moe-flex-dispatcher-backend ${MOE_FLEX_DISPATCHER_BACKEND:-deepep}
        )
        if [ -n "${MOE_DEEPEP_NUM_SMS:-}" ]; then
            MOE_ARGS+=(--moe-deepep-num-sms ${MOE_DEEPEP_NUM_SMS})
        fi
        if [ -n "${MOE_HYBRIDEP_NUM_SMS:-}" ]; then
            MOE_ARGS+=(--moe-hybridep-num-sms ${MOE_HYBRIDEP_NUM_SMS})
        fi
    fi
    if [ "${MOE_OVERLAP_EP_COMM:-false}" = "true" ]; then
        MOE_ARGS+=(
            --overlap-moe-expert-parallel-comm
        )
        if [ "${MOE_DELAY_WGRAD_COMPUTE:-false}" = "true" ]; then
            MOE_ARGS+=(
                --delay-wgrad-compute
            )
        fi
    fi
    if [[ -n "${MOE_EXPERT_CAPACITY_FACTOR}" ]] \
        && [[ "${MOE_EXPERT_CAPACITY_FACTOR}" != "0" ]] \
        && [[ "${MOE_EXPERT_CAPACITY_FACTOR}" != "0.0" ]]; then
        MOE_ARGS+=(
            --moe-expert-capacity-factor ${MOE_EXPERT_CAPACITY_FACTOR}
        )
    fi
    if [ "${MOE_ROUTER_NORM_TOPK_PROB}" = "false" ]; then
        # See: https://github.com/NVIDIA/Megatron-LM/issues/1593#issuecomment-2908620372
        MOE_ARGS+=(
            --moe-router-pre-softmax
        )
    fi
    if [ "${MOE_ROUTER_PER_EXPERT_PROBE}" = "true" ]; then
        MOE_ARGS+=(
            --moe-router-per-expert-probe
        )
    fi
    if [ "${MOE_USE_WEIGHT_AWARE_ROUTER}" = "true" ]; then
        MOE_ARGS+=(
            --use-weight-aware-router
            --moe-router-grad-mode ${MOE_ROUTER_GRAD_MODE}
        )
        if [ "${MOE_ROUTER_GATE_UP_INTERACTION}" = "true" ]; then
            MOE_ARGS+=(
                --moe-router-gate-up-interaction
            )
        fi
        if [ "${MOE_USE_EAR_HYBRID}" = "true" ]; then
            MOE_ARGS+=(
                --moe-router-add-static-router
            )
        fi
        if [ "${MOE_ROUTER_EAR_NORMALIZE}" = "true" ]; then
            MOE_ARGS+=(
                --moe-router-ear-normalize
                --moe-router-ear-normalize-tau ${MOE_ROUTER_EAR_NORMALIZE_TAU}
            )
        fi
    fi
    MOE_ARGS+=(
        --moe-router-orthogonal-preprocess ${MOE_ROUTER_ORTHOGONAL_PREPROCESS}
    )
    if [ "${MOE_ROUTER_ORTHOGONAL_PREPROCESS}" = "pca" ]; then
        MOE_ARGS+=(
            --moe-router-orthogonal-pca-rank ${MOE_ROUTER_ORTHOGONAL_PCA_RANK}
        )
    fi
    if [ -n "${MOE_ROUTER_ORTHOGONAL_LOSS_COEFF}" ]; then
        MOE_ARGS+=(
            --moe-router-orthogonal-loss-coeff ${MOE_ROUTER_ORTHOGONAL_LOSS_COEFF}
        )
    fi
    if [ -n "${MOE_LAYER_FREQ}" ]; then
        MOE_ARGS+=(
            --moe-layer-freq "${MOE_LAYER_FREQ}"
        )
    fi
    if [ -n "${SHARED_EXPERTS_HIDDEN_SIZE}" ] && [ "${SHARED_EXPERTS_HIDDEN_SIZE}" -gt "1" ]; then
        MOE_ARGS+=(
            --moe-shared-expert-intermediate-size ${SHARED_EXPERTS_HIDDEN_SIZE}
        )
        if [ "${MOE_SHARED_EXPERTS_GATE}" = "true" ]; then
            MOE_ARGS+=(
                --moe-shared-expert-gate
            )
        fi
    fi
else
    MOE_ARGS=()
fi

TRAINING_ARGS=(
    --micro-batch-size ${BATCH_SIZE}
    --global-batch-size ${GLOBAL_BATCH_SIZE}
    --lr ${LR}
    --train-samples ${TRAIN_SAMPLES}
    --lr-warmup-samples ${LR_WARMUP_SAMPLES}
    --lr-decay-samples ${LR_DECAY_SAMPLES}
    --lr-decay-style ${LR_DECAY_STYLE}
    --min-lr ${MIN_LR}
    --dataloader-type ${DATALOADER_TYPE:-cyclic}
    --weight-decay 0.1
    --clip-grad ${CLIP_GRAD}
    --num-workers ${NUM_WORKERS}
    --bf16
    --ckpt-format torch_dist
    --ckpt-fully-parallel-load
    --auto-detect-ckpt-format
    --save "${CHECKPOINT_PATH}"
    --seed "${SEED}"
)
if [ "${ASYNC_SAVE:-false}" = true ]; then
    TRAINING_ARGS+=(--async-save)
fi
    # --manual-gc
    # --manual-gc-interval 1000

if [ "${USE_DDP_OVERLAP}" = "true" ]; then
    TRAINING_ARGS+=(
        --overlap-grad-reduce
        --ddp-bucket-size ${DDP_BUCKET_SIZE}
        --ddp-average-in-collective
        --ddp-pad-buckets-for-high-nccl-busbw
    )
else
    echo -e "\033[33m\n[INFO] Disable DDP overlap\033[0m"
fi

# Add load checkpoint path if not none
if [ "${LOAD_CHECKPOINT_PATH}" != "none" ]; then
    TRAINING_ARGS+=(
        --load "${LOAD_CHECKPOINT_PATH}"
    )
fi

DATA_ARGS=()
if [ "${CREATE_ATTENTION_MASK_IN_DATALOADER:-false}" != "true" ]; then
    DATA_ARGS+=(--no-create-attention-mask-in-dataloader)
fi
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${MP_SIZE}
    --pipeline-model-parallel-size ${PP_SIZE}
    --context-parallel-size ${CP_SIZE}
    --expert-model-parallel-size ${EP_SIZE}
    --expert-tensor-parallel-size 1
    # --sequence-parallel
)

if [ "$OPTIMIZER" == "muon" ]; then
    TRAINING_ARGS+=(
        --optimizer muon
        --muon-momentum 0.95
        --muon-num-ns-steps ${MUON_NUM_NS_STEPS}
        --muon-scale-mode ${MUON_SCALE_MODE}
        --muon-tp-mode ${MUON_TP_MODE}
    )
elif [ "$OPTIMIZER" == "spectral_ball" ]; then
    TRAINING_ARGS+=(
        --optimizer spectral_ball
    )
elif [ "$OPTIMIZER" == "adam" ]; then
    TRAINING_ARGS+=(
        --optimizer adam
    )
    if [ "${USE_DDP_OVERLAP}" = "true" ] && [ "${USE_DISTRIBUTED_OPTIMIZER:-false}" = true ]; then
        TRAINING_ARGS+=(
            --overlap-param-gather
        )
    fi
    if [ "${USE_DISTRIBUTED_OPTIMIZER:-false}" = true ]; then
        MODEL_PARALLEL_ARGS+=(--use-distributed-optimizer)
        TRAINING_ARGS+=(--dist-ckpt-optim-fully-reshardable)
    fi
else
    echo -e "\033[31m[ERROR] Unknown OPTIMIZER: ${OPTIMIZER}. Supported: muon, adam, spectral_ball.\033[0m"
    exit 1
fi

LOGGING_ARGS=(
    --log-interval ${LOG_INTERVAL}
    --log-throughput
    --async-log-writer
    --save-interval ${SAVE_INTERVAL}
    --eval-interval ${EVAL_INTERVAL}
    --eval-iters ${EVAL_ITERS:-10}
    --tensorboard-dir "${LOG_DIR}"
    --tensorboard-log-interval ${TENSORBOARD_LOG_INTERVAL}
    --log-metrics-interval ${LOG_METRICS_INTERVAL:-50}
    --log-moe-metrics load_balance
    --timing-log-level ${TIMING_LOG_LEVEL:-0}
)

if [ "${LOG_TIMERS_TO_TENSORBOARD:-1}" = "1" ]; then
    LOGGING_ARGS+=(
        --log-timers-to-tensorboard
    )
fi

if [ "${LOG_MEMORY_TO_TENSORBOARD:-1}" = "1" ]; then
    LOGGING_ARGS+=(
        --log-memory-to-tensorboard
    )
fi

# YuLan benchmark evaluation is not migrated; normal validation uses official eval-iters.

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project "${WANDB_PROJECT:-MoE}"
        --wandb-exp-name "${NAME}"
        --wandb-save-dir "${WANDB_SAVE_DIR:-${OUTPUT_CHECKPOINT_PATH}/wandb/${NAME}}"
    )
fi

# Logging uses the migrated TensorBoard/WandB writers.

if [ ! "1${ACTIVATION_CHECKPOINT}" = "1false" ]; then
    RECOMPUTE_METHOD=block
    EXTRA_ARGS="${EXTRA_ARGS} \
        --recompute-granularity full \
        --recompute-method ${RECOMPUTE_METHOD} \
        --recompute-num-layers $ACTIVATION_CHECKPOINT \
        "
fi

# ============================================================================
# Launch Training
# ============================================================================

# Use the experiment module so imports resolve to the official root pretrain_gpt.
PRETRAIN_ARGS=(--module experiments.xcx.training.pretrain_gpt)
if [ -n "${PRETRAIN_SCRIPT}" ]; then
    PRETRAIN_ARGS=("$(realpath "${PRETRAIN_SCRIPT}")")
fi
CMD=("${PYTHON:-python}" -m torch.distributed.run)
if [ "${NSYS:-0}" = 1 ]; then
    NSIGHT_PREFIX="${LOG_DIR}/nsys-report-rank${RANK:-0}-$(date +%Y%m%d-%H%M%S)"
    CMD=(nsys profile -s none -t nvtx,cuda,cublas,cudnn,osrt
        --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop
        "--duration=${NSYS_DURATION:-300}" -o "${NSIGHT_PREFIX}" "${CMD[@]}")
fi

TRAIN_TEMPLATE_PATH=$(realpath "$0")
cp "${TRAIN_TEMPLATE_PATH}" "${CHECKPOINT_PATH}/"
if [ -n "${LAUNCH_SCRIPT_PATH}" ]; then
    cp "${LAUNCH_SCRIPT_PATH}" "${CHECKPOINT_PATH}/"
fi

# Preserve YuLan EXTRA_ARGS syntax, including quoted paths, without shell eval.
EXTRA_ARGS_FILE=$(mktemp) || exit 1
if ! "${PYTHON:-python}" - "${EXTRA_ARGS}" > "${EXTRA_ARGS_FILE}" <<'PY_ARGS'
import shlex
import sys
for arg in shlex.split(sys.argv[1]):
    sys.stdout.buffer.write(arg.encode() + b"\0")
PY_ARGS
then
    rm -f "${EXTRA_ARGS_FILE}"
    exit 2
fi
mapfile -d '' -t EXTRA_ARGS_ARRAY < "${EXTRA_ARGS_FILE}"
rm -f "${EXTRA_ARGS_FILE}"
RUN_COMMAND=("${CMD[@]}" "${DISTRIBUTED_ARGS[@]}" "${PRETRAIN_ARGS[@]}"
    "${MODEL_ARGS[@]}" "${DATA_PATH_ARGS[@]}" "${DATA_ARGS[@]}" "${MOE_ARGS[@]}"
    "${TRAINING_ARGS[@]}" "${MODEL_PARALLEL_ARGS[@]}" "${LOGGING_ARGS[@]}"
    "${TOKENIZER_ARGS[@]}" "${EXTRA_ARGS_ARRAY[@]}" "$@")
{
    echo "$(pwd) $(git rev-parse HEAD)"
    git status --porcelain
    printf '%q ' "${RUN_COMMAND[@]}"
    printf '\n'
} >> "${RANK_LOG}"

"${RUN_COMMAND[@]}" 2>&1 | tee -a "${RANK_LOG}"
training_status=$?
cp "${RANK_LOG}" "${CHECKPOINT_PATH}/"
if [ "${training_status}" -ne 0 ]; then
    exit "${training_status}"
fi
if [ -n "${MCORE2HF_SCRIPT_PATH}" ]; then
    MEGATRON_PATH=$(pwd) bash "${MCORE2HF_SCRIPT_PATH}" "${CHECKPOINT_PATH}"
fi
