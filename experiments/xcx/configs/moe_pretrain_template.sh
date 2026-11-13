#!/usr/bin/env bash
# Shared BF16 MoE recipe. Presets and caller arguments override these defaults.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
output_dir="${OUTPUT_DIR:-${repo_root}/runs/xcx/${RUN_NAME:-moe}}"

# Paths can come from the environment or ordinary Megatron CLI arguments.
has_data=false
has_tokenizer=false
per_split_data=false
data_args=(--tokenizer-type HuggingFaceTokenizer)
if [[ -n "${DATA_PATH:-}" ]]; then
    data_args+=(--data-path "$DATA_PATH")
    has_data=true
fi
if [[ -n "${TOKENIZER_MODEL_PATH:-}" ]]; then
    data_args+=(--tokenizer-model "$TOKENIZER_MODEL_PATH")
    has_tokenizer=true
fi
for arg in "$@"; do
    case "$arg" in
        --data-path|--data-path=*|--data-args-path|--data-args-path=*|--mock-data)
            has_data=true ;;
        --train-data-path|--train-data-path=*|--valid-data-path|--valid-data-path=*|--test-data-path|--test-data-path=*|--per-split-data-args-path|--per-split-data-args-path=*)
            has_data=true
            per_split_data=true ;;
        --tokenizer-type|--tokenizer-type=*|--tokenizer-model|--tokenizer-model=*)
            has_tokenizer=true ;;
    esac
done
if [[ "$per_split_data" == false ]]; then
    data_args+=(--split "${SPLIT:-999,1,0}")
fi
if [[ "$has_data" == false ]]; then
    echo "Set DATA_PATH to an indexed dataset prefix, or pass --data-path/--train-data-path/--mock-data." >&2
    exit 2
fi
if [[ "$has_tokenizer" == false ]]; then
    echo "Set TOKENIZER_MODEL_PATH, or pass Megatron tokenizer arguments." >&2
    exit 2
fi

# Single-node torchrun needs no rendezvous setup; multi-node uses its normal variables.
launch_args=(--nproc-per-node "${NPROC_PER_NODE:-1}" --nnodes "${NNODES:-1}")
if [[ "${NNODES:-1}" == 1 ]]; then
    launch_args+=(--standalone)
else
    : "${MASTER_ADDR:?Set MASTER_ADDR for multi-node training}"
    : "${NODE_RANK:?Set NODE_RANK on each node}"
    launch_args+=(--master-addr "$MASTER_ADDR" --master-port "${MASTER_PORT:-6000}"
                  --node-rank "$NODE_RANK")
fi

model_args=(
    --num-layers "${NUM_LAYERS:-8}"
    --hidden-size "${HIDDEN_SIZE:-512}"
    --num-attention-heads "${NUM_ATTN_HEADS:-8}"
    --ffn-hidden-size "${FFN_HIDDEN_SIZE:-2048}"
    --seq-length "${SEQ_LEN:-2048}"
    --max-position-embeddings "${SEQ_LEN:-2048}"
    --num-experts "${NUM_EXPERTS:-8}"
    --moe-ffn-hidden-size "${MOE_FFN_HIDDEN_SIZE:-512}"
    --moe-router-topk "${MOE_TOPK:-2}"
    --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 0.005
    --moe-token-dispatcher-type alltoall
    --tensor-model-parallel-size "${TP_SIZE:-1}"
    --pipeline-model-parallel-size "${PP_SIZE:-1}"
    --context-parallel-size "${CP_SIZE:-1}"
    --expert-model-parallel-size "${EP_SIZE:-1}"
    --expert-tensor-parallel-size 1
    --bf16 --disable-bias-linear --swiglu
    --normalization RMSNorm --norm-epsilon 1e-6
    --position-embedding-type rope --rotary-base 10000
    --untie-embeddings-and-output-weights
    --attention-dropout 0 --hidden-dropout 0
    --transformer-impl "${TRANSFORMER_IMPL:-local}"
    --no-gradient-accumulation-fusion
)
if [[ "${TRANSFORMER_IMPL:-local}" == local ]]; then
    model_args+=(--no-persist-layer-norm --no-masked-softmax-fusion --no-bias-gelu-fusion
                 --no-rope-fusion)
fi

train_args=(
    --micro-batch-size "${MICRO_BATCH_SIZE:-1}"
    --global-batch-size "${GLOBAL_BATCH_SIZE:-32}"
    --train-iters "${TRAIN_ITERS:-10000}"
    --optimizer "${OPTIMIZER:-adam}"
    --lr "${LR:-3e-4}" --min-lr "${MIN_LR:-3e-5}"
    --lr-warmup-iters "${LR_WARMUP_ITERS:-100}" --lr-decay-style cosine
    --weight-decay 0.1 --clip-grad 1 --seed "${SEED:-25}"
    --num-workers "${NUM_WORKERS:-4}"
    --data-cache-path "${DATA_PATH_CACHE:-${output_dir}/data_cache}"
    --save "${output_dir}/checkpoints" --save-interval "${SAVE_INTERVAL:-1000}"
    --ckpt-format torch_dist
    --eval-iters "${EVAL_ITERS:-10}" --eval-interval "${EVAL_INTERVAL:-1000}"
    --log-interval "${LOG_INTERVAL:-10}"
    --tensorboard-dir "${output_dir}/tensorboard" --tensorboard-log-interval 10
    --log-moe-metrics load_balance --log-metrics-interval "${LOG_METRICS_INTERVAL:-100}"
)

exec "${PYTHON:-python}" -m torch.distributed.run "${launch_args[@]}" \
    --module experiments.xcx.training.pretrain_gpt \
    "${model_args[@]}" "${train_args[@]}" "${data_args[@]}" "$@"
