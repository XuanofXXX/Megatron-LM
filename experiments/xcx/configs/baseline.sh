#!/usr/bin/env bash
# Small upstream GPT baseline; append ordinary Megatron arguments to override defaults.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
nproc_per_node="${NPROC_PER_NODE:-1}"

exec "${PYTHON:-python}" -m torch.distributed.run \
    --standalone \
    --nproc-per-node "$nproc_per_node" \
    --module experiments.xcx.training.pretrain_gpt \
    --num-layers 2 \
    --hidden-size 128 \
    --num-attention-heads 4 \
    --ffn-hidden-size 512 \
    --seq-length 128 \
    --max-position-embeddings 128 \
    --micro-batch-size 1 \
    --global-batch-size "$((2 * nproc_per_node))" \
    --train-iters 4 \
    --lr 1e-4 \
    --lr-decay-style constant \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size 256 \
    --transformer-impl local \
    --no-persist-layer-norm \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --num-workers 0 \
    --split 100,0,0 \
    --eval-iters 0 \
    --eval-interval 1000 \
    --log-interval 1 \
    "$@"
