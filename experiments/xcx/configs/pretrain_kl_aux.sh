#!/usr/bin/env bash
set -euo pipefail

export RUN_NAME="${RUN_NAME:-kl_aux}"
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/moe_pretrain_template.sh" \
    --moe-router-load-balancing-type global_sym_kl_aux_loss \
    --moe-router-sym-kl-loss-beta 0.7 --moe-router-sym-kl-loss-rho 0 "$@"
