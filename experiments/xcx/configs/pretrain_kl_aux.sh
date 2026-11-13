#!/usr/bin/env bash
set -euo pipefail

export NAME_PREFIX="${NAME_PREFIX:-kl_aux_}"
export LAUNCH_SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
export LOAD_BALANCING=${LOAD_BALANCING:-global_sym_kl_aux_loss}
export MOE_ROUTER_SYM_KL_LOSS_BETA=${MOE_ROUTER_SYM_KL_LOSS_BETA:-0.7}
export MOE_ROUTER_SYM_KL_LOSS_RHO=${MOE_ROUTER_SYM_KL_LOSS_RHO:-0}
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/moe_pretrain_template.sh" "$@"
