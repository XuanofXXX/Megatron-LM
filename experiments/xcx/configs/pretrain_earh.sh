#!/usr/bin/env bash
set -euo pipefail

export NAME_PREFIX="${NAME_PREFIX:-earh_}"
export LAUNCH_SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
export MOE_USE_WEIGHT_AWARE_ROUTER=${MOE_USE_WEIGHT_AWARE_ROUTER:-true}
export MOE_USE_EAR_HYBRID=${MOE_USE_EAR_HYBRID:-true}
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/moe_pretrain_template.sh" "$@"
