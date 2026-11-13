#!/usr/bin/env bash
set -euo pipefail

export NAME_PREFIX="${NAME_PREFIX:-moe_}"
export LAUNCH_SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/moe_pretrain_template.sh" "$@"
