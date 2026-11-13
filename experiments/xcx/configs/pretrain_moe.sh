#!/usr/bin/env bash
set -euo pipefail

export RUN_NAME="${RUN_NAME:-moe}"
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/moe_pretrain_template.sh" "$@"
