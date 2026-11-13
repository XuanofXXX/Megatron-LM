#!/usr/bin/env bash
set -euo pipefail

export RUN_NAME="${RUN_NAME:-earh}"
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/moe_pretrain_template.sh" \
    --use-weight-aware-router --moe-router-add-static-router "$@"
