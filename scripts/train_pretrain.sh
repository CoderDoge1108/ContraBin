#!/usr/bin/env bash
# Launch ContraBin pre-training from the default config.
#
# Usage:
#   ./scripts/train_pretrain.sh [config=configs/pretrain.yaml]

set -euo pipefail

CONFIG=${1:-configs/pretrain.yaml}

contrabin pretrain --config "$CONFIG"
