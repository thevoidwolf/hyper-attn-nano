#!/usr/bin/env bash
# Quick-activate from anywhere: source ~/hyper-attn-nano/activate.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES=0          # pin to your 3060 Ti
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"
cd "$SCRIPT_DIR"
echo "  hyper-attn-nano env active — Python $(python --version 2>&1 | awk '{print $2}')"
echo "  Project: $SCRIPT_DIR"
