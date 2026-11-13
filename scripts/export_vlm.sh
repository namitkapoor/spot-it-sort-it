#!/usr/bin/env bash
# Convert a small VLM to OpenVINO IR with INT4 weights (fits the NUC's iGPU, ~2-3 GB).
# Run once, on any machine; copy models/vlm/ to the NUC.
#   uv sync --extra export && ./scripts/export_vlm.sh
set -euo pipefail
MODEL="${1:-Qwen/Qwen2.5-VL-3B-Instruct}"
OUT="${2:-models/vlm/qwen2.5-vl-3b-int4}"
uv run optimum-cli export openvino --model "$MODEL" --weight-format int4 --trust-remote-code "$OUT"
echo "VLM -> $OUT"
