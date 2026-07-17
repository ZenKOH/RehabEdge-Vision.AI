#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:-yolo26n_openvino_model}"
SOURCE="${2:-0}"
rehabedge run --model "$MODEL" --runtime openvino --source "$SOURCE" --config configs/default_zones.yaml --events events.csv
