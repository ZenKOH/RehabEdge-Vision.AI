#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:-yolo26n.pt}"
IMGSZ="${2:-640}"
rehabedge export-yolo --model "$MODEL" --format openvino --imgsz "$IMGSZ"
