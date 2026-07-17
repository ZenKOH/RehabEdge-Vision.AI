#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y python3-venv python3-pip git libglib2.0-0 libgl1
python3 -m venv ~/venvs/rehabedge
source ~/venvs/rehabedge/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[vision]
rehabedge verify-openvino
