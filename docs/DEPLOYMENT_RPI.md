# Raspberry Pi Deployment

## Recommended hardware

- Raspberry Pi 5 for sustained camera workloads and local dashboard use.
- Raspberry Pi 4 for prototypes and lighter single-camera services.
- 64-bit Raspberry Pi OS.
- Official power supply and active cooling for Pi 5.
- Raspberry Pi Camera Module or USB camera.

## Install

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git libglib2.0-0 libgl1
python3 -m venv ~/venvs/rehabedge
source ~/venvs/rehabedge/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[vision]
rehabedge verify-openvino
```

If OpenVINO does not show `CPU`, check Python version, architecture, and OS age:

```bash
uname -m
python --version
```

The intended target is Linux AArch64 with Python 3.10+.

## Export model

```bash
rehabedge export-yolo --model yolo26n.pt --format openvino --imgsz 640
```

Do not commit model weights or exported model artefacts into this repository by default.

## Warm the OpenVINO cache

```bash
rehabedge compile-ir --xml yolo26n_openvino_model/yolo26n.xml --device CPU --cache-dir ov_cache
```

## Run as a service

```bash
sudo cp systemd/rehabedge.service /etc/systemd/system/rehabedge.service
sudo systemctl daemon-reload
sudo systemctl enable rehabedge
sudo systemctl start rehabedge
sudo journalctl -u rehabedge -f
```

Adjust the service file paths to match your installation directory and virtual environment.
