# RehabEdge-Vision.AI

**Make rehabilitation visible at the edge.**

RehabEdge-Vision.AI is a Raspberry Pi edge-AI reference stack for privacy-preserving rehabilitation observation. It demonstrates how to run Ultralytics YOLO models with OpenVINO on Raspberry Pi 4/5, then translate local detections into clinically bounded events for therapy-station utilisation, home-practice review, and rehabilitation robotics context sensing.

> **Safety boundary**: this repository is not a medical device, diagnostic tool, autonomous fall detector, or AI therapist. It is a research and prototyping toolkit for local computer-vision workflows in rehabilitation settings. Clinical use requires validation, risk management, privacy review, and regulatory assessment.

## Why this exists

Most Raspberry Pi vision demos stop at object detection. Rehabilitation needs something more specific: a local observer that can see context around therapy without sending raw video to the cloud by default.

The repository implements a small, practical loop:

```text
Camera input
  -> YOLO runtime adapter: OpenVINO / NCNN / PyTorch fallback
  -> Zone-aware rule engine
  -> Local event store
  -> Clinician-review dashboard and CSV/JSONL exports
```

The design follows the Raspberry Pi + OpenVINO deployment pattern: clean Python environment, exported OpenVINO IR when startup matters, explicit CPU target, persistent model cache, service packaging, and benchmarks that treat Raspberry Pi 4 and Raspberry Pi 5 as different targets.

## Initial use cases

### 1. Therapy-station utilisation and safety observer

Place a Raspberry Pi Camera near a gait lane, parallel bars, upper-limb table, robotic device, cycle ergometer, balance area, or FES station. The node emits simple non-diagnostic events:

- `session_started`
- `station_occupied`
- `patient_left_frame`
- `therapist_or_caregiver_present`
- `chair_ready`
- `walker_detected`
- `review_needed`

### 2. Home-practice adherence logger

Run a bounded home-practice observer that logs whether a prescribed task appears to have been attempted. Early versions intentionally avoid unsupported clinical claims. They provide timestamps, clips, and clinician-review flags.

### 3. Rehab robotics companion observer

Use an external Raspberry Pi vision node near a robot/FES/device station to capture environmental context the robot may not know: person present, object visible, therapy zone occupied, support chair in frame, therapist nearby, or interruption detected.

## Repository layout

```text
rehabedge_vision/
  camera/          Camera adapters for USB/OpenCV and PiCamera2
  inference/       YOLO/OpenVINO/NCNN runtime adapters and benchmarking
  rules/           Config-driven rehab zone and event rules
  storage/         CSV/JSONL local event stores
  dashboard/       Local FastAPI dashboard
  main.py          CLI entrypoint
configs/           Example zones, object maps, runtime config
scripts/           Raspberry Pi setup/export/run helpers
systemd/           Example Linux service file
docker/            Optional Dockerfile
docs/              Architecture, deployment, safety, benchmarking, roadmap
tests/             Unit tests for geometry/rules/storage
```

## Quick start on Raspberry Pi OS 64-bit

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git libglib2.0-0 libgl1
python3 -m venv ~/venvs/rehabedge
source ~/venvs/rehabedge/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Verify OpenVINO can see the CPU device:

```bash
rehabedge verify-openvino
```

Export a YOLO model to OpenVINO format:

```bash
rehabedge export-yolo --model yolo26n.pt --format openvino --imgsz 640
```

Run with a USB camera:

```bash
rehabedge run \
  --model yolo26n_openvino_model \
  --runtime openvino \
  --source 0 \
  --config configs/default_zones.yaml \
  --events events.csv
```

Run the local dashboard:

```bash
rehabedge dashboard --events events.csv --host 0.0.0.0 --port 8000
```

Open: `http://<raspberry-pi-ip>:8000`

## Benchmarks

The project includes benchmark scaffolding because OpenVINO, NCNN, ONNX, PyTorch, model size, image size, cooling, and power supply all affect real performance on Raspberry Pi.

```bash
rehabedge benchmark \
  --model yolo26n_openvino_model \
  --source sample.mp4 \
  --runtime openvino \
  --frames 300 \
  --output benchmark_pi5_openvino.csv
```

Track at minimum:

| Runtime | Model | Input | Pi | FPS | CPU % | Temp | Cold start | Warm start | Notes |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| OpenVINO IR | YOLO26n | 640 | Pi 5 | TBD | TBD | TBD | TBD | TBD | cache enabled |
| NCNN | YOLO26n | 640 | Pi 5 | TBD | TBD | TBD | TBD | TBD | ARM-optimised |
| PyTorch | YOLO26n | 640 | Pi 5 | TBD | TBD | TBD | TBD | TBD | development only |

## Example rule configuration

```yaml
zones:
  gait_lane:
    polygon: [[80, 120], [1200, 120], [1200, 650], [80, 650]]
    required_classes: [person]
    optional_classes: [chair, walker]

rules:
  - name: session_started
    type: presence_in_zone
    zone: gait_lane
    class_name: person
    min_duration_s: 5

  - name: patient_left_frame
    type: absence
    class_name: person
    min_duration_s: 3

  - name: therapist_or_caregiver_present
    type: min_count
    class_name: person
    min_count: 2
    min_duration_s: 5
```

## Clinical and privacy boundaries

- Run locally by default.
- Do not send raw video to the cloud unless explicitly configured.
- Store event logs rather than raw video whenever possible.
- Use clipped video only for clinician review and local audit.
- Do not infer diagnosis, impairment severity, fall risk, or therapy quality without validated clinical models.
- Treat all AI outputs as review prompts, not clinical decisions.

## Licence

AGPL-3.0. Ultralytics YOLO uses AGPL-3.0 by default; proprietary/internal/commercial deployments may require an Ultralytics Enterprise licence.

## Status

`v0.1.0` reference build: therapy-station observer, OpenVINO/Ultralytics runtime path, local event rules, CSV/JSONL storage, dashboard, Raspberry Pi deployment helpers, and benchmark scaffolding.
