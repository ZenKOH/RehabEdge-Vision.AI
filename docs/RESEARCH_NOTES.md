# Research notes

This build was motivated by the Raspberry Pi/OpenVINO/Ultralytics deployment pattern:

- Raspberry Pi OS 64-bit on Pi 4 or Pi 5.
- OpenVINO installed from PyPI.
- `openvino.Core().available_devices` should expose `CPU`.
- OpenVINO IR is preferred when load latency matters.
- `cache_dir` is a deployment feature because small edge devices often restart as services.
- Pi 4 and Pi 5 should be benchmarked separately.
- Power and cooling are runtime dependencies.

The clinical interpretation is deliberately conservative: the repository is a local edge observer for rehab context, not a diagnosis or autonomous therapy system.
