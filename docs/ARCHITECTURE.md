# Architecture

RehabEdge-Vision.AI is deliberately small. Its purpose is to make edge computer-vision deployments in rehabilitation reproducible and clinically bounded.

## Control flow

```text
Camera
  -> Detector
  -> FrameResult
  -> RehabRuleEngine
  -> RuleEvent
  -> CSV/JSONL storage
  -> Local dashboard / export
```

## Design principles

1. **Local first**: raw frames stay on the device unless a developer explicitly enables review-clip workflows.
2. **Review prompts, not decisions**: rules emit events such as `session_started` or `patient_left_frame`. They do not diagnose, score impairment, infer fall risk, or make therapy decisions.
3. **Runtime adapters are replaceable**: OpenVINO, NCNN, and PyTorch/ONNX are runtime choices, not product claims.
4. **Rules are explicit**: zone and event logic live in YAML so clinicians and engineers can review what the system is allowed to say.
5. **Benchmark before claims**: Raspberry Pi 4 and Raspberry Pi 5 have different performance envelopes; each deployment should report actual FPS, warm/cold start, CPU load, and temperature.

## Runtime choices

### OpenVINO

Best default for a repeatable deployment path on Raspberry Pi OS 64-bit: export to OpenVINO IR, compile for CPU, enable cache, and run as a service.

### NCNN

Useful as a benchmark path on ARM. It may outperform other formats for some models and Pi configurations. Treat it as an empirical choice.

### PyTorch / ONNX direct

Useful during development. Less attractive for repeatable edge service deployment unless benchmark data justifies it.

## Safety architecture

The safety layer is explicit logic:

- Which zone is observed?
- Which class is expected?
- How long must a condition persist?
- What severity is emitted?
- What does the event mean?
- What does it explicitly not mean?

The default configuration avoids unsupported clinical inference.
