# Benchmarking

Benchmarking is part of the product, not an afterthought.

## Minimum benchmark fields

| Field | Reason |
|---|---|
| Pi model | Pi 4 and Pi 5 are not equivalent targets. |
| OS/Python | Wheel availability and runtime behaviour depend on user space. |
| Runtime | OpenVINO, NCNN, ONNX and PyTorch behave differently. |
| Model | Nano/small/pose/segmentation models have different costs. |
| Input size | Large images change speed and accuracy. |
| FPS | Basic throughput. |
| Cold start | Service restart behaviour. |
| Warm start | Cache usefulness. |
| CPU load | Sustained service feasibility. |
| Temperature | Cooling and throttling risk. |
| Power supply | Stability risk. |

## Run

```bash
rehabedge benchmark --model yolo26n_openvino_model --runtime openvino --source 0 --frames 300 --output benchmarks/pi5_openvino.csv
```

## Interpreting results

The fastest runtime is not automatically the best runtime. Choose the runtime that gives the right mix of:

- sufficient FPS
- stable thermals
- reproducible startup
- packaging simplicity
- maintainable dependencies
- acceptable model accuracy for the target task
