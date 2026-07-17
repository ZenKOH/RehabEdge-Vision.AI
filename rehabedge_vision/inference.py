from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from rehabedge_vision.core import Detection


class USBCameraSource:
    """OpenCV video source for USB cameras or video files."""

    def __init__(self, source: int | str = 0, width: int | None = None, height: int | None = None):
        import cv2  # type: ignore

        self.cv2 = cv2
        self.capture = cv2.VideoCapture(source)
        if width:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self.capture.read()
            if not ok:
                break
            yield frame

    def close(self) -> None:
        self.capture.release()


class UltralyticsDetector:
    """Ultralytics runtime wrapper.

    Supports `.pt`, exported OpenVINO directories, NCNN directories, ONNX files and other
    Ultralytics-supported model formats. Ultralytics handles preprocessing and postprocessing.
    """

    def __init__(self, model_path: str | Path, conf: float = 0.25, imgsz: int = 640):
        from ultralytics import YOLO  # type: ignore

        self.model = YOLO(str(model_path))
        self.conf = conf
        self.imgsz = imgsz

    def predict(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)
        if not results:
            return []
        result = results[0]
        names = result.names or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        detections: list[Detection] = []
        for box in boxes:
            cls_id = int(box.cls[0]) if box.cls is not None else -1
            class_name = str(names.get(cls_id, cls_id))
            confidence = float(box.conf[0]) if box.conf is not None else 0.0
            xyxy = tuple(float(v) for v in box.xyxy[0].detach().cpu().numpy().tolist())
            detections.append(Detection(class_name=class_name, confidence=confidence, xyxy=xyxy))
        return detections


def verify_openvino_devices() -> list[str]:
    from openvino import Core  # type: ignore

    return list(Core().available_devices)


def configure_openvino_cache(cache_dir: str | Path) -> list[str]:
    import openvino as ov  # type: ignore
    import openvino.properties as props  # type: ignore

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    core = ov.Core()
    core.set_property({props.cache_dir: str(cache_path)})
    return list(core.available_devices)


def compile_ir(model_xml: str | Path, device: str = "CPU", cache_dir: str | Path = "ov_cache") -> str:
    import openvino as ov  # type: ignore
    import openvino.properties as props  # type: ignore

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    core = ov.Core()
    core.set_property({props.cache_dir: str(cache_path)})
    model = core.read_model(str(model_xml))
    compiled_model = core.compile_model(model, device)
    try:
        return str(compiled_model.get_property("EXECUTION_DEVICES"))
    except Exception:
        return device


def export_yolo_model(
    model: str = "yolo26n.pt",
    export_format: str = "openvino",
    imgsz: int = 640,
    half: bool = False,
    int8: bool = False,
    dynamic: bool = False,
) -> str:
    from ultralytics import YOLO  # type: ignore

    yolo = YOLO(model)
    output = yolo.export(format=export_format, imgsz=imgsz, half=half, int8=int8, dynamic=dynamic)
    return str(output)


@dataclass(slots=True)
class BenchmarkResult:
    runtime: str
    model: str
    frames: int
    elapsed_s: float
    fps: float
    cold_start_s: float


def run_benchmark(
    model: str,
    source: int | str = 0,
    runtime: str = "openvino",
    frames: int = 300,
    output: str | Path | None = None,
) -> BenchmarkResult:
    cold_start_begin = time.perf_counter()
    detector = UltralyticsDetector(model)
    cold_start_s = time.perf_counter() - cold_start_begin

    camera = USBCameraSource(source)
    processed = 0
    begin = time.perf_counter()
    try:
        for frame in camera.frames():
            detector.predict(frame)
            processed += 1
            if processed >= frames:
                break
    finally:
        camera.close()

    elapsed_s = max(time.perf_counter() - begin, 1e-9)
    result = BenchmarkResult(runtime, model, processed, elapsed_s, processed / elapsed_s, cold_start_s)
    if output:
        write_benchmark(output, result)
    return result


def write_benchmark(path: str | Path, result: BenchmarkResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.__dict__.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(result.__dict__)
