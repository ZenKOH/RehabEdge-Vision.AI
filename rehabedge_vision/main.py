from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from rehabedge_vision.core import (
    CSVEventWriter,
    Detection,
    FrameResult,
    JSONLEventWriter,
    RehabRuleEngine,
    load_config,
    read_csv_events,
)
from rehabedge_vision.inference import (
    USBCameraSource,
    compile_ir,
    configure_openvino_cache,
    export_yolo_model,
    run_benchmark,
    verify_openvino_devices,
    UltralyticsDetector,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rehabedge")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("verify-openvino", help="Show OpenVINO runtime devices.")
    p.add_argument("--cache-dir", default="ov_cache")

    p = sub.add_parser("compile-ir", help="Compile an OpenVINO IR model and warm cache.")
    p.add_argument("--xml", required=True)
    p.add_argument("--device", default="CPU")
    p.add_argument("--cache-dir", default="ov_cache")

    p = sub.add_parser("export-yolo", help="Export a YOLO model using Ultralytics.")
    p.add_argument("--model", default="yolo26n.pt")
    p.add_argument("--format", default="openvino")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--dynamic", action="store_true")

    p = sub.add_parser("run", help="Run camera inference and emit local rehab events.")
    p.add_argument("--model", required=True)
    p.add_argument("--runtime", default="openvino")
    p.add_argument("--source", default="0")
    p.add_argument("--config", default="configs/default_zones.yaml")
    p.add_argument("--events", default="events.csv")
    p.add_argument("--jsonl", default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--max-frames", type=int, default=0)

    p = sub.add_parser("benchmark", help="Run a simple FPS benchmark.")
    p.add_argument("--model", required=True)
    p.add_argument("--source", default="0")
    p.add_argument("--runtime", default="openvino")
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--output", default="benchmarks/benchmark.csv")

    p = sub.add_parser("dashboard", help="Run local event dashboard.")
    p.add_argument("--events", default="events.csv")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    p = sub.add_parser("demo-rules", help="Run a synthetic rules-engine demo without a camera.")
    p.add_argument("--config", default="configs/default_zones.yaml")
    return parser


def coerce_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def cmd_run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    engine = RehabRuleEngine(config)
    csv_writer = CSVEventWriter(args.events)
    jsonl_writer = JSONLEventWriter(args.jsonl) if args.jsonl else None
    detector = UltralyticsDetector(args.model, conf=args.conf, imgsz=args.imgsz)
    camera = USBCameraSource(coerce_source(args.source))

    processed = 0
    try:
        for frame in camera.frames():
            detections = detector.predict(frame)
            frame_result = FrameResult(
                frame_index=processed,
                timestamp=datetime.now(timezone.utc),
                detections=detections,
                source=str(args.source),
            )
            events = engine.evaluate(frame_result)
            csv_writer.write(events)
            if jsonl_writer:
                jsonl_writer.write(events)
            for event in events:
                print(f"{event.timestamp.isoformat()} {event.name}: {event.message}")
            processed += 1
            if args.max_frames and processed >= args.max_frames:
                break
    finally:
        camera.close()


def cmd_dashboard(args: argparse.Namespace) -> None:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="RehabEdge-Vision.AI")

    @app.get("/api/events")
    def events(limit: int = 200) -> list[dict]:
        return read_csv_events(args.events, limit=limit)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        rows = []
        for event in reversed(read_csv_events(args.events, limit=200)):
            rows.append(
                "<tr>"
                f"<td>{event.get('timestamp', '')}</td>"
                f"<td>{event.get('name', '')}</td>"
                f"<td>{event.get('severity', '')}</td>"
                f"<td>{event.get('zone', '')}</td>"
                f"<td>{event.get('message', '')}</td>"
                f"<td><code>{event.get('data', '')}</code></td>"
                "</tr>"
            )
        html = """
        <!doctype html><html><head><meta charset='utf-8'><title>RehabEdge-Vision.AI</title>
        <style>body{font-family:system-ui;margin:2rem;background:#0f172a;color:#e2e8f0}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #334155;padding:.5rem;text-align:left}th{color:#93c5fd}</style>
        </head><body><h1>RehabEdge-Vision.AI</h1><p>Local review events. Not clinical decisions.</p>
        <table><thead><tr><th>Time</th><th>Event</th><th>Severity</th><th>Zone</th><th>Message</th><th>Data</th></tr></thead><tbody>{rows}</tbody></table>
        </body></html>
        """.format(rows="\n".join(rows))
        return HTMLResponse(html)

    uvicorn.run(app, host=args.host, port=args.port)


def cmd_demo_rules(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    engine = RehabRuleEngine(config)
    now = datetime.now(timezone.utc)
    frames = [
        FrameResult(0, now, [Detection("person", 0.9, (100, 150, 250, 550))]),
        FrameResult(1, now + timedelta(seconds=6), [Detection("person", 0.9, (100, 150, 250, 550))]),
        FrameResult(2, now + timedelta(seconds=12), []),
        FrameResult(3, now + timedelta(seconds=16), []),
    ]
    for frame in frames:
        for event in engine.evaluate(frame):
            print(event.to_dict())


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "verify-openvino":
        print("OpenVINO devices:", verify_openvino_devices())
        print("OpenVINO cache configured. Devices:", configure_openvino_cache(args.cache_dir))
    elif args.command == "compile-ir":
        print("Compiled execution devices:", compile_ir(args.xml, args.device, args.cache_dir))
    elif args.command == "export-yolo":
        print("Exported to:", export_yolo_model(args.model, args.format, args.imgsz, args.half, args.int8, args.dynamic))
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "benchmark":
        print(run_benchmark(args.model, coerce_source(args.source), args.runtime, args.frames, args.output))
    elif args.command == "dashboard":
        cmd_dashboard(args)
    elif args.command == "demo-rules":
        cmd_demo_rules(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
