from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass(slots=True)
class Detection:
    """Single object detection in image coordinates."""

    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    track_id: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(slots=True)
class FrameResult:
    """Inference result for one frame."""

    frame_index: int
    timestamp: datetime
    detections: list[Detection]
    source: str = "camera"
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuleEvent:
    """Clinician-review event emitted by the bounded rules engine."""

    name: str
    timestamp: datetime
    severity: str = "info"
    message: str = ""
    zone: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.astimezone(timezone.utc).isoformat(),
            "name": self.name,
            "severity": self.severity,
            "message": self.message,
            "zone": self.zone or "",
            "data": self.data,
        }


@dataclass(slots=True)
class RuntimeConfig:
    zones: dict[str, dict[str, Any]] = field(default_factory=dict)
    rules: list[dict[str, Any]] = field(default_factory=list)
    class_aliases: dict[str, str] = field(default_factory=dict)
    save_review_clips: bool = False
    no_cloud_mode: bool = True


def load_config(path: str | Path) -> RuntimeConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return RuntimeConfig(
        zones=raw.get("zones", {}) or {},
        rules=raw.get("rules", []) or [],
        class_aliases=raw.get("class_aliases", {}) or {},
        save_review_clips=bool(raw.get("save_review_clips", False)),
        no_cloud_mode=bool(raw.get("no_cloud_mode", True)),
    )


def point_in_polygon(point: tuple[float, float], polygon: list[list[float]] | list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test for simple image-coordinate polygons."""

    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    px1, py1 = polygon[0]
    for i in range(n + 1):
        px2, py2 = polygon[i % n]
        if min(py1, py2) <= y <= max(py1, py2) and x <= max(px1, px2):
            xinters = px1 if py1 == py2 else (y - py1) * (px2 - px1) / (py2 - py1) + px1
            if px1 == px2 or x <= xinters:
                inside = not inside
        px1, py1 = px2, py2
    return inside


class RehabRuleEngine:
    """Stateful rule engine for clinically bounded review events.

    The engine emits prompts such as `session_started` or `patient_left_frame`. It does not
    diagnose, score impairment, infer fall risk, or make therapy decisions.
    """

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self._first_true_at: dict[str, datetime] = {}
        self._active: set[str] = set()

    def evaluate(self, frame: FrameResult) -> list[RuleEvent]:
        detections = self._normalise(frame.detections)
        events: list[RuleEvent] = []
        for rule in self.config.rules:
            ok, data = self._evaluate_rule(rule, detections)
            event = self._duration_gate(rule, ok, frame.timestamp, data)
            if event:
                events.append(event)
        return events

    def _normalise(self, detections: list[Detection]) -> list[Detection]:
        out: list[Detection] = []
        for det in detections:
            class_name = self.config.class_aliases.get(det.class_name, det.class_name)
            out.append(Detection(class_name, det.confidence, det.xyxy, det.track_id))
        return out

    def _evaluate_rule(self, rule: dict[str, Any], detections: list[Detection]) -> tuple[bool, dict]:
        rule_type = rule.get("type")
        class_name = rule.get("class_name")
        min_count = int(rule.get("min_count", 1))

        if rule_type == "absence":
            count = sum(1 for det in detections if det.class_name == class_name)
            return count == 0, {"class_name": class_name, "count": count}

        if rule_type == "min_count":
            count = sum(1 for det in detections if det.class_name == class_name)
            return count >= min_count, {"class_name": class_name, "count": count, "min_count": min_count}

        if rule_type in {"presence_in_zone", "object_in_zone"}:
            zone_name = rule.get("zone")
            zone = self.config.zones.get(zone_name)
            if not zone:
                return False, {"error": f"unknown zone: {zone_name}"}
            polygon = zone.get("polygon", [])
            in_zone = [
                det
                for det in detections
                if det.class_name == class_name and point_in_polygon(det.center, polygon)
            ]
            return len(in_zone) >= min_count, {
                "class_name": class_name,
                "zone": zone_name,
                "count": len(in_zone),
                "min_count": min_count,
            }

        return False, {"error": f"unknown rule type: {rule_type}"}

    def _duration_gate(
        self, rule: dict[str, Any], ok: bool, timestamp: datetime, data: dict[str, Any]
    ) -> RuleEvent | None:
        name = str(rule.get("name", "unnamed_rule"))
        min_duration_s = float(rule.get("min_duration_s", 0))
        if ok:
            if name not in self._first_true_at:
                self._first_true_at[name] = timestamp
                if min_duration_s > 0:
                    return None
            elapsed = (timestamp - self._first_true_at[name]).total_seconds()
            if elapsed >= min_duration_s and name not in self._active:
                self._active.add(name)
                return RuleEvent(
                    name=name,
                    timestamp=timestamp,
                    severity=str(rule.get("severity", "info")),
                    message=str(rule.get("message") or default_message(rule)),
                    zone=rule.get("zone"),
                    data={"elapsed_s": round(elapsed, 3), **data},
                )
        else:
            self._first_true_at.pop(name, None)
            self._active.discard(name)
        return None


def default_message(rule: dict[str, Any]) -> str:
    rule_type = rule.get("type")
    if rule_type == "presence_in_zone":
        return f"{rule.get('class_name')} present in {rule.get('zone')}; review context if needed."
    if rule_type == "absence":
        return f"No {rule.get('class_name')} detected; check camera framing or patient location."
    if rule_type == "min_count":
        return f"At least {rule.get('min_count', 1)} {rule.get('class_name')} detections present."
    return "Rule triggered."


def summarise_detections_by_zone(detections: list[Detection], config: RuntimeConfig) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for zone_name, zone in config.zones.items():
        polygon = zone.get("polygon", [])
        for det in detections:
            if point_in_polygon(det.center, polygon):
                summary[zone_name][det.class_name] += 1
    return {zone: dict(counts) for zone, counts in summary.items()}


class CSVEventWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, events: Iterable[RuleEvent]) -> None:
        events = list(events)
        if not events:
            return
        new_file = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "name", "severity", "message", "zone", "data"])
            if new_file:
                writer.writeheader()
            for event in events:
                row = event.to_dict()
                row["data"] = json.dumps(row["data"], ensure_ascii=False)
                writer.writerow(row)


class JSONLEventWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, events: Iterable[RuleEvent]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            for event in events:
                f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")


def read_csv_events(path: str | Path, limit: int = 200) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))[-limit:]
