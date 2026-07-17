from datetime import datetime, timedelta, timezone

from rehabedge_vision.core import Detection, FrameResult, RehabRuleEngine, RuntimeConfig, point_in_polygon


def test_point_in_polygon_inside():
    polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), polygon)


def test_point_in_polygon_outside():
    polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert not point_in_polygon((15, 5), polygon)


def test_presence_in_zone_duration_gate():
    config = RuntimeConfig(
        zones={"lane": {"polygon": [(0, 0), (100, 0), (100, 100), (0, 100)]}},
        rules=[
            {
                "name": "session_started",
                "type": "presence_in_zone",
                "zone": "lane",
                "class_name": "person",
                "min_duration_s": 5,
            }
        ],
    )
    engine = RehabRuleEngine(config)
    now = datetime.now(timezone.utc)
    det = Detection("person", 0.9, (10, 10, 40, 90))
    assert engine.evaluate(FrameResult(0, now, [det])) == []
    events = engine.evaluate(FrameResult(1, now + timedelta(seconds=6), [det]))
    assert len(events) == 1
    assert events[0].name == "session_started"


def test_absence_duration_gate():
    config = RuntimeConfig(
        rules=[
            {
                "name": "patient_left_frame",
                "type": "absence",
                "class_name": "person",
                "min_duration_s": 3,
            }
        ]
    )
    engine = RehabRuleEngine(config)
    now = datetime.now(timezone.utc)
    assert engine.evaluate(FrameResult(0, now, [])) == []
    events = engine.evaluate(FrameResult(1, now + timedelta(seconds=4), []))
    assert len(events) == 1
    assert events[0].name == "patient_left_frame"
