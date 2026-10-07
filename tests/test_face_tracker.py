"""Presence timing, camera coordinates and capture shutdown behavior."""

from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from glados.vision.face_tracker import FaceTracker, face_from_model


@pytest.fixture
def tracker(monkeypatch: pytest.MonkeyPatch) -> FaceTracker:
    front = Mock()
    profile = Mock()
    for cascade in (front, profile):
        cascade.empty.return_value = False
        cascade.detectMultiScale.return_value = ()
    monkeypatch.setattr(cv2, "CascadeClassifier", Mock(side_effect=[front, profile]))
    return FaceTracker(backend="haar", interval_s=1)


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def test_yunet_uses_cpu_and_native_boxes_at_camera_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = Mock()
    # YuNet supplies x/y/width/height, five landmarks, then confidence.
    detector.detect.return_value = (1, np.array([[200, 60, 80, 120, *([0] * 10), 0.9]], dtype=np.float32))
    create = Mock(return_value=detector)
    monkeypatch.setattr(cv2, "FaceDetectorYN", Mock(create=create))
    tracker = FaceTracker()
    assert create.call_args.args[-2:] == (cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
    tracker.update(FRAME, 100)
    detector.setInputSize.assert_called_once_with((320, 240))
    face = tracker.snapshot(100)
    assert face["source"] == "yunet" and face["present"]
    assert face["bbox"] == pytest.approx([0.625, 0.25, 0.25, 0.5])
    assert face["x"] == pytest.approx(0.5) and face["y"] == pytest.approx(0)
    tracker.update(FRAME, 100.01)
    assert detector.detect.call_count == 1
    tracker.update(FRAME, 100.05)
    assert detector.detect.call_count == 2
    detector.detect.return_value = (1, None)
    tracker.update(FRAME, 100.1)
    assert tracker.snapshot(100.1)["present"] is False
    assert "bbox" not in tracker.snapshot(100.1)
    detector.detect.side_effect = cv2.error("failed")
    tracker.update(FRAME, 100.15)
    assert tracker.snapshot(100.15)["present"] is None
    assert tracker.snapshot(100.15)["absent_seconds"] == 0


def test_bundled_yunet_loads_and_detects_empty_frame() -> None:
    tracker = FaceTracker()
    face = tracker.locate(FRAME, 100)
    assert face["available"] and face["present"] is False
    assert tracker.snapshot()["error"] is None


def test_largest_face_coordinates_and_throttling(tracker: FaceTracker) -> None:
    front, profile = tracker._cascades
    front.detectMultiScale.return_value = [(20, 30, 40, 40), (240, 60, 100, 120)]
    profile.detectMultiScale.return_value = [(30, 50, 50, 50)]
    tracker.update(FRAME, 100)
    face = tracker.snapshot(100)
    assert face["available"] and face["present"]
    assert face["x"] == pytest.approx(0.45)
    assert face["y"] == pytest.approx(-0.2)
    assert face["bbox"] == pytest.approx([0.6, 0.2, 0.25, 0.4])
    assert face["closeness"] == pytest.approx(0.4)
    tracker.update(FRAME, 100.5)
    assert front.detectMultiScale.call_count == 1
    tracker.update(FRAME, 101)
    assert front.detectMultiScale.call_count == 2


def test_mirrored_profile_coordinates(tracker: FaceTracker) -> None:
    tracker._cascades[1].detectMultiScale.side_effect = [(), [(20, 30, 80, 80)]]
    tracker.update(FRAME, 100)
    face = tracker.snapshot(100)
    assert face["present"]
    assert face["bbox"] == pytest.approx([0.75, 0.1, 0.2, 80 / 300])
    assert face["x"] == pytest.approx(0.7)


def test_absence_timer_resets_on_return_and_capture_reset(tracker: FaceTracker) -> None:
    tracker.update(FRAME, 100)
    assert tracker.snapshot(104)["absent_seconds"] == 4
    tracker._cascades[0].detectMultiScale.return_value = [(20, 30, 40, 40)]
    tracker.update(FRAME, 105)
    assert tracker.snapshot(106)["absent_seconds"] == 0
    tracker._cascades[0].detectMultiScale.return_value = ()
    tracker.update(FRAME, 106)
    tracker.update(FRAME, 107)
    assert tracker.snapshot(111)["absent_seconds"] == 5
    tracker.reset()
    assert not tracker.snapshot(120)["available"]
    tracker.update(FRAME, 120)
    assert tracker.snapshot(120)["absent_seconds"] == 0


def test_pause_during_detection_discards_presence(tracker: FaceTracker) -> None:
    def detect(*args: object, **kwargs: object) -> list[tuple[int, int, int, int]]:
        tracker.reset()
        return [(20, 30, 40, 40)]

    tracker._cascades[0].detectMultiScale.side_effect = detect
    tracker.update(FRAME, 100)
    assert not tracker.snapshot(100)["available"]


def test_failed_or_disabled_detector_is_unknown_not_absent(tracker: FaceTracker) -> None:
    tracker._cascades[0].detectMultiScale.side_effect = cv2.error("test failure")
    tracker.update(FRAME, 100)
    assert not tracker.snapshot(106)["available"]
    assert tracker.snapshot(106)["present"] is None
    disabled = FaceTracker(enabled=False)
    disabled.update(FRAME, 100)
    assert not disabled.snapshot(106)["available"]


def test_e4b_presence_requires_consecutive_confirmed_empty_observations(monkeypatch: pytest.MonkeyPatch) -> None:
    cascade = Mock(side_effect=AssertionError("E4B tracking must not load Haar"))
    monkeypatch.setattr(cv2, "CascadeClassifier", cascade)
    tracker = FaceTracker(backend="e4b")
    tracker.update(FRAME, 100)
    cascade.assert_not_called()
    tracker.publish(face_from_model({"presence": "present", "box_2d": [200, 600, 600, 800]}, 100))
    face = tracker.snapshot(100)
    assert face["source"] == "e4b" and face["present"]
    assert face["x"] == pytest.approx(0.4) and face["y"] == pytest.approx(-0.2)
    assert face["bbox"] == pytest.approx([0.6, 0.2, 0.2, 0.4])
    legacy = face_from_model({"presence": "present", "box": [600, 200, 800, 600]}, 100)
    assert legacy["bbox"] == face["bbox"]
    tracker.publish(face_from_model({"presence": "absent", "box": None}, 101))
    tracker.publish(face_from_model({"presence": "absent", "box": None}, 104))
    assert tracker.snapshot(106)["absent_seconds"] == 5
    tracker.publish(face_from_model({"presence": "uncertain", "box": None}, 106))
    assert tracker.snapshot(106)["present"] is None
    assert tracker.snapshot(106)["absent_seconds"] == 0
    tracker.publish(face_from_model({"presence": "absent", "box": None}, 107))
    assert tracker.snapshot(107)["absent_seconds"] == 0
    tracker.publish(face_from_model(None, 110))
    assert not tracker.snapshot(110)["available"] and tracker.snapshot(110)["absent_seconds"] == 0
    tracker.publish(face_from_model({"presence": "absent", "box": None}, 111))
    tracker.publish(face_from_model({"presence": "absent", "box": None}, 120))
    assert tracker.snapshot(120)["absent_seconds"] == 0, "Stale inference gaps restart absence timing"
    tracker.publish(face_from_model({"presence": "present", "box": None}, 121))
    assert tracker.snapshot(121)["present"] is True, "A hidden face must not make a visible person absent"
    assert "x" not in tracker.snapshot(121)


@pytest.mark.parametrize(
    "box",
    [
        [800, 200, 600, 400],
        [0, 0, 1001, 500],
        [True, 0, 500, 500],
        [0, 0, float("nan"), 500],
        [0, 0, 500],
        "600,200,800,600",
    ],
)
def test_invalid_model_coordinates_cannot_drive_the_eye(box: object) -> None:
    face = face_from_model({"presence": "present", "box_2d": box}, 100)
    assert face["present"] is True and "bbox" not in face and "x" not in face
    assert "error" in face
    contradiction = face_from_model({"presence": "absent", "box": [200, 200, 600, 600]}, 100)
    assert contradiction["present"] is None, "Contradictory model results must not trigger sleep"


def test_all_people_are_reported_without_duplicate_haar_detections(tracker: FaceTracker) -> None:
    tracker._cascades[0].detectMultiScale.return_value = [(20, 30, 60, 80), (250, 40, 100, 120)]
    tracker._cascades[1].detectMultiScale.side_effect = [[(21, 31, 60, 80)], ()]
    tracker.update(FRAME, 100)
    face = tracker.snapshot(100)
    assert len(face["faces"]) == 2
    assert face["bbox"] == face["faces"][0]["bbox"]
    assert face["faces"][0]["x"] > 0 and face["faces"][1]["x"] < 0
    tracker._cascades[0].detectMultiScale.return_value = ()
    tracker._cascades[1].detectMultiScale.side_effect = None
    tracker.update(FRAME, 102)
    assert tracker.snapshot(102)["faces"] == []


def test_yunet_reports_multiple_people_and_ignores_boxes_outside_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = Mock()
    detector.detect.return_value = (3, np.array([
        [20, 40, 60, 90, *([0] * 10), 0.95],
        [200, 60, 80, 120, *([0] * 10), 0.9],
        [500, 400, 80, 120, *([0] * 10), 0.9],
    ], dtype=np.float32))
    monkeypatch.setattr(cv2, "FaceDetectorYN", Mock(create=Mock(return_value=detector)))
    result = FaceTracker().locate(FRAME, 100)
    assert result["available"] and result["present"]
    assert len(result["faces"]) == 2
    assert result["faces"][0]["bbox"] == pytest.approx([0.625, 0.25, 0.25, 0.5])
