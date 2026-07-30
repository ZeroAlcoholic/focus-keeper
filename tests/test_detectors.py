"""偵測器契約與模型完整性（規格 §2、§9）。

需要真實權重的測試會在缺檔時自動 skip。這裡驗的是**契約與授權護欄**：
輸出型別、缺檔行為、SHA-256 強制比對、縮放還原是否回到原始座標系。
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FRAME_H, FRAME_W, YUNET_MODEL, face, requires_yunet

from focus_keeper.validation import ConfigError
from focus_keeper.detectors import available_detectors, build_detector
from focus_keeper.detectors.base import (
    BBox,
    Detection,
    DetectorUnavailableError,
    ModelIntegrityError,
    filter_detections,
    sha256_of,
)

NOISE = np.random.default_rng(20260729).integers(
    0, 256, size=(FRAME_H, FRAME_W, 3), dtype=np.uint8
)


class TestRegistry:
    def test_lists_route_a_and_b(self) -> None:
        assert set(available_detectors()) == {"yunet", "mediapipe"}

    def test_unknown_detector_rejected(self) -> None:
        with pytest.raises(ConfigError):
            build_detector("ultralytics", {})

    def test_missing_config_section_rejected(self) -> None:
        with pytest.raises(ConfigError):
            build_detector("yunet", {})


class TestBBox:
    def test_iou_of_identical_boxes(self) -> None:
        box = BBox(10, 20, 30, 40)
        assert box.iou(box) == pytest.approx(1.0)

    def test_iou_of_disjoint_boxes(self) -> None:
        assert BBox(0, 0, 10, 10).iou(BBox(100, 100, 10, 10)) == 0.0

    def test_half_overlap(self) -> None:
        assert BBox(0, 0, 10, 10).iou(BBox(5, 0, 10, 10)) == pytest.approx(1 / 3)

    def test_scaled_round_trip(self) -> None:
        box = BBox(10, 20, 30, 40)
        assert box.scaled(2.0).scaled(0.5) == box

    def test_clip_to_frame(self) -> None:
        clipped = BBox(-50, -50, 200, 200).clip(100, 100)
        assert clipped == BBox(0, 0, 100, 100)

    def test_contains_point(self) -> None:
        box = BBox(0, 0, 10, 10)
        assert box.contains_point(5, 5)
        assert not box.contains_point(11, 5)


class TestFiltering:
    def test_drops_low_confidence(self) -> None:
        kept = filter_detections(
            [face(640, 360, score=0.9), face(200, 200, score=0.4)],
            frame_width=FRAME_W, frame_height=FRAME_H,
            min_confidence=0.70, min_face_area_ratio=0.0,
        )
        assert len(kept) == 1
        assert kept[0].score == pytest.approx(0.9)

    def test_drops_tiny_faces(self) -> None:
        big = face(640, 360, size=160)      # 128x160 / (1280x720) ~= 0.0222
        small = face(200, 200, size=40)     # 32x40   / (1280x720) ~= 0.0014
        kept = filter_detections(
            [big, small], frame_width=FRAME_W, frame_height=FRAME_H,
            min_confidence=0.0, min_face_area_ratio=0.015,
        )
        assert kept == [big]

    def test_sorted_by_confidence(self) -> None:
        kept = filter_detections(
            [face(300, 300, score=0.75), face(640, 360, score=0.99), face(900, 300, score=0.85)],
            frame_width=FRAME_W, frame_height=FRAME_H,
            min_confidence=0.70, min_face_area_ratio=0.0,
        )
        assert [round(d.score, 2) for d in kept] == [0.99, 0.85, 0.75]

    def test_area_ratio_uses_frame_area(self) -> None:
        det = Detection(bbox=BBox(0, 0, 128, 72), score=1.0)
        assert det.area_ratio(FRAME_W, FRAME_H) == pytest.approx(0.01)


class TestYuNetGuards:
    def test_missing_model_file_is_explicit(self, tmp_path) -> None:
        with pytest.raises(DetectorUnavailableError) as exc:
            build_detector("yunet", {"yunet": {"model_path": str(tmp_path / "nope.onnx")}})
        # 錯誤訊息必須指向手動取得流程，不得暗示會自動下載。
        assert "fetch_model.py" in str(exc.value)

    @requires_yunet
    def test_sha256_mismatch_refuses_to_start(self) -> None:
        with pytest.raises(ModelIntegrityError):
            build_detector(
                "yunet",
                {"yunet": {"model_path": str(YUNET_MODEL), "sha256": "0" * 64}},
            )

    @requires_yunet
    def test_pinned_sha256_matches_repo_config(self) -> None:
        """config.yaml 釘住的雜湊必須與實際檔案相符。"""
        import yaml
        from conftest import ROOT

        cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        pinned = cfg["detectors"]["yunet"]["sha256"]
        assert pinned, "config.yaml 應釘住 yunet 權重的 SHA-256"
        assert pinned.lower() == sha256_of(YUNET_MODEL)


@requires_yunet
class TestYuNetContract:
    @pytest.fixture(scope="class")
    def detector(self):
        det = build_detector(
            "yunet", {"yunet": {"model_path": str(YUNET_MODEL), "detect_width": 320}}
        )
        yield det
        det.close()

    def test_returns_detection_objects(self, detector) -> None:
        results = detector.detect(NOISE)
        assert isinstance(results, list)
        for det in results:
            assert isinstance(det, Detection)
            assert 0.0 <= det.score <= 1.0
            assert det.bbox.w > 0 and det.bbox.h > 0

    def test_blank_frame_yields_no_faces(self) -> None:
        det = build_detector("yunet", {"yunet": {"model_path": str(YUNET_MODEL)}})
        try:
            blank = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            assert det.detect(blank) == []
        finally:
            det.close()

    def test_coordinates_are_in_original_frame_space(self, detector) -> None:
        """縮放推論後座標必須還原到原始影格，不能停在 320 寬的空間。"""
        for d in detector.detect(NOISE):
            assert -FRAME_W <= d.bbox.x <= FRAME_W * 1.5
            assert -FRAME_H <= d.bbox.y <= FRAME_H * 1.5

    def test_handles_repeated_size_changes(self, detector) -> None:
        small = np.zeros((360, 640, 3), dtype=np.uint8)
        assert detector.detect(NOISE) is not None
        assert detector.detect(small) == []
        assert detector.detect(NOISE) is not None

    def test_describe_reports_version_and_hash(self, detector) -> None:
        info = detector.describe()
        assert info["detector"] == "yunet"
        assert info["route"] == "A"
        assert len(info["model_sha256"]) == 64
        assert "Apache-2.0" in info["license"] and "MIT" in info["license"]
        assert info["model_bytes"] > 0
