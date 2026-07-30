"""主角追蹤行為（規格 §4）。"""

from __future__ import annotations

import pytest
from conftest import FRAME_H, FRAME_W, face

from focus_keeper.validation import ConfigError
from focus_keeper.tracker import PrimarySubjectTracker, TrackerConfig

FRAME = (FRAME_W, FRAME_H)


def make_tracker(**overrides) -> PrimarySubjectTracker:
    return PrimarySubjectTracker(TrackerConfig(**overrides))


class TestInitialization:
    def test_picks_largest_and_most_central(self) -> None:
        tracker = make_tracker(smoothing=0.0)
        detections = [
            face(200, 200, size=100),                      # 小、偏角落
            face(FRAME_W / 2, FRAME_H / 2, size=180),      # 大、正中央 -> 應中選
            face(1100, 600, size=170),                     # 幾乎一樣大但偏角落
        ]
        result = tracker.update(detections, timestamp_ms=0, frame_size=FRAME)

        assert result.initialized is True
        assert result.primary is not None
        assert result.primary.bbox.cx == pytest.approx(FRAME_W / 2)
        assert result.person_count == 3

    def test_central_beats_slightly_larger_offcenter(self) -> None:
        """中央性與大小同權重：中央的中等臉框應贏過角落的稍大臉框。"""
        tracker = make_tracker(smoothing=0.0)
        result = tracker.update(
            [face(FRAME_W / 2, FRAME_H / 2, size=150), face(120, 120, size=175)],
            timestamp_ms=0,
            frame_size=FRAME,
        )
        assert result.primary is not None
        assert result.primary.bbox.cx == pytest.approx(FRAME_W / 2)

    def test_no_detection_means_no_primary(self) -> None:
        tracker = make_tracker()
        result = tracker.update([], timestamp_ms=0, frame_size=FRAME)
        assert result.primary is None
        assert result.person_count == 0
        assert result.initialized is False


class TestMaintenance:
    def test_does_not_reselect_each_frame(self) -> None:
        """規格明文：不得每幀重新選主角。

        主角先在中央被選中，之後畫面出現一個更大更靠中央的第二人，
        主角身分必須留在原本那位（track_id 不變、框仍跟著原本的人）。
        """
        tracker = make_tracker(smoothing=0.0)
        tracker.update([face(FRAME_W / 2, FRAME_H / 2, size=150)], timestamp_ms=0, frame_size=FRAME)
        original_id = tracker.primary.track_id

        for step in range(1, 11):
            ts = step * 100
            primary_face = face(FRAME_W / 2 + step, FRAME_H / 2, size=150)
            intruder = face(FRAME_W / 2 + 260, FRAME_H / 2 + 10, size=240)  # 更大、也很中央
            result = tracker.update([primary_face, intruder], timestamp_ms=ts, frame_size=FRAME)

        assert result.primary is not None
        assert result.primary.track_id == original_id
        assert result.primary.bbox.cx == pytest.approx(FRAME_W / 2 + 10, abs=2.0)
        assert result.person_count == 2

    def test_tracks_through_gradual_motion(self) -> None:
        tracker = make_tracker(smoothing=0.0)
        tracker.update([face(400, 360)], timestamp_ms=0, frame_size=FRAME)
        original_id = tracker.primary.track_id

        for step in range(1, 16):
            result = tracker.update(
                [face(400 + step * 20, 360)], timestamp_ms=step * 100, frame_size=FRAME
            )
            assert result.primary is not None
            assert result.primary.visible is True
            assert result.primary.track_id == original_id

    def test_rejects_teleporting_candidate(self) -> None:
        """位移超過門檻的候選不得被當成同一人。"""
        tracker = make_tracker(smoothing=0.0, max_center_shift_ratio=0.10)
        tracker.update([face(300, 360)], timestamp_ms=0, frame_size=FRAME)
        result = tracker.update([face(1200, 360)], timestamp_ms=100, frame_size=FRAME)

        assert result.primary is not None
        assert result.primary.visible is False  # 沒匹配到，進入寬限期
        assert result.primary.bbox.cx == pytest.approx(300)

    def test_rejects_wildly_different_size(self) -> None:
        tracker = make_tracker(smoothing=0.0, max_area_ratio=2.0)
        tracker.update([face(640, 360, size=100)], timestamp_ms=0, frame_size=FRAME)
        result = tracker.update([face(640, 360, size=300)], timestamp_ms=100, frame_size=FRAME)
        assert result.primary is not None
        assert result.primary.visible is False


class TestMissingAndRelease:
    def test_brief_miss_keeps_identity(self) -> None:
        tracker = make_tracker(smoothing=0.0, lost_after_ms=3000)
        tracker.update([face(640, 360)], timestamp_ms=0, frame_size=FRAME)
        original_id = tracker.primary.track_id

        gap = tracker.update([], timestamp_ms=100, frame_size=FRAME)
        assert gap.primary is not None
        assert gap.primary.visible is False
        assert gap.primary.missing_ms == 100

        back = tracker.update([face(645, 362)], timestamp_ms=200, frame_size=FRAME)
        assert back.primary is not None
        assert back.primary.visible is True
        assert back.primary.missing_ms == 0
        assert back.primary.track_id == original_id

    def test_releases_after_lost_timeout(self) -> None:
        tracker = make_tracker(lost_after_ms=1000)
        tracker.update([face(640, 360)], timestamp_ms=0, frame_size=FRAME)

        assert tracker.update([], timestamp_ms=1000, frame_size=FRAME).primary is not None
        released = tracker.update([], timestamp_ms=1100, frame_size=FRAME)
        assert released.released is True
        assert released.primary is None

    def test_does_not_hand_over_identity_on_the_release_frame(self) -> None:
        """釋放主角的那一幀不得立刻改認在場的別人。"""
        tracker = make_tracker(lost_after_ms=1000)
        tracker.update([face(300, 200, size=120)], timestamp_ms=0, frame_size=FRAME)
        first_id = tracker.primary.track_id

        bystander = [face(1000, 600, size=200)]
        release_frame = tracker.update(bystander, timestamp_ms=1100, frame_size=FRAME)
        assert release_frame.released is True
        assert release_frame.primary is None

        next_frame = tracker.update(bystander, timestamp_ms=1200, frame_size=FRAME)
        assert next_frame.initialized is True
        assert next_frame.primary is not None
        assert next_frame.primary.track_id != first_id


class TestConfig:
    def test_rejects_unknown_field(self) -> None:
        with pytest.raises(ConfigError):
            TrackerConfig.from_dict({"nope": 1})

    def test_from_dict_roundtrip(self) -> None:
        cfg = TrackerConfig.from_dict({"lost_after_ms": 2500, "smoothing": 0.0})
        assert cfg.lost_after_ms == 2500
        assert cfg.smoothing == 0.0
