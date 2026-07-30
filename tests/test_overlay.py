"""預覽疊圖（規格 §5：即時顯示框線、人物數、狀態、FPS、延遲）。

補洞說明：此檔是 2026-07-29 證偽輪的產物。在此之前，`draw_overlay`
**零測試覆蓋且從未被執行過**——所有驗證都走 `--no-display`，
而預覽正是 DEMO 會用到的路徑。
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FRAME_H, FRAME_W, TimelineDetector, face

import cv2

from focus_keeper.config import PipelineConfig
from focus_keeper.metrics import LatencyStats
from focus_keeper.overlay import draw_overlay
from focus_keeper.pipeline import AnalysisPipeline
from focus_keeper.detectors.base import BBox
from focus_keeper.rules import EventType, RuleConfig, TemporalRuleEngine
from focus_keeper.sources import FramePacket
from focus_keeper.tracker import PrimarySubjectTracker, TrackerConfig

BASE = np.full((FRAME_H, FRAME_W, 3), 210, np.uint8)


def content(timestamp_ms: int) -> list:
    """在場 -> 離場 -> 回來且多一人。

    離場窗必須 > primary_left_ms(2000)，否則 PRIMARY_LEFT 永遠不會出現，
    這個狀態的疊圖也就測不到。
    """
    if 1000 <= timestamp_ms < 4500:
        return []
    if timestamp_ms >= 6000:
        return [face(FRAME_W / 2, FRAME_H / 2), face(260, 320, size=150)]
    return [face(FRAME_W / 2, FRAME_H / 2)]


def render_until(status: EventType, *, max_ms: int = 9000):
    """跑到指定狀態出現，回傳 (疊圖結果, FrameResult)。"""
    pipeline = AnalysisPipeline(
        detector=TimelineDetector(content),
        tracker=PrimarySubjectTracker(TrackerConfig(smoothing=0.0)),
        # 本檔每格都餵同一張 BASE 影像（才能驗「不得改到原始影格」），
        # 那在真實情境下就是來源停格，FEED_FROZEN 會正確地蓋掉其他狀態。
        # 停格本身有自己的測試（test_rules.py::TestFeedFrozen），這裡關掉。
        rules=TemporalRuleEngine(RuleConfig(feed_frozen_ms=0)),
        config=PipelineConfig(inference_fps=10.0),
    )
    stats = LatencyStats()
    for frame_id in range(int(max_ms / 1000 * 30)):
        ts = int(round(frame_id * 1000 / 30))
        pipeline.detector.now_ms = ts
        result = pipeline.process(
            FramePacket(image=BASE, timestamp_ms=ts, frame_id=frame_id, capture_monotonic=0.0)
        )
        if result is None:
            continue
        stats.add(result)
        if result.rule_update.status is status:
            return draw_overlay(BASE, result, stats, detector_name="yunet"), result
    pytest.fail(f"情境未進入 {status.value} 狀態")


@pytest.mark.parametrize(
    "status",
    [
        EventType.NORMAL,
        EventType.PRIMARY_MISSING,
        EventType.PRIMARY_LEFT,
        EventType.MULTI_PERSON,
    ],
)
def test_renders_every_status_without_error(status: EventType) -> None:
    canvas, _ = render_until(status)
    assert canvas.shape == BASE.shape
    assert canvas.dtype == BASE.dtype


def test_does_not_mutate_source_frame() -> None:
    """疊圖必須畫在副本上——原始影格不得被改到。"""
    before = BASE.copy()
    render_until(EventType.MULTI_PERSON)
    assert np.array_equal(BASE, before)


def test_overlay_actually_draws_something() -> None:
    """不能只確認「沒拋例外」：畫面必須真的和原圖不同。"""
    canvas, _ = render_until(EventType.MULTI_PERSON)
    assert not np.array_equal(canvas, BASE)
    changed = np.count_nonzero(np.any(canvas != BASE, axis=2))
    assert changed > 2000, f"只改了 {changed} 個像素，疊圖可能沒畫出來"


def test_predicted_primary_marks_stale_confidence() -> None:
    """漏偵測寬限期顯示的是殘值信心，必須標明，否則畫面會誤導。"""
    import inspect

    from focus_keeper import overlay as app

    source = inspect.getsource(app.draw_overlay)
    assert "predicted, last" in source
    canvas, result = render_until(EventType.PRIMARY_MISSING)
    assert result.primary is not None
    assert result.primary.visible is False
    assert canvas is not None


def test_handles_missing_primary_and_empty_roi() -> None:
    """主角已被釋放（primary=None）時仍要能畫，不能 crash。"""
    pipeline = AnalysisPipeline(
        detector=TimelineDetector(lambda _ms: []),
        tracker=PrimarySubjectTracker(TrackerConfig()),
        rules=TemporalRuleEngine(RuleConfig()),
        config=PipelineConfig(inference_fps=10.0),
    )
    stats = LatencyStats()
    result = pipeline.process(
        FramePacket(image=BASE, timestamp_ms=0, frame_id=0, capture_monotonic=0.0)
    )
    assert result is not None and result.primary is None
    stats.add(result)
    assert draw_overlay(BASE, result, stats, detector_name="yunet") is not None


def test_hud_fits_inside_small_frames() -> None:
    """HUD 面板不得超出畫面（低解析度是規格 §7 的降載手段）。"""
    small = np.zeros((240, 426, 3), np.uint8)
    pipeline = AnalysisPipeline(
        detector=TimelineDetector(lambda _ms: [face(213, 120, size=90)]),
        tracker=PrimarySubjectTracker(TrackerConfig()),
        rules=TemporalRuleEngine(RuleConfig()),
        config=PipelineConfig(inference_fps=10.0, min_face_area_ratio=0.0),
    )
    stats = LatencyStats()
    result = pipeline.process(
        FramePacket(image=small, timestamp_ms=0, frame_id=0, capture_monotonic=0.0)
    )
    assert result is not None
    stats.add(result)
    canvas = draw_overlay(small, result, stats, detector_name="yunet")
    assert canvas.shape == small.shape


def test_gui_backend_is_available() -> None:
    """DEMO 會用到 imshow；若 OpenCV 沒帶 GUI backend 要在這裡就發現。"""
    window = "focus_keeper_selftest"
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.imshow(window, np.zeros((32, 64, 3), np.uint8))
        cv2.waitKey(1)
    except cv2.error as exc:  # pragma: no cover - 無頭環境
        pytest.skip(f"此環境無 GUI backend：{exc}")
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


class TestBoxDrawing:
    def test_draw_box_handles_clipped_geometry(self) -> None:
        from focus_keeper.overlay import draw_box, draw_dashed_box

        canvas = BASE.copy()
        # 肩上構圖框常常被畫面邊界裁切，甚至退化成零寬高。
        draw_box(canvas, BBox(-50, -50, 200, 200), (0, 255, 0))
        draw_dashed_box(canvas, BBox(0, 0, 0, 0), (0, 255, 0))
        draw_dashed_box(canvas, BBox(FRAME_W - 5, FRAME_H - 5, 100, 100), (0, 255, 0))
        assert canvas.shape == BASE.shape
