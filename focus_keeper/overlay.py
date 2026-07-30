"""預覽疊圖。這是唯一需要 OpenCV GUI 能力的模組。"""

from __future__ import annotations

import cv2
import numpy as np

from .detectors.base import BBox
from .metrics import LatencyStats
from .pipeline import FrameResult
from .rules import EventType

__all__ = ["draw_overlay", "draw_box", "draw_dashed_box"]

COLOR_NORMAL = (80, 220, 80)
COLOR_ALERT = (60, 60, 240)
COLOR_WARN = (0, 190, 255)
COLOR_OTHER = (170, 170, 170)
COLOR_ROI = (200, 140, 60)
COLOR_SHOULDER = (120, 200, 255)

STATUS_COLOR = {
    EventType.NORMAL: COLOR_NORMAL,
    EventType.MULTI_PERSON: COLOR_WARN,
    EventType.PRIMARY_OUTSIDE_ROI: COLOR_WARN,
    EventType.PRIMARY_MISSING: COLOR_ALERT,
    EventType.PRIMARY_LEFT: COLOR_ALERT,
    # 停格用洋紅，與「人不在」的紅色明確區分——這是系統故障，不是人的行為。
    EventType.FEED_FROZEN: (200, 60, 220),
}


def draw_box(image: np.ndarray, box, color, thickness: int = 2) -> None:
    x, y, w, h = box.to_int()
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)


def draw_dashed_box(image: np.ndarray, box, color, dash: int = 10) -> None:
    x, y, w, h = box.to_int()
    for i in range(x, x + w, dash * 2):
        cv2.line(image, (i, y), (min(i + dash, x + w), y), color, 1, cv2.LINE_AA)
        cv2.line(image, (i, y + h), (min(i + dash, x + w), y + h), color, 1, cv2.LINE_AA)
    for i in range(y, y + h, dash * 2):
        cv2.line(image, (x, i), (x, min(i + dash, y + h)), color, 1, cv2.LINE_AA)
        cv2.line(image, (x + w, i), (x + w, min(i + dash, y + h)), color, 1, cv2.LINE_AA)


def draw_overlay(
    image: np.ndarray,
    result: FrameResult,
    stats: LatencyStats,
    *,
    detector_name: str,
) -> np.ndarray:
    """在影格副本上畫出框線、狀態、FPS 與延遲（OpenCV 字型限 ASCII）。"""
    canvas = image.copy()
    update = result.rule_update
    status_color = STATUS_COLOR.get(update.status, COLOR_NORMAL)

    if update.roi_box is not None:
        draw_box(canvas, update.roi_box, COLOR_ROI, 1)

    primary_box = result.primary.bbox if result.primary else None
    # 畫在場門檻那一組：人數說 2 人，畫面就必須看得到 2 個框。
    for det in result.presence:
        if primary_box is not None and det.bbox.iou(primary_box) > 0.5:
            continue
        draw_box(canvas, det.bbox, COLOR_OTHER, 1)
        x, y, _, _ = det.bbox.to_int()
        cv2.putText(
            canvas, f"{det.score:.2f}", (x, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_OTHER, 1, cv2.LINE_AA,
        )

    if result.primary is not None:
        if update.shoulder_box is not None:
            draw_dashed_box(canvas, update.shoulder_box, COLOR_SHOULDER)
        draw_box(canvas, result.primary.bbox, status_color, 2)
        x, y, _, _ = result.primary.bbox.to_int()
        # 漏偵測寬限期內主角並未被偵測到，信心值是上次的殘值——必須標明，
        # 否則畫面會讓人以為當下仍有 0.86 的偵測。
        label = (
            f"PRIMARY {result.primary.score:.2f}"
            if result.primary.visible
            else f"PRIMARY (predicted, last {result.primary.score:.2f})"
        )
        cv2.putText(
            canvas, label, (x, max(12, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA,
        )

    lines = [
        f"STATUS  {update.status.value}",
        f"PERSONS {update.person_count}",
        f"FPS     {stats.recent_fps():.1f} (max {stats.headroom_fps():.0f})",
        f"LATENCY {result.end_to_end_ms:.0f} ms  p95 {stats.hud_p95_end_to_end():.0f} ms",
        f"MODEL   {detector_name}",
    ]
    box_h = 18 * len(lines) + 12
    overlay = canvas.copy()
    cv2.rectangle(overlay, (8, 8), (352, 8 + box_h), (28, 28, 28), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    for i, text in enumerate(lines):
        color = status_color if i == 0 else (235, 235, 235)
        cv2.putText(
            canvas, text, (18, 30 + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )

    active = [e.type.value for e in update.active]
    if active:
        cv2.putText(
            canvas, " | ".join(active), (18, canvas.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA,
        )
    return canvas
