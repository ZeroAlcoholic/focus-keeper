"""來源無關的分析核心：偵測 → 過濾 → 追蹤 → 時間規則。

本模組不含 CLI、不含疊圖、不含檔案輸出，因此可獨立測試與嵌入其他程式。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import cv2
import numpy as np

from .config import AppConfig, PipelineConfig
from .detectors import Detector, build_detector
from .detectors.base import Detection, filter_detections, filter_presence
from .rules import Event, RuleConfig, RuleUpdate, TemporalRuleEngine
from .sources import FramePacket
from .tracker import PrimarySubject, PrimarySubjectTracker, TrackerConfig

__all__ = ["FrameResult", "AnalysisPipeline", "build_pipeline"]

@dataclass
class FrameResult:
    """單一「已處理」影格的完整結果。"""

    frame_id: int
    timestamp_ms: int
    #: 通過**主角資格**門檻的偵測（追蹤器的候選）。
    detections: tuple[Detection, ...]
    #: 通過**在場**門檻的偵測，門檻較寬，是 person_count 的依據。
    #: 疊圖要畫這一組——否則畫面會出現「示警說 2 人、只看到 1 個框」的矛盾。
    presence: tuple[Detection, ...]
    primary: PrimarySubject | None
    rule_update: RuleUpdate
    #: 偵測＋追蹤＋規則的耗時。
    pipeline_ms: float
    #: 自影格擷取完成到本結果產生的耗時（含排隊）。
    end_to_end_ms: float


class AnalysisPipeline:
    """來源無關的分析核心：偵測 → 過濾 → 追蹤 → 時間規則。"""

    def __init__(
        self,
        detector: Detector,
        tracker: PrimarySubjectTracker,
        rules: TemporalRuleEngine,
        config: PipelineConfig | None = None,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.rules = rules
        self.config = config or PipelineConfig()
        if self.config.opencv_threads > 0:
            cv2.setNumThreads(self.config.opencv_threads)
        self._period_ms = 1000.0 / self.config.inference_fps
        self._next_due_ms: float | None = None
        self.processed_frames = 0
        self.skipped_frames = 0
        #: 上一格的縮圖，用來偵測來源停格。縮圖只有 32x32 灰階，成本可忽略，
        #: 且不構成影像留存（不落地、單格覆寫、無法還原臉部）。
        self._prev_thumb: np.ndarray | None = None
        self.last_frame_diff: float = 0.0

    # ------------------------------------------------------------------ #

    def should_process(self, timestamp_ms: int) -> bool:
        """依 ``inference_fps`` 節流；不累積影格，落後太多直接重設節拍。"""
        if self._next_due_ms is None:
            return True
        return timestamp_ms >= self._next_due_ms

    def _advance_due(self, timestamp_ms: int) -> None:
        if self._next_due_ms is None:
            self._next_due_ms = timestamp_ms + self._period_ms
            return
        nxt = self._next_due_ms + self._period_ms
        # 落後超過一個週期（例如處理變慢）→ 以當下時間重新起算，避免補跑。
        self._next_due_ms = nxt if nxt > timestamp_ms else timestamp_ms + self._period_ms

    def _frame_is_static(self, image: np.ndarray) -> bool:
        """來源是否已停止更新（USB 當掉、驅動凍結、虛擬攝影機停格）。

        停格是監測系統最危險的失效：畫面停在有人的那一格，
        所有判定都會持續回報 NORMAL，監測早就死了卻毫無跡象。
        """
        n = self.config.static_frame_thumb
        # INTER_NEAREST 而非 INTER_AREA：實測 0.009 ms vs 2.028 ms（225 倍），
        # 而且鑑別力更好（AMI 最小影格差 0.0488 vs 0.0427）。
        # 原因：NEAREST 只取樣 4096 個原始像素、保留像素級雜訊，
        # AREA 要讀完 92 萬像素做平均、反而把細微變化抹平。
        thumb = cv2.cvtColor(
            cv2.resize(image, (n, n), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2GRAY
        )
        previous, self._prev_thumb = self._prev_thumb, thumb
        if previous is None:
            self.last_frame_diff = 255.0  # 第一格沒有比較對象，不視為停格
            return False
        self.last_frame_diff = float(
            np.mean(cv2.absdiff(thumb, previous).astype(np.float32))
        )
        return self.last_frame_diff < self.config.static_frame_diff

    def process(self, packet: FramePacket) -> FrameResult | None:
        """處理一格；若被節流跳過則回傳 ``None``。"""
        if not self.should_process(packet.timestamp_ms):
            self.skipped_frames += 1
            return None

        started = time.perf_counter()
        frame_width, frame_height = packet.size
        frame_static = self._frame_is_static(packet.image)

        raw = self.detector.detect(packet.image)
        # 主角資格：夠大夠清楚才適合被追蹤。
        detections = filter_detections(
            raw,
            frame_width=frame_width,
            frame_height=frame_height,
            min_confidence=self.config.min_confidence,
            min_face_area_ratio=self.config.min_face_area_ratio,
        )
        # 在場人數：門檻較寬，站得遠的第二人也要算進來（產品語意＝單人主構圖）。
        presence = filter_presence(
            raw,
            frame_width=frame_width,
            frame_height=frame_height,
            min_confidence=self.config.effective_presence_confidence,
            min_face_area_ratio=self.config.effective_presence_area_ratio,
        )

        tracked = self.tracker.update(
            detections,
            timestamp_ms=packet.timestamp_ms,
            frame_size=(frame_width, frame_height),
        )
        rule_update = self.rules.update(
            timestamp_ms=packet.timestamp_ms,
            person_count=len(presence),
            primary=tracked.primary,
            frame_size=(frame_width, frame_height),
            frame_static=frame_static,
        )

        finished = time.perf_counter()
        self.processed_frames += 1
        self._advance_due(packet.timestamp_ms)

        return FrameResult(
            frame_id=packet.frame_id,
            timestamp_ms=packet.timestamp_ms,
            detections=tuple(detections),
            presence=tuple(presence),
            primary=tracked.primary,
            rule_update=rule_update,
            pipeline_ms=(finished - started) * 1000.0,
            end_to_end_ms=(finished - packet.capture_monotonic) * 1000.0,
        )

    def flush(self, timestamp_ms: int) -> tuple[Event, ...]:
        return self.rules.flush(timestamp_ms)


def build_pipeline(
    cfg: "AppConfig", *, detector_name: str | None = None
) -> tuple[AnalysisPipeline, Detector]:
    """由已驗證的 :class:`~focus_keeper.config.AppConfig` 建立分析核心。"""
    detector = build_detector(detector_name or cfg.detector, cfg.detectors)
    pipeline = AnalysisPipeline(
        detector=detector,
        tracker=PrimarySubjectTracker(cfg.tracker),
        rules=TemporalRuleEngine(cfg.rules),
        config=cfg.pipeline,
    )
    return pipeline, detector


# --------------------------------------------------------------------------- #
# 量測與記錄
# --------------------------------------------------------------------------- #

