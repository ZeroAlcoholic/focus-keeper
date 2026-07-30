"""延遲與頻率統計（規格 §6 驗收依據）。"""

from __future__ import annotations

import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Sequence

from .pipeline import FrameResult

__all__ = ["percentile", "LatencyStats"]

def percentile(values: Sequence[float], pct: float) -> float:
    """最近秩法（nearest-rank），小樣本下不做插值以免高估效能。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(pct / 100.0 * len(ordered))))
    return ordered[rank - 1]


#: 百分位取樣視窗（判定次數）。36,000 次 ≈ 10 FPS 下的 1 小時。
#:
#: 為什麼要有上界：本系統設計上會連續運轉數小時。若把每一次判定都留著，
#: (1) 記憶體無上界成長；(2) 更嚴重的是 `draw_overlay` 每格都要對整份清單
#: 排序算 p95——實測 8 小時後單格 6.29 ms，比整個偵測成本（2.5 ms）還高，
#: 監測本身會愈跑愈慢。
DEFAULT_WINDOW = 36_000


@dataclass
class LatencyStats:
    """延遲與有效 FPS 統計（規格 §6 驗收依據）。

    **計數、平均、最大值對整段執行是精確的**（串流累加，不受視窗影響）；
    **百分位是視窗內的**（見 :data:`DEFAULT_WINDOW`），``summary()`` 會
    一併回報視窗大小，避免把視窗值誤讀成全程值。
    """

    window: int = DEFAULT_WINDOW
    pipeline_ms: deque[float] = field(default_factory=lambda: deque(maxlen=DEFAULT_WINDOW))
    end_to_end_ms: deque[float] = field(default_factory=lambda: deque(maxlen=DEFAULT_WINDOW))
    #: 每次判定的來源時間戳，用來算**實際達成**的判定頻率。
    timestamps_ms: deque[int] = field(default_factory=lambda: deque(maxlen=DEFAULT_WINDOW))
    _wall_start: float = field(default_factory=time.perf_counter)
    # 全程精確累加值（不受視窗影響）
    _n: int = 0
    _sum_pipeline: float = 0.0
    _sum_end_to_end: float = 0.0
    _max_pipeline: float = 0.0
    _max_end_to_end: float = 0.0
    # HUD 用的 p95 快取：疊圖每格都要顯示，但沒必要每格重算。
    _hud_p95: float = 0.0
    _hud_p95_at: int = -1

    def __post_init__(self) -> None:
        if self.window != DEFAULT_WINDOW:
            self.pipeline_ms = deque(self.pipeline_ms, maxlen=self.window)
            self.end_to_end_ms = deque(self.end_to_end_ms, maxlen=self.window)
            self.timestamps_ms = deque(self.timestamps_ms, maxlen=self.window)

    def add(self, result: FrameResult) -> None:
        self.pipeline_ms.append(result.pipeline_ms)
        self.end_to_end_ms.append(result.end_to_end_ms)
        self.timestamps_ms.append(result.timestamp_ms)
        self._n += 1
        self._sum_pipeline += result.pipeline_ms
        self._sum_end_to_end += result.end_to_end_ms
        self._max_pipeline = max(self._max_pipeline, result.pipeline_ms)
        self._max_end_to_end = max(self._max_end_to_end, result.end_to_end_ms)

    @property
    def count(self) -> int:
        """全程判定次數（精確，不受視窗限制）。"""
        return self._n

    @property
    def effective_fps(self) -> float:
        elapsed = time.perf_counter() - self._wall_start
        return self.count / elapsed if elapsed > 0 else 0.0

    def recent_fps(self, window: int = 30) -> float:
        """近期**實際達成**的判定頻率。

        不能用 ``1000 / 處理耗時`` —— 那是「機器最多跑得多快」，忽略了
        ``inference_fps`` 節流，畫面上會顯示系統從未達到的速率
        （實測會顯示 ~200 FPS，而實際是 10 FPS）。
        """
        # deque 不支援切片；islice(reversed(...)) 只走訪尾端 n 筆，
        # 不會複製整個視窗（視窗上限 36,000）。
        recent = list(islice(reversed(self.timestamps_ms), max(2, window)))[::-1]
        if len(recent) < 2:
            return 0.0
        span_ms = recent[-1] - recent[0]
        return (len(recent) - 1) * 1000.0 / span_ms if span_ms > 0 else 0.0

    def headroom_fps(self) -> float:
        """純算力上限（1000 / 處理耗時），與實際節流後的頻率分開報。"""
        recent = list(islice(reversed(self.pipeline_ms), 30))
        if not recent:
            return 0.0
        mean_ms = statistics.fmean(recent)
        return 1000.0 / mean_ms if mean_ms > 0 else 0.0

    def hud_p95_end_to_end(self, refresh_every: int = 10) -> float:
        """疊圖用的端到端 p95，每 ``refresh_every`` 次判定才重算一次。

        疊圖每格都要顯示這個數字，但沒必要每格排序整個視窗——10 FPS 下
        每秒更新一次對操作者已足夠，卻把成本降為 1/10。
        """
        if self._n - self._hud_p95_at >= refresh_every or self._hud_p95_at < 0:
            self._hud_p95 = percentile(self.end_to_end_ms, 95)
            self._hud_p95_at = self._n
        return self._hud_p95

    def summary(self) -> dict[str, Any]:
        """效能摘要。

        ``mean`` / ``max`` / ``processed_frames`` 對**整段執行**精確；
        ``p50`` / ``p95`` 只涵蓋最近 ``percentile_window_samples`` 次判定
        （見 :data:`DEFAULT_WINDOW`）。欄位名刻意標明，避免誤讀成全程值。
        """
        return {
            "processed_frames": self.count,
            "wall_seconds": round(time.perf_counter() - self._wall_start, 3),
            "effective_inference_fps": round(self.effective_fps, 2),
            "percentile_window_samples": len(self.pipeline_ms),
            "percentile_window_capacity": self.window,
            "pipeline_ms": {
                "mean": round(self._sum_pipeline / self._n, 2) if self._n else 0.0,
                "p50": round(percentile(self.pipeline_ms, 50), 2),
                "p95": round(percentile(self.pipeline_ms, 95), 2),
                "max": round(self._max_pipeline, 2),
            },
            "end_to_end_ms": {
                "mean": round(self._sum_end_to_end / self._n, 2) if self._n else 0.0,
                "p50": round(percentile(self.end_to_end_ms, 50), 2),
                "p95": round(percentile(self.end_to_end_ms, 95), 2),
                "max": round(self._max_end_to_end, 2),
            },
        }

