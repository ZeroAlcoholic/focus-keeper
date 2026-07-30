"""影像來源（規格 §3）。

統一輸出 :class:`FramePacket`（``image, timestamp_ms, frame_id``），
讓攝影機與 MP4 共用同一分析核心。

* :class:`CameraSource`：背景執行緒擷取，**槽位固定為 1**，
  新影格覆蓋舊影格 → 只處理最新影格，不累積延遲（規格 §3、§7）。
* :class:`VideoSource`：循序讀取，時間戳由影片本身決定 → 可重現。
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import cv2
import numpy as np

if TYPE_CHECKING:  # 僅供型別檢查，避免與 config 形成循環匯入
    from .config import SourcesConfig

__all__ = [
    "FramePacket",
    "FrameSource",
    "CameraSource",
    "VideoSource",
    "SourceError",
    "open_source",
]


class SourceError(RuntimeError):
    """來源無法開啟或讀取。"""


@dataclass(frozen=True)
class FramePacket:
    """單一影格。

    ``timestamp_ms``
        分析用時間軸。攝影機＝自開始擷取起的經過毫秒；影片＝影片內時間。
        規則層一律只看這個值，因此兩種來源行為一致。
    ``capture_monotonic``
        擷取完成當下的 :func:`time.perf_counter` 讀值，用於量測端到端延遲。
        影片模式下此值仍是真實時鐘，僅供效能統計，不影響判定。
    """

    image: np.ndarray
    timestamp_ms: int
    frame_id: int
    capture_monotonic: float

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)``。"""
        height, width = self.image.shape[:2]
        return width, height


class FrameSource(ABC):
    """影像來源介面。"""

    #: 即時來源（攝影機）為 True；離線來源（影片）為 False。
    realtime: bool = False

    @abstractmethod
    def read(self) -> FramePacket | None:
        """回傳下一格；來源結束回傳 ``None``。"""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        ...

    def close(self) -> None:  # pragma: no cover - 由子類覆寫
        return None

    def __iter__(self) -> Iterator[FramePacket]:
        while True:
            packet = self.read()
            if packet is None:
                return
            yield packet

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class CameraSource(FrameSource):
    """攝影機來源；背景執行緒擷取、槽位固定為 1。"""

    realtime = True

    _BACKENDS = {
        "auto": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "v4l2": cv2.CAP_V4L2,
    }

    def __init__(
        self,
        index: int = 0,
        *,
        width: int | None = 1280,
        height: int | None = 720,
        fps: int | None = 30,
        backend: str = "auto",
        warmup_timeout_s: float = 5.0,
    ) -> None:
        backend_id = self._BACKENDS.get(backend.lower())
        if backend_id is None:
            raise SourceError(
                f"未知的攝影機後端 {backend!r}，可用：{', '.join(sorted(self._BACKENDS))}"
            )

        self._index = index
        self._backend = backend
        self._cap = cv2.VideoCapture(index, backend_id)
        if not self._cap.isOpened():
            raise SourceError(f"無法開啟攝影機 index={index}（backend={backend}）")

        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self._cap.set(cv2.CAP_PROP_FPS, fps)
        # 驅動層 buffer 也壓到最小，避免在 OpenCV 之外累積影格。
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 幾何資訊在建構時抓一次就好。describe() 若在主執行緒即時查詢，
        # 會與讀取執行緒的 cap.read() 併發存取同一個 VideoCapture。
        self._geometry = {
            "frame_width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "frame_height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "capture_fps": self._cap.get(cv2.CAP_PROP_FPS),
        }

        self._lock = threading.Lock()
        self._available = threading.Condition(self._lock)
        self._slot: FramePacket | None = None
        self._stop = threading.Event()
        self._grabbed = 0
        self._dropped = 0
        self._delivered = 0
        self._error: str | None = None
        self._t0 = time.perf_counter()

        self._thread = threading.Thread(target=self._reader, name="camera-reader", daemon=True)
        self._thread.start()

        if not self._wait_first_frame(warmup_timeout_s):
            self.close()
            raise SourceError(
                f"攝影機 index={index} 在 {warmup_timeout_s} 秒內沒有輸出影格"
                + (f"：{self._error}" if self._error else "")
            )

    # ------------------------------------------------------------------ #

    #: 連續讀取失敗時的重試間隔上限（秒）。
    #: 固定 10 ms 重試會在裝置被占用／拔除時變成每秒 100 次失敗呼叫，
    #: 既燒 CPU 又讓 OpenCV 的警告洗滿 stderr。改為指數退避。
    _MAX_RETRY_DELAY_S = 0.5

    def _reader(self) -> None:
        frame_id = 0
        retry_delay = 0.01
        consecutive_failures = 0
        while not self._stop.is_set():
            ok, image = self._cap.read()
            if not ok or image is None:
                consecutive_failures += 1
                self._error = (
                    f"VideoCapture.read() 失敗（連續 {consecutive_failures} 次）"
                )
                # 用 Event.wait 而非 sleep，close() 時可立即中斷退出。
                self._stop.wait(retry_delay)
                retry_delay = min(retry_delay * 2, self._MAX_RETRY_DELAY_S)
                continue
            if consecutive_failures:
                consecutive_failures = 0
                retry_delay = 0.01
            now = time.perf_counter()
            packet = FramePacket(
                image=image,
                timestamp_ms=int(round((now - self._t0) * 1000.0)),
                frame_id=frame_id,
                capture_monotonic=now,
            )
            frame_id += 1
            with self._available:
                if self._slot is not None:
                    # 消費端還沒取走上一格 → 直接丟棄，只保留最新。
                    self._dropped += 1
                self._slot = packet
                self._grabbed += 1
                self._available.notify()

    def _wait_first_frame(self, timeout_s: float) -> bool:
        deadline = time.perf_counter() + timeout_s
        with self._available:
            while self._slot is None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return False
                self._available.wait(timeout=min(0.1, remaining))
        return True

    def read(self, timeout_s: float = 2.0) -> FramePacket | None:
        with self._available:
            if self._slot is None:
                self._available.wait(timeout=timeout_s)
            packet, self._slot = self._slot, None
        if packet is not None:
            self._delivered += 1
        return packet

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "camera",
            "index": self._index,
            "backend": self._backend,
            **self._geometry,
            "grabbed": self._grabbed,
            "delivered": self._delivered,
            "dropped_stale": self._dropped,
            "last_error": self._error,
        }

    @property
    def last_error(self) -> str | None:
        return self._error

    def close(self) -> None:
        self._stop.set()
        thread = getattr(self, "_thread", None)
        if thread is not None and thread.is_alive():
            # 讀取執行緒可能卡在 cap.read()（裝置被占用、驅動異常）。
            # 在它還活著時 release()，OpenCV 屬於未定義行為，可能當掉整個行程；
            # 寧可讓 capture 隨行程結束被作業系統回收，也不要在這裡冒險。
            thread.join(timeout=3.0)
            if thread.is_alive():
                self._error = "讀取執行緒未能結束，略過 VideoCapture.release() 以免當掉"
                return
        if getattr(self, "_cap", None) is not None:
            self._cap.release()


class VideoSource(FrameSource):
    """影片檔來源；時間戳取自影片，結果可重現。"""

    realtime = False

    def __init__(self, path: str | Path, *, pace_realtime: bool = False) -> None:
        self._path = Path(path)
        if not self._path.is_file():
            raise SourceError(f"找不到影片檔：{self._path}")
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise SourceError(f"無法開啟影片：{self._path}")

        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 0.0
        self._frame_id = 0
        self._pace_realtime = pace_realtime
        self._t0 = time.perf_counter()

    def read(self) -> FramePacket | None:
        ok, image = self._cap.read()
        if not ok or image is None:
            return None

        # 優先用容器提供的時間；缺值時退回 frame_id / fps，保持嚴格單調。
        pos_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_ms is None or pos_ms <= 0.0:
            pos_ms = (self._frame_id / self._fps * 1000.0) if self._fps > 0 else 0.0
        timestamp_ms = int(round(pos_ms))

        if self._pace_realtime:
            target = self._t0 + timestamp_ms / 1000.0
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

        packet = FramePacket(
            image=image,
            timestamp_ms=timestamp_ms,
            frame_id=self._frame_id,
            capture_monotonic=time.perf_counter(),
        )
        self._frame_id += 1
        return packet

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "video",
            "path": str(self._path),
            "fps": self._fps,
            "frame_count": int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "frame_width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "frame_height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frames_read": self._frame_id,
            "pace_realtime": self._pace_realtime,
        }

    def close(self) -> None:
        if getattr(self, "_cap", None) is not None:
            self._cap.release()


def open_source(spec: str, cfg: SourcesConfig | None = None) -> FrameSource:
    """``--source`` 的解析：純數字＝攝影機 index，其餘＝影片路徑。

    設定以型別化的 :class:`~focus_keeper.config.SourcesConfig` 傳入；
    欄位驗證統一由 config 層負責，本模組不再自備 key set。
    """
    from .config import SourcesConfig

    resolved = cfg or SourcesConfig()
    text = str(spec).strip()
    if text.isdigit():
        cam = resolved.camera
        return CameraSource(
            index=int(text), width=cam.width, height=cam.height, fps=cam.fps,
            backend=cam.backend, warmup_timeout_s=cam.warmup_timeout_s,
        )
    return VideoSource(text, pace_realtime=resolved.video.pace_realtime)
