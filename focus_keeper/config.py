"""設定的載入、驗證與型別。

所有可調參數的**唯一**載入入口。跨區塊一致性檢查也在這裡，
避免各層各自驗證造成慣例不一致。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

from .rules import RuleConfig
from .tracker import TrackerConfig
from .validation import ConfigError, as_mapping, check_fields

__all__ = [
    "AppConfig",
    "PipelineConfig",
    "CameraConfig",
    "VideoConfig",
    "SourcesConfig",
    "ConfigError",
    "load_config",
    "DEFAULT_CONFIG_PATH",
]

def _app_dir() -> Path:
    """設定檔與模型的預設所在目錄。

    PyInstaller 打包後 ``__file__`` 指向暫存解壓目錄，那裡不會有 config.yaml
    也不會有 models/。獨立執行檔的正確基準是**執行檔本身所在的目錄**，
    使用者把 config.yaml 與 models/ 放在 exe 旁邊即可。
    """
    if getattr(sys, "frozen", False):  # PyInstaller / cx_Freeze
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


APP_DIR = _app_dir()
DEFAULT_CONFIG_PATH = APP_DIR / "config.yaml"

@dataclass(frozen=True)
class PipelineConfig:
    """分析核心的取樣與過濾參數（規格 §4）。"""

    inference_fps: float = 10.0
    #: 主角資格門檻（規格 §4）。
    min_confidence: float = 0.70
    min_face_area_ratio: float = 0.015
    #: 在場人數門檻，刻意比主角資格寬鬆——產品語意是「畫面必須是單人主構圖」，
    #: 站得較遠、不足以當主角的第二個人**仍然**要觸發 MULTI_PERSON。
    #: 設為 None 表示沿用主角門檻（等同舊行為）。
    presence_min_confidence: float | None = 0.60
    presence_min_face_area_ratio: float | None = 0.004
    #: 判定「單格畫面沒有變化」的平均像素差門檻（64x64 灰階縮圖，0~255）。
    #:
    #: 這個值是量出來的，不是猜的（2026-07-29）：
    #:   完全相同的影格          -> 0.0000
    #:   AMI 低解析度壓縮影片    -> 最小 0.0410（編碼器會複製相同區塊，差值很低）
    #:   本機攝影機              -> 偶爾也會出現 0.0000（驅動重複送格）
    #:
    #: 因此**單格相同不能當停格證據**；真正的判準是「連續」靜止
    #: `rules.feed_frozen_ms`（預設 3 秒）。此門檻只需把「相同」與
    #: 「壓縮影片的低幅變化」分開，取 0.02 對 AMI 最小值留約 2 倍餘裕。
    static_frame_diff: float = 0.02
    #: 停格判定用的縮圖邊長。32 太小會讓壓縮影片的差值掉到 0.025，
    #: 與停格的 0 太接近；64 可拉開到 0.041。
    static_frame_thumb: int = 64
    #: OpenCV 的運算執行緒數。實測（2026-07-29，1280x720 輸入）：
    #:   1 條 -> 3.42 ms（可支撐 292 FPS）
    #:   4 條 -> 2.52 ms
    #:   16 條（預設）-> 2.65 ms
    #: 模型很小，1->8 條只快 1.4 倍，擴展性極差。單執行緒只多約 1 ms，
    #: 卻少了 15 條執行緒與前景程式競爭 —— 對「與其他應用共存的背景監測」
    #: 這個交換明確划算（10 FPS 需求下仍有 29 倍餘裕）。
    #: 設為 0 表示不干預 OpenCV 的預設值。
    opencv_threads: int = 1

    @property
    def effective_presence_confidence(self) -> float:
        return (
            self.min_confidence
            if self.presence_min_confidence is None
            else self.presence_min_confidence
        )

    @property
    def effective_presence_area_ratio(self) -> float:
        return (
            self.min_face_area_ratio
            if self.presence_min_face_area_ratio is None
            else self.presence_min_face_area_ratio
        )

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | None) -> "PipelineConfig":
        cfg = as_mapping("pipeline", cfg)
        check_fields("pipeline", cfg, cls.__dataclass_fields__)
        instance = cls(**cfg)
        if instance.inference_fps <= 0:
            raise ConfigError("inference_fps 必須 > 0")
        if not (0.0 < instance.min_confidence <= 1.0):
            raise ConfigError("min_confidence 必須落在 (0, 1]")
        if not (0.0 < instance.effective_presence_confidence <= 1.0):
            raise ConfigError("presence_min_confidence 必須落在 (0, 1]")
        if instance.effective_presence_confidence > instance.min_confidence:
            raise ConfigError(
                "presence_min_confidence 不得高於 min_confidence"
                "（在場門檻比主角門檻嚴格會造成主角不被計入人數）"
            )
        if instance.effective_presence_area_ratio > instance.min_face_area_ratio:
            raise ConfigError("presence_min_face_area_ratio 不得高於 min_face_area_ratio")
        return instance


# --------------------------------------------------------------------------- #
# 影像來源
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CameraConfig:
    width: int | None = 1280
    height: int | None = 720
    fps: int | None = 30
    backend: str = "auto"
    warmup_timeout_s: float = 5.0

    @classmethod
    def from_dict(cls, cfg: Any) -> "CameraConfig":
        cfg = as_mapping("sources.camera", cfg)
        check_fields("sources.camera", cfg, cls.__dataclass_fields__)
        return cls(**cfg)


@dataclass(frozen=True)
class VideoConfig:
    pace_realtime: bool = False

    @classmethod
    def from_dict(cls, cfg: Any) -> "VideoConfig":
        cfg = as_mapping("sources.video", cfg)
        check_fields("sources.video", cfg, cls.__dataclass_fields__)
        return cls(**cfg)


@dataclass(frozen=True)
class SourcesConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    video: VideoConfig = field(default_factory=VideoConfig)

    @classmethod
    def from_dict(cls, cfg: Any) -> "SourcesConfig":
        cfg = as_mapping("sources", cfg)
        check_fields("sources", cfg, {"camera", "video"})
        return cls(
            camera=CameraConfig.from_dict(cfg.get("camera")),
            video=VideoConfig.from_dict(cfg.get("video")),
        )


# --------------------------------------------------------------------------- #
# 整份設定
# --------------------------------------------------------------------------- #

#: 偵測器的**程式內建預設**。
#:
#: 為什麼需要：其他每個區塊（pipeline / rules / tracker / sources）都有程式
#: 內建預設，只有 detectors 沒有——導致在缺 config.yaml 的目錄執行時，
#: 錯誤是「缺少 detectors.yunet 區塊」，而真正的問題是找不到設定檔。
#: 有了內建預設，缺設定檔時會走到準確的「找不到模型權重」訊息。
DEFAULT_DETECTORS: Mapping[str, Mapping[str, Any]] = {
    "yunet": {
        "model_path": "models/face_detection_yunet_2023mar.onnx",
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "detect_width": 320,
        "score_threshold": 0.50,
        "nms_threshold": 0.30,
        "top_k": 50,
    },
    "mediapipe": {
        "model_path": "models/blaze_face_short_range.tflite",
        "sha256": "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f",
        "detect_width": 320,
        "score_threshold": 0.50,
        "nms_threshold": 0.30,
    },
}

#: 每個偵測器區塊允許的欄位。放在這裡而非各偵測器模組，
#: 是為了讓「未知欄位」的檢查與其他區塊走同一條路。
_DETECTOR_FIELDS = {
    "model_path",
    "sha256",
    "detect_width",
    "score_threshold",
    "nms_threshold",
    "top_k",
}


@dataclass(frozen=True)
class AppConfig:
    """整份設定的**唯一**載入與驗證入口。

    在此之前，各層各自 ``from_dict``、``sources`` 用另一套 key set、
    頂層 key 完全不驗——把 ``rules`` 打成 ``rulez`` 會讓整組時間門檻
    悄悄退回預設值而程式看起來正常。現在所有區塊走同一條驗證路徑，
    且錯誤都是 :class:`ConfigError`。
    """

    source: str = "0"
    detector: str = "yunet"
    event_log: Path | None = Path("logs/events.jsonl")
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    rules: RuleConfig = field(default_factory=RuleConfig)
    sources: SourcesConfig = field(default_factory=SourcesConfig)
    #: 各偵測器的原始設定區塊（欄位已驗證，值交由偵測器解讀）。
    detectors: Mapping[str, Mapping[str, Any]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_DETECTORS.items()}
    )
    #: 跨區塊一致性警告。不是錯誤，但呼叫端應顯示給使用者。
    warnings: tuple[str, ...] = ()
    #: 實際讀取的設定檔路徑；``None`` 表示全部使用預設值。
    path: Path | None = None

    TOP_LEVEL_KEYS = frozenset(
        {"source", "detector", "event_log", "pipeline", "tracker", "rules", "sources", "detectors"}
    )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AppConfig":
        raw = load_config(path)
        check_fields("設定檔頂層", raw, cls.TOP_LEVEL_KEYS)

        # 未提供 detectors 區塊時使用內建預設（與其他區塊一致）。
        raw_detectors = as_mapping("detectors", raw.get("detectors"))
        if raw_detectors:
            detectors: dict[str, dict[str, Any]] = {}
            for name, section in raw_detectors.items():
                section = as_mapping(f"detectors.{name}", section)
                check_fields(f"detectors.{name}", section, _DETECTOR_FIELDS)
                detectors[name] = section
        else:
            detectors = {k: dict(v) for k, v in DEFAULT_DETECTORS.items()}

        # 相對的 model_path 以 APP_DIR 為基準，而非行程的 CWD。
        # 獨立執行檔可能從任意目錄啟動；相對 CWD 會找不到 models/。
        for section in detectors.values():
            mp = section.get("model_path")
            if mp and not Path(mp).is_absolute():
                section["model_path"] = str(APP_DIR / mp)

        # `event_log:` 留空或空字串是使用者關閉記錄的自然寫法。
        configured_log = raw.get("event_log", "logs/events.jsonl")
        event_log = Path(configured_log) if configured_log else None

        pipeline = PipelineConfig.from_dict(raw.get("pipeline"))
        tracker = TrackerConfig.from_dict(raw.get("tracker"))
        rules = RuleConfig.from_dict(raw.get("rules"))

        warnings: list[str] = []
        if tracker.lost_after_ms < rules.primary_left_ms:
            warnings.append(
                f"tracker.lost_after_ms({tracker.lost_after_ms}) < "
                f"rules.primary_left_ms({rules.primary_left_ms})："
                "主角身分會在 PRIMARY_LEFT 示警前被釋放，可能改認他人為主角。"
            )

        return cls(
            source=str(raw.get("source", "0")),
            detector=str(raw.get("detector", "yunet")),
            event_log=event_log,
            pipeline=pipeline,
            tracker=tracker,
            rules=rules,
            sources=SourcesConfig.from_dict(raw.get("sources")),
            detectors=detectors,
            warnings=tuple(warnings),
            path=Path(path) if path else (DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.is_file() else None),
        )

    def as_report(self) -> dict[str, Any]:
        """供交付回報使用的可序列化摘要。"""
        return {
            "config_path": str(self.path) if self.path else None,
            "configured_detector": self.detector,
            "pipeline": asdict(self.pipeline),
            "tracker": asdict(self.tracker),
            "rules": asdict(self.rules),
            "sources": asdict(self.sources),
        }


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """讀取 YAML 為原始 dict。一般請用 :meth:`AppConfig.load`。"""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        if path:
            raise ConfigError(f"找不到設定檔：{config_path}")
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"設定檔格式錯誤（應為 mapping）：{config_path}")
    return data
