"""命令列介面。組裝各層並處理啟動失敗、資源釋放與回報輸出。"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from . import __version__
from .config import AppConfig, ConfigError
from .detectors import Detector, DetectorUnavailableError, ModelIntegrityError, available_detectors
from .eventlog import EventLogger
from .metrics import LatencyStats
from .overlay import draw_overlay
from .pipeline import build_pipeline
from .rules import Event
from .sources import FrameSource, SourceError, open_source

__all__ = ["run", "main"]

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="focus-keeper",
        description="即時主人物頭像監測 MVP（全地端，不做人臉辨識或身分比對）",
    )
    parser.add_argument(
        "--source", default=None,
        help="攝影機 index（例：0）或影片路徑（例：sample.mp4）；預設取 config.yaml 的 source",
    )
    parser.add_argument(
        "--config", default=None,
        help="設定檔路徑，預設 ./config.yaml。未指定的欄位與區塊一律沿用程式內建預設，"
             "因此可以只寫要覆蓋的部分",
    )
    parser.add_argument(
        "--detector", default=None, choices=available_detectors(),
        help="覆寫設定檔的偵測器（A=yunet / B=mediapipe）",
    )
    parser.add_argument(
        "--log", default=None,
        help="JSONL 事件記錄輸出路徑；給 'none' 可停用",
    )
    parser.add_argument("--no-display", action="store_true", help="不開預覽視窗（無頭環境／測試用）")
    parser.add_argument("--duration", type=float, default=None, help="執行秒數上限")
    parser.add_argument("--max-frames", type=int, default=None, help="處理影格數上限")
    parser.add_argument("--report", default=None, help="把效能與版本回報寫成 JSON 檔")
    parser.add_argument("--quiet", action="store_true", help="不在 stdout 印出事件")
    return parser


def build_session_info(
    source: FrameSource, detector: Detector, cfg: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "app_version": __version__,
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "processor": platform.processor(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "source": source.describe(),
        "detector": detector.describe(),
        **cfg.as_report(),
    }


def run(args: argparse.Namespace) -> int:
    # 設定的載入與驗證收斂在 AppConfig.load()，錯誤一律是 ConfigError。
    try:
        cfg = AppConfig.load(args.config)
    except ConfigError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    except TypeError as exc:  # dataclass 收到型別不符的值
        print(f"[error] 設定值型別有誤：{exc}", file=sys.stderr)
        return 2

    for warning in cfg.warnings:
        print(f"[warn] {warning}", file=sys.stderr)

    source_spec = args.source if args.source is not None else cfg.source

    try:
        pipeline, detector = build_pipeline(cfg, detector_name=args.detector)
    except DetectorUnavailableError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    except ModelIntegrityError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    except (ConfigError, KeyError, TypeError) as exc:
        print(f"[error] 設定有誤：{exc}", file=sys.stderr)
        return 2

    try:
        source = open_source(source_spec, cfg.sources)
    except SourceError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        detector.close()
        return 2

    log_path: Path | None
    if args.log is not None:
        log_path = None if args.log.lower() == "none" else Path(args.log)
    else:
        # `event_log:` 留空／空字串代表關閉記錄，已在 AppConfig 解析。
        log_path = cfg.event_log

    try:
        logger = EventLogger(log_path, pipeline.rules)
    except OSError as exc:
        # 記錄檔開不起來時，前面已經開好的攝影機與模型必須先關掉，
        # 否則行程以 traceback 結束會把攝影機裝置一直占住。
        print(f"[error] 無法建立事件記錄 {log_path}：{exc}", file=sys.stderr)
        source.close()
        detector.close()
        return 2
    stats = LatencyStats()
    display = not args.no_display
    window = "focus_keeper"
    last_timestamp_ms = 0
    interrupted = False

    session_info = build_session_info(source, detector, cfg)
    logger.session_start(session_info)
    if not args.quiet:
        print(f"[info] source={source_spec} detector={detector.name} log={log_path or 'disabled'}")
        print("[info] 按 q 或 ESC 結束")

    started_wall = time.perf_counter()
    #: 即時來源沒有「結尾」——讀不到影格代表停頓（USB 重新列舉、驅動異常），
    #: 不是正常結束。連續停頓超過此上限才放棄，並以非零狀態碼收場。
    max_consecutive_stalls = 5
    stalls = 0
    stalled_out = False
    try:
        while True:
            packet = source.read()
            if packet is None:
                if not source.realtime:
                    break  # 影片讀完了，這才是正常結束
                stalls += 1
                detail = getattr(source, "last_error", None)
                print(
                    f"[warn] 攝影機停頓（第 {stalls}/{max_consecutive_stalls} 次）"
                    + (f"：{detail}" if detail else ""),
                    file=sys.stderr,
                )
                if stalls >= max_consecutive_stalls:
                    print("[error] 攝影機持續無影格，結束監測", file=sys.stderr)
                    stalled_out = True
                    break
                continue
            stalls = 0

            result = pipeline.process(packet)
            if result is not None:
                stats.add(result)
                last_timestamp_ms = result.timestamp_ms
                logger.log_update(result.rule_update)
                if not args.quiet:
                    for event in result.rule_update.started:
                        threshold = pipeline.rules.threshold_for(event.type) or 0
                        print(
                            f"[event] {event.type.value} @ {event.triggered_ms} ms "
                            f"(條件起於 {event.condition_start_ms} ms，"
                            f"超出門檻 {event.alert_latency_ms(threshold)} ms，"
                            f"人數 {event.max_person_count})"
                        )
                    for event in result.rule_update.ended:
                        print(f"[event] {event.type.value} 結束 @ {event.end_ms} ms")

                if display:
                    try:
                        cv2.imshow(window, draw_overlay(packet.image, result, stats, detector_name=detector.name))
                    except cv2.error as exc:  # pragma: no cover - 無頭環境
                        print(f"[warn] 無法開啟預覽視窗，改為無頭模式：{exc}", file=sys.stderr)
                        display = False

            if display:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            if args.max_frames is not None and pipeline.processed_frames >= args.max_frames:
                break
            if args.duration is not None and (time.perf_counter() - started_wall) >= args.duration:
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        closed = pipeline.flush(last_timestamp_ms)
        logger.log_closed(closed)
        summary = stats.summary()
        report = {
            **session_info,
            "metrics": summary,
            "skipped_frames": pipeline.skipped_frames,
            "events_logged": logger.event_count,
            "interrupted": interrupted,
            "stalled_out": stalled_out,
            "source_final": source.describe(),
        }
        logger.session_end({"metrics": summary, "events_logged": logger.event_count})
        logger.close()
        source.close()
        detector.close()
        if display:
            cv2.destroyAllWindows()

        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[info] 回報已寫入 {report_path}")

        if not args.quiet:
            print("\n=== 效能回報 ===")
            print(f"處理影格數      : {summary['processed_frames']}（節流跳過 {pipeline.skipped_frames}）")
            print(f"有效推論 FPS    : {summary['effective_inference_fps']}")
            print(f"pipeline 延遲   : mean {summary['pipeline_ms']['mean']} ms / P95 {summary['pipeline_ms']['p95']} ms")
            print(f"端到端延遲      : mean {summary['end_to_end_ms']['mean']} ms / P95 {summary['end_to_end_ms']['p95']} ms")
            print(f"事件數          : {logger.event_count}")
            if stalled_out:
                print("狀態            : 攝影機停頓而中止（非正常結束）")

    # 攝影機掛掉不能回 0——監測工作沒做完，呼叫端必須看得出來。
    return 3 if stalled_out else 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
