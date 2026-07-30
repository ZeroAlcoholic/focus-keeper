"""真實素材評測（規格 §6 準確度驗收的最小可行版本）。

用途：關閉「偵測準確度未以真實素材驗收」這個不確定性。合成臉只能驗流程，
真實會議影片才會出現低頭、側臉、手擋臉、講話時轉頭等真正的失效模式。

**單人 closeup 素材的關鍵性質**：主角全程在場，所以任何
``PRIMARY_MISSING`` / ``PRIMARY_LEFT`` 事件都是**誤報**。這讓我們不需要
逐幀人工標註，就能直接量出誤報率——這是最省力、資訊量最高的一次量測。

用法::

    python scripts/evaluate_real.py media/ami/IS1000a.Closeup1.avi --expect always-present

素材授權：AMI Meeting Corpus 為 CC BY 4.0（允許商用，需標示出處）。
本腳本**不儲存任何影格或臉部裁切**，只輸出統計數字。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from focus_keeper.config import PipelineConfig, load_config
from focus_keeper.metrics import LatencyStats
from focus_keeper.pipeline import AnalysisPipeline  # noqa: E402
from focus_keeper.detectors import build_detector  # noqa: E402
from focus_keeper.rules import EventType, RuleConfig, TemporalRuleEngine  # noqa: E402
from focus_keeper.sources import VideoSource  # noqa: E402
from focus_keeper.tracker import PrimarySubjectTracker, TrackerConfig  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="以真實影片評測偵測率與誤報率")
    parser.add_argument("video", help="影片路徑")
    parser.add_argument(
        "--expect", choices=["always-present", "unknown"], default="unknown",
        help="always-present = 主角全程在場，任何離場事件皆為誤報。"
             "⚠️ 這是承重前提，未經目視確認**不要**使用——"
             "多數真實素材的頭尾都是空景。",
    )
    parser.add_argument(
        "--absent", default=None,
        help="已目視確認的空景區間（秒），格式 '0-148,571-731'。"
             "這些區間會被排除在漏偵測率之外，並單獨檢查是否正確判為離場。",
    )
    parser.add_argument("--detector", default="yunet")
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit-seconds", type=float, default=None, help="只評測前 N 秒")
    parser.add_argument("--json", default=None, help="把結果寫成 JSON")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    detector = build_detector(args.detector, cfg["detectors"])
    pipeline = AnalysisPipeline(
        detector=detector,
        tracker=PrimarySubjectTracker(TrackerConfig.from_dict(cfg.get("tracker"))),
        rules=TemporalRuleEngine(RuleConfig.from_dict(cfg.get("rules"))),
        config=PipelineConfig.from_dict(cfg.get("pipeline")),
    )
    stats = LatencyStats()

    absent: list[tuple[float, float]] = []
    if args.absent:
        for chunk in args.absent.split(","):
            lo, hi = chunk.split("-")
            absent.append((float(lo) * 1000.0, float(hi) * 1000.0))

    def is_absent(ms: int) -> bool:
        return any(lo <= ms <= hi for lo, hi in absent)

    judged = 0
    detected = 0            # 至少一張有效臉
    primary_visible = 0     # 主角當下真的被偵測到
    scores: list[float] = []
    person_counts: Counter[int] = Counter()
    status_counts: Counter[str] = Counter()
    started, last_ts = [], 0

    # 分開統計「已知有人」與「已知空景」兩段，避免把空景的正確離場算成漏偵測。
    present_judged = present_miss = 0
    absent_judged = absent_correct = 0
    miss_timestamps: list[int] = []

    with VideoSource(args.video) as source:
        for packet in source:
            if args.limit_seconds and packet.timestamp_ms > args.limit_seconds * 1000:
                break
            result = pipeline.process(packet)
            if result is None:
                continue
            stats.add(result)
            judged += 1
            last_ts = result.timestamp_ms
            person_counts[result.rule_update.person_count] += 1
            status_counts[result.rule_update.status.value] += 1
            if result.detections:
                detected += 1
                scores.append(max(d.score for d in result.detections))
            if result.primary is not None and result.primary.visible:
                primary_visible += 1
            started.extend(result.rule_update.started)

            if is_absent(result.rule_update.timestamp_ms):
                absent_judged += 1
                if not result.detections:
                    absent_correct += 1
            elif absent:
                present_judged += 1
                if not result.detections:
                    present_miss += 1
                    miss_timestamps.append(result.timestamp_ms)

    # 連續漏偵測區段（偵測層診斷）。
    #
    # ⚠️ 這是**診斷數字，不是示警數**。示警數一律以引擎實際發出的
    # events_started 為準——腳本自己重寫一套合併規則去推算示警，
    # 只要規則和引擎不一致，得到的就是另一個系統的數字。
    #
    # 合併間隔取自實際判定週期（引擎採嚴格連續：任何一次偵測到就重新起算），
    # 不寫死常數，才不會在 inference_fps 改動後悄悄失準。
    gap_ms = int(1000.0 / PipelineConfig.from_dict(cfg.get("pipeline")).inference_fps * 1.5)
    miss_runs: list[int] = []
    run_start: int | None = None
    run_end: int = 0
    for ts in miss_timestamps:
        if run_start is None:
            run_start = run_end = ts
        elif ts - run_end <= gap_ms:
            run_end = ts
        else:
            miss_runs.append(run_end - run_start)
            run_start = run_end = ts
    if run_start is not None:
        miss_runs.append(run_end - run_start)

    closed = pipeline.flush(last_ts)
    detector.close()

    by_type = Counter(e.type.value for e in started)
    false_alarm_types = {
        EventType.PRIMARY_MISSING.value,
        EventType.PRIMARY_LEFT.value,
        EventType.MULTI_PERSON.value,
    }
    false_alarms = (
        sum(by_type[t] for t in false_alarm_types) if args.expect == "always-present" else None
    )

    duration_s = last_ts / 1000.0
    summary = {
        "video": str(args.video),
        "detector": detector.name,
        "expect": args.expect,
        "duration_s": round(duration_s, 1),
        "judgements": judged,
        "frame_detection_rate": round(detected / judged, 4) if judged else 0.0,
        "primary_visible_rate": round(primary_visible / judged, 4) if judged else 0.0,
        "confidence": {
            "mean": round(statistics.fmean(scores), 3) if scores else None,
            "min": round(min(scores), 3) if scores else None,
            "p05": round(sorted(scores)[max(0, int(0.05 * len(scores)) - 1)], 3) if scores else None,
            "max": round(max(scores), 3) if scores else None,
        },
        "person_count_distribution": dict(sorted(person_counts.items())),
        "status_distribution": dict(status_counts),
        "events_started": dict(by_type),
        "events_open_at_end": [e.type.value for e in closed],
        "metrics": stats.summary(),
    }
    if false_alarms is not None:
        summary["false_alarms"] = false_alarms
        summary["false_alarms_per_minute"] = (
            round(false_alarms / (duration_s / 60.0), 3) if duration_s > 0 else None
        )

    if absent:
        missing_hold = RuleConfig.from_dict(cfg.get("rules")).primary_missing_warn_ms
        left_hold = RuleConfig.from_dict(cfg.get("rules")).primary_left_ms
        summary["ground_truth"] = {
            "absent_intervals_s": [[lo / 1000, hi / 1000] for lo, hi in absent],
            "present": {
                "judgements": present_judged,
                "missed": present_miss,
                "miss_rate": round(present_miss / present_judged, 4) if present_judged else None,
                # 以下為偵測層診斷（推算），示警數請看 events_started（引擎實測）
                "_diagnostic_note": "miss_runs 為推算值；實際示警數以 events_started 為準",
                "miss_run_gap_ms": gap_ms,
                "miss_runs": len(miss_runs),
                "longest_miss_runs_ms": sorted(miss_runs, reverse=True)[:10],
                "runs_absorbed_by_hold": sum(1 for d in miss_runs if d < missing_hold),
                "runs_reaching_missing": sum(1 for d in miss_runs if d >= missing_hold),
                "runs_reaching_left": sum(1 for d in miss_runs if d >= left_hold),
            },
            # 交叉檢核（同基準）：只比「有人時段」內起算的 PRIMARY_MISSING。
            # 推算值與引擎實測若不符，代表腳本的合併規則與引擎已經脫節，
            # 這時所有以推算值寫成的結論都要作廢。
            "cross_check": {
                "predicted_missing_alerts": sum(1 for d in miss_runs if d >= missing_hold),
                "engine_missing_alerts_in_present": sum(
                    1
                    for e in started
                    if e.type is EventType.PRIMARY_MISSING
                    and not is_absent(e.condition_start_ms)
                ),
                "engine_missing_alerts_total": by_type.get(
                    EventType.PRIMARY_MISSING.value, 0
                ),
            },
            "absent": {
                "judgements": absent_judged,
                "correctly_no_face": absent_correct,
                "accuracy": round(absent_correct / absent_judged, 4) if absent_judged else None,
            },
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
