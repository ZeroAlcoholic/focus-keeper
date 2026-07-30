"""JSONL 事件記錄。只寫中繼資料，不寫影像。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .rules import Event, RuleUpdate, TemporalRuleEngine

__all__ = ["EventLogger"]

class EventLogger:
    """JSONL 事件記錄；只寫中繼資料，不寫影像。

    每筆記錄都帶 ``logged_at``（含時區的 ISO 8601 絕對時間）。事件本身的
    ``*_ms`` 是相對於來源起點的毫秒，事後要把某個事件對應到當天的某個時段
    必須有絕對時間，否則整份記錄無法與任何外部時間軸對照。

    注意：這是**應用層記錄**，不是防篡改稽核軌跡。純文字檔可被任意編輯，
    沒有簽章也沒有 append-only 保證。若需要具證據力的軌跡，須另行設計。
    """

    def __init__(self, path: str | Path | None, rules: TemporalRuleEngine) -> None:
        self._rules = rules
        self._handle = None
        self.path = Path(path) if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = open(self.path, "a", encoding="utf-8")
        self.event_count = 0

    @staticmethod
    def _now() -> str:
        return datetime.now().astimezone().isoformat(timespec="milliseconds")

    def _write(self, record: Mapping[str, Any]) -> None:
        if self._handle is None:
            return
        self._handle.write(
            json.dumps({"logged_at": self._now(), **record}, ensure_ascii=False) + "\n"
        )
        self._handle.flush()

    def session_start(self, info: Mapping[str, Any]) -> None:
        self._write({"type": "session_start", "started_at": self._now(), **info})

    def log_update(self, update: RuleUpdate) -> None:
        for event in update.started:
            self.event_count += 1
            self._write(
                {
                    "type": "event_start",
                    **event.to_record(self._rules.threshold_for(event.type)),
                }
            )
        for event in update.ended:
            self._write(
                {"type": "event_end", **event.to_record(self._rules.threshold_for(event.type))}
            )

    def log_closed(self, events: Sequence[Event]) -> None:
        for event in events:
            self._write(
                {
                    "type": "event_end",
                    "closed_by": "session_end",
                    **event.to_record(self._rules.threshold_for(event.type)),
                }
            )

    def session_end(self, info: Mapping[str, Any]) -> None:
        self._write({"type": "session_end", **info})

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


# --------------------------------------------------------------------------- #
# 疊圖
# --------------------------------------------------------------------------- #

