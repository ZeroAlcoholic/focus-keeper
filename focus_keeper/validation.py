"""設定驗證的共用基礎。

刻意做成**葉模組**（不匯入任何專案內模組），讓 config／rules／tracker／
sources 都能引用而不產生循環匯入。

所有設定錯誤都收斂成單一 :class:`ConfigError`，呼叫端只需要 catch 一種例外。
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

__all__ = ["ConfigError", "check_fields", "as_mapping"]


class ConfigError(ValueError):
    """設定有誤。

    繼承 ``ValueError`` 而非 ``KeyError``：未知欄位代表**值寫錯了**，
    不是「查詢某個 key 卻不存在」，語意上屬於 ValueError。
    """


def as_mapping(section: str, value: Any) -> dict[str, Any]:
    """把設定區塊轉成 dict；``None`` 視為空區塊。"""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{section} 應為 mapping，實得 {type(value).__name__}")
    return dict(value)


def check_fields(section: str, cfg: Mapping[str, Any], allowed: Iterable[str]) -> None:
    """未知欄位一律報錯，不得靜默忽略。

    靜默忽略打錯的欄位名是最難察覺的設定錯誤：把 ``rules`` 打成 ``rulez``
    會讓整組時間門檻悄悄退回預設值，而程式看起來完全正常。
    """
    allowed = set(allowed)
    unknown = sorted(set(cfg) - allowed)
    if unknown:
        raise ConfigError(
            f"{section} 含未知欄位：{unknown}；可用欄位：{sorted(allowed)}"
        )
