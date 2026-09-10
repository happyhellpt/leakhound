"""Best-effort inference of target / time / group columns, and pre-split advice."""
from __future__ import annotations

import pandas as pd

from .report import Finding

_TARGET_NAMES = {"target", "label", "y", "class", "outcome", "response", "churn", "is_fraud"}
_TIME_NAMES = ("date", "time", "timestamp", "datetime", "created", "_at", "ds")
_GROUP_HINTS = ("_id", "id", "patient", "user", "device", "session", "customer",
                "account", "group", "subject")


def infer_roles(df: pd.DataFrame) -> dict[str, str | None]:
    roles: dict[str, str | None] = {"target": None, "time_col": None, "group_col": None}
    low = {c: str(c).lower() for c in df.columns}

    for c in df.columns:
        if low[c] in _TARGET_NAMES:
            roles["target"] = c
            break

    for c in df.columns:
        if any(t in low[c] for t in _TIME_NAMES):
            roles["time_col"] = c
            break
    if roles["time_col"] is None:
        for c in df.columns:
            if df[c].dtype == "object":
                if pd.to_datetime(df[c], errors="coerce").notna().mean() > 0.8:
                    roles["time_col"] = c
                    break

    n = len(df)
    for c in df.columns:
        if c in (roles["target"], roles["time_col"]):
            continue
        if any(h in low[c] for h in _GROUP_HINTS) and 1 < df[c].nunique(dropna=True) < n:
            roles["group_col"] = c
            break

    return roles


def build_advisories(roles: dict[str, str | None]) -> list[Finding]:
    """Advice when you only have one file and haven't split yet."""
    out: list[Finding] = []
    if roles.get("time_col"):
        out.append(Finding(
            "temporal", "medium",
            f"column '{roles['time_col']}' looks like a time column — a random split would "
            f"let the model train on the future",
            {"column": roles["time_col"]},
            fix="Split chronologically instead of randomly.",
        ))
    if roles.get("group_col"):
        out.append(Finding(
            "group_split", "medium",
            f"column '{roles['group_col']}' looks like a group id — a random split would put "
            f"the same group on both sides",
            {"column": roles["group_col"]},
            fix="Use a group-aware splitter (GroupShuffleSplit / GroupKFold).",
        ))
    return out
