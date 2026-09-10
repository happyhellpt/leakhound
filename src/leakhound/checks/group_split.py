"""The same group (patient, user, device) appearing in both train and test."""
from __future__ import annotations

import pandas as pd

from ..report import Finding

_FIX = ("Use a group-aware splitter (sklearn GroupShuffleSplit / GroupKFold) so "
        "each group stays entirely on one side of the split.")


def check_group_split(train: pd.DataFrame, test: pd.DataFrame, group_col: str) -> Finding:
    if group_col not in train.columns or group_col not in test.columns:
        return Finding("group_split", "low", f"group column '{group_col}' not in both sets")

    tr = set(train[group_col].dropna().unique())
    te = set(test[group_col].dropna().unique())
    shared = tr & te
    if not shared:
        return Finding("group_split", "ok",
                       f"no '{group_col}' appears in both train and test")

    frac = len(shared) / max(len(te), 1)
    sev = "high" if frac > 0.1 else "medium"
    sample = list(map(str, list(shared)[:5]))
    return Finding(
        "group_split", sev,
        f"{len(shared)} values of '{group_col}' ({frac:.1%} of test groups) are in both sets — "
        f"rows from the same group leak across the split",
        {"group_col": group_col, "n_shared_groups": len(shared), "examples": sample},
        fix=_FIX,
    )
