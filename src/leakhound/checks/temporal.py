"""Training rows dated at or after the earliest test row (look-ahead leakage)."""
from __future__ import annotations

import pandas as pd

from ..report import Finding

_FIX = "Split chronologically: sort by the time column and put later rows in test."


def check_temporal(train: pd.DataFrame, test: pd.DataFrame, time_col: str) -> Finding:
    if time_col not in train.columns or time_col not in test.columns:
        return Finding("temporal", "low", f"time column '{time_col}' not in both sets")

    tr = pd.to_datetime(train[time_col], errors="coerce")
    te = pd.to_datetime(test[time_col], errors="coerce")
    if tr.notna().sum() == 0 or te.notna().sum() == 0:
        return Finding("temporal", "low", f"could not parse '{time_col}' as dates")

    test_min = te.min()
    future = int((tr >= test_min).sum())
    if future == 0:
        return Finding("temporal", "ok", "all training rows predate the test window")

    frac = future / len(train)
    sev = "high" if frac > 0.05 else "medium"
    return Finding(
        "temporal", sev,
        f"{future} training rows ({frac:.1%}) are dated at/after the earliest test row — "
        f"the model trains on the future",
        {"earliest_test": str(test_min), "latest_train": str(tr.max()),
         "offending_train_rows": future},
        fix=_FIX,
    )
