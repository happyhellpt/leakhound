# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joel Gomes
"""Features that encode the target almost perfectly (leaked columns)."""
from __future__ import annotations

import pandas as pd

from ..report import Finding

_FIX = ("Drop this feature, or replace it with information that is actually "
        "available at prediction time.")


def check_target_encoding(train: pd.DataFrame, target_col: str,
                         corr_threshold: float = 0.99,
                         purity_threshold: float = 0.99) -> list[Finding]:
    if target_col not in train.columns:
        return [Finding("target_encoding", "low", f"target '{target_col}' not found")]

    y = train[target_col]
    n = len(train)
    findings: list[Finding] = []

    for col in train.columns:
        if col == target_col:
            continue
        s = train[col]

        # Numeric feature that is almost a linear copy of the target.
        if pd.api.types.is_numeric_dtype(s) and pd.api.types.is_numeric_dtype(y):
            c = s.corr(y)
            if pd.notna(c) and abs(c) >= corr_threshold:
                findings.append(Finding(
                    "target_encoding", "high",
                    f"feature '{col}' correlates {c:.3f} with the target — it likely leaks it",
                    {"feature": col, "correlation": round(float(c), 4)},
                    fix=_FIX,
                ))
            continue

        # Datetime columns are the temporal check's job, not target-encoding.
        if pd.api.types.is_datetime64_any_dtype(s):
            continue

        # Near-unique columns (IDs, timestamps, uuids) are trivially "pure" within
        # a single set — that is NOT evidence of leakage on its own, so skip them.
        nunique = int(s.nunique(dropna=True))
        if nunique > 0.5 * n:
            continue

        # A genuine low/medium-cardinality feature that pins down the target.
        grp_purity = train.groupby(col)[target_col].transform(
            lambda x: x.value_counts(normalize=True).max()
        )
        if float(grp_purity.mean()) >= purity_threshold:
            findings.append(Finding(
                "target_encoding", "high",
                f"feature '{col}' determines the target "
                f"({float(grp_purity.mean()):.1%} pure) — likely leakage",
                {"feature": col, "target_purity": round(float(grp_purity.mean()), 4),
                 "distinct_values": nunique},
                fix=_FIX,
            ))

    if not findings:
        findings.append(Finding("target_encoding", "ok",
                                "no single feature trivially predicts the target"))
    return findings
