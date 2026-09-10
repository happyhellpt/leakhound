"""Features that encode the target almost perfectly (IDs, leaked columns)."""
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
    findings: list[Finding] = []

    for col in train.columns:
        if col == target_col:
            continue
        s = train[col]

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

        grp_purity = train.groupby(col)[target_col].transform(
            lambda x: x.value_counts(normalize=True).max()
        )
        mean_purity = float(grp_purity.mean())
        nunique = int(s.nunique(dropna=True))
        id_like = nunique > 0.5 * len(train)

        if mean_purity >= purity_threshold:
            sev = "high" if id_like else "medium"
            label = "row-id-like feature" if id_like else "feature"
            findings.append(Finding(
                "target_encoding", sev,
                f"{label} '{col}' determines the target ({mean_purity:.1%} pure) — likely leakage",
                {"feature": col, "target_purity": round(mean_purity, 4),
                 "distinct_values": nunique},
                fix=_FIX,
            ))

    if not findings:
        findings.append(Finding("target_encoding", "ok",
                                "no single feature trivially predicts the target"))
    return findings
