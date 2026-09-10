# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joel Gomes
"""Rows that appear in both the training and the test set (exact and near-exact)."""
from __future__ import annotations

import pandas as pd

from ..report import Finding

_FIX = ("Remove duplicates before splitting (df.drop_duplicates()) and make sure "
        "no test row also exists in train.")


def _exact_keys(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].astype(str).agg("|".join, axis=1)


def _rounded_keys(df: pd.DataFrame, cols: list[str], decimals: int) -> pd.Series:
    out = df[cols].copy()
    num = out.select_dtypes(include="number").columns
    if len(num):
        out[num] = out[num].round(decimals)
    return out.astype(str).agg("|".join, axis=1)


def check_duplicates(train: pd.DataFrame, test: pd.DataFrame,
                     feature_cols: list[str] | None = None,
                     near_dup_decimals: int = 4) -> list[Finding]:
    cols = feature_cols or [c for c in train.columns if c in test.columns]
    if not cols:
        return [Finding("duplicates", "low", "no shared columns to compare")]

    findings: list[Finding] = []

    train_exact = set(_exact_keys(train, cols))
    test_exact = _exact_keys(test, cols)
    exact_mask = test_exact.isin(train_exact)
    n_exact = int(exact_mask.sum())

    if n_exact == 0:
        findings.append(Finding("duplicates", "ok",
                                "no identical rows shared between train and test"))
    else:
        frac = n_exact / len(test)
        sev = "high" if frac > 0.01 else "medium"
        findings.append(Finding(
            "duplicates", sev,
            f"{n_exact} test rows ({frac:.1%}) also appear in training — the model has seen them",
            {"matched_on": cols, "n_shared_rows": n_exact, "fraction_of_test": round(frac, 4)},
            fix=_FIX,
        ))

    # Near-duplicates: identical after rounding numeric columns, excluding exact matches.
    has_numeric = len(train[cols].select_dtypes(include="number").columns) > 0
    if has_numeric:
        train_round = set(_rounded_keys(train, cols, near_dup_decimals))
        test_round = _rounded_keys(test, cols, near_dup_decimals)
        near_mask = test_round.isin(train_round) & (~exact_mask)
        n_near = int(near_mask.sum())
        if n_near > 0:
            frac = n_near / len(test)
            findings.append(Finding(
                "near_duplicates", "medium",
                f"{n_near} test rows ({frac:.1%}) are near-identical to training rows "
                f"(match after rounding to {near_dup_decimals} decimals) — likely copies",
                {"n_near_dup_rows": n_near, "rounded_decimals": near_dup_decimals},
                fix="Check your pipeline for copied/augmented rows; dedupe with a tolerance.",
            ))

    return findings
