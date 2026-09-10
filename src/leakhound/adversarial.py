# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joel Gomes
"""Adversarial validation: can a model tell your train set from your test set?

If it can, your split is not a random sample of one distribution — that's either
distribution shift or a column that encodes the split (an id, a timestamp). Either
way, your evaluation may not mean what you think.
"""
from __future__ import annotations

import pandas as pd

from .report import Finding


def adversarial_check(train: pd.DataFrame, test: pd.DataFrame, *,
                      target: str | None = None,
                      time_col: str | None = None,
                      group_col: str | None = None,
                      random_state: int = 0) -> list[Finding]:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import roc_auc_score
    except Exception:
        return []  # a bonus check; silently skip without scikit-learn
    try:
        return _adv(train, test, {target, time_col, group_col}, random_state,
                    HistGradientBoostingClassifier, cross_val_predict, roc_auc_score)
    except Exception:
        return []


def _adv(train, test, exclude, rs, Clf, cvp, auc):
    feats = [c for c in train.select_dtypes(include="number").columns
             if c not in exclude and c in test.columns]
    if not feats or min(len(train), len(test)) < 20:
        return []

    X = pd.concat([train[feats], test[feats]], ignore_index=True)
    y = pd.Series([0] * len(train) + [1] * len(test))
    proba = cvp(Clf(random_state=rs), X, y, cv=5, method="predict_proba")[:, 1]
    a = float(auc(y, proba))

    if a < 0.65:
        return [Finding("adversarial", "ok",
                        f"train and test look like one distribution (adversarial AUC {a:.3f})",
                        {"adversarial_auc": round(a, 3)})]

    corrs = {c: abs(float(X[c].corr(y))) for c in feats if X[c].notna().any()}
    top = [c for c, _ in sorted(corrs.items(), key=lambda kv: kv[1], reverse=True)[:3]]
    sev = "high" if a >= 0.8 else "medium"
    return [Finding(
        "adversarial", sev,
        f"a model tells train from test apart (adversarial AUC {a:.3f}) — "
        f"distribution shift, or a column that encodes the split",
        {"adversarial_auc": round(a, 3), "top_separating_features": top},
        fix="Inspect the top separating features; drop split-identifying columns "
            "(ids, timestamps) and make train and test come from the same distribution.",
    )]
