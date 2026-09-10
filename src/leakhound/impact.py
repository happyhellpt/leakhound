# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joel Gomes
"""Quantify the damage: fit a quick baseline and measure how much each leak inflates the score."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .report import Finding

_NEED_SKLEARN = ("install scikit-learn to measure leak impact: "
                 "pip install 'leakhound-ml[impact]'")
_EPS = 0.005  # ignore inflation smaller than this (noise)


def _severity(inflation: float) -> str:
    return "high" if inflation >= 0.02 else "medium"


def measure_leak_impact(train: pd.DataFrame, test: pd.DataFrame, target: str,
                        leaks: list[Finding], *,
                        time_col: str | None = None,
                        group_col: str | None = None,
                        random_state: int = 0) -> list[Finding]:
    try:
        from sklearn.ensemble import (HistGradientBoostingClassifier,
                                      HistGradientBoostingRegressor)
        from sklearn.metrics import roc_auc_score, r2_score
    except Exception:
        return [Finding("impact", "low", _NEED_SKLEARN)]
    try:
        return _impact(train, test, target, leaks, time_col, group_col, random_state,
                       HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                       roc_auc_score, r2_score)
    except Exception as e:  # never let impact crash the whole audit
        return [Finding("impact", "low", f"could not measure impact ({type(e).__name__}: {e})")]


def _impact(train, test, target, leaks, time_col, group_col, rs, Clf, Reg, auc, r2):
    if target not in train.columns or target not in test.columns:
        return [Finding("impact", "low", "target not in both sets; cannot measure impact")]

    y = train[target]
    nun = int(y.nunique(dropna=True))
    is_binary = nun == 2 and pd.api.types.is_numeric_dtype(y)
    is_reg = pd.api.types.is_float_dtype(y) and nun > 10
    if not (is_binary or is_reg):
        return [Finding("impact", "low",
                        f"impact needs a binary (numeric 0/1) or continuous target; "
                        f"'{target}' has {nun} distinct values")]

    exclude = {target, time_col, group_col}
    feats = [c for c in train.select_dtypes(include="number").columns
             if c not in exclude and c in test.columns]
    if not feats:
        return [Finding("impact", "low", "no numeric features available to fit a baseline")]

    metric = "AUC" if is_binary else "R2"

    def score(features, tr_df, te_df):
        Xtr, Xte = tr_df[features], te_df[features]
        if is_binary:
            m = Clf(random_state=rs).fit(Xtr, tr_df[target])
            return float(auc(te_df[target], m.predict_proba(Xte)[:, 1]))
        m = Reg(random_state=rs).fit(Xtr, tr_df[target])
        return float(r2(te_df[target], m.predict(Xte)))

    findings: list[Finding] = []
    base = score(feats, train, test)              # the score on the split as given
    full = pd.concat([train, test], ignore_index=True)
    n_test = len(test)

    def add(inflation, msg, evidence, fix):
        if inflation >= _EPS:
            findings.append(Finding("impact", _severity(inflation), msg, evidence, fix=fix))

    # 1) Target-encoding: drop the leaking feature(s) and re-score on the same split.
    leaky = [f.evidence.get("feature") for f in leaks
             if f.check == "target_encoding" and f.evidence.get("feature") in feats]
    leaky = [c for c in leaky if c]
    if leaky:
        honest_feats = [c for c in feats if c not in leaky]
        if honest_feats:
            honest = score(honest_feats, train, test)
            add(base - honest,
                f"removing leaking feature(s) {leaky} drops {metric} "
                f"from {base:.3f} to {honest:.3f} — {base - honest:+.3f} of fake performance",
                {f"{metric}_with_leak": round(base, 3), f"{metric}_without_leak": round(honest, 3),
                 "inflation": round(base - honest, 3)},
                "Drop the leaking feature(s) and re-evaluate honestly.")

    # 2) Duplicates: evaluate only on test rows the model has never seen.
    if any(f.check in ("duplicates", "near_duplicates") for f in leaks):
        shared = [c for c in train.columns if c in test.columns]
        tr_keys = set(train[shared].astype(str).agg("|".join, axis=1))
        te_keys = test[shared].astype(str).agg("|".join, axis=1)
        unseen = ~te_keys.isin(tr_keys)
        if 0 < int(unseen.sum()) < len(test):
            honest = score(feats, train, test[unseen])
            add(base - honest,
                f"on the {int(unseen.sum())} test rows the model has NOT seen, {metric} is "
                f"{honest:.3f} (vs {base:.3f} on the full test) — {base - honest:+.3f} inflated",
                {f"{metric}_full_test": round(base, 3), f"{metric}_unseen_only": round(honest, 3),
                 "inflation": round(base - honest, 3)},
                "Evaluate only on rows the model never saw during training.")

    # 3) Temporal: re-split the combined data chronologically and re-score.
    if time_col and any(f.check == "temporal" for f in leaks) and time_col in full.columns:
        order = pd.to_datetime(full[time_col], errors="coerce").argsort(kind="stable")
        ordered = full.iloc[order].reset_index(drop=True)
        honest = score(feats, ordered.iloc[:len(full) - n_test], ordered.iloc[len(full) - n_test:])
        add(base - honest,
            f"an honest chronological split scores {metric} {honest:.3f} "
            f"(vs {base:.3f} on your split) — {base - honest:+.3f} inflated",
            {f"{metric}_your_split": round(base, 3), f"{metric}_chronological": round(honest, 3),
             "inflation": round(base - honest, 3)},
            "Split by time: train on the past, test on the future.")

    # 4) Group: re-split the combined data so no group spans the split, and re-score.
    if group_col and any(f.check == "group_split" for f in leaks) and group_col in full.columns:
        rng = np.random.default_rng(rs)
        groups = full[group_col]
        sizes = groups.value_counts()
        test_groups, count = set(), 0
        for g in rng.permutation(groups.dropna().unique()):
            if count >= n_test:
                break
            test_groups.add(g)
            count += int(sizes[g])
        mask = groups.isin(test_groups).to_numpy()
        if 0 < int(mask.sum()) < len(full):
            honest = score(feats, full[~mask], full[mask])
            add(base - honest,
                f"a group-aware split scores {metric} {honest:.3f} "
                f"(vs {base:.3f} on your split) — {base - honest:+.3f} inflated",
                {f"{metric}_your_split": round(base, 3), f"{metric}_group_aware": round(honest, 3),
                 "inflation": round(base - honest, 3)},
                "Keep each group on one side (GroupShuffleSplit / GroupKFold).")

    if not findings:
        findings.append(Finding("impact", "ok",
                                f"baseline {metric} {base:.3f}; no measurable inflation from detected leaks"))
    return findings
