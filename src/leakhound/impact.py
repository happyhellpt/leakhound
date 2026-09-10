"""Quantify the damage: fit a quick baseline and measure how much each leak inflates the score."""
from __future__ import annotations

import pandas as pd

from .report import Finding

_NEED_SKLEARN = ("install scikit-learn to measure leak impact: "
                 "pip install 'leakhound[impact]'")
_EPS = 0.005  # ignore inflation smaller than this (noise)


def _numeric_features(df: pd.DataFrame, target: str) -> list[str]:
    return [c for c in df.select_dtypes(include="number").columns if c != target]


def _severity(inflation: float) -> str:
    return "high" if inflation >= 0.02 else "medium"


def measure_leak_impact(train: pd.DataFrame, test: pd.DataFrame, target: str,
                        leaks: list[Finding], *, random_state: int = 0) -> list[Finding]:
    try:
        from sklearn.ensemble import (HistGradientBoostingClassifier,
                                      HistGradientBoostingRegressor)
        from sklearn.metrics import roc_auc_score, r2_score
    except Exception:
        return [Finding("impact", "low", _NEED_SKLEARN)]
    try:
        return _impact(train, test, target, leaks, random_state,
                       HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                       roc_auc_score, r2_score)
    except Exception as e:  # never let impact crash the whole audit
        return [Finding("impact", "low", f"could not measure impact ({type(e).__name__}: {e})")]


def _impact(train, test, target, leaks, rs, Clf, Reg, auc, r2) -> list[Finding]:
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

    feats = [c for c in _numeric_features(train, target) if c in test.columns]
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
    base = score(feats, train, test)

    leaky = [f.evidence.get("feature") for f in leaks
             if f.check == "target_encoding" and f.evidence.get("feature") in feats]
    leaky = [c for c in leaky if c]
    if leaky:
        honest_feats = [c for c in feats if c not in leaky]
        if honest_feats:
            honest = score(honest_feats, train, test)
            infl = base - honest
            if infl >= _EPS:
                findings.append(Finding(
                    "impact", _severity(infl),
                    f"removing leaking feature(s) {leaky} drops {metric} "
                    f"from {base:.3f} to {honest:.3f} — {infl:+.3f} of fake performance",
                    {f"{metric}_with_leak": round(base, 3),
                     f"{metric}_without_leak": round(honest, 3),
                     "inflation": round(infl, 3)},
                    fix="Drop the leaking feature(s) and re-evaluate honestly.",
                ))

    if any(f.check in ("duplicates", "near_duplicates") for f in leaks):
        shared = [c for c in train.columns if c in test.columns]
        tr_keys = set(train[shared].astype(str).agg("|".join, axis=1))
        te_keys = test[shared].astype(str).agg("|".join, axis=1)
        unseen = ~te_keys.isin(tr_keys)
        n_unseen = int(unseen.sum())
        if 0 < n_unseen < len(test):
            full = score(feats, train, test)
            honest = score(feats, train, test[unseen])
            infl = full - honest
            if infl >= _EPS:
                findings.append(Finding(
                    "impact", _severity(infl),
                    f"on the {n_unseen} test rows the model has NOT seen, {metric} is "
                    f"{honest:.3f} (vs {full:.3f} on the full test) — {infl:+.3f} inflated",
                    {f"{metric}_full_test": round(full, 3),
                     f"{metric}_unseen_only": round(honest, 3),
                     "inflation": round(infl, 3)},
                    fix="Evaluate only on rows the model never saw during training.",
                ))

    if not findings:
        findings.append(Finding("impact", "ok",
                                f"baseline {metric} {base:.3f}; no measurable inflation from detected leaks"))
    return findings
