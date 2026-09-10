# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joel Gomes
"""Real-world case study: a random split of a continuous EEG recording lies to you.

EEG Eye State (OpenML, ~15,000 rows) is ONE continuous recording sampled at 128 Hz
— each row is a moment in time. Split it randomly (the default everyone reaches
for) and your model looks brilliant. Split it honestly (past -> future) and it
collapses, because neighbouring samples are almost identical.

Requires:  pip install 'leakhound-ml[impact]'  and an internet connection.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from leakhound import audit

print("Fetching EEG Eye State from OpenML ...")
d = fetch_openml("eeg-eye-state", version=1, as_frame=True, parser="pandas")
df = d.frame.copy()
tgt = df.columns[-1]
df["label"] = df[tgt].astype("category").cat.codes.astype(int)
df = df.drop(columns=[tgt])
# real recording cadence: 128 Hz. The row order is genuine time.
df["time"] = pd.Timestamp("2011-06-15 14:00:00") + pd.to_timedelta(np.arange(len(df)) / 128.0, unit="s")

rng = np.random.default_rng(0)
idx = rng.permutation(len(df))
cut = int(0.8 * len(df))
train = df.iloc[idx[:cut]].reset_index(drop=True)
test = df.iloc[idx[cut:]].reset_index(drop=True)

print("\n== What LeakHound says about the naive random split ==")
print(audit(train, test, target="label", time_col="time").render())

# The honest cost — using only the real EEG features (NOT the time column).
X = df.drop(columns=["label", "time"])
y = df["label"]
def auc(tr, te):
    m = HistGradientBoostingClassifier(random_state=0).fit(X.iloc[tr], y.iloc[tr])
    return roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1])

r = auc(idx[:cut], idx[cut:])
c = auc(np.arange(cut), np.arange(cut, len(df)))
print("== The honest cost ==")
print(f"  AUC you'd report from the random split: {r:.3f}")
print(f"  AUC from an honest chronological split: {c:.3f}")
print(f"  >> {r - c:+.3f} AUC of pure fiction.\n")
