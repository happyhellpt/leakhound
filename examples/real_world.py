# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joel Gomes
"""Real-world case study: a random split of a continuous EEG recording lies to you.

EEG Eye State (OpenML, ~15,000 rows) is ONE continuous recording sampled at 128 Hz.
Split it randomly (the default everyone reaches for) and your model looks brilliant.
LeakHound flags the leak AND measures exactly how much it inflated your score.

Requires:  pip install 'leakhound-ml[impact]'  and an internet connection.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from leakhound import audit

print("Fetching EEG Eye State from OpenML ...")
d = fetch_openml("eeg-eye-state", version=1, as_frame=True, parser="pandas")
df = d.frame.copy()
tgt = df.columns[-1]
df["label"] = df[tgt].astype("category").cat.codes.astype(int)
df = df.drop(columns=[tgt])
df["time"] = pd.Timestamp("2011-06-15 14:00:00") + pd.to_timedelta(np.arange(len(df)) / 128.0, unit="s")

rng = np.random.default_rng(0)
idx = rng.permutation(len(df))
cut = int(0.8 * len(df))
train = df.iloc[idx[:cut]].reset_index(drop=True)
test = df.iloc[idx[cut:]].reset_index(drop=True)

# LeakHound both DETECTS the leak and MEASURES what it costs.
print(audit(train, test, target="label", time_col="time", measure_impact=True).render())
