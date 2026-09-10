# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joel Gomes
"""Homology leakage in protein data — the leak generic tools miss.

We build protein families (near-identical sequences), then split them the naive
way (random). LeakHound flags that similar sequences span the split. Then we split
with safesplit's sequence-aware mode and LeakHound confirms it's clean.

Requires:  pip install leakhound-ml safesplit
"""
import numpy as np
import pandas as pd
from leakhound import audit

rng = np.random.default_rng(0)
AA = list("ACDEFGHIKLMNPQRSTVWY")

def mutate(seq, n):
    seq = list(seq)
    for _ in range(n):
        seq[rng.integers(0, len(seq))] = rng.choice(AA)
    return "".join(seq)

families = ["".join(rng.choice(AA, 80)) for _ in range(40)]
rows = [{"seq": mutate(f, rng.integers(1, 4)), "activity": int(i % 2)}
        for i, f in enumerate(families) for _ in range(8)]
df = pd.DataFrame(rows)

# naive random split
idx = rng.permutation(len(df)); cut = int(0.75 * len(df))
train, test = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
print("== Naive random split ==")
print(audit(train, test, seq_col="seq").render())

# sequence-aware split (the fix)
try:
    from safesplit import safe_split
    s_train, s_test = safe_split(df, seq_col="seq", test_size=0.25)
    print("== safesplit(seq_col=...) — homology-aware ==")
    print(audit(s_train, s_test, seq_col="seq").render())
except ImportError:
    print("(pip install safesplit to see the sequence-aware fix)")
