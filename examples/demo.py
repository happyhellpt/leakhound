"""A 20-second demo: build a dataset with planted leakage and watch LeakHound catch it."""
import numpy as np
import pandas as pd

from leakhound import audit

rng = np.random.default_rng(0)
n = 2000

df = pd.DataFrame({
    "patient_id": rng.integers(0, 600, n),          # groups
    "age": rng.integers(18, 90, n),
    "biomarker": rng.normal(0, 1, n),
    "date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
})
df["label"] = (df["biomarker"] + rng.normal(0, 0.3, n) > 0).astype(int)
# Planted leak: a column that is just the label with noise on top.
df["leaky_feature"] = df["label"] + rng.normal(0, 0.001, n)

# A naive random split (wrong: ignores patient groups and time)
train = df.sample(frac=0.7, random_state=1)
test = df.drop(train.index)
# Plant duplicate rows across the split too
test = pd.concat([test, train.iloc[:40]], ignore_index=True)

report = audit(train, test,
               target="label", time_col="date", group_col="patient_id")
print(report.render())
