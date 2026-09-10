import numpy as np
import pandas as pd

from leakhound import audit


def _leaky():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({
        "gid": rng.integers(0, 100, n),
        "x": rng.normal(0, 1, n),
    })
    df["label"] = (df["x"] > 0).astype(int)
    df["leak"] = df["label"] + rng.normal(0, 0.001, n)
    train = df.sample(frac=0.7, random_state=1)
    test = pd.concat([df.drop(train.index), train.iloc[:10]], ignore_index=True)
    return train, test


def test_catches_duplicates_and_target_leak():
    train, test = _leaky()
    report = audit(train, test, target="label", group_col="gid")
    assert not report.clean
    checks = {f.check for f in report.leaks}
    assert "duplicates" in checks
    assert "target_encoding" in checks
    assert "group_split" in checks


def test_clean_split_is_clean():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(0, 1, 400)})
    df["label"] = (df["x"] + rng.normal(0, 1, 400) > 0).astype(int)
    train = df.iloc[:300]
    test = df.iloc[300:]
    report = audit(train, test, target="label")
    assert report.clean


def test_ascii_fallback_is_pure_ascii():
    train, test = _leaky()
    report = audit(train, test, target="label", group_col="gid")
    ascii_out = report.render(ascii=True)
    # Must be encodable as ASCII on any legacy console, with no crash.
    ascii_out.encode("ascii")
    assert "[X]" in ascii_out
    for ch in "✗✓─·—":
        assert ch not in ascii_out
    # Unicode mode still uses the nice symbols.
    assert "✗" in report.render(ascii=False)
