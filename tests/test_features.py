import numpy as np
import pandas as pd
import pytest

from leakhound import audit, infer_roles


def test_near_duplicates_detected():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
    train["label"] = (train["a"] > 0).astype(int)

    near = train.iloc[:5].copy()
    near["a"] = near["a"] + 1e-6  # tiny perturbation -> near, not exact
    fresh = pd.DataFrame({"a": rng.normal(size=10) + 50, "b": rng.normal(size=10)})
    fresh["label"] = (fresh["a"] > 0).astype(int)
    test = pd.concat([fresh, near], ignore_index=True)

    report = audit(train, test)
    checks = {f.check for f in report.findings}
    assert "near_duplicates" in checks


def test_infer_roles():
    df = pd.DataFrame({
        "patient_id": [1, 1, 2, 2, 3, 3],
        "date": pd.date_range("2024-01-01", periods=6).astype(str),
        "biomarker": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "label": [0, 1, 0, 1, 0, 1],
    })
    roles = infer_roles(df)
    assert roles["target"] == "label"
    assert roles["time_col"] == "date"
    assert roles["group_col"] == "patient_id"


def test_html_report_is_self_contained():
    rng = np.random.default_rng(1)
    train = pd.DataFrame({"x": rng.normal(size=200)})
    train["label"] = (train["x"] > 0).astype(int)
    train["leak"] = train["label"] + rng.normal(0, 0.001, 200)
    test = pd.concat([train.iloc[:150], train.iloc[:20]], ignore_index=True)

    html = audit(train, test, target="label").to_html()
    assert html.startswith("<!doctype html>")
    assert "class=\"card\"" in html
    assert "http://" not in html and "https://" not in html  # no external assets


def test_impact_quantifies_inflation():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(2)
    n = 600
    df = pd.DataFrame({"signal": rng.normal(size=n)})
    df["label"] = (0.4 * df["signal"] + rng.normal(0, 1, n) > 0).astype(int)
    df["leak"] = df["label"] + rng.normal(0, 0.001, n)  # near-perfect fake predictor
    train = df.iloc[:400]
    test = df.iloc[400:]

    report = audit(train, test, target="label", measure_impact=True)
    impacts = [f for f in report.findings if f.check == "impact"]
    assert any(f.evidence.get("inflation", 0) > 0.05 for f in impacts)


def test_temporal_impact_is_measured():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(3)
    n = 1500
    # a drifting time series: the honest (chronological) score is much worse
    t = pd.date_range("2024-01-01", periods=n, freq="h")
    trend = np.linspace(-3, 3, n)
    x = trend + rng.normal(0, 0.5, n)
    label = (rng.normal(0, 1, n) + trend > 0).astype(int)
    df = pd.DataFrame({"time": t, "x": x, "label": label})
    ridx = rng.permutation(n); c = int(0.8 * n)
    train = df.iloc[ridx[:c]].reset_index(drop=True)
    test = df.iloc[ridx[c:]].reset_index(drop=True)
    report = audit(train, test, target="label", time_col="time", measure_impact=True)
    assert any(f.check == "impact" and "chronological" in f.message for f in report.findings)


def test_adversarial_flags_distribution_shift():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(4)
    train = pd.DataFrame({"x": rng.normal(0, 1, 300), "label": rng.integers(0, 2, 300)})
    test = pd.DataFrame({"x": rng.normal(3, 1, 200), "label": rng.integers(0, 2, 200)})  # shifted
    report = audit(train, test, target="label", measure_impact=True)
    adv = [f for f in report.findings if f.check == "adversarial"]
    assert adv and adv[0].severity in ("high", "medium")


def test_homology_leakage_detected():
    rng = np.random.default_rng(5)
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    def seq(L=60): return "".join(rng.choice(aa, L))
    train = pd.DataFrame({"seq": [seq() for _ in range(120)], "y": rng.integers(0, 2, 120)})
    # 20 near-copies of training seqs (1 residue mutated) + 30 fresh sequences
    near = []
    for s in train["seq"].iloc[:20]:
        s = list(s); s[10] = rng.choice(aa); near.append("".join(s))
    fresh = [seq() for _ in range(30)]
    test = pd.DataFrame({"seq": near + fresh, "y": rng.integers(0, 2, 50)})
    report = audit(train, test, seq_col="seq")
    hom = [f for f in report.findings if f.check == "homology"]
    assert hom and hom[0].severity in ("high", "medium")
    assert hom[0].evidence["n_similar"] >= 15


def test_homology_clean_when_distinct():
    rng = np.random.default_rng(6)
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    def seq(L=60): return "".join(rng.choice(aa, L))
    train = pd.DataFrame({"seq": [seq() for _ in range(80)]})
    test = pd.DataFrame({"seq": [seq() for _ in range(40)]})
    report = audit(train, test, seq_col="seq")
    hom = [f for f in report.findings if f.check == "homology"][0]
    assert hom.severity == "ok"
