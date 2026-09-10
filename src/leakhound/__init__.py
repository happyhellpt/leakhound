# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joel Gomes
"""LeakHound — sniff out data leakage in your ML train/test split before it fools you."""
from __future__ import annotations

import pandas as pd

from .report import Finding, Report
from .checks import (
    check_duplicates,
    check_temporal,
    check_target_encoding,
    check_group_split,
)
from .autodetect import infer_roles, build_advisories
from .homology import check_homology

__version__ = "0.5.0"
__all__ = ["audit", "Report", "Finding", "infer_roles", "build_advisories", "__version__"]


def audit(train: pd.DataFrame,
          test: pd.DataFrame | None = None,
          *,
          target: str | None = None,
          time_col: str | None = None,
          group_col: str | None = None,
          seq_col: str | None = None,
          seq_k: int | None = None,
          feature_cols: list[str] | None = None,
          measure_impact: bool = False) -> Report:
    """Run every applicable leakage check and return a Report.

    Only the checks whose inputs you supply are run. Set measure_impact=True to
    also fit a quick baseline and quantify how much each leak inflates the score
    (requires scikit-learn; a test set and a target).
    """
    report = Report()

    if test is not None:
        for f in check_duplicates(train, test, feature_cols):
            report.add(f)
    if target is not None:
        for f in check_target_encoding(train, target):
            report.add(f)
    if test is not None and time_col is not None:
        report.add(check_temporal(train, test, time_col))
    if test is not None and group_col is not None:
        report.add(check_group_split(train, test, group_col))
    if test is not None and seq_col is not None:
        report.add(check_homology(train, test, seq_col, k=seq_k))

    if measure_impact and test is not None:
        from .adversarial import adversarial_check
        for f in adversarial_check(train, test, target=target,
                                   time_col=time_col, group_col=group_col):
            report.add(f)
        if target is not None:
            from .impact import measure_leak_impact
            for f in measure_leak_impact(train, test, target, report.leaks,
                                         time_col=time_col, group_col=group_col):
                report.add(f)

    if not report.findings:
        report.add(Finding("audit", "low",
                           "nothing to check — pass a test set, a target, a time_col or a group_col"))
    return report
