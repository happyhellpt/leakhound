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

__version__ = "0.2.0"
__all__ = ["audit", "Report", "Finding", "infer_roles", "build_advisories", "__version__"]


def audit(train: pd.DataFrame,
          test: pd.DataFrame | None = None,
          *,
          target: str | None = None,
          time_col: str | None = None,
          group_col: str | None = None,
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

    if measure_impact and test is not None and target is not None:
        from .impact import measure_leak_impact
        for f in measure_leak_impact(train, test, target, report.leaks):
            report.add(f)

    if not report.findings:
        report.add(Finding("audit", "low",
                           "nothing to check — pass a test set, a target, a time_col or a group_col"))
    return report
