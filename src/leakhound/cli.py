# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joel Gomes
"""leakhound command-line interface."""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import __version__, audit, infer_roles, build_advisories


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="leakhound",
        description="Sniff out data leakage in your ML train/test split.",
    )
    p.add_argument("--train", required=True, help="path to the training CSV")
    p.add_argument("--test", help="path to the test/validation CSV")
    p.add_argument("--target", help="name of the target/label column")
    p.add_argument("--time-col", help="timestamp column (checks look-ahead leakage)")
    p.add_argument("--group-col", help="group id column, e.g. patient/user/device")
    p.add_argument("--seq-col", help="sequence column (protein/DNA) for homology-leakage check")
    p.add_argument("--auto", action="store_true",
                   help="auto-detect target/time/group columns (and advise on a single file)")
    p.add_argument("--measure-impact", action="store_true",
                   help="fit a baseline and quantify how much each leak inflates the score")
    p.add_argument("--html", metavar="PATH", help="also write a shareable HTML report")
    p.add_argument("--ascii", action="store_true",
                   help="force plain ASCII output (for legacy terminals)")
    p.add_argument("--version", action="version", version=f"leakhound {__version__}")
    args = p.parse_args(argv)

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test) if args.test else None

    target, time_col, group_col = args.target, args.time_col, args.group_col
    if args.auto:
        roles = infer_roles(train)
        target = target or roles["target"]
        time_col = time_col or roles["time_col"]
        group_col = group_col or roles["group_col"]
        print(f"  Auto-detected -> target={target!r}, time_col={time_col!r}, "
              f"group_col={group_col!r}")

    report = audit(train, test, target=target, time_col=time_col, group_col=group_col,
                   seq_col=args.seq_col, measure_impact=args.measure_impact)

    if args.auto and test is None:
        for f in build_advisories({"time_col": time_col, "group_col": group_col}):
            report.add(f)

    print(report.render(ascii=args.ascii or None))

    if args.html:
        with open(args.html, "w", encoding="utf-8") as fh:
            fh.write(report.to_html())
        print(f"  HTML report written to {args.html}")

    return 1 if not report.clean else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
