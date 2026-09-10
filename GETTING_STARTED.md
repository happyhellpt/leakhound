# Getting Started with LeakHound

A complete, copy-paste guide — from zero to your first leakage report — on
**Windows, macOS and Linux**. No prior experience assumed. If you can open a
terminal and paste a line, you can run this.

> **What it does, in one sentence:** you give LeakHound your training and test
> files, and it tells you whether your model's score is real — and how much of it
> is fake — by finding the ways the test set was contaminated by the training data.

---

## Table of contents

1. [Do you have Python?](#1-do-you-have-python)
2. [Install LeakHound](#2-install-leakhound)
3. [Run your first check (sample data included)](#3-run-your-first-check)
4. [Run it on your own data](#4-run-it-on-your-own-data)
5. [Read the report](#5-read-the-report)
6. [Fix each kind of leak](#6-fix-each-kind-of-leak)
7. [Measure the damage (`--measure-impact`)](#7-measure-the-damage)
8. [Auto mode (`--auto`)](#8-auto-mode)
9. [Shareable HTML report (`--html`)](#9-shareable-html-report)
10. [Use it from Python](#10-use-it-from-python)
11. [Use it in CI (fail a bad merge)](#11-use-it-in-ci)
12. [Troubleshooting](#12-troubleshooting)
13. [FAQ](#13-faq)

---

## 1. Do you have Python?

LeakHound needs **Python 3.9 or newer**. Check it first.

**Windows** (Command Prompt or PowerShell):
```bat
python --version
```
**macOS / Linux** (Terminal):
```bash
python3 --version
```

If you see something like `Python 3.11.5`, you're set — skip to step 2.
If not, install Python:

- **Windows:** download from <https://www.python.org/downloads/> and, on the
  first installer screen, **tick "Add python.exe to PATH"**. (Or run
  `winget install Python.Python.3.12`.)
- **macOS:** `brew install python`, or download from python.org.
- **Linux (Debian/Ubuntu/Pop!_OS):** `sudo apt install python3 python3-pip`

---

## 2. Install LeakHound

> While LeakHound is pre-release, install it from source. Once it's published,
> `pip install leakhound` will be all you need.

```bash
git clone https://github.com/happyhellpt/leakhound.git
cd leakhound
pip install -e '.[impact]'
```

`[impact]` also installs scikit-learn, which powers the `--measure-impact`
feature (step 7). Leave it out (`pip install -e .`) if you only want the
detectors.

On Windows, if `pip` isn't found, use `py -m pip install -e ".[impact]"`.

Check it works:
```bash
leakhound --version
```

> If `leakhound` is "not found" after install, you can always run it as
> `python -m leakhound.cli ...` — see [Troubleshooting](#12-troubleshooting).

---

## 3. Run your first check

The repo ships with example data that already contains planted leaks. From inside
the `leakhound` folder:

```bash
leakhound --train examples/train_sample.csv --test examples/test_sample.csv --target label --time-col date --group-col patient_id
```

You'll see (trimmed):

```
  LeakHound report
  ────────────────────────────────────────────────
  ✗ [high  ] duplicates: 12 test rows (13.3%) also appear in training — the model has seen them
        → fix: Remove duplicates before splitting (df.drop_duplicates()) ...
  ✗ [high  ] target_encoding: feature 'leaky_feature' correlates 1.000 with the target — it likely leaks it
        → fix: Drop this feature, or replace it with information available at prediction time.
  ✗ [high  ] temporal: 168 training rows (100.0%) are dated at/after the earliest test row — the model trains on the future
        → fix: Split chronologically: sort by the time column and put later rows in test.
  ✗ [high  ] group_split: 49 values of 'patient_id' (89.1% of test groups) are in both sets ...
        → fix: Use a group-aware splitter (GroupShuffleSplit / GroupKFold) ...
  ! [medium] near_duplicates: 5 test rows (5.6%) are near-identical to training rows ...
  ────────────────────────────────────────────────
  5 likely leaks found. Your reported metric is probably optimistic.
```

That's it — you just ran LeakHound. 🎉 Every finding tells you what it found **and
how to fix it**.

---

## 4. Run it on your own data

You need **two CSV files**: your training set and your test (or validation) set,
each with a header row. Then:

```bash
leakhound --train path/to/train.csv --test path/to/test.csv --target label
```

Add the optional flags when your data has them — each switches on more:

| Flag | Give it | Turns on |
|---|---|---|
| `--train` (required) | your training CSV | — |
| `--test` | your test/validation CSV | duplicate + near-duplicate checks |
| `--target` | the column you're predicting | target-encoding check |
| `--time-col` | a date/timestamp column | look-ahead (temporal) check |
| `--group-col` | an id column (patient, user, device) | group-split check |
| `--measure-impact` | *(no value)* | quantify how much each leak inflates the score |
| `--auto` | *(no value)* | auto-detect the columns above / advise on one file |
| `--html PATH` | a file path | also write a shareable HTML report |
| `--ascii` | *(no value)* | plain-text output for old terminals |

Only the checks whose inputs you provide will run.

---

## 5. Read the report

Each line is one finding, with a severity, the evidence, and a fix.

- **`✗ high`** — almost certainly real leakage. Fix before trusting any score.
- **`! medium`** — likely leakage; worth investigating.
- **`· low`** — informational (e.g. a check couldn't run).
- **`✓ ok`** — that specific check found nothing.

| Check | What it means |
|---|---|
| **duplicates** | Exact same rows in both train and test — tested on memorised data. |
| **near_duplicates** | Rows identical after tiny rounding — copies you didn't notice. |
| **temporal** | Training rows dated at/after your test rows — the model "saw the future". |
| **target_encoding** | A feature (often an ID) basically *is* the answer. |
| **group_split** | The same subject is in both sets — it recognises the subject, not the pattern. |
| **impact** | *How much* score each leak is inflating (see step 7). |

> ⚠️ A clean report is **not proof** your split is perfect. It means these
> common, high-impact leaks aren't present.

---

## 6. Fix each kind of leak

**Duplicates / near-duplicates across train/test**
```python
df = df.drop_duplicates()   # then split once, cleanly
```
Also check your pipeline for copied or augmented rows.

**Temporal (look-ahead) leakage** — split by time, not randomly:
```python
df = df.sort_values("date")
cut = int(len(df) * 0.8)
train, test = df.iloc[:cut], df.iloc[cut:]
```

**Target-encoding leakage** — drop the leaking feature (often an ID or a value
recorded *after* the outcome):
```python
train = train.drop(columns=["leaky_feature"])
test = test.drop(columns=["leaky_feature"])
```

**Group split leakage** — keep every group on one side:
```python
from sklearn.model_selection import GroupShuffleSplit
splitter = GroupShuffleSplit(test_size=0.2, random_state=0)
train_idx, test_idx = next(splitter.split(X, y, groups=df["patient_id"]))
```

---

## 7. Measure the damage

Detecting a leak is one thing; seeing what it's worth is another. Add
`--measure-impact` and LeakHound fits a quick baseline model, then reports your
honest score next to the inflated one:

```bash
leakhound --train examples/train_sample.csv --test examples/test_sample.csv --target label --measure-impact
```

```
  ✗ [high  ] impact: removing leaking feature(s) ['leaky_feature'] drops AUC from 1.000 to 0.669 — +0.331 of fake performance
        AUC_with_leak: 1.0
        AUC_without_leak: 0.669
        inflation: 0.331
        → fix: Drop the leaking feature(s) and re-evaluate honestly.
```

Your `1.000` was really `0.669`. It works for **binary** (0/1) and **continuous**
targets (AUC and R² respectively), using numeric features. Needs scikit-learn
(`pip install 'leakhound[impact]'`). Treat the number as a quick baseline
estimate, not gospel.

---

## 8. Auto mode

Not sure which columns are the target, the timestamp or the group id? Add
`--auto` and LeakHound guesses them:

```bash
leakhound --train examples/train_sample.csv --auto
```

```
  Auto-detected -> target='label', time_col='date', group_col='patient_id'
  ...
  ! [medium] temporal: column 'date' looks like a time column — a random split would let the model train on the future
  ! [medium] group_split: column 'patient_id' looks like a group id — a random split would put the same group on both sides
```

Run on a **single file** (no `--test`), it becomes a *pre-split linter*: it warns
you which columns will leak **before** you split. Run it with a test set too and
it just fills in any columns you didn't name.

---

## 9. Shareable HTML report

Add `--html` to also write a self-contained report you can send to a colleague or
attach to a pull request (no internet or extra files needed to open it):

```bash
leakhound --train train.csv --test test.csv --target label --measure-impact --html leakhound_report.html
```

Open `leakhound_report.html` in any browser. It adapts to light and dark mode.

---

## 10. Use it from Python

```python
import pandas as pd
from leakhound import audit

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

report = audit(train, test, target="label", time_col="date",
               group_col="patient_id", measure_impact=True)
print(report.render())

# Save the shareable report
with open("report.html", "w", encoding="utf-8") as f:
    f.write(report.to_html())

if not report.clean:
    raise SystemExit("Leakage detected — fix the split before training.")
```

`report.leaks` gives the findings as objects (`.check`, `.severity`, `.message`,
`.evidence`, `.fix`).

---

## 11. Use it in CI

LeakHound exits non-zero when it finds leakage, so it can **fail a pull request**
that would ship a contaminated split (`.github/workflows/leakhound.yml`):

```yaml
name: data-leakage-check
on: [push, pull_request]
jobs:
  leakhound:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install 'leakhound[impact]'
      - run: leakhound --train data/train.csv --test data/test.csv --target label --measure-impact
```

If a leak is found, the step fails and the merge is blocked.

---

## 12. Troubleshooting

**`leakhound: command not found`** — run it as a module instead:
```bash
python -m leakhound.cli --train examples/train_sample.csv --test examples/test_sample.csv --target label
```

**`install scikit-learn to measure leak impact`** — `--measure-impact` needs it:
```bash
pip install 'leakhound[impact]'
```

**`FileNotFoundError`** — wrong path. Use the full path, or `cd` into the folder
first. On Windows, paths use backslashes: `data\train.csv`.

**Strange symbols / `UnicodeEncodeError` on Windows** — old `cmd.exe` can't print
the report symbols. Add `--ascii` (LeakHound auto-detects this in most cases):
```bat
leakhound --train train.csv --test test.csv --target label --ascii
```

**`UnicodeDecodeError` reading the CSV** — re-save your CSV as UTF-8.

**`target 'label' not found`** — the `--target` name must match a column header
exactly (case-sensitive). Or use `--auto` to let LeakHound guess it.

---

## 13. FAQ

**Does LeakHound change my data?** No. It only reads your files and prints a
report. It never writes to them (except the `--html` file you ask for).

**Does a clean report guarantee my model is fine?** No. It rules out the most
common, highest-impact leaks — not every possible one.

**Which operating systems are supported?** Windows, macOS and Linux, identically
(bar the `python` vs `python3` detail in step 1).

**What file formats does it read?** CSV, for now. Data is loaded into memory with
pandas, so very large files need enough RAM.

**Can I run just one check?** Yes — only supply the inputs for the checks you
want.
