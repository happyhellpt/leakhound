# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joel Gomes
"""Homology leakage: test sequences that are *similar* (not identical) to training ones.

In biological ML (proteins, DNA/RNA), near-homologous sequences across the split
inflate results dramatically — and generic tools miss it because the rows aren't
duplicates. This estimates k-mer Jaccard similarity with MinHash (pure Python, no
MMseqs2/CD-HIT needed) and flags test sequences close to any training sequence.
"""
from __future__ import annotations

import hashlib
import random
from collections import defaultdict

import pandas as pd

from .report import Finding

_PRIME = (1 << 61) - 1


def _kmers(seq: str, k: int) -> set[str]:
    seq = str(seq).upper().strip()
    if len(seq) <= k:
        return {seq} if seq else set()
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def _base_hash(kmer: str) -> int:
    return int.from_bytes(hashlib.blake2b(kmer.encode(), digest_size=8).digest(), "big")


def _perms(num_perm: int, seed: int = 0):
    rng = random.Random(seed)
    return [(rng.randrange(1, _PRIME), rng.randrange(0, _PRIME)) for _ in range(num_perm)]


def _signature(kmers: set[str], perms) -> list[int]:
    if not kmers:
        return [0] * len(perms)
    bases = [_base_hash(km) for km in kmers]
    return [min((a * h + b) % _PRIME for h in bases) for a, b in perms]


_NUCLEOTIDES = set("ACGTUN")


def _auto_k(seqs, k):
    """Pick a k-mer size: larger for nucleotides (4-letter alphabet), smaller for protein."""
    if k is not None:
        return k, ("nucleotide" if False else "custom")
    sample = "".join(str(x).upper() for x in list(seqs)[:50])
    if sample and sum(c in _NUCLEOTIDES for c in sample) / len(sample) > 0.9:
        return 6, "nucleotide"
    return 3, "protein"


def check_homology(train: pd.DataFrame, test: pd.DataFrame, seq_col: str,
                   k: int | None = None, threshold: float = 0.7, num_perm: int = 64) -> Finding:
    if seq_col not in train.columns or seq_col not in test.columns:
        return Finding("homology", "low", f"sequence column '{seq_col}' not in both sets")

    k, alphabet = _auto_k(train[seq_col], k)
    perms = _perms(num_perm)
    tr_sig = [_signature(_kmers(s, k), perms) for s in train[seq_col].astype(str)]

    index = defaultdict(list)  # (position, minhash value) -> train indices
    for ti, sig in enumerate(tr_sig):
        for pos, val in enumerate(sig):
            index[(pos, val)].append(ti)

    n_hits, examples = 0, []
    for s in test[seq_col].astype(str):
        sig = _signature(_kmers(s, k), perms)
        cand = set()
        for pos, val in enumerate(sig):
            cand.update(index.get((pos, val), ()))
        best = 0.0
        for ti in cand:
            sim = sum(1 for a, b in zip(sig, tr_sig[ti]) if a == b) / num_perm
            if sim > best:
                best = sim
        if best >= threshold:
            n_hits += 1
            if len(examples) < 5:
                examples.append(round(best, 3))

    if n_hits == 0:
        return Finding("homology", "ok",
                       f"no test sequences are highly similar to training sequences (k={k})")

    frac = n_hits / len(test)
    sev = "high" if frac > 0.05 else "medium"
    return Finding(
        "homology", sev,
        f"{n_hits} test sequences ({frac:.1%}) are highly similar to training sequences "
        f"(estimated k-mer Jaccard ≥ {threshold}) — homology leakage inflates biological ML",
        {"seq_col": seq_col, "alphabet": alphabet, "k": k, "threshold": threshold,
         "n_similar": n_hits, "example_similarities": examples},
        fix="Split by sequence similarity so near-homologues stay on one side "
            "(safesplit(df, seq_col=...), or CD-HIT / MMseqs2).",
    )
