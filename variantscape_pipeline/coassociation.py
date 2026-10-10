"""Consensus label per association over per-paper labels.

The per-paper labels come from the association verifier (``verification.py``),
which replaced the 06.02.5 co-association prompt. The consensus resolves one
label per key (a variant-treatment pair, or a verified variant-cancer-treatment
triple); see ``resolve_label`` for the rules.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from .verification import DIRECTIONAL

# Labels with a definite meaning vs. the weak "reported, direction unclear" label
VALID_LABELS = set(DIRECTIONAL)
WEAK_LABELS = {"Reported"}
SENSITIVE_LABELS = {"Sensitive"}
AGAINST_LABELS = {"Resistant", "No response"}   # the treatment does not work for the variant
MIXED = "Mixed"
MAJORITY = 0.60


def pair_key(variant: str, treatment: str) -> str:
    """Key used by EvidenceDb: ``f"{variant} + {treatment}".lower()``."""
    return f"{variant} + {treatment}".lower()


def resolve_label(counts: Counter) -> str:
    """One label from the label counts of an association's papers.

    1. a single label (all papers agree) is the consensus;
    2. only "Reported": Reported;
    3. treatment response labels decide when present: Sensitive if at least 60% of
       the response votes are Sensitive; Resistant / No response (the more frequent)
       if at least 60% are against; otherwise "Mixed" (e.g. papers on response and
       papers on acquired resistance in the same population);
    4. otherwise the most frequent of the remaining definite labels (Prognostic,
       Diagnostic); a tie is "No consensus".
    """
    counts = Counter({k: v for k, v in counts.items() if v > 0})
    if len(counts) == 1:
        return next(iter(counts))
    if set(counts) <= WEAK_LABELS:
        return "Reported"
    sensitive = sum(counts[label] for label in SENSITIVE_LABELS)
    against = sum(counts[label] for label in AGAINST_LABELS)
    if sensitive + against:
        if sensitive >= MAJORITY * (sensitive + against):
            return "Sensitive"
        if against >= MAJORITY * (sensitive + against):
            return max(sorted(AGAINST_LABELS), key=lambda label: counts[label])   # tie -> No response
        return MIXED
    definite = Counter({k: v for k, v in counts.items() if k in VALID_LABELS})
    if not definite:
        return "No consensus"
    (top, n), *rest = definite.most_common()
    return "No consensus" if rest and rest[0][1] == n else top


def compute_consensus(votes: pd.DataFrame, key: str = "Variant_Treatment_Pair") -> pd.DataFrame:
    """One ``Resolved_Prediction`` per ``key`` from the per-paper ``Prediction`` votes
    (``resolve_label``)."""
    columns = [key, "Resolved_Prediction"]
    if votes.empty:
        return pd.DataFrame(columns=columns)
    rows = [(value, resolve_label(Counter(group["Prediction"]))) for value, group in votes.groupby(key)]
    return pd.DataFrame(rows, columns=columns).sort_values(key).reset_index(drop=True)
