"""Variant-treatment consensus over per-paper labels (06.02.5_LLM-based_coassociation).

The per-paper labels now come from the association verifier
(``verification.py``), which replaced the 06.02.5 co-association prompt; the
consensus rules are unchanged.
"""

from __future__ import annotations

import pandas as pd

from .verification import DIRECTIONAL

# Labels with a definite meaning vs. the weak "reported, direction unclear" label
VALID_LABELS = set(DIRECTIONAL)
WEAK_LABELS = {"Reported"}


def pair_key(variant: str, treatment: str) -> str:
    """Key used by EvidenceDb: ``f"{variant} + {treatment}".lower()``."""
    return f"{variant} + {treatment}".lower()


def compute_consensus(votes: pd.DataFrame, key: str = "Variant_Treatment_Pair") -> pd.DataFrame:
    """Resolve one label per key from all per-paper votes.

    ``votes`` has columns ``key`` and ``Prediction``. Rules (as in 06.02.5):

    1. hard consensus - all votes agree;
    2. soft consensus - dominant label is a definite label with >= 60% of >= 3 votes;
    3. fallback - only weak labels: "Reported"; a definite label mixed with weak
       labels: the most frequent definite label; otherwise "No consensus".
    """
    columns = [key, "Resolved_Prediction"]
    if votes.empty:
        return pd.DataFrame(columns=columns)

    counts = votes.groupby([key, "Prediction"]).size().reset_index(name="Count")
    totals = votes.groupby(key).size().reset_index(name="Total")
    merged = counts.merge(totals, on=key)
    merged["Is_Consensus"] = merged["Count"] == merged["Total"]
    merged["Proportion"] = merged["Count"] / merged["Total"]

    hard = (merged[merged["Is_Consensus"]]
            .sort_values([key, "Count"], ascending=[True, False])
            .drop_duplicates(key)[[key, "Prediction"]]
            .rename(columns={"Prediction": "Resolved_Prediction"}))

    dominant = (merged.sort_values([key, "Proportion"], ascending=[True, False])
                .drop_duplicates(key).copy())
    dominant["Soft_Consensus"] = (
        (dominant["Proportion"] >= 0.60) & (dominant["Total"] >= 3) & dominant["Prediction"].isin(VALID_LABELS)
    )
    soft = (dominant[dominant["Soft_Consensus"]][[key, "Prediction"]]
            .rename(columns={"Prediction": "Resolved_Prediction"}))
    soft = soft[~soft[key].isin(hard[key])]

    resolved = set(hard[key]) | set(soft[key])
    fallback = []
    for value, group in votes[~votes[key].isin(resolved)].groupby(key):
        label_counts = group["Prediction"].value_counts()
        unique = label_counts.index.tolist()
        present_valid = [label for label in unique if label in VALID_LABELS]
        present_weak = [label for label in unique if label in WEAK_LABELS]
        if set(unique).issubset(WEAK_LABELS):
            fallback.append((value, "Reported"))
        elif len(unique) == 1:
            fallback.append((value, unique[0]))
        elif present_valid and present_weak:
            fallback.append((value, label_counts.loc[present_valid].idxmax()))
        else:
            fallback.append((value, "No consensus"))

    result = pd.concat([hard, soft, pd.DataFrame(fallback, columns=columns)], ignore_index=True)
    return result[columns].sort_values(key).reset_index(drop=True)
