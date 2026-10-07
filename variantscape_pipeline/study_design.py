"""Study design classification (04.3_General_classifier_for_study_design_categorization).

04.3 used the ``general-classifier`` package (Llama-3.3-70B on DeepInfra) with
the prompt below; here the same prompt is sent directly to the LLM client and
the answer is matched to the allowed categories.
"""

from __future__ import annotations

import re
from typing import Sequence

from rapidfuzz import fuzz, process

PROMPT_ID = "study_design_gc"


def build_prompt(title: str, abstract: str, categories: Sequence[str]) -> str:
    text = f"{title} {abstract}"
    return (
        "Prompt: INSTRUCTION: You are a helpful classifier. You are given the abstract of a "
        "scientific, biomedical publication and you have to select the correct of the possible categories. "
        f"The topic of the classification is 'Study Type'. The allowed categories are '{', '.join(categories)}'. "
        f"QUESTION: The abstract to be classified is '{text}'. "
        'ANSWER: The correct category for this abstract is "".\n'
        "Respond only with the category name, exactly as written in the list of allowed categories."
    )


def parse_label(response: str | None, categories: Sequence[str], fallback: str = "undefined") -> str:
    if not isinstance(response, str) or not response.strip():
        return fallback
    answer = response.strip().strip('."\'` ').lower()
    by_lower = {c.lower(): c for c in categories}
    if answer in by_lower:
        return by_lower[answer]
    # Longest category name mentioned in the answer
    for lower, category in sorted(by_lower.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(lower)}\b", answer):
            return category
    match = process.extractOne(answer, list(by_lower), scorer=fuzz.ratio, score_cutoff=80)
    return by_lower[match[0]] if match else fallback
