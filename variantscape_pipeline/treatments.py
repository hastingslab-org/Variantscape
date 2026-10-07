"""String matching of CIViC therapies and aliases (04.2_SM_for_therapy_categorization)."""

from __future__ import annotations

import re
from typing import Sequence

# Aliases of <= 4 characters are too unspecific (e.g. "RT" matching RT-qPCR)
ALIAS_MIN_LENGTH = 5
# Therapies that are noise when string matched
REMOVED_THERAPY_NAMES = {"lysine", "inhibitor"}


class TreatmentMatcher:
    def __init__(self, therapies: Sequence[dict]):
        term_to_names: dict[str, set[str]] = {}
        for therapy in therapies:
            name = therapy["name"]
            if not name or name.lower() in REMOVED_THERAPY_NAMES:
                continue
            terms = [name] + [a for a in therapy.get("aliases") or [] if isinstance(a, str) and len(a) >= ALIAS_MIN_LENGTH]
            for term in terms:
                term_to_names.setdefault(term.lower().strip(), set()).add(name)
        term_to_names.pop("", None)
        self._terms = [(term, re.compile(rf"\b{re.escape(term)}\b"), names) for term, names in term_to_names.items()]

    def match(self, title: str, abstract: str) -> set[str]:
        text = f"{title} {abstract}".lower()
        found: set[str] = set()
        for term, pattern, names in self._terms:
            # Cheap substring test before the word-boundary regex
            if term in text and pattern.search(text):
                found |= names
        return found
