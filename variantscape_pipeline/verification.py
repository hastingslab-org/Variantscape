"""LLM verification of mined associations against the abstract (replaces 06.02.5).

For every included paper, the candidate associations built from its mined
entities (variant x cancer x treatment triples, and variant x cancer pairs) are
sent to the LLM with the title and abstract. For each candidate it decides
whether the abstract *actually reports* the association, and with which
relation, and must give a verbatim supporting quote. A verdict only counts if
the quote is found in the title/abstract.

Verified associations become the literature edges of the graph; associations
that were only co-mentioned are kept as a separate, lower tier
(``cooccurrence_weight``).
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

from rapidfuzz import fuzz

PROMPT_ID = "verify_v1"

NOT_REPORTED = "Not reported"
RELATIONS = ("Sensitive", "Resistant", "No response", "Diagnostic", "Prognostic", "Reported", NOT_REPORTED)
# Labels that give a treatment direction (used for the variant-treatment consensus)
DIRECTIONAL = {"Sensitive", "Resistant", "No response", "Diagnostic", "Prognostic"}

_RELATION_SYNONYMS = {
    "sensitive": "Sensitive", "sensitivity": "Sensitive", "response": "Sensitive", "benefit": "Sensitive",
    "resistant": "Resistant", "resistance": "Resistant",
    "no response": "No response", "noresponse": "No response", "no benefit": "No response",
    "not effective": "No response", "ineffective": "No response",
    "diagnostic": "Diagnostic", "prognostic": "Prognostic",
    "reported": "Reported", "associated": "Reported", "association": "Reported",
    "not reported": NOT_REPORTED, "notreported": NOT_REPORTED, "not supported": NOT_REPORTED,
    "unrelated": NOT_REPORTED, "none": NOT_REPORTED, "no": NOT_REPORTED,
}

MAX_PAIR_CANDIDATES = 20
MAX_CANDIDATES = 80


@dataclass(frozen=True)
class Candidate:
    variant: str      # graph node, e.g. v600e_BRAF
    cancer: str       # graph node
    treatment: str    # graph node, "" for a variant-cancer pair

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.variant, self.cancer, self.treatment)


@dataclass(frozen=True)
class Verdict:
    candidate: Candidate
    relation: str
    quote: str
    quote_ok: bool

    @property
    def verified(self) -> bool:
        return self.relation != NOT_REPORTED and self.quote_ok


def make_candidates(variants: Iterable[str], cancers: Iterable[str], treatments: Iterable[str]
                    ) -> tuple[list[Candidate], bool]:
    """Variant-cancer pairs first, then variant-cancer-treatment triples, capped.

    Returns the candidates and whether the cap truncated them.
    """
    variants, cancers, treatments = sorted(variants), sorted(cancers), sorted(treatments)
    all_pairs = [Candidate(v, c, "") for v, c in product(variants, cancers)]
    pairs = all_pairs[:MAX_PAIR_CANDIDATES]
    triples = [Candidate(v, c, t) for v, c, t in product(variants, cancers, treatments)]
    room = MAX_CANDIDATES - len(pairs)
    return pairs + triples[:room], len(all_pairs) > len(pairs) or len(triples) > room


def display_variant(node: str) -> str:
    """v600e_BRAF -> BRAF V600E"""
    variant, _, gene = node.rpartition("_")
    return f"{gene} {variant.upper()}" if variant else node


def build_prompt(title: str, abstract: str, candidates: Sequence[Candidate]) -> str:
    lines = []
    for i, c in enumerate(candidates, start=1):
        text = f"{i}. variant: {display_variant(c.variant)} | cancer: {c.cancer}"
        if c.treatment:
            text += f" | treatment: {c.treatment}"
        lines.append(text)
    return (
        "You check associations that were automatically mined from a biomedical publication. "
        "For each numbered candidate, decide whether the title or abstract itself REPORTS that association "
        "(a finding of the study or a statement made by the authors), not merely mentions the entities.\n\n"
        f"Title: {title}\nAbstract: {abstract}\n\n"
        "Candidates:\n" + "\n".join(lines) + "\n\n"
        "Relations:\n"
        "- Sensitive: the variant is reported to predict response or benefit to the treatment in this cancer\n"
        "- Resistant: the variant is reported to cause or predict resistance to the treatment in this cancer\n"
        "- No response: the treatment is reported as not effective for this variant in this cancer\n"
        "- Diagnostic: the variant is reported as diagnostic for this cancer\n"
        "- Prognostic: the variant is reported as prognostic in this cancer\n"
        "- Reported: the association is reported, but none of the above applies\n"
        "- Not reported: the association is not reported (only co-mentioned, mentioned in background, "
        "a different variant/cancer/treatment, or an entity that is wrong for this text)\n\n"
        "For candidates without a treatment, judge the variant-cancer association only.\n"
        "Answer with only a JSON array, one object per candidate:\n"
        '[{"id": 1, "relation": "<relation>", "quote": "<shortest verbatim excerpt from the title or abstract '
        'that supports it; empty for Not reported>"}]'
    )


def max_tokens_for(candidates: Sequence[Candidate]) -> int:
    return min(8192, 300 + 90 * len(candidates))


# --------------------------------------------------------------------------- #
# Parsing and quote check
# --------------------------------------------------------------------------- #
def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def quote_in_text(quote: str, text: str) -> bool:
    # An ellipsis separates excerpts; mark it with a double space to split on below
    quote = _normalize_text(quote).strip(" .\"'`…").replace("...", "  ").replace("…", "  ")
    if len(quote) < 10:
        return False
    norm = _normalize_text(text)
    if quote in norm:
        return True
    # Tolerate small differences (punctuation, a dropped word); quotes stitched from
    # several sentences are checked part by part
    parts = [p.strip() for p in re.split(r"\s{2,}", quote) if len(p.strip()) >= 10] or [quote]
    return all(p in norm or fuzz.partial_ratio(p, norm) >= 92 for p in parts)


def normalize_relation(value) -> str | None:
    if not isinstance(value, str):
        return None
    key = re.sub(r"[_\-]+", " ", value.strip().lower()).strip(" .")
    if key in _RELATION_SYNONYMS:
        return _RELATION_SYNONYMS[key]
    for relation in RELATIONS:
        if key == relation.lower():
            return relation
    return None


def _json_array(response: str) -> list:
    text = re.sub(r"<think>.*?</think>", "", response or "", flags=re.S)
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            pass
    # Fall back to individual objects (e.g. a truncated array)
    items = []
    for match in re.finditer(r"\{[^{}]*\}", text):
        try:
            items.append(json.loads(match.group(0)))
        except json.JSONDecodeError:
            continue
    return items


def parse_response(response: str | None, candidates: Sequence[Candidate], title: str, abstract: str
                   ) -> list[Verdict]:
    """Verdicts for the candidates the model answered (by ``id``); others stay unverified."""
    if not isinstance(response, str):
        return []
    text = f"{title} {abstract}"
    verdicts: dict[int, Verdict] = {}
    for item in _json_array(response):
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("id")) - 1
        except (TypeError, ValueError):
            continue
        relation = normalize_relation(item.get("relation"))
        if not (0 <= idx < len(candidates)) or relation is None or idx in verdicts:
            continue
        quote = str(item.get("quote") or "")
        ok = relation == NOT_REPORTED or quote_in_text(quote, text)
        verdicts[idx] = Verdict(candidates[idx], relation, quote, ok)
    return [verdicts[i] for i in sorted(verdicts)]


def candidates_to_json(candidates: Sequence[Candidate]) -> str:
    return json.dumps([list(c.key) for c in candidates])


def candidates_from_json(value: str | None) -> list[Candidate]:
    return [Candidate(*item) for item in json.loads(value or "[]")]
