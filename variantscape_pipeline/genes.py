"""Gene sets and gene mention detection, the first relevance filter (03_gene_extraction).

Gene sets (``VARIANTSCAPE_GENE_SET`` / ``--gene-set``):

- ``oncology`` (default): all oncology-relevant genes, i.e. the OncoKB Cancer
  Gene List merged with all CIViC genes. Variants of any gene reported by the
  LLM are kept.
- ``civic``: the CIViC genes only (the notebooks' gene list).
- a path to a panel file (one gene symbol per line, or a CSV whose first column
  holds the symbols, e.g. ``notebooks/05_LLM_variant_extraction/oncomine_ngs_panel.csv``).

For ``civic`` and panels, papers must mention a gene of the set *and* only
variants in genes of the set become graph nodes.

Detection always covers the oncology set (plus any panel genes outside it) and
the active gene set is applied when the graph is built, so switching gene sets
needs no reprocessing.

Two methods are available:

- ``biobert`` (default, as in 03.1): BioBERT genetic NER on title and abstract
  separately, normalized to the CIViC gene list (exact match, token match, or
  fuzzy match > 85).
- ``string`` (as in 03.0): case-sensitive whole-word matching of CIViC gene
  symbols; much faster, useful for testing or CPU-only machines.

Note: 03.1 fetched MyGene.info synonyms, but the expansion never took effect
(``requests`` was not imported and only the dict keys were ever matched), so
only the gene symbols themselves are used here.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

log = logging.getLogger(__name__)

ONCOLOGY_GENE_SET = "oncology"
CIVIC_GENE_SET = "civic"


@dataclass(frozen=True)
class GeneSet:
    name: str
    genes: frozenset[str]
    restrict_variants: bool  # only keep variants in genes of the set

    def contains(self, gene: str) -> bool:
        return gene.upper() in self.genes


def read_panel(path: Path) -> set[str]:
    """Gene symbols from a panel file: one per line, or the first column of a CSV."""
    genes = set()
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if not row or not row[0].strip() or row[0].strip().startswith("#"):
                continue
            symbol = row[0].strip().upper()
            if re.fullmatch(r"[A-Z0-9][A-Z0-9\-\.]*", symbol) and symbol not in {"GENE", "SYMBOL", "HUGOSYMBOL"}:
                genes.add(symbol)
    if not genes:
        raise ValueError(f"No gene symbols found in panel file {path}")
    return genes


def oncology_genes(reference) -> set[str]:
    return {g.upper() for g in reference.genes} | {g.upper() for g in reference.oncokb_genes}


def resolve_gene_set(spec: str, reference) -> GeneSet:
    if spec == ONCOLOGY_GENE_SET:
        return GeneSet(ONCOLOGY_GENE_SET, frozenset(oncology_genes(reference)), restrict_variants=False)
    if spec == CIVIC_GENE_SET:
        return GeneSet(CIVIC_GENE_SET, frozenset(g.upper() for g in reference.genes), restrict_variants=True)
    path = Path(spec)
    return GeneSet(f"panel:{path.name}", frozenset(read_panel(path)), restrict_variants=True)


class StringGeneMatcher:
    def __init__(self, gene_symbols: Sequence[str]):
        symbols = sorted(set(gene_symbols), key=len, reverse=True)
        self._pattern = re.compile(r"\b(?:" + "|".join(re.escape(s) for s in symbols) + r")\b")

    def extract(self, titles: Sequence[str], abstracts: Sequence[str]) -> list[set[str]]:
        return [set(self._pattern.findall(f"{t} {a}")) for t, a in zip(titles, abstracts)]


class BioBertGeneMatcher:
    MAX_TOKENS = 510  # 512 minus [CLS]/[SEP]
    STRIDE = 256

    def __init__(self, gene_symbols: Sequence[str], model_name: str, device: int = -1, batch_size: int = 16):
        from transformers import pipeline

        self.gene_keys = {g.upper() for g in gene_symbols}
        self._gene_keys_list = sorted(self.gene_keys)
        self.batch_size = batch_size
        log.info("Loading BioBERT model %s ...", model_name)
        self.ner = pipeline("ner", model=model_name, tokenizer=model_name, device=device)
        self.tokenizer = self.ner.tokenizer

    def _chunks(self, text: str) -> list[str]:
        """Split text into overlapping token windows.

        03.1 dropped the final, shorter window of long texts; it is kept here.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= self.MAX_TOKENS:
            return [text]
        chunks = []
        for start in range(0, len(tokens), self.STRIDE):
            window = tokens[start:start + self.MAX_TOKENS]
            chunks.append(self.tokenizer.decode(window, skip_special_tokens=True))
            if start + self.MAX_TOKENS >= len(tokens):
                break
        return chunks

    @staticmethod
    def _terms_from_entities(entities: list[dict]) -> set[str]:
        terms: set[str] = set()
        current: list[str] = []
        for ent in entities:
            word = ent["word"].replace("##", "")
            if ent["entity"].startswith("B-"):
                if current:
                    terms.add("".join(current))
                current = [word]
            elif ent["entity"].startswith("I-"):
                current.append(word)
        if current:
            terms.add("".join(current))
        return terms

    def normalize(self, terms: set[str]) -> set[str]:
        from rapidfuzz import fuzz, process

        genes: set[str] = set()
        for term in terms:
            upper = term.upper()
            if upper in self.gene_keys:
                genes.add(upper)
                continue
            words = re.sub(r"[\[\]\(\),-]", " ", upper).split()
            matched = [w for w in words if w in self.gene_keys]
            genes.update(matched)
            if not matched:
                match = process.extractOne(upper, self._gene_keys_list, scorer=fuzz.ratio, score_cutoff=85.01)
                if match:
                    genes.add(match[0])
        return genes

    def extract(self, titles: Sequence[str], abstracts: Sequence[str]) -> list[set[str]]:
        if not titles:
            return []
        texts = [str(t or "") for t in titles] + [str(a or "") for a in abstracts]
        owners: list[int] = []
        chunks: list[str] = []
        for i, text in enumerate(texts):
            if not text.strip():
                continue
            for chunk in self._chunks(text):
                owners.append(i % len(titles))
                chunks.append(chunk)
        results: list[set[str]] = [set() for _ in titles]
        if not chunks:
            return results
        outputs = self.ner(chunks, batch_size=self.batch_size)
        raw_terms: list[set[str]] = [set() for _ in titles]
        for owner, entities in zip(owners, outputs):
            raw_terms[owner] |= self._terms_from_entities(entities)
        return [self.normalize(terms) for terms in raw_terms]


def make_gene_matcher(method: str, gene_symbols: Sequence[str], model_name: str, device: int):
    if method == "string":
        return StringGeneMatcher(gene_symbols)
    if method == "biobert":
        return BioBertGeneMatcher(gene_symbols, model_name, device)
    raise ValueError(f"Unknown gene filter method: {method!r} (use 'biobert' or 'string')")
