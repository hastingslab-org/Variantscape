"""Cancer type extraction and normalization.

- Extraction (04.1 section 3): two SciSpaCy NER models, then term cleaning.
  Cleaned raw terms are stored per paper.
- Mapping (04.1 section 4): terms -> CIViC disease names via names and synonyms.
- Harmonization: CIViC diseases -> OncoTree tumour types, the cancer nodes of the
  network (``oncotree.py``; replaces the notebooks' keyword mapping, which lumped
  e.g. all lung cancers together and produced site-less nodes).

Mapping and harmonization depend on reference data only and are recomputed on
every build from the stored raw terms.
"""

from __future__ import annotations

import logging
from collections import defaultdict
import re
import unicodedata
from typing import Iterable, Sequence

from .oncotree import OncoTreeMapper, do_class, load_do_terms
from .reference import Reference

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
EXCLUDE_TERMS = {"anticancer", "anti cancer", "anti-cancer", "anti-tumor",
                 "antitumor", "anti tumor", "cancerous", "non-cancerous", "precancerous", "cancer-related"}

PREFIXES_TO_REMOVE = [
    "age", "aggressive", "advance", "advanced", "alk positive", "alk+", "ampullary", "anti", "antitumor",
    "brca associated", "brca1 associated", "brca2 associated", "brca1", "brca2", "brca 1", "brca 2",
    "brca-mutated", "brca-positive", "braf", "brca1-mutated", "brca mutant", "brca1 mutant", "brca2 mutant",
    "brca2 altered", "brca1/2", "brca2-mutated", "brca deficient", "brca linked", "brca negative",
    "brca positive", "cell line", "cell lines", "iii", "iiii", "iiiii", "iiiv", "iiv",
    "chemoresistant", "circulating", "disease", "diseases", "cisplatin", "dna alterations",
    "double negative", "early stage", "germline", "human", "kras mutant", "kras", "hypoxic",
    "intercellular", "late stage", "line", "lines", "mcf", "mcf 7", "mediastinal", "membrane",
    "methylated", "mice", "mouse", "murine", "mutation", "n myc", "organoids", "pain", "parp",
    "patient", "patients", "platinum-sensitive", "predisposition", "sample", "samples", "senescent",
    "silenced", "somatic", "specific", "specimen", "specimens", "stage", "tissue", "tissues", "tnbc",
    "tumor dna", "tp53", "p53", "tumor specimen", "tumorigenic", "xenograft", "xenografts",
    "brcawt", "dna", "biopsis", "abstract", "therapy related", "moderate",
    "advance stage", "advanced stage", "chemo", "chemotherapy", "ilc",
]
# Escape the prefixes (04.1 used them unescaped, so "alk+" and "brca1/2" were regexes)
_PREFIX_RES = [re.compile(rf"\b{re.escape(p)}\b\s*") for p in PREFIXES_TO_REMOVE]

WORD_REPLACEMENTS = {
    "cancers": "cancer", "tumour": "tumor", "tumours": "tumor", "tumors": "tumor",
    "carcinomas": "carcinoma", "gliomas": "glioma", "adenocarcinomas": "adenocarcinoma",
}


def _clean_entity(term: str) -> str:
    term = unicodedata.normalize("NFKC", term.lower().strip())
    return re.sub(r"[-‐–—]", " ", term)


def clean_cancer_terms(terms: Iterable[str]) -> list[str]:
    cleaned = []
    for term in terms:
        term = term.strip().lower()
        term = re.sub(r"^[\+\.,\d\(\)\-\s]+", "", term)
        for pattern in _PREFIX_RES:
            term = pattern.sub("", term)
        for old, new in WORD_REPLACEMENTS.items():
            term = term.replace(old, new)
        term = " ".join(dict.fromkeys(term.split()))
        if term.strip():
            cleaned.append(term.strip())
    return sorted(set(cleaned))


class CancerExtractor:
    def __init__(self, batch_size: int = 64):
        import spacy

        log.info("Loading SciSpaCy models ...")
        self.nlp_cancer = spacy.load("en_ner_bionlp13cg_md")   # CANCER entities
        self.nlp_disease = spacy.load("en_ner_bc5cdr_md")      # DISEASE entities
        self.batch_size = batch_size

    def extract(self, texts: Sequence[str]) -> list[list[str]]:
        texts = [str(t) for t in texts]
        found: list[set[str]] = [set() for _ in texts]
        for i, doc in enumerate(self.nlp_cancer.pipe(texts, batch_size=self.batch_size)):
            found[i].update(_clean_entity(e.text) for e in doc.ents if e.label_ == "CANCER")
        for i, doc in enumerate(self.nlp_disease.pipe(texts, batch_size=self.batch_size)):
            for e in doc.ents:
                if e.label_ == "DISEASE":
                    term = _clean_entity(e.text)
                    if "cancer" in term or "tumor" in term:
                        found[i].add(term)
        return [clean_cancer_terms(t for t in terms if t not in EXCLUDE_TERMS) for terms in found]


# --------------------------------------------------------------------------- #
# Mapping to CIViC diseases (04.1 section 4)
# --------------------------------------------------------------------------- #
GENERIC_DISEASE_NAMES = {"cancer", "carcinoma", "tumor", "tumour", "solid tumors, advanced", "solid tumor"}

MANUAL_DISEASES = [
    {
        "name": "metastatic castration-resistant prostate cancer",
        "synonyms": ["mCRPC", "advanced prostate cancer", "CRPC", "castration-resistant PC",
                     "advanced-stage prostate cancer", "androgen-independent prostate cancer",
                     "androgen independent prostate cancer", "metastatic castrate resistant prostate cancer",
                     "metastatic castrate,resistant prostate cancer", "hormone-refractory prostate cancer",
                     "bone metastatic castration resistant prostate cancer",
                     "bone metastatic castration-resistant prostate cancer",
                     "metastatic prostate cancer castration resistant", "bone metastatic crpc",
                     "metastatic prostate cancer castration-resistant",
                     "metastatic castration-resistance prostate cancer",
                     "metastatic castrate-resistant prostate cancer",
                     "brca1 mutated metastatic castration resistant prostate cancer"],
    },
    {
        "name": "metastatic hormone-sensitive prostate cancer",
        "synonyms": ["mHSPC", "castrationsensitive prostate cancer", "hormone-sensitive metastatic prostate cancer",
                     "HSPC", "hormone sensitive prostate cancer", "androgen-dependent prostate cancer",
                     "androgen dependent prostate cancer", "metastatic castration-sensitive prostate cancer",
                     "androgen-dependent metastatic prostate cancer", "hormone-naïve prostate cancer"],
    },
]


def normalize_cancer_term(term: str) -> str:
    term = unicodedata.normalize("NFKC", term.lower().strip())
    term = re.sub(r"\(.*?\)", "", term).strip()
    term = re.sub(r"\b(cancers|carcinoma|carcinomas|tumor|tumors)\b", "cancer", term)
    term = re.sub(r"[\.\d/]+$", "", term)
    return re.sub(r"\s+", " ", term).strip()


def _clean_synonym(name: str) -> str:
    """04.1 clean_individual_cancer_name: drop hyphens and parenthesized text."""
    name = re.sub(r"[-]+", " ", name)
    name = re.sub(r"\(.*?\)", "", name)
    return " ".join(name.split())


def _fix_disease_name(name: str) -> str:
    # CIViC data issue handled in 04.1
    return "Adenoid Cystic Carcinoma" if name.lower() == "doid:0080202" else name


def disease_synonyms(disease: dict) -> list[str]:
    raw = set(disease.get("aliases") or []) | set(disease.get("mondo_synonyms") or [])
    seen, out = set(), []
    for name in sorted(raw):
        if name.lower() in {"familial", "n/a"}:
            continue
        cleaned = _clean_synonym(name)
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            out.append(cleaned)
    return out


# --------------------------------------------------------------------------- #
# Harmonization: CIViC disease -> OncoTree cancer node (oncotree.py)
# --------------------------------------------------------------------------- #
_METHOD_RANK = {"manual": 0, "xref": 1, "name": 2, "ancestor": 3, "fallback": 4, "organ": 5, "unmapped": 6}


class CancerMapper:
    """Maps cleaned raw cancer terms to OncoTree cancer node names."""

    def __init__(self, reference: Reference, use_synonyms: bool = True):
        rows = [
            {"name": _fix_disease_name(d["name"]), "synonyms": disease_synonyms(d),
             "doid": f"DOID:{d['doid']}" if d.get("doid") else None}
            for d in reference.diseases
            if d["name"] and _fix_disease_name(d["name"]).lower() not in GENERIC_DISEASE_NAMES
        ] + [{**m, "doid": None} for m in MANUAL_DISEASES]
        # 04.1 round-tripped the synonym lists through a CSV and then ignored them
        # (they were no longer Python lists), so only disease names were matched.
        # Synonyms are used here by default; names take precedence over synonyms.
        mapping: dict[str, str] = {}
        if use_synonyms:
            for row in rows:
                standard = normalize_cancer_term(row["name"])
                for syn in row["synonyms"]:
                    mapping[normalize_cancer_term(syn)] = standard
        for row in rows:
            standard = normalize_cancer_term(row["name"])
            mapping[standard] = standard
        mapping.pop("cancer", None)
        mapping.pop("", None)
        self.mapping = mapping
        self.synonym_rows = rows

        do_terms = load_do_terms(reference.source_dir / "doid.json") if reference.source_dir else {}
        self.oncotree = OncoTreeMapper(reference.oncotree, do_terms)
        if not reference.oncotree:
            log.warning("No OncoTree in the reference snapshot: no cancer types can be mapped")
        # Normalized CIViC name -> mapping; when several CIViC diseases normalize to the
        # same name (Lung Cancer, Lung Carcinoma) the best-founded mapping wins
        self.disease_mappings = {}
        for row in rows:
            m = self.oncotree.map_disease(row["name"], row["doid"], row["synonyms"])
            key = normalize_cancer_term(row["name"])
            old = self.disease_mappings.get(key)
            if old is None or (old.code is None, _METHOD_RANK[old.method]) > (m.code is None, _METHOD_RANK[m.method]):
                self.disease_mappings[key] = m

    def to_civic(self, terms: Iterable[str]) -> set[str]:
        mapped = set()
        for term in terms:
            normalized = normalize_cancer_term(term)
            if normalized in self.mapping:
                mapped.add(self.mapping[normalized])
                continue
            stripped = re.sub(r"^metastatic\s+|\s+metastatic$", "", normalized).strip()
            if stripped in self.mapping:
                mapped.add(self.mapping[stripped])
        mapped.discard("cancer")
        return mapped

    def harmonize(self, civic_name: str) -> str | None:
        """Cancer node for a (normalized) CIViC disease name; None if it is dropped."""
        m = self.disease_mappings.get(normalize_cancer_term(civic_name))
        if m is None:
            m = self.oncotree.map_disease(civic_name)
        return m.node

    def map_terms(self, terms: Iterable[str]) -> set[str]:
        return {node for name in self.to_civic(terms) if (node := self.harmonize(name))}

    def node_attributes(self, node: str) -> dict[str, str]:
        """OncoTree attributes of a cancer node (code, lineage of codes, tissue, main type)."""
        code = self._code_of_node().get(node)
        if not code:
            return {}
        t = self.oncotree.tree.by_code[code]
        return {"oncotree_code": code, "oncotree_lineage": ";".join(self.oncotree.tree.lineage(code)),
                "oncotree_tissue": t.get("tissue") or "", "oncotree_main_type": t.get("mainType") or ""}

    def _code_of_node(self) -> dict[str, str]:
        if not hasattr(self, "_node_codes"):
            self._node_codes = {m.node: m.code for m in self.disease_mappings.values() if m.node}
        return self._node_codes

    def mapping_table(self) -> list[dict]:
        """One row per CIViC disease name: its OncoTree mapping (``cancer_mapping.csv``)."""
        tree = self.oncotree.tree
        rows = []
        for key, m in sorted(self.disease_mappings.items()):
            t = tree.by_code.get(m.code) if m.code else None
            rows.append({"civic_disease": m.disease, "doid": m.doid, "node": m.node or "",
                         "oncotree_code": m.code or "", "matched_code": m.matched_code or "",
                         "oncotree_tissue": t["tissue"] if t else "", "method": m.method,
                         "do_class": do_class(m.doid or None, self.oncotree.do_terms), "review": m.reason})
        return rows

    def synonym_table(self) -> dict[str, list[str]]:
        """Cancer node -> names that should find it: CIViC names and synonyms mapped to
        it, its OncoTree name and code."""
        table: dict[str, set[str]] = defaultdict(set)
        for row in self.synonym_rows:
            node = self.harmonize(normalize_cancer_term(row["name"]))
            if node:
                table[node].update([row["name"], *row["synonyms"]])
        for node, code in self._code_of_node().items():
            table[node].update([self.oncotree.tree.by_code[code]["name"], code])
        return {k: sorted(v) for k, v in table.items()}
