"""Cancer type extraction and normalization.

- Extraction (04.1 section 3): two SciSpaCy NER models, then term cleaning.
  Cleaned raw terms are stored per paper.
- Mapping (04.1 section 4): terms -> CIViC disease names via names and synonyms.
- Harmonization (06.02 step 1): CIViC names -> the cancer node names of the
  network (DO-derived "final parents", keyword mapping, leukemia/lymphoma).

Mapping and harmonization depend on reference data only and are recomputed on
every build from the stored raw terms.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Iterable, Sequence

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
# Harmonization (04.1 section 5 final parents, 06.02 step 1)
# --------------------------------------------------------------------------- #
# 04.1 keyword mapping (its dict literal listed "glioblastoma" twice; the later value wins)
FINAL_PARENT_KEYWORDS = {
    "skin": "skin cancer", "breast": "breast cancer", "mammary": "breast cancer",
    "mucinous": "mucinous cancer", "lung": "lung cancer", "bronchio": "lung cancer",
    "spindle cell": "spindle cell cancer", "acute myeloid leukemia": "acute myeloid leukemia",
    "salivary gland": "salivary gland cancer", "renal": "renal cancer", "prostate": "prostate cancer",
    "pancreatic": "pancreatic cancer", "medulloblastoma": "medulloblastoma",
    "lymphoblastic leukemia": "lymphoblastic leukemia", "myeloid": "myeloid cancer",
    "kidney": "kidney cancer", "head and neck": "head and neck cancer",
    "gastrointestinal": "gastrointestinal cancer", "neurofibroma": "neurofibroma",
    "ovarian": "ovarian cancer", "ovary": "ovarian cancer", "supratentorial ependymoma": "supratentorial ependymoma",
    "cervix": "cervix cancer", "cervical": "cervix cancer", "colorectal": "colon cancer", "colon": "colon cancer",
    "endometri": "endometrial cancer", "melano": "melanoma", "laryngeal": "laryngeal cancer", "glioma": "glioma",
    "bone": "bone cancer", "osteo": "bone cancer", "peritoneal": "peritoneal cancer", "astrocytoma": "astrocytoma",
    "glioblastoma": "glioma", "gastric": "gastric cancer", "mesothelioma": "mesothelioma",
    "esophag": "esophagus cancer", "thyroid": "thyroid cancer", "thymus": "thymus cancer",
    "uterus": "uterine cancer", "spinal": "spinal cancer", "hepatocellular": "liver cancer",
    "cholangio": "cholangio cancer", "bile duct": "biliary tract cancer", "gliosarcoma": "glioma",
    "myeloid cancer": "hematologic cancer", "myeloproliferative cancer": "hematologic cancer",
    "myelodysplastic syndrome": "hematologic cancer", "essential thrombocythemia": "hematologic cancer",
    "myelofibrosis": "hematologic cancer", "barrett": "esophagus cancer", "fraumeni": "li-fraumeni syndrome",
    "liposarcoma": "liposarcoma", "papillary": "papillary cancer",
}

# 06.02 keyword mapping used to harmonize cancer columns
HARMONIZE_KEYWORDS = dict(FINAL_PARENT_KEYWORDS)
HARMONIZE_KEYWORDS["glioblastoma"] = "glioblastoma"

_UNWANTED_PARENTS = {"cancer", "carcinoma", "adenocarcinoma", "unknown", "cell type cancer",
                     "cell type benign neoplasm", "autosomal dominant disease", "autosomal recessive disease",
                     "syndrome"}
_GENERIC_PARENT = {"none", "cancer", "carcinoma", "solid cancer", "solid tumor", "solid tumors, advanced"}


def _classify_leukemia_lymphoma(name: str) -> str | None:
    lower = name.lower()
    leukemia = "leukemia" in lower or "leukemic" in lower
    lymphoma = "lymphoma" in lower
    if leukemia and lymphoma:
        return "leukemia/lymphoma"
    if leukemia:
        return "leukemia"
    if lymphoma:
        return "lymphoma"
    return None


def compute_final_parents(diseases: Sequence[dict]) -> set[str]:
    """Port of 04.1 'Cancer parent mapping' producing the set of final parents."""
    parents: set[str] = set()
    for d in diseases:
        name = _fix_disease_name(d["name"])
        parent_names = ", ".join(d.get("do_parent_names") or [])
        parent_names = re.sub(r"\b(?:malignant|childhood|adult|juvenile)\b", "", parent_names, flags=re.IGNORECASE)
        parent_names = re.sub(r"\s+", " ", parent_names).strip()
        final = name
        if parent_names:
            for parent in reversed([p.strip() for p in parent_names.split(",")]):
                if parent.lower() not in _UNWANTED_PARENTS:
                    final = parent
                    break
        if name.lower() in GENERIC_DISEASE_NAMES | {"doid:", "solid cancer"} and final.lower() in _GENERIC_PARENT:
            continue
        final = re.sub(r"\b(?:malignant|childhood|adult|juvenile|benign)\b", "", final, flags=re.IGNORECASE).strip()
        for old in ("neoplasm", "carcinoma", "adenocarcinoma", "adenocancer"):
            final = re.sub(old, "cancer", final, flags=re.IGNORECASE)
        final = re.sub("leukaemia", "leukemia", final, flags=re.IGNORECASE)
        final = next((v for k, v in FINAL_PARENT_KEYWORDS.items() if k in final.lower()), final)
        final = _classify_leukemia_lymphoma(final) or final
        lower = final.lower()
        if re.search(r"\bbladder\b", lower) and "gallbladder" not in lower:
            final = "bladder cancer"
        elif "gallbladder" in lower:
            final = "gallbladder cancer"
        else:
            final = next((v for k, v in FINAL_PARENT_KEYWORDS.items() if re.search(rf"\b{re.escape(k)}\b", lower)), final)
        parents.add(final.lower())
    return parents


class CancerMapper:
    """Maps cleaned raw cancer terms to harmonized cancer node names."""

    def __init__(self, reference: Reference, use_synonyms: bool = True):
        self.final_parents = compute_final_parents(reference.diseases)
        rows = [
            {"name": _fix_disease_name(d["name"]), "synonyms": disease_synonyms(d)}
            for d in reference.diseases
            if d["name"] and _fix_disease_name(d["name"]).lower() not in GENERIC_DISEASE_NAMES
        ] + MANUAL_DISEASES
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

    def harmonize(self, civic_name: str) -> str:
        lower = civic_name.lower()
        if lower in self.final_parents:
            return lower
        for keyword, replacement in HARMONIZE_KEYWORDS.items():
            if re.search(rf"\b{re.escape(keyword)}\b", lower):
                return replacement
        return _classify_leukemia_lymphoma(lower) or civic_name

    def map_terms(self, terms: Iterable[str]) -> set[str]:
        return {self.harmonize(name) for name in self.to_civic(terms)}

    def synonym_table(self) -> dict[str, list[str]]:
        """Harmonized cancer name -> CIViC names and synonyms that map to it."""
        table: dict[str, set[str]] = {}
        for row in self.synonym_rows:
            node = self.harmonize(normalize_cancer_term(row["name"]))
            table.setdefault(node, set()).update([row["name"], *row["synonyms"]])
        return {k: sorted(v) for k, v in table.items()}
