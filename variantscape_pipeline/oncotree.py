"""Cancer types as OncoTree tumour types.

Cancer terms found in abstracts are mapped to CIViC diseases (``cancers.py``); each
CIViC disease is mapped here to an OncoTree tumour type, which becomes the
cancer node of the network. OncoTree (MSK; used by OncoKB, GENIE, cBioPortal) is
organised by tissue -> main type -> subtype (Lung -> NSCLC -> Lung Adenocarcinoma),
so every node has an organ site and a lineage that EvidenceDb uses to roll
subtypes up into broader searches.

Mapping steps for a CIViC disease, first match wins:

1. ``manual``: MANUAL_RULES (cases the automatic steps get wrong; ``None`` = drop);
2. ``xref``: DOID -> Disease Ontology NCIt / UMLS cross-references -> OncoTree;
3. ``name``: CIViC / DO / MONDO names equal to an OncoTree name;
4. ``ancestor``: the nearest Disease Ontology ancestor that maps by 2. or 3.;
5. ``organ``: the OncoTree tissue named in the disease (``lung cancer`` -> Lung),
   also replacing an ancestor mapping in another tissue;
6. ``fallback``: FALLBACK_RULES (molecular subtypes missing from OncoTree).

Candidates in another organ than the one a disease is named after are skipped
(OncoTree's cervical "Mucinous Carcinoma" carries the generic NCIt code).

Dropped (not cancer nodes): site-less histologies (OncoTree "..., NOS" types without
a site, e.g. squamous cell carcinoma NOS), names too generic to be clinically
meaningful ("leukemia", "hematologic cancer") and unmapped diseases. These are
listed in ``cancer_mapping.csv`` of each build for review.

The rules below are a first, unreviewed version; they are meant to be checked by
a clinician and extended.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

ONCOTREE_URL = "https://oncotree.mskcc.org/api/tumorTypes?version=oncotree_latest_stable"
OBO = "http://purl.obolibrary.org/obo/"
DO_CELL_PROLIFERATION = "DOID:14566"   # disease of cellular proliferation
DO_BENIGN = "DOID:0060072"             # benign neoplasm
DO_SYNDROME = "DOID:225"               # syndrome

# OncoTree codes too generic to map to ("MT" = Malignant Tumor and "MLYM" = Malignant
# Lymphoma sit under CNS/Brain but carry the NCIt codes of malignant neoplasm / lymphoma)
GENERIC_CODES = {"MT", "MLYM"}
# Tissues that are not one organ: fine for diseases named after an organ (hepatosplenic
# lymphoma, gastrointestinal stromal tumor) and for names without a site
SITE_AGNOSTIC_TISSUES = {"Other", "Soft Tissue", "Lymphoid", "Myeloid", "Bone", "Skin",
                         "Peripheral Nervous System", "CNS/Brain"}

# Regex on the lower-case CIViC disease name -> OncoTree code; None drops the disease
MANUAL_RULES = [
    (r"^chronic lymphocytic leukemia$", "CLLSLL"),
    (r"^(childhood )?b[- ]cell (adult |childhood )?acute lymphoblastic leukemia$"
     r"|^b[- ]cell (adult |childhood )?acute lymphocytic leukemia$", "BLL"),
    (r"^t[- ]cell (acute )?lymphoblastic leukemia(/lymphoma)?$", "TLL"),
    (r"^(childhood )?acute lymphoblastic leukemia$|^(childhood )?acute lymphocytic leukemia$"
     r"|^lymphoid leukemia$", "LNM"),
    (r"^acute promyelocytic leukemia$", "APLPMLRARA"),
    (r"^acute biphenotypic leukemia$", "ALAL"),
    (r"^lymphoma$", "LNM"),
    (r"^b[- ]cell non[- ]hodgkin lymphoma$", "MBN"),
    # too generic / no site
    (r"^(acute |chronic )?leukemia$|^hematologic cancer$", None),
    (r"^mucinous adenocarcinoma$|^papillary adenocarcinoma$", None),
]
# Applied only when nothing else maps
FALLBACK_RULES = [
    (r"^b[- ]lymphoblastic leuka?emia", "BLL"),
    (r"^(acute myeloid leuka?emia|aml) with", "AML"),
]

# Organ / site words in disease names -> OncoTree tissue (level 1) code
ORGAN_TISSUE = [
    (r"gallbladder|bile duct|biliary|cholangio", "BILIARY_TRACT"),
    (r"lung|bronch|pulmonary", "LUNG"), (r"breast|mammary", "BREAST"),
    (r"colon|colorectal|rectum|rectal|bowel|intestin|appendi|anal\b|anus", "BOWEL"),
    (r"esophag|oesophag|stomach|gastric|gastroesophageal|cardia\b", "STOMACH"),
    (r"ovar|fallopian", "OVARY"), (r"kidney|renal|nephr", "KIDNEY"),
    (r"bladder|urothelial|transitional cell|ureter|urethra|urinary", "BLADDER"), (r"prostat", "PROSTATE"),
    (r"pancrea", "PANCREAS"), (r"liver|hepat", "LIVER"), (r"thyroid", "THYROID"),
    (r"skin|cutaneous", "SKIN"), (r"cervix|cervical", "CERVIX"), (r"uter|endometri", "UTERUS"),
    (r"brain|central nervous system|\bcns\b|glio|astrocyt|ependym|medulloblast", "BRAIN"),
    (r"head and neck|oral|pharyn|laryn|salivary|tongue|sinonasal|nasal", "HEAD_NECK"),
    (r"\bbone\b|osteo", "BONE"), (r"pheochromocyt|adrenal", "ADRENAL_GLAND"), (r"soft tissue", "SOFT_TISSUE"),
    (r"\beye\b|ocular|uveal|retin|conjunctiv", "EYE"),
    (r"pleura|mesothelioma", "PLEURA"), (r"peritone", "PERITONEUM"), (r"testi", "TESTIS"),
    (r"penis|penile", "PENIS"), (r"thym", "THYMUS"), (r"vulva|vagina", "VULVA"),
    (r"ampulla", "AMPULLA_OF_VATER"),
]


def fetch_oncotree() -> list[dict]:
    import requests

    resp = requests.get(ONCOTREE_URL, timeout=120)
    resp.raise_for_status()
    types = resp.json()
    log.info("OncoTree: %d tumour types", len(types))
    return types


def load_do_terms(path: Path | None) -> dict[str, dict]:
    """Disease Ontology terms with names, synonyms, cross-references and parents."""
    if not path or not Path(path).exists():
        return {}
    graph = json.loads(Path(path).read_text())["graphs"][0]
    terms = {}
    for node in graph["nodes"]:
        if not node["id"].startswith(OBO + "DOID_") or node.get("type") != "CLASS":
            continue
        meta = node.get("meta") or {}
        if meta.get("deprecated"):
            continue
        doid = node["id"].rsplit("/", 1)[1].replace("_", ":")
        terms[doid] = {"name": node.get("lbl", ""), "synonyms": [s["val"] for s in meta.get("synonyms", [])],
                       "xrefs": [x["val"] for x in meta.get("xrefs", [])], "parents": []}
    for edge in graph["edges"]:
        if edge["pred"] != "is_a":
            continue
        sub, obj = (e.rsplit("/", 1)[1].replace("_", ":") for e in (edge["sub"], edge["obj"]))
        if sub in terms and obj in terms:
            terms[sub]["parents"].append(obj)
    return terms


def _norm(name: str) -> str:
    name = name.lower().replace("carcinoma", "cancer").replace("tumour", "tumor").replace("neoplasm", "tumor")
    name = re.sub(r"[^a-z0-9]+", " ", name)
    name = re.sub(r"\b(malignant|nos|primary)\b", " ", name)
    return " ".join(name.split())


def organ_tissue(names) -> str | None:
    """OncoTree tissue code of the first organ word in ``names``."""
    for name in names:
        lower = name.lower()
        for pattern, code in ORGAN_TISSUE:
            if re.search(pattern, lower):
                return code
    return None


def do_ancestors(doid: str, do_terms: dict) -> set[str]:
    seen, queue = set(), deque([doid])
    while queue:
        for p in do_terms.get(queue.popleft(), {}).get("parents", []):
            if p not in seen:
                seen.add(p)
                queue.append(p)
    return seen


def do_class(doid: str | None, do_terms: dict) -> str:
    """neoplasm, benign, syndrome, non-neoplastic or unknown (from DO ancestry; a hint
    for review only: DO calls low-grade gliomas benign)."""
    if not doid or doid not in do_terms:
        return "unknown"
    anc = do_ancestors(doid, do_terms) | {doid}
    if DO_BENIGN in anc:
        return "benign"
    if DO_SYNDROME in anc and DO_CELL_PROLIFERATION not in anc:
        return "syndrome"
    if DO_CELL_PROLIFERATION not in anc:
        return "non-neoplastic"
    return "neoplasm"


@dataclass(frozen=True)
class DiseaseMapping:
    disease: str
    doid: str
    code: str | None          # OncoTree code of the cancer node (None = dropped)
    matched_code: str | None  # code the disease matched before NOS / drop handling
    method: str
    steps: int
    node: str | None
    reason: str               # why dropped / what to review


class OncoTree:
    """OncoTree tumour types with lookups by NCIt / UMLS code and name."""

    def __init__(self, types: list[dict]):
        self.by_code = {t["code"]: t for t in types}
        self.by_nci, self.by_umls, self.by_name = defaultdict(set), defaultdict(set), defaultdict(set)
        for t in types:
            if t["code"] in GENERIC_CODES:
                continue
            refs = t.get("externalReferences") or {}
            for c in refs.get("NCI") or []:
                self.by_nci[c].add(t["code"])
            for c in refs.get("UMLS") or []:
                self.by_umls[c].add(t["code"])
            self.by_name[_norm(t["name"])].add(t["code"])

    def best(self, codes, organ: str | None = None) -> str | None:
        """Most specific candidate (deepest level), skipping other organs than ``organ``."""
        if organ:
            tissue = self.by_code[organ]["tissue"]
            codes = {c for c in codes if self.by_code[c]["tissue"] in {tissue} | SITE_AGNOSTIC_TISSUES}
        return sorted(codes, key=lambda c: (-self.by_code[c]["level"], c))[0] if codes else None

    def from_xrefs(self, xrefs, organ=None) -> str | None:
        codes = set()
        for x in xrefs:
            if x.startswith("NCI:"):
                codes |= self.by_nci.get(x[4:], set())
            elif x.startswith("UMLS_CUI:"):
                codes |= self.by_umls.get(x[9:], set())
        return self.best(codes, organ)

    def from_names(self, names, organ=None) -> str | None:
        codes = set()
        for n in names:
            codes |= self.by_name.get(_norm(n), set())
        return self.best(codes, organ)

    def lineage(self, code: str) -> list[str]:
        """Codes from the tissue down to ``code`` (LUNG, NSCLC, LUAD)."""
        out = []
        while code in self.by_code:
            out.append(code)
            code = self.by_code[code].get("parent")
        return out[::-1]

    def node_name(self, code: str) -> str:
        """Cancer node name: the OncoTree name, or the main type for tissues ("lung cancer")."""
        t = self.by_code[code]
        return (t.get("mainType") if t["level"] == 1 else t["name"]).strip().lower()


class OncoTreeMapper:
    """Maps CIViC diseases to OncoTree cancer nodes."""

    def __init__(self, oncotree_types: list[dict], do_terms: dict | None = None):
        self.tree = OncoTree(oncotree_types)
        self.do_terms = do_terms or {}

    def _do_term(self, doid, organ):
        term = self.do_terms.get(doid)
        if not term:
            return None, None
        code = self.tree.from_xrefs(term["xrefs"], organ)
        if code:
            return code, "xref"
        code = self.tree.from_names([term["name"], *term["synonyms"]], organ)
        return (code, "name") if code else (None, None)

    def match(self, name: str, doid: str | None = None, synonyms=()) -> tuple[str | None, str, int]:
        """(OncoTree code or None, method, DO steps up) for a disease."""
        lower = name.lower().strip()
        for pattern, manual in MANUAL_RULES:
            if re.search(pattern, lower):
                return manual, "manual", 0
        do_name = [self.do_terms[doid]["name"]] if doid in self.do_terms else []
        organ = organ_tissue([name, *do_name])
        organ = organ if organ in self.tree.by_code else None
        code, method, steps = None, None, 0
        if doid:
            code, method = self._do_term(doid, organ)
        if not code:
            code = self.tree.from_names([name, *synonyms], organ)
            method = "name" if code else None
        if not code and doid in self.do_terms:
            level, frontier, seen = 0, [doid], {doid}
            while frontier and not code:
                level += 1
                nxt = [p for t in frontier for p in self.do_terms[t]["parents"] if p not in seen]
                seen.update(nxt)
                found = {c for c in (self._do_term(p, organ)[0] for p in nxt) if c}
                if found:
                    code, method, steps = self.tree.best(found), "ancestor", level
                frontier = nxt
        if organ and (not code or (method == "ancestor" or self.tree.by_code[code]["tissue"] == "Other")
                      and self.tree.by_code[code]["tissue"] != self.tree.by_code[organ]["tissue"]):
            code, method, steps = organ, "organ", 0
        if not code:
            code = next((c for pattern, c in FALLBACK_RULES if re.search(pattern, lower)), None)
            method = "fallback" if code else "unmapped"
        return code, method, steps

    def map_disease(self, name: str, doid: str | None = None, synonyms=()) -> DiseaseMapping:
        code, method, steps = self.match(name, doid, synonyms)
        matched, reason = code, ""
        t = self.tree.by_code.get(code) if code else None
        if t and t["tissue"] == "Other" and "NOS" in code:
            code, reason = None, "no site (OncoTree NOS type)"
        elif t and code.endswith("NOS") and ", NOS" in t["name"] and t.get("parent") in self.tree.by_code:
            code = t["parent"]   # "AML, NOS" -> AML, "Breast Neoplasm, NOS" -> Breast
        if not t:
            reason = "too generic" if method == "manual" else "unmapped"
        if method == "ancestor" and steps > 1:
            reason = f"coarse: {steps} Disease Ontology levels up"
        return DiseaseMapping(name, doid or "", code, matched, method, steps,
                              self.tree.node_name(code) if code else None, reason)
