"""LLM-based variant extraction (05.1) and rule/CIViC-based normalization (05.2).

The raw LLM response is stored per paper; parsing and normalization run on
every build so that rule changes apply retroactively without new LLM calls.
Variant entities are named ``<variant>_<GENE>`` (e.g. ``v600e_BRAF``), the node
naming used by EvidenceDb.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Sequence

PROMPT_ID = "variant_prompt4"


def build_prompt(title: str, abstract: str) -> str:
    """Prompt #3 of 05.1 (best performing in the LLM evaluation), extended with
    alteration classes: fusions, amplifications, ITD and codon-level hotspots.

    Note: the published evaluation used prompt #3; this extension is not yet evaluated.
    """
    return (
        f"Extract only specific genetic variants from the text. Return strictly:\n"
        f"- **HGVS Notation** (c., p., g.) e.g., c.2138C>G, p.Arg713Trp\n"
        f"- **Protein changes** (e.g., V600E, Arg713Trp)\n"
        f"- **rsIDs** (e.g., rs121913529)\n"
        f"- **Gene fusions / rearrangements** (e.g., EML4-ALK fusion, ALK rearrangement): "
        f"'Variant: fusion, Gene: <5' partner>-<3' partner>' or 'Variant: fusion, Gene: <gene>'\n"
        f"- **Gene amplifications** (e.g., ERBB2/HER2 amplification): 'Variant: amplification, Gene: <gene>'\n"
        f"- **Internal tandem duplications** (e.g., FLT3-ITD): 'Variant: ITD, Gene: <gene>'\n"
        f"- **Hotspot codons** reported without the exact change (e.g., BRAF V600, KRAS G12): "
        f"'Variant: <codon>, Gene: <gene>'\n"
        f"- Ignore vague terms (e.g., 'mutation found', 'expression', 'loss').\n\n"
        f"### Format:\n"
        f"- Variant: 'Variant: <mutation>, Gene: <gene>' per line\n"
        f"- If none, return: 'No variant'\n"
        f"- No extra text, no explanations.\n\n"
        f"Title: {title}\nAbstract: {abstract}"
    )


# The 05.2 regex only allowed [\w.*-] in variant names, so every variant
# containing '>', '+', '(' or ')' (e.g. c.2138C>G, IVS2+1G>A, p.(Gly12Cys)) was
# silently dropped.
_PAIR_RE = re.compile(r"Variant:\s*([^,\n]+?)\s*,\s*Gene:\s*([A-Za-z0-9][A-Za-z0-9\-/:]*)", re.IGNORECASE)


def parse_variant_gene_pairs(response: str | None) -> list[tuple[str, str]]:
    if not isinstance(response, str):
        return []
    pairs = []
    for variant, gene in _PAIR_RE.findall(response):
        variant = re.sub(r"^\*+\s+", "", variant).strip(" '\"`")
        if len(variant.split()) > 4:  # sentences, not variant names
            continue
        # "V600E (c.1799T>A)" -> "V600E"; "p.(Gly12Cys)" -> "p.Gly12Cys"
        m = re.match(r"^(.*?)\s*\((.*)\)\s*$", variant)
        if m:
            outer = re.sub(r"^[cpg]\.$", "", m.group(1).strip(), flags=re.IGNORECASE)
            variant = m.group(1).strip() if len(re.sub(r"\W", "", outer)) >= 2 else m.group(1) + m.group(2)
        pairs.append((variant, gene.strip()))
    return pairs


# --------------------------------------------------------------------------- #
# Rule-based cleaning (05.2)
# --------------------------------------------------------------------------- #
AA_3_TO_1 = {
    "val": "V", "gly": "G", "glu": "E", "asp": "D", "thr": "T", "met": "M",
    "ala": "A", "leu": "L", "ser": "S", "pro": "P", "cys": "C", "phe": "F",
    "his": "H", "lys": "K", "asn": "N", "tyr": "Y", "trp": "W", "ile": "I",
    "arg": "R", "gln": "Q",
}
_AA_RE = re.compile(rf"(?i)\b({'|'.join(AA_3_TO_1)})(\d+)({'|'.join(AA_3_TO_1)})\b")

# Exon-level events, matched on the whole name with spaces/punctuation removed.
# 05.2 used substring tests on the raw name, which missed "Exon 19 Deletion" and
# would turn e.g. c.1219del into exon19del.
EXON_MAP = {
    "exon19del": re.compile(r"^(?:ex(?:on)?19del(?:etion)?s?|19del|del19)$"),
    "exon20ins": re.compile(r"^(?:ex(?:on)?20ins(?:ertion)?s?|ins20)$"),
    "exon14skipping": re.compile(r"^(?:ex(?:on)?14skip(?:ping)?(?:mutation|variant)?s?)$"),
}

UNWANTED_VARIANTS = {"vus", "c.", "loss", "indel"}
REMOVE_IF_CONTAINS = {"mutation", "mutated", "mut", "mutant", "sensitive", "deficient", "null", "amplified",
                      "deficiency", "deletion", "loxp", "loss", "knockdown", "flox"}
MAX_VARIANT_LENGTH, MAX_GENE_LENGTH = 40, 10
MIN_VARIANT_LENGTH, MIN_GENE_LENGTH = 2, 2
SPECIFIC_REMOVALS = {("p53", "TP53")}

GENE_ALIAS_MAP = {
    "CASPASE": "CASP1", "CBRAF": "BRAF", "CHECK2": "CHEK2", "CKIT": "KIT", "CMET": "MET", "CMYC": "MYC",
    "CTNNB": "CTNNB1", "CYP19": "CYP19A1", "EGFR2": "ERBB2", "ER": "ESR1", "ERB": "ERBB2", "ERB2": "ERBB2",
    "ERBB": "ERBB2", "ERRB2": "ERBB2", "ERRB3": "ERBB3", "ERRB4": "ERBB4", "ESR": "ESR1", "FACND2": "FANCD2",
    "FCR": "FCGR3A", "FCRIIIA": "FCGR3A", "FP": "PTGFR", "GAL": "GAL1", "GALECTIN": "LGALS1", "GCSF": "CSF3",
    "GLUT3": "SLC2A3", "GNB2L1": "RACK1", "GNAS1": "GNAS", "GP130": "IL6ST", "GPX": "GPX1", "GQ": "GNAQ",
    "HER-2": "ERBB2", "HER-3": "ERBB3", "HER-4": "ERBB4", "HER2": "ERBB2", "HER2-NEU": "ERBB2",
    "HER2/NEU": "ERBB2", "HER2NEU": "ERBB2", "HER3": "ERBB3", "HER4": "ERBB4", "TELOMERASE": "TERT",
    "ATM1": "ATM", "ATR1": "ATR", "BRCA2A": "BRCA2", "CDN1A": "CDKN1A", "CDN2A": "CDKN2A", "CDN2B": "CDKN2B",
    "CNNA1": "CCNA1", "CNND1": "CCND1", "COLI": "COL1A1", "EGFRVIII": "EGFR", "FLT3ITD": "FLT3", "INSR1": "INSR",
    "KRASG12C": "KRAS", "KRASG13D": "KRAS", "NRAS1": "NRAS", "PDGFR": "PDGFRA", "PI3K": "PIK3CA",
    "PI3KCA": "PIK3CA", "PTEN1": "PTEN", "RASA1A": "RASA1", "RB1CC1A": "RB1CC1", "SMAD": "SMAD4", "SRC1": "SRC",
    "STAT5": "STAT5A", "TERC1": "TERC", "TERT1": "TERT", "TP5": "TP53", "TP53A": "TP53", "TP53B": "TP53",
    # Point mutations reported on a fusion gene (e.g. BCR-ABL T315I) belong to the driver gene
    "BCRABL": "ABL1", "BCRABL1": "ABL1", "EML4ALK": "ALK",
}
FUSION_GENE_ALIASES = {"BCRABL", "BCRABL1", "EML4ALK"}

INVALID_GENES = {
    "ANDROGEN", "CYP450", "CATENIN", "CSF", "14", "2Q35", "6Q25", "8Q24", "9Q31", "VARIANT",
    "FCR", "FCRIIIA", "GCSF", "GAL", "GALECTIN", "GQ", "FNACA", "FOPNL", "FP", "GM", "G11",
    "VITAMIN", "ISOCITRATE", "MICRORNA", "LOC146880", "LOC643714", "LOC730100", "HISTONE3",
    "CIRCHIBADH", "SURVIVIN", "LINC00951", "LINC01614", "LINC02183", "LINC02869",
    "NONE", "NO", "NOT", "UNKNOWN", "",
}

# Patterns and manual removals added after reviewer feedback (05.2)
_REMOVE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    r"^[a-z]{1}\d{2,5}$",   # e.g. Q61 (position only)
    r"^\d{2,5}[a-z]{1}$",   # e.g. 12D
    r".*wt$",               # wild type
    r"^\d{2,5}$",           # digits only
    r"^[a-z]{2,}$",         # letters only
)]
MANUAL_REMOVALS = {
    "*1", "*3", "*4", "*6", "17991801hetdeltga", "186edel", "195053ctgdel", "558560agadel",
    "d770indelsdsvd", "del1518", "del17p", "del18", "del19", "del22352249", "del22362250",
    "del52", "del5395", "del5395bp", "del719", "del746750", "del866c", "dele709",
    "dele709t710insd", "dele746750", "dele746a750", "dele746t751", "dele746t751insv",
    "dele747a750", "dele748752", "dele748a752", "delex19", "dell747e749", "dell747p753",
    "dell747p753inss", "dell747s752ins", "dell747s75linss", "dell747t751", "delp704",
    "delpv104p", "dels752i759", "delt751i759inss", "delw557k558", "ex19indel", "exon15cndel",
    "ckras", "cbraf", "craf", "cgfr", "cegfr", "cnras", "cmet", "cmek1", "cher2",
    "cher3", "cher4", "cher", "chk2", "cpten", "cdlkb", "cher1", "cher5",
}
_PATTERN_EXCEPTIONS = {"v3", "v7"}

_ALIASES_BY_GENE: dict[str, list[str]] = defaultdict(list)
for _alias, _canonical in GENE_ALIAS_MAP.items():
    _ALIASES_BY_GENE[_canonical].append(_alias.lower())


def _normalize_amino_acids(variant: str) -> str:
    return _AA_RE.sub(lambda m: f"{AA_3_TO_1[m.group(1).lower()]}{m.group(2)}{AA_3_TO_1[m.group(3).lower()]}", variant)


def _map_exon(variant: str) -> str:
    compact = re.sub(r"[\s_\-.]", "", variant.lower())
    for canonical, pattern in EXON_MAP.items():
        if pattern.match(compact):
            return canonical
    return variant


def clean_gene(gene: str) -> str:
    gene = re.sub(r"[-_\s]", "", gene.strip().upper())
    return GENE_ALIAS_MAP.get(gene, gene)


def clean_variant_string(variant: str) -> str:
    variant = _normalize_amino_acids(variant.strip())
    variant = _map_exon(variant)
    variant = re.sub(r"^[cp]\.", "", variant, flags=re.IGNORECASE)
    return re.sub(r"[-_\s\.()'\"`]", "", variant).lower()


# --------------------------------------------------------------------------- #
# Alteration classes: fusions, amplifications, ITD and hotspot codons
# --------------------------------------------------------------------------- #
# Kinases / drivers that define a fusion's clinical meaning; other fusions are
# named after their 3' partner
FUSION_DRIVERS = {
    "ALK", "ROS1", "RET", "NTRK1", "NTRK2", "NTRK3", "FGFR1", "FGFR2", "FGFR3", "NRG1", "BRAF", "RAF1",
    "ABL1", "PDGFRA", "PDGFRB", "MET", "EGFR", "ERBB2", "JAK2", "RARA", "KMT2A", "ERG", "ETV6", "EWSR1",
    "FLI1", "NUTM1", "MAML2", "PAX3", "PAX8", "TFE3", "CIC", "BCOR", "NR4A3", "MYB", "CSF1R", "ESR1",
}

# Hotspot codons represented as codon-level classes (e.g. "BRAF V600 mutation" -> v600_BRAF)
HOTSPOT_CODONS = {
    "BRAF": {"v600", "k601", "g469"},
    "KRAS": {"g12", "g13", "q61", "k117", "a146"},
    "NRAS": {"g12", "g13", "q61"},
    "HRAS": {"g12", "g13", "q61"},
    "IDH1": {"r132"},
    "IDH2": {"r140", "r172"},
    "PIK3CA": {"e542", "e545", "h1047"},
    "AKT1": {"e17"},
    "EGFR": {"g719"},
    "ERBB2": {"s310"},
    "KIT": {"d816"},
    "PDGFRA": {"d842"},
    "FLT3": {"d835"},
    "DNMT3A": {"r882"},
    "ESR1": {"y537", "d538"},
    "FGFR3": {"s249"},
    "GNAQ": {"q209"},
    "GNA11": {"q209"},
    "SF3B1": {"k700"},
    "U2AF1": {"s34"},
    "MYD88": {"l265"},
    "EZH2": {"y641"},
    "CTNNB1": {"s33", "s37", "t41", "s45"},
    "TP53": {"r175", "r248", "r273"},
    "H3-3A": {"k27", "g34"},
    "H3F3A": {"k27", "g34"},
    "SPOP": {"f133"},
    "MAP2K1": {"k57"},
}

# Short forms used for fusion partners
FUSION_PARTNER_ALIASES = {"ABL": "ABL1", "MLL": "KMT2A", "PDGFR": "PDGFRB", "NTRK": "NTRK1"}

_FUSION_WORDS_RE = re.compile(r"fusion|rearrang|translocation|::", re.IGNORECASE)
_PARTNERS_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,9})\s*(?:::|-|/|–)\s*([A-Z][A-Z0-9]{1,9})\b")
_AMPLIFICATION = {"amp", "amplification", "amplifications", "amplified", "geneamplification",
                  "focalamplification", "copynumbergain", "copynumberamplification", "cngain", "cnamplification",
                  "highlevelamplification"}
_ITD = {"itd", "itdmutation", "itdmutations", "internaltandemduplication", "internaltandemduplications"}
_CODON_RE = re.compile(rf"^(?:p)?({'|'.join(AA_3_TO_1)}|[a-z])(\d{{1,4}})(?:x|mut|mutant|mutation|mutations|any)?$")
_CODON_ONLY_RE = re.compile(r"^codon(\d{1,4})(?:mut|mutant|mutation|mutations)?$")


def fusion_partners(variant: str, gene: str) -> tuple[str, str] | None:
    """5' and 3' partner symbols if the text names them (EML4-ALK, BCR::ABL1, ...)."""
    match = _PARTNERS_RE.search(f"{gene} {variant}".upper())
    return (match.group(1), match.group(2)) if match else None


def alteration_class(variant: str, gene: str) -> tuple[str, str] | None:
    """Map an alteration-level description to a class node ``(class, GENE)``.

    - fusions -> ("fusion", driver gene), e.g. EML4-ALK -> (fusion, ALK)
    - amplifications -> ("amplification", gene)
    - internal tandem duplications -> ("itd", gene)
    - hotspot codons without a specific change -> e.g. ("v600", "BRAF")
    """
    raw_variant, raw_gene = variant.strip(), gene.strip()
    compact = re.sub(r"[\s_\-.()'\"`/]", "", raw_variant.lower())
    if compact.startswith("p") and _CODON_RE.match(compact[1:]) and not _CODON_RE.match(compact):
        compact = compact[1:]

    # A hyphenated gene pair (e.g. Gene: BCR-ABL) without a specific change is a fusion,
    # unless it is a known alias such as HER2-neu
    compact_gene = re.sub(r"[-_\s/]", "", raw_gene.upper())
    gene_partners = fusion_partners("", raw_gene)
    remainder = compact
    for partner in gene_partners or ():
        remainder = remainder.replace(partner.lower(), "")
    gene_pair = (re.search(r"::|-|/", raw_gene) and gene_partners
                 and (compact_gene not in GENE_ALIAS_MAP or compact_gene in FUSION_GENE_ALIASES)
                 and not re.search(r"\d", remainder)          # no specific change such as T315I
                 and not compact.endswith(("amplification", "amplified", "itd")))
    if _FUSION_WORDS_RE.search(raw_variant) or "::" in raw_gene or gene_pair:
        # CIViC writes "fusion with any/unknown partner" as v::ALK or FGFR2::?
        if "::" in raw_gene:
            known = [p.strip() for p in raw_gene.split("::") if p.strip() not in {"v", "V", "?", ""}]
            if len(known) == 1:
                return ("fusion", FUSION_PARTNER_ALIASES.get(known[0].upper(), clean_gene(known[0])))
        partners = fusion_partners(raw_variant, raw_gene)
        stated = None if re.search(r"::|-|/", raw_gene) else clean_gene(raw_gene)
        if partners:
            five, three = (FUSION_PARTNER_ALIASES.get(p, clean_gene(p)) for p in partners)
            if three in FUSION_DRIVERS:
                driver = three
            elif five in FUSION_DRIVERS:
                driver = five
            elif stated and any(stated.startswith(p) or p.startswith(stated) for p in (five, three)):
                driver = stated      # the gene the text attributes the fusion to
            else:
                driver = three
        else:
            driver = stated
        return ("fusion", driver) if driver else None

    single_gene = clean_gene(raw_gene)
    # "Amplification", "HER2-neu amplification", "FLT3-ITD", ... (a short gene prefix is allowed)
    for words, token, max_prefix in ((_AMPLIFICATION, "amplification", 12), (_ITD, "itd", 8)):
        if compact in words or any(compact.endswith(w) and 0 < len(compact) - len(w) <= max_prefix
                                   and (len(w) > 3 or token == "itd") for w in words):
            return (token, single_gene)

    hotspots = HOTSPOT_CODONS.get(single_gene)
    if hotspots:
        m = _CODON_RE.match(compact)
        if m:
            aa = AA_3_TO_1.get(m.group(1), m.group(1)).lower()
            codon = f"{aa}{m.group(2)}"
            if codon in hotspots:
                return (codon, single_gene)
        m = _CODON_ONLY_RE.match(compact)
        if m:
            codon = next((c for c in hotspots if c[1:] == m.group(1)), None)
            if codon:
                return (codon, single_gene)
    return None


def variant_kind(node: str) -> str:
    """specific | fusion | amplification | itd | codon, for a variant node such as v600_BRAF."""
    variant, _, gene = node.rpartition("_")
    if variant in {"fusion", "amplification", "itd"}:
        return variant
    if variant in HOTSPOT_CODONS.get(gene, set()):
        return "codon"
    return "specific"


def codon_class(variant: str, gene: str) -> str | None:
    """Hotspot codon class of a specific variant, e.g. v600e/BRAF -> v600."""
    m = re.match(r"^([a-z])(\d{1,4})[a-z*]", variant)
    if m and f"{m.group(1)}{m.group(2)}" in HOTSPOT_CODONS.get(gene, set()):
        return f"{m.group(1)}{m.group(2)}"
    return None


def clean_pair(variant: str, gene: str) -> tuple[str, str] | None:
    """Clean one LLM (variant, gene) pair; ``None`` if it should be discarded.

    Alteration classes (fusions, amplifications, ITD, hotspot codons) are
    recognized first; other non-specific descriptions are filtered out.
    """
    alteration = alteration_class(variant, gene)
    if alteration:
        token, gene = alteration
        if gene in INVALID_GENES or len(gene) < MIN_GENE_LENGTH or len(gene) > MAX_GENE_LENGTH:
            return None
        return alteration
    gene = clean_gene(gene)
    if gene in INVALID_GENES:
        return None
    variant = variant.strip()
    if gene == "TP53" and variant.lower().startswith("p53"):
        variant = variant[3:]
    v = clean_variant_string(variant)

    if v == gene.lower() or len(v) < MIN_VARIANT_LENGTH or len(gene) < MIN_GENE_LENGTH:
        return None
    if (v, gene) in SPECIFIC_REMOVALS or v in UNWANTED_VARIANTS or v == "none":
        return None
    if any(term in v for term in REMOVE_IF_CONTAINS):
        return None
    if len(v) > MAX_VARIANT_LENGTH or len(gene) > MAX_GENE_LENGTH:
        return None

    # Remove a leading gene symbol or gene alias from the variant (e.g. "brafv600e")
    g = gene.lower()
    if v.startswith(g) and not re.match(rf"^{re.escape(g)}[*.]", v):
        v = v[len(g):]
    for alias in _ALIASES_BY_GENE.get(gene, []):
        if v.startswith(alias):
            v = v[len(alias):]
            break

    if len(v) < MIN_VARIANT_LENGTH:
        return None
    if v not in _PATTERN_EXCEPTIONS and (v in MANUAL_REMOVALS or any(p.match(v) for p in _REMOVE_PATTERNS)):
        return None
    return v, gene


# --------------------------------------------------------------------------- #
# CIViC-based normalization (replaces the v2/v3 alias merging of 05.2)
# --------------------------------------------------------------------------- #
class VariantNormalizer:
    """Maps CIViC aliases, HGVS descriptions and rsIDs onto the CIViC variant name.

    05.2 merged alias columns via a join of variants and molecular profiles on
    unrelated IDs and compared raw (uncleaned) aliases with cleaned names. Here
    CIViC's own gene, aliases and HGVS descriptions are used, all passed through
    the same cleaning as the LLM output, and aliases only merge within a gene.
    """

    def __init__(self, civic_variants: Sequence[dict]):
        canonical: dict[tuple[str, str], str] = {}
        canonical_names: set[tuple[str, str]] = set()
        prepared = []
        for row in civic_variants:
            if not row.get("gene") or not row.get("name"):
                continue
            name, extra = row["name"], []
            m = re.match(r"^(.*?)\s*\((.*)\)\s*$", name)
            if m:  # e.g. "L89P (c.266T>A)"
                name, extra = m.group(1), [m.group(2)]
            cleaned = clean_pair(name, row["gene"])
            if cleaned is None:
                continue
            canonical_names.add(cleaned)
            aliases = list(row.get("aliases") or []) + extra + [h.split(":", 1)[-1] for h in row.get("hgvs") or []]
            prepared.append((cleaned, aliases))

        ambiguous: set[tuple[str, str]] = set()
        for (variant, gene), aliases in prepared:
            for alias in aliases:
                cleaned_alias = clean_pair(alias, gene)
                if cleaned_alias is None or cleaned_alias == (variant, gene) or cleaned_alias in canonical_names:
                    continue
                if cleaned_alias in canonical and canonical[cleaned_alias] != variant:
                    ambiguous.add(cleaned_alias)
                canonical[cleaned_alias] = variant
        for key in ambiguous:
            canonical.pop(key, None)
        self.alias_map = canonical
        # Fusion partners seen per fusion node, e.g. fusion_ALK -> {"EML4::ALK", "KIF5B::ALK"}
        self.fusion_partners: dict[str, set[str]] = defaultdict(set)

    def record_fusion(self, node: str, variant: str, gene: str) -> None:
        partners = fusion_partners(variant, gene)
        if partners and node.startswith("fusion_"):
            five, three = (FUSION_PARTNER_ALIASES.get(p, p) for p in partners)
            self.fusion_partners[node].add(f"{five}::{three}")

    def canonicalize(self, variant: str, gene: str) -> str:
        return self.alias_map.get((variant, gene), variant)

    def node_id(self, variant: str, gene: str) -> str:
        return f"{self.canonicalize(variant, gene)}_{gene}"

    def canonical_node(self, node: str) -> str:
        """Canonicalize an existing ``variant_GENE`` name (e.g. variants named in co-association pairs)."""
        if "_" not in node:
            return node
        variant, gene = node.rsplit("_", 1)
        return self.node_id(variant, gene)

    def nodes_from_response(self, response: str | None) -> set[str]:
        nodes = set()
        for variant, gene in parse_variant_gene_pairs(response):
            cleaned = clean_pair(variant, gene)
            if cleaned:
                node = self.node_id(*cleaned)
                nodes.add(node)
                if cleaned[0] == "fusion":
                    self.record_fusion(node, variant, gene)
        return nodes

    def nodes_from_responses(self, responses: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
        out = {}
        for paper_id, response in responses:
            nodes = self.nodes_from_response(response)
            if nodes:
                out[paper_id] = nodes
        return out
