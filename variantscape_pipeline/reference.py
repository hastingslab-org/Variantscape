"""Reference data from CIViC, OncoKB, MONDO (via OLS) and the Disease Ontology.

Fetched once per run (cached per day under ``reference/<date>/``). MONDO synonym
lookups are cached across runs under ``reference/cache/``. Disease Ontology
parents come from the bulk ``doid.json`` release (one ~25 MB download) instead
of per-term API calls, which the DO API rate-limits heavily.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import requests

log = logging.getLogger(__name__)

CIVIC_GRAPHQL_URL = "https://civicdb.org/api/graphql"
# HGNC complete set: approved gene symbols with their aliases and previous symbols
HGNC_URL = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
# OncoKB Cancer Gene List (public, no token): ~1,200 genes aggregated from OncoKB,
# MSK-IMPACT, FoundationOne, Vogelstein et al. and the Sanger Cancer Gene Census
ONCOKB_CANCER_GENES_URL = "https://www.oncokb.org/api/v1/utils/cancerGeneList"
OLS_MONDO_URL = "https://www.ebi.ac.uk/ols4/api/ontologies/mondo/terms"
DOID_JSON_URL = "http://purl.obolibrary.org/obo/doid.json"
_OBO_PREFIX = "http://purl.obolibrary.org/obo/"


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def _post_graphql(query: str, variables: dict | None = None, retries: int = 6) -> dict:
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(CIVIC_GRAPHQL_URL, json={"query": query, "variables": variables or {}}, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                if "errors" in data:
                    raise RuntimeError(f"CIViC GraphQL errors: {data['errors']}")
                return data["data"]
            log.warning("CIViC returned %s (attempt %d/%d)", resp.status_code, attempt, retries)
        except requests.RequestException as exc:
            log.warning("CIViC request failed: %s (attempt %d/%d)", exc, attempt, retries)
        time.sleep(delay)
        delay = min(delay * 2, 60)
    raise RuntimeError("CIViC GraphQL request failed repeatedly")


def _paginate(query: str, root: str) -> list[dict]:
    """Collect all ``nodes`` (or ``edges.node``) of a cursor-paginated CIViC connection."""
    items: list[dict] = []
    after = None
    while True:
        data = _post_graphql(query, {"after": after})[root]
        if "nodes" in data:
            items.extend(data["nodes"])
        else:
            items.extend(edge["node"] for edge in data["edges"])
        if not data["pageInfo"]["hasNextPage"]:
            return items
        after = data["pageInfo"]["endCursor"]


class _NotFound:
    pass


NOT_FOUND = _NotFound()


def _get_json(url: str, params: dict | None = None, retries: int = 8):
    """Return the JSON body, ``NOT_FOUND`` for a 404, or ``None`` if all attempts failed."""
    delay = 5.0
    for attempt in range(1, retries + 1):
        wait = delay
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return NOT_FOUND
            retry_after = resp.headers.get("Retry-After", "")
            if retry_after.isdigit():
                wait = max(wait, float(retry_after))
            log.warning("GET %s returned %s (attempt %d/%d)", url, resp.status_code, attempt, retries)
        except requests.RequestException as exc:
            log.warning("GET %s failed: %s (attempt %d/%d)", url, exc, attempt, retries)
        time.sleep(wait)
        delay = min(delay * 2, 120)
    return None


# --------------------------------------------------------------------------- #
# CIViC queries
# --------------------------------------------------------------------------- #
GENES_QUERY = """
query ($after: String) {
  genes(first: 100, after: $after) { pageInfo { hasNextPage endCursor } nodes { name } }
}"""

THERAPIES_QUERY = """
query ($after: String) {
  browseTherapies(first: 100, after: $after) {
    pageInfo { hasNextPage endCursor }
    edges { node { id name ncitId therapyAliases } }
  }
}"""

BROWSE_DISEASES_QUERY = """
query ($after: String) {
  browseDiseases(first: 100, after: $after) {
    pageInfo { hasNextPage endCursor }
    edges { node { id name doid diseaseUrl diseaseAliases } }
  }
}"""

DISEASE_MONDO_QUERY = """
query ($after: String) {
  diseases(first: 100, after: $after) {
    pageInfo { hasNextPage endCursor }
    nodes { id myDiseaseInfo { mondoId } }
  }
}"""

_ASSOCIATION_FIELDS = """
      disease { name doid }
      therapies { name }
      molecularProfile { id name variants { id name feature { name } } }
"""

EVIDENCE_QUERY = """
query ($after: String) {
  evidenceItems(first: 100, after: $after, status: ACCEPTED) {
    pageInfo { hasNextPage endCursor }
    nodes {
      id evidenceType evidenceLevel evidenceRating evidenceDirection significance therapyInteractionType
      source { citationId sourceType }
""" + _ASSOCIATION_FIELDS + """
    }
  }
}"""

ASSERTIONS_QUERY = """
query ($after: String) {
  assertions(first: 100, after: $after, status: ACCEPTED) {
    pageInfo { hasNextPage endCursor }
    nodes {
      id assertionType assertionDirection ampLevel significance therapyInteractionType
""" + _ASSOCIATION_FIELDS + """
    }
  }
}"""

VARIANTS_QUERY = """
query ($after: String) {
  variants(first: 100, after: $after) {
    pageInfo { hasNextPage endCursor }
    nodes {
      id name variantAliases
      feature { name }
      ... on GeneVariant { hgvsDescriptions clinvarIds }
    }
  }
}"""


def _clean_mondo_synonym(name: str) -> str:
    """From 04.1: remove internal commas/hyphens/pipes from MONDO synonyms."""
    return " ".join(re.sub(r"[,|-]+", " ", name).split())


@dataclass
class Reference:
    genes: list[str]
    therapies: list[dict]
    diseases: list[dict]
    variants: list[dict]
    oncokb_genes: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)     # accepted CIViC evidence items
    assertions: list[dict] = field(default_factory=list)   # accepted CIViC assertions
    hgnc: dict = field(default_factory=dict)                # {"approved": [...], "aliases": {alias: symbol}}
    source_dir: Path | None = field(default=None)

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, reference_dir: Path, refresh: bool = True) -> "Reference":
        """Load today's reference snapshot, fetching it if missing.

        With ``refresh=False`` the most recent snapshot on disk is reused.
        """
        reference_dir.mkdir(parents=True, exist_ok=True)
        snapshots = sorted(p for p in reference_dir.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name))
        today_dir = reference_dir / date.today().isoformat()
        if not refresh and snapshots:
            return cls._read(snapshots[-1])
        if (today_dir / "complete").exists():
            return cls._read(today_dir)
        return cls._fetch(today_dir, reference_dir / "cache")

    @classmethod
    def _read(cls, snapshot: Path) -> "Reference":
        log.info("Using reference snapshot %s", snapshot)

        def rd(name: str):
            return json.loads((snapshot / f"{name}.json").read_text())

        # Snapshots written before these sources were added get them on first use
        backfill = {
            "oncokb_genes": _fetch_oncokb_genes,
            "civic_evidence": lambda: _paginate(EVIDENCE_QUERY, "evidenceItems"),
            "civic_assertions": lambda: _paginate(ASSERTIONS_QUERY, "assertions"),
            "hgnc": _fetch_hgnc,
        }
        for name, fetch in backfill.items():
            if not (snapshot / f"{name}.json").exists():
                (snapshot / f"{name}.json").write_text(json.dumps(fetch(), indent=1))
        return cls(rd("genes"), rd("therapies"), rd("diseases"), rd("variants"), rd("oncokb_genes"),
                   rd("civic_evidence"), rd("civic_assertions"), rd("hgnc"), snapshot)

    @classmethod
    def _fetch(cls, snapshot: Path, cache_dir: Path) -> "Reference":
        snapshot.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        log.info("Fetching CIViC genes ...")
        genes = sorted({g["name"] for g in _paginate(GENES_QUERY, "genes") if g.get("name")})

        log.info("Fetching OncoKB cancer gene list ...")
        oncokb_genes = _fetch_oncokb_genes()

        log.info("Fetching CIViC therapies ...")
        therapies = [
            {"id": t["id"], "name": t["name"], "ncitId": t.get("ncitId"), "aliases": t.get("therapyAliases") or []}
            for t in _paginate(THERAPIES_QUERY, "browseTherapies")
        ]

        log.info("Fetching CIViC diseases ...")
        diseases = _paginate(BROWSE_DISEASES_QUERY, "browseDiseases")
        mondo_ids = {
            d["id"]: (d.get("myDiseaseInfo") or {}).get("mondoId")
            for d in _paginate(DISEASE_MONDO_QUERY, "diseases")
        }
        mondo_cache = _JsonCache(cache_dir / "mondo_synonyms.json")
        log.info("Downloading Disease Ontology ...")
        do_terms = _load_disease_ontology(snapshot / "doid.json")
        disease_rows = []
        for d in diseases:
            mondo_id = mondo_ids.get(d["id"])
            disease_rows.append({
                "id": d["id"],
                "name": (d.get("name") or "").strip(),
                "doid": d.get("doid"),
                "diseaseUrl": d.get("diseaseUrl"),
                "aliases": d.get("diseaseAliases") or [],
                "mondoId": mondo_id,
                "mondo_synonyms": _mondo_synonyms(mondo_id, mondo_cache) if mondo_id else [],
                "do_parent_names": _do_parent_names(d.get("diseaseUrl"), do_terms),
            })
        mondo_cache.save()

        log.info("Fetching CIViC variants ...")
        variants = [
            {
                "id": v["id"],
                "name": (v.get("name") or "").strip(),
                "gene": ((v.get("feature") or {}).get("name") or "").strip(),
                "aliases": v.get("variantAliases") or [],
                "hgvs": v.get("hgvsDescriptions") or [],
                "clinvar_ids": v.get("clinvarIds") or [],
            }
            for v in _paginate(VARIANTS_QUERY, "variants")
        ]

        log.info("Downloading HGNC gene symbols ...")
        hgnc = _fetch_hgnc()

        log.info("Fetching CIViC evidence items and assertions ...")
        evidence = _paginate(EVIDENCE_QUERY, "evidenceItems")
        assertions = _paginate(ASSERTIONS_QUERY, "assertions")

        for name, obj in (("genes", genes), ("oncokb_genes", oncokb_genes), ("therapies", therapies),
                          ("diseases", disease_rows), ("variants", variants),
                          ("civic_evidence", evidence), ("civic_assertions", assertions), ("hgnc", hgnc)):
            (snapshot / f"{name}.json").write_text(json.dumps(obj, indent=1))
        (snapshot / "complete").write_text("ok")
        log.info("Reference snapshot written to %s (%d CIViC genes, %d OncoKB genes, %d therapies, %d diseases, "
                 "%d variants, %d evidence items, %d assertions)", snapshot, len(genes), len(oncokb_genes),
                 len(therapies), len(disease_rows), len(variants), len(evidence), len(assertions))
        return cls(genes, therapies, disease_rows, variants, oncokb_genes, evidence, assertions, hgnc, snapshot)


def _fetch_hgnc() -> dict:
    """Approved HGNC symbols and an alias/previous-symbol -> approved symbol map.

    Aliases that point to several genes (e.g. ER -> ESR1/EREG) are left out, as are
    aliases that are themselves approved symbols (e.g. AR).
    """
    import csv
    import io

    resp = requests.get(HGNC_URL, timeout=300)
    resp.raise_for_status()
    rows = [r for r in csv.DictReader(io.StringIO(resp.text), delimiter="\t") if r.get("status") == "Approved"]
    approved = {r["symbol"].upper(): r["symbol"] for r in rows}
    targets: dict[str, set[str]] = {}
    for r in rows:
        for column in ("alias_symbol", "prev_symbol"):
            for alias in filter(None, (r.get(column) or "").strip('"').split("|")):
                targets.setdefault(alias.strip().upper(), set()).add(r["symbol"])
    aliases = {a: next(iter(s)) for a, s in targets.items() if len(s) == 1 and a not in approved}
    log.info("HGNC: %d approved symbols, %d unambiguous aliases", len(approved), len(aliases))
    return {"approved": sorted(approved.values()), "aliases": aliases}


def _fetch_oncokb_genes() -> list[str]:
    data = _get_json(ONCOKB_CANCER_GENES_URL)
    if not isinstance(data, list):
        raise RuntimeError("Could not download the OncoKB cancer gene list")
    return sorted({g["hugoSymbol"].strip() for g in data if g.get("hugoSymbol")})


class _JsonCache:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict = json.loads(path.read_text()) if path.exists() else {}
        self._dirty = 0

    def get(self, key: str):
        return self.data.get(key)

    def set(self, key: str, value) -> None:
        self.data[key] = value
        self._dirty += 1
        if self._dirty >= 25:  # persist progress during long first fetches
            self.save()

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=1))
        self._dirty = 0


def _mondo_synonyms(mondo_id: str, cache: _JsonCache) -> list[str]:
    cached = cache.get(mondo_id)
    if cached is not None:
        return cached
    iri = f"http://purl.obolibrary.org/obo/{mondo_id.replace(':', '_')}"
    data = _get_json(OLS_MONDO_URL, params={"iri": iri})
    if data is None:  # transient failure: do not cache, retry next run
        log.warning("Could not fetch MONDO synonyms for %s", mondo_id)
        return []
    synonyms: list[str] = []
    if data is not NOT_FOUND:
        try:
            term = data["_embedded"]["terms"][0]
            synonyms = [_clean_mondo_synonym(s) for s in (term.get("synonyms") or [])]
        except (KeyError, IndexError):
            synonyms = []
    cache.set(mondo_id, synonyms)
    return synonyms


def _load_disease_ontology(path: Path) -> dict[str, dict]:
    """Return ``{'DOID:x': {'name': ..., 'parents': [...]}}`` from the DO OBO-graph JSON release.

    Parents are sorted by descending ID, which reproduces the order returned by
    the DO API used in 04.1 (that order matters for choosing the final parent).
    """
    resp = requests.get(DOID_JSON_URL, timeout=300)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    graph = resp.json()["graphs"][0]

    def curie(iri: str) -> str | None:
        if not iri.startswith(_OBO_PREFIX + "DOID_"):
            return None
        return iri[len(_OBO_PREFIX):].replace("_", ":")

    terms = {curie(n["id"]): {"name": n.get("lbl"), "parents": []} for n in graph["nodes"] if curie(n["id"])}
    for edge in graph["edges"]:
        sub, obj = curie(edge["sub"]), curie(edge["obj"])
        if edge["pred"] == "is_a" and sub in terms and obj:
            terms[sub]["parents"].append(obj)
    for term in terms.values():
        term["parents"].sort(reverse=True)
    log.info("Disease Ontology loaded: %d terms", len(terms))
    return terms


def _do_parent_names(disease_url: str | None, do_terms: dict[str, dict]) -> list[str]:
    if not disease_url:
        return []
    m = re.search(r"DOID:(\d+)", disease_url)
    term = do_terms.get(f"DOID:{m.group(1)}") if m else None
    if not term:
        return []
    return [(do_terms.get(p) or {}).get("name") or "Unknown" for p in term["parents"]]
