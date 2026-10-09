"""Curated (expert-reviewed) associations from CIViC.

Accepted CIViC evidence items (levels A-D) and assertions (AMP/ASCO/CAP tiers
I-II) are mapped onto the node names of the literature graph and added to it
with their own provenance and score, so that well-established associations are
present and rank first even when the literature mining misses them.

- Predictive records give variant-cancer, variant-treatment and cancer-treatment
  edges; diagnostic and prognostic records give variant-cancer edges.
- "Does not support" records are kept as negative evidence (e.g. a therapy that
  does not work for a variant in a cancer type).
- Molecular profiles combining several variants, non-specific variants
  ("Mutation", "Amplification", fusions) and generic diseases ("Cancer") cannot
  be mapped to single nodes and are skipped (counted in the coverage report).
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

import networkx as nx
import pandas as pd

from .cancers import GENERIC_DISEASE_NAMES, CancerMapper, normalize_cancer_term
from .coassociation import pair_key
from .genes import GeneSet
from .reference import Reference
from .variants import VariantNormalizer, clean_pair

log = logging.getLogger(__name__)

SOURCE_LITERATURE = "literature"
SOURCE_CIVIC = "civic"

INCLUDED_TYPES = {"PREDICTIVE", "DIAGNOSTIC", "PROGNOSTIC"}

# Ranking score: assertions (consensus guideline-level summaries) above single
# evidence items; within evidence items the 1-5 star rating breaks ties.
EVIDENCE_LEVEL_SCORES = {"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0}   # level E (inferential) excluded
AMP_LEVEL_SCORES = {"TIER_I_LEVEL_A": 6.0, "TIER_I_LEVEL_B": 5.5, "TIER_II_LEVEL_C": 3.5, "TIER_II_LEVEL_D": 2.5}
AMP_LEVEL_LABELS = {"TIER_I_LEVEL_A": "Tier I-A", "TIER_I_LEVEL_B": "Tier I-B",
                    "TIER_II_LEVEL_C": "Tier II-C", "TIER_II_LEVEL_D": "Tier II-D"}

# Predictions shown by EvidenceDb in the sensitive / resistant lists
SENSITIVE_PREDICTIONS = {"Sensitive"}
RESISTANT_PREDICTIONS = {"Resistant", "Reduced sensitivity", "No response"}


def prediction_label(evidence_type: str, significance: str | None, direction: str | None) -> str:
    supports = direction != "DOES_NOT_SUPPORT"
    significance = significance or ""
    if evidence_type == "PREDICTIVE":
        if significance == "SENSITIVITYRESPONSE":
            return "Sensitive" if supports else "No response"
        if significance == "RESISTANCE":
            return "Resistant" if supports else "Not resistant"
        if significance == "REDUCED_SENSITIVITY":
            return "Reduced sensitivity" if supports else "Not reduced sensitivity"
        if significance == "ADVERSE_RESPONSE":
            return "Adverse response" if supports else "No adverse response"
        return significance.replace("_", " ").capitalize()
    if evidence_type == "DIAGNOSTIC":
        return f"Diagnostic ({significance.lower()})" if supports else f"Not diagnostic ({significance.lower()})"
    if evidence_type == "PROGNOSTIC":
        outcome = significance.replace("_", " ").lower()
        return f"Prognostic ({outcome})" if supports else f"Not prognostic ({outcome})"
    return significance.capitalize()


@dataclass
class CuratedRecord:
    record_id: str           # e.g. EID238 / AID3
    kind: str                # evidence | assertion
    evidence_type: str       # PREDICTIVE | DIAGNOSTIC | PROGNOSTIC
    variant: str             # graph node, e.g. t790m_EGFR
    gene: str
    cancer: str              # graph node, e.g. lung cancer
    treatment: str           # graph node ("" for diagnostic/prognostic)
    combination: bool
    prediction: str          # Sensitive, Resistant, No response, Diagnostic (positive), ...
    significance: str
    direction: str
    level: str               # A-D or Tier I-A ...
    score: float
    rating: int | None
    pmid: str
    civic_disease: str
    civic_variant_id: int | None
    civic_profile: str


def build_curated_records(reference: Reference, mapper: CancerMapper, normalizer: VariantNormalizer,
                          gene_set: GeneSet | None = None) -> tuple[list[CuratedRecord], Counter]:
    """Map CIViC evidence items and assertions to graph entities; returns records and skip reasons."""
    records: list[CuratedRecord] = []
    skipped: Counter = Counter()

    items = [("evidence", e) for e in reference.evidence] + [("assertion", a) for a in reference.assertions]
    for kind, item in items:
        evidence_type = item.get("evidenceType") or item.get("assertionType")
        if evidence_type not in INCLUDED_TYPES:
            skipped[f"type_{str(evidence_type).lower()}"] += 1
            continue
        if kind == "evidence":
            level = item.get("evidenceLevel")
            if level not in EVIDENCE_LEVEL_SCORES:
                skipped[f"level_{level}"] += 1
                continue
            rating = item.get("evidenceRating")
            score = EVIDENCE_LEVEL_SCORES[level] + (rating or 0) / 10
            direction = item.get("evidenceDirection")
            record_id = f"EID{item['id']}"
            source = item.get("source") or {}
            pmid = source.get("citationId") if source.get("sourceType") == "PUBMED" else ""
        else:
            amp = item.get("ampLevel")
            if amp not in AMP_LEVEL_SCORES:
                skipped[f"assertion_level_{amp}"] += 1
                continue
            level, rating, score = AMP_LEVEL_LABELS[amp], None, AMP_LEVEL_SCORES[amp]
            direction = item.get("assertionDirection")
            record_id = f"AID{item['id']}"
            pmid = ""

        profile = item.get("molecularProfile") or {}
        variants = profile.get("variants") or []
        if len(variants) != 1:
            skipped["multi_variant_profile"] += 1
            continue
        civic_variant = variants[0]
        gene = ((civic_variant.get("feature") or {}).get("name") or "").strip()
        cleaned = clean_pair(civic_variant.get("name") or "", gene)
        if cleaned is None:
            skipped["non_specific_variant"] += 1
            continue
        if gene_set is not None and gene_set.restrict_variants and not gene_set.contains(cleaned[1]):
            skipped["outside_gene_set"] += 1
            continue
        variant_node = normalizer.node_id(*cleaned)
        if cleaned[0] == "fusion":
            normalizer.record_fusion(variant_node, civic_variant.get("name") or "", gene)

        disease_name = ((item.get("disease") or {}).get("name") or "").strip()
        normalized = normalize_cancer_term(disease_name)
        if not normalized or disease_name.lower() in GENERIC_DISEASE_NAMES or normalized in {"cancer", "solid cancer"}:
            skipped["generic_disease"] += 1
            continue
        cancer_node = mapper.harmonize(normalized)
        if cancer_node is None:   # no OncoTree type (site-less, too generic, unmapped)
            skipped["unmapped_disease"] += 1
            continue

        therapies = [t["name"] for t in item.get("therapies") or [] if t.get("name")]
        if evidence_type == "PREDICTIVE" and not therapies:
            skipped["predictive_without_therapy"] += 1
            continue
        combination = item.get("therapyInteractionType") == "COMBINATION" and len(therapies) > 1
        prediction = prediction_label(evidence_type, item.get("significance"), direction)
        for treatment in (therapies if evidence_type == "PREDICTIVE" else [""]):
            records.append(CuratedRecord(
                record_id=record_id, kind=kind, evidence_type=evidence_type, variant=variant_node,
                gene=cleaned[1], cancer=cancer_node, treatment=treatment, combination=combination,
                prediction=prediction, significance=item.get("significance") or "", direction=direction or "",
                level=level, score=round(score, 2), rating=rating, pmid=pmid or "", civic_disease=disease_name,
                civic_variant_id=civic_variant.get("id"), civic_profile=profile.get("name") or "",
            ))
    log.info("Curated CIViC records mapped: %d (skipped: %s)", len(records), dict(skipped))
    return records, skipped


# --------------------------------------------------------------------------- #
# Graph and consensus integration
# --------------------------------------------------------------------------- #
def _record_edges(record: CuratedRecord) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    """Edges (as (node, category) pairs) asserted by a record."""
    v, c = (record.variant, "Variant"), (record.cancer, "Cancer")
    if record.treatment:
        t = (record.treatment, "Treatment")
        return [(v, c), (v, t), (c, t)]
    return [(v, c)]


def add_curated_to_graph(G: nx.Graph, records: list[CuratedRecord]) -> dict:
    """Annotate the literature graph with curated provenance (in place).

    ``sources`` of nodes and edges gains ``civic`` (e.g. ``literature;civic`` or
    ``cooccurrence;civic``). Edges keep ``weight`` (verified literature) and
    ``cooccurrence_weight`` (0 for curated-only edges) and gain ``curated_score``
    (best record), ``curated_level``, ``curated_count`` and ``curated_ids``.
    """
    for _, data in G.nodes(data=True):
        data.setdefault("sources", SOURCE_LITERATURE)
    for _, _, data in G.edges(data=True):
        data.setdefault("sources", SOURCE_LITERATURE)
        data.update(curated_score=0.0, curated_level="", curated_count=0, curated_ids="")

    edge_records: dict[tuple[str, str], list[CuratedRecord]] = defaultdict(list)
    conflicts = 0
    for record in records:
        for (a, cat_a), (b, cat_b) in _record_edges(record):
            endpoints = ((a, cat_a), (b, cat_b))
            if any(node in G and G.nodes[node]["category"] != category for node, category in endpoints):
                conflicts += 1   # same name used for another entity type
                continue
            for node, category in endpoints:
                if node not in G:
                    G.add_node(node, category=category, sources=SOURCE_CIVIC)
                elif SOURCE_CIVIC not in G.nodes[node]["sources"]:
                    G.nodes[node]["sources"] += ";" + SOURCE_CIVIC
            edge_records[tuple(sorted((a, b)))].append(record)
        if record.civic_variant_id is not None and record.variant in G:
            G.nodes[record.variant]["civic_variant_id"] = int(record.civic_variant_id)

    for (a, b), recs in edge_records.items():
        best = max(recs, key=lambda r: r.score)
        ids = sorted({r.record_id for r in recs})
        if G.has_edge(a, b):
            data = G[a][b]
            data["sources"] = f"{data['sources']};{SOURCE_CIVIC}"
        else:
            G.add_edge(a, b, weight=0.0, cooccurrence_weight=0.0, verified_papers=0, sources=SOURCE_CIVIC)
            data = G[a][b]
        data.update(curated_score=best.score, curated_level=best.level, curated_count=len(ids),
                    curated_ids=";".join(ids))
    return {"curated_edges": len(edge_records), "category_conflicts": conflicts,
            "curated_only_edges": sum(1 for _, _, d in G.edges(data=True) if d["sources"] == SOURCE_CIVIC)}


def records_frame(records: list[CuratedRecord], literature: nx.Graph, consensus: pd.DataFrame) -> pd.DataFrame:
    """The ``curated_associations.csv`` artifact, with literature support per record."""
    llm = dict(zip(consensus["Variant_Treatment_Pair"], consensus["Resolved_Prediction"])) if len(consensus) else {}

    def lit_weight(a: str, b: str) -> float:
        return float(literature[a][b]["weight"]) if a and b and literature.has_edge(a, b) else 0.0

    rows = []
    for r in records:
        row = asdict(r)
        row["literature_variant_cancer_weight"] = lit_weight(r.variant, r.cancer)
        row["literature_variant_treatment_weight"] = lit_weight(r.variant, r.treatment)
        row["literature_cancer_treatment_weight"] = lit_weight(r.cancer, r.treatment)
        row["llm_consensus"] = llm.get(pair_key(r.variant, r.treatment), "") if r.treatment else ""
        rows.append(row)
    columns = list(CuratedRecord.__dataclass_fields__) + [
        "literature_variant_cancer_weight", "literature_variant_treatment_weight",
        "literature_cancer_treatment_weight", "llm_consensus"]
    df = pd.DataFrame(rows, columns=columns)
    return df.sort_values(["score", "variant", "cancer", "treatment"], ascending=[False, True, True, True])


def merge_consensus(consensus: pd.DataFrame, records: list[CuratedRecord]) -> pd.DataFrame:
    """Add curated provenance columns to the variant-treatment consensus.

    ``Resolved_Prediction`` stays the literature (LLM) consensus and is empty for
    curated-only pairs. ``Curated_Prediction`` is the curated label if all
    diseases agree, otherwise ``Mixed``; ``Curated_Detail`` lists the
    disease-specific labels.
    """
    by_pair: dict[str, list[CuratedRecord]] = defaultdict(list)
    for r in records:
        if r.treatment:
            by_pair[pair_key(r.variant, r.treatment)].append(r)

    curated_rows = []
    for key, recs in by_pair.items():
        predictions = {r.prediction for r in recs}
        best = max(recs, key=lambda r: r.score)
        detail: dict[tuple[str, str], CuratedRecord] = {}
        for r in sorted(recs, key=lambda r: -r.score):
            detail.setdefault((r.cancer, r.prediction), r)
        curated_rows.append({
            "Variant_Treatment_Pair": key,
            "Curated_Prediction": predictions.pop() if len(predictions) == 1 else "Mixed",
            "Curated_Level": best.level,
            "Curated_Score": best.score,
            "Curated_Detail": "; ".join(f"{c}: {p} ({r.level})" for (c, p), r in detail.items()),
            "Curated_IDs": ";".join(sorted({r.record_id for r in recs})),
        })
    curated = pd.DataFrame(curated_rows, columns=["Variant_Treatment_Pair", "Curated_Prediction", "Curated_Level",
                                                  "Curated_Score", "Curated_Detail", "Curated_IDs"])
    merged = consensus.merge(curated, on="Variant_Treatment_Pair", how="outer")
    in_lit = merged["Resolved_Prediction"].notna()
    in_cur = merged["Curated_Prediction"].notna()
    merged["Source"] = [
        f"{SOURCE_LITERATURE};{SOURCE_CIVIC}" if lit and cur else (SOURCE_LITERATURE if lit else SOURCE_CIVIC)
        for lit, cur in zip(in_lit, in_cur)
    ]
    return merged.sort_values("Variant_Treatment_Pair").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Coverage report
# --------------------------------------------------------------------------- #
def coverage_report(records: list[CuratedRecord], literature: nx.Graph, consensus: pd.DataFrame,
                    skipped: Counter) -> tuple[pd.DataFrame, dict]:
    """How much curated knowledge the literature-only graph recovers.

    "Supported by literature" uses verified edges (``weight``); "supported by
    co-occurrence" the unverified tier (``cooccurrence_weight``), i.e. what the
    pipeline found before verification.

    Unique associations are (variant, cancer, treatment, prediction); each takes
    its best level. Supported means the literature graph has the variant-cancer
    edge (diagnostic/prognostic) or all three pairwise edges (predictive).
    Label agreement compares Sensitive/Resistant curated labels with the LLM
    consensus of the variant-treatment pair.
    """
    llm = dict(zip(consensus["Variant_Treatment_Pair"], consensus["Resolved_Prediction"])) if len(consensus) else {}

    def has(a: str, b: str, attr: str = "weight") -> bool:
        return literature.has_edge(a, b) and literature[a][b].get(attr, 0) > 0

    best: dict[tuple, CuratedRecord] = {}
    for r in records:
        key = (r.variant, r.cancer, r.treatment, r.prediction)
        if key not in best or r.score > best[key].score:
            best[key] = r

    rows = []
    for (variant, cancer, treatment, prediction), r in best.items():
        supported = has(variant, cancer) and (not treatment or (has(variant, treatment) and has(cancer, treatment)))
        co = "cooccurrence_weight"
        co_supported = has(variant, cancer, co) and (
            not treatment or (has(variant, treatment, co) and has(cancer, treatment, co)))
        llm_label = llm.get(pair_key(variant, treatment), "") if treatment else ""
        comparable = prediction in {"Sensitive", "Resistant"} and llm_label in {"Sensitive", "Resistant"}
        rows.append({
            "level": r.level, "evidence_type": r.evidence_type,
            "variant_in_literature": variant in literature,
            "supported_by_literature": supported,
            "supported_by_cooccurrence": co_supported,
            "label_comparable": comparable,
            "label_agrees": comparable and llm_label == prediction,
        })
    df = pd.DataFrame(rows, columns=["level", "evidence_type", "variant_in_literature", "supported_by_literature",
                                     "supported_by_cooccurrence", "label_comparable", "label_agrees"])
    table = (df.groupby(["evidence_type", "level"])
             .agg(associations=("level", "size"), variant_in_literature=("variant_in_literature", "sum"),
                  supported_by_literature=("supported_by_literature", "sum"),
                  supported_by_cooccurrence=("supported_by_cooccurrence", "sum"),
                  label_comparable=("label_comparable", "sum"), label_agrees=("label_agrees", "sum"))
             .reset_index())
    if len(table):
        table["supported_pct"] = (100 * table["supported_by_literature"] / table["associations"]).round(1)
    n = len(df)
    summary = {
        "curated_associations": n,
        "variant_in_literature_pct": round(100 * df["variant_in_literature"].mean(), 1) if n else None,
        "supported_by_literature_pct": round(100 * df["supported_by_literature"].mean(), 1) if n else None,
        "supported_by_cooccurrence_pct": round(100 * df["supported_by_cooccurrence"].mean(), 1) if n else None,
        "label_agreement_pct": (round(100 * df["label_agrees"].sum() / df["label_comparable"].sum(), 1)
                                if n and df["label_comparable"].sum() else None),
        "civic_records_skipped": dict(skipped),
    }
    return table, summary
