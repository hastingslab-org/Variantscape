"""Assemble per-paper entities and build the EvidenceDb artifacts (06.01, 06.02, 06.04)."""

from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import networkx as nx
import pandas as pd

from .cancers import CancerMapper
from .coassociation import compute_consensus, pair_key
from .genes import GeneSet
from .config import study_weight
from .store import Store
from .variants import VariantNormalizer, variant_kind
from .verification import Candidate, Verdict, candidates_from_json, parse_response

log = logging.getLogger(__name__)

CATEGORIES = ("Variant", "Cancer", "Treatment")
SOURCE_LITERATURE = "literature"      # verified in the abstract
SOURCE_COOCCURRENCE = "cooccurrence"  # only co-mentioned
ARTIFACTS = ("network_graph_weighted.gml", "final_variant_treatment_consensus.csv", "metadata_mapping_transposed.csv",
             "curated_associations.csv", "verified_associations.csv", "gene_aliases.csv", "cancer_synonyms.csv")


@dataclass
class PipelineState:
    """Per-paper entities derived from the stored stage results."""

    genes: set[str] = field(default_factory=set)
    cancers: dict[str, set[str]] = field(default_factory=dict)
    treatments: dict[str, set[str]] = field(default_factory=dict)
    variants: dict[str, set[str]] = field(default_factory=dict)

    @property
    def variant_candidates(self) -> set[str]:
        """Papers that need LLM variant extraction: gene, cancer and treatment found."""
        return self.genes & set(self.cancers) & set(self.treatments)

    @property
    def included(self) -> dict[str, dict[str, set[str]]]:
        """Papers with all of gene, cancer, treatment and variant mentions (06.01 filter)."""
        ids = self.variant_candidates & set(self.variants)
        return {
            pid: {"Variant": self.variants[pid], "Cancer": self.cancers[pid], "Treatment": self.treatments[pid]}
            for pid in ids
        }


def compute_pipeline_state(store: Store, mapper: CancerMapper, normalizer: VariantNormalizer,
                           gene_set: GeneSet | None = None) -> PipelineState:
    """Per-paper entities under the active gene set.

    Papers count only if they mention a gene of ``gene_set``; for restricting sets
    (CIViC, panels) variants in other genes are dropped as well.
    """
    state = PipelineState()
    state.genes = {pid for pid, gene in store.conn.execute("SELECT paper_id, gene FROM gene_hits")
                   if gene_set is None or gene_set.contains(gene)}
    state.treatments = {pid: t for pid, t in store.pairs_by_paper("treatment_hits", "treatment").items() if pid in state.genes}

    # Map each distinct raw term once, then assemble per paper
    term_map: dict[str, set[str]] = {}
    for pid, term in store.conn.execute("SELECT paper_id, term FROM cancer_terms"):
        if pid not in state.genes:
            continue
        if term not in term_map:
            term_map[term] = mapper.map_terms([term])
        if term_map[term]:
            state.cancers.setdefault(pid, set()).update(term_map[term])

    # Grounding check: only variants that occur in the paper's title/abstract count
    responses = store.conn.execute("SELECT paper_id, response FROM variant_llm").fetchall()
    texts = store.texts([pid for pid, _ in responses])
    text_by_id = dict(zip(texts["paper_id"], texts["title"].fillna("") + " " + texts["abstract"].fillna("")))
    state.variants = normalizer.nodes_from_responses(responses, text_by_id)
    if gene_set is not None and gene_set.restrict_variants:
        restricted = {pid: {v for v in nodes if gene_set.contains(v.rsplit("_", 1)[-1])}
                      for pid, nodes in state.variants.items()}
        state.variants = {pid: nodes for pid, nodes in restricted.items() if nodes}
    return state


def collect_verdicts(store: Store, normalizer: VariantNormalizer,
                     entities: dict[str, dict[str, set[str]]]) -> dict[str, list[Verdict]]:
    """Parsed verifier verdicts per included paper, restricted to its current candidates.

    Stored candidate variants are re-canonicalized, so normalization changes keep
    their verdicts; verdicts for associations no longer among the paper's mined
    entities are ignored.
    """
    rows = [r for r in store.conn.execute("SELECT paper_id, candidates_json, response FROM verify_llm")
            if r[0] in entities]
    texts = store.texts([r[0] for r in rows]).set_index("paper_id") if rows else None
    out: dict[str, list[Verdict]] = {}
    for pid, candidates_json, response in rows:
        ents = entities[pid]
        current = {(v, c, t) for v in ents["Variant"] for c in ents["Cancer"] for t in ents["Treatment"] | {""}}
        title, abstract = (texts.loc[pid, "title"] or "", texts.loc[pid, "abstract"] or "") if pid in texts.index else ("", "")
        verdicts = []
        for verdict in parse_response(response, candidates_from_json(candidates_json), title, abstract):
            c = verdict.candidate
            candidate = Candidate(normalizer.canonical_node(c.variant), c.cancer, c.treatment)
            if candidate.key in current:
                verdicts.append(Verdict(candidate, verdict.relation, verdict.quote, verdict.quote_ok))
        out[pid] = verdicts
    return out


def collect_votes(verdicts: dict[str, list[Verdict]]) -> pd.DataFrame:
    """Per-paper variant-treatment labels from verified associations (for the consensus)."""
    rows = sorted({(pid, pair_key(v.candidate.variant, v.candidate.treatment), v.relation)
                   for pid, vs in verdicts.items() for v in vs if v.verified and v.candidate.treatment})
    return pd.DataFrame(rows, columns=["PaperId", "Variant_Treatment_Pair", "Prediction"])


def _cross_category_pairs(ents: dict[str, set[str]]) -> set[tuple[str, str]]:
    pairs = set()
    for cat_a, cat_b in (("Variant", "Cancer"), ("Variant", "Treatment"), ("Cancer", "Treatment")):
        for a in ents[cat_a]:
            for b in ents[cat_b]:
                pairs.add(tuple(sorted((a, b))))
    return pairs


def _verified_pairs(verdicts: list[Verdict]) -> set[tuple[str, str]]:
    pairs = set()
    for v in verdicts:
        if not v.verified:
            continue
        c = v.candidate
        pairs.add(tuple(sorted((c.variant, c.cancer))))
        if c.treatment:
            pairs.add(tuple(sorted((c.variant, c.treatment))))
            pairs.add(tuple(sorted((c.cancer, c.treatment))))
    return pairs


def build_graph(entities: dict[str, dict[str, set[str]]], designs: dict[str, str],
                verdicts: dict[str, list[Verdict]] | None = None) -> tuple[nx.Graph, dict]:
    """Literature graph weighted by study design (06.04), split into two tiers.

    - ``weight``: study-weighted count of papers whose abstract was verified to
      report the association (``verified_papers`` papers).
    - ``cooccurrence_weight``: study-weighted count of papers merely mentioning
      both entities (the notebooks' weight; the unverified tier).

    Only variant-cancer, variant-treatment and cancer-treatment edges are built
    (06.04 also linked variants to variants, cancers to cancers, ...).
    """
    verdicts = verdicts or {}
    category_of: dict[str, str] = {}
    cooccurrence: dict[tuple[str, str], float] = defaultdict(float)
    verified: dict[tuple[str, str], float] = defaultdict(float)
    verified_papers: dict[tuple[str, str], int] = defaultdict(int)
    conflicts = missing_design = papers_verified = 0
    for pid, ents in entities.items():
        if pid not in designs:
            missing_design += 1
        w = study_weight(designs.get(pid))
        clean: dict[str, set[str]] = {}
        for category in CATEGORIES:
            clean[category] = set()
            for entity in ents[category]:
                if category_of.setdefault(entity, category) != category:
                    conflicts += 1
                    continue
                clean[category].add(entity)
        pairs = _cross_category_pairs(clean)
        for pair in pairs:
            cooccurrence[pair] += w
        paper_verified = _verified_pairs(verdicts.get(pid, [])) & pairs
        papers_verified += bool(paper_verified)
        for pair in paper_verified:
            verified[pair] += w
            verified_papers[pair] += 1

    G = nx.Graph()
    for (a, b), co in cooccurrence.items():
        for n in (a, b):
            if n not in G:
                G.add_node(n, category=category_of[n], sources=SOURCE_COOCCURRENCE)
        w = round(verified.get((a, b), 0.0), 6)
        source = SOURCE_LITERATURE if w > 0 else SOURCE_COOCCURRENCE
        G.add_edge(a, b, weight=w, cooccurrence_weight=round(co, 6), verified_papers=verified_papers.get((a, b), 0),
                   sources=source)
        if w > 0:
            G.nodes[a]["sources"] = G.nodes[b]["sources"] = SOURCE_LITERATURE
    stats = {
        "papers": len(entities),
        "papers_with_verified_associations": papers_verified,
        "papers_without_study_design": missing_design,
        "entity_category_conflicts": conflicts,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "verified_edges": sum(1 for _, _, d in G.edges(data=True) if d["weight"] > 0),
        **{f"{c.lower()}_nodes": sum(1 for _, d in G.nodes(data=True) if d["category"] == c) for c in CATEGORIES},
    }
    return G, stats


def annotate_variant_nodes(G: nx.Graph, normalizer: VariantNormalizer) -> None:
    """Add ``variant_class`` (specific, fusion, amplification, itd, codon) and, for
    fusion nodes, the observed ``fusion_partners`` to the variant nodes."""
    for node, data in G.nodes(data=True):
        if data.get("category") != "Variant":
            continue
        data["variant_class"] = variant_kind(node)
        if data["variant_class"] == "fusion":
            data["fusion_partners"] = ";".join(sorted(normalizer.fusion_partners.get(node, ())))


def annotate_cancer_nodes(G: nx.Graph, mapper) -> None:
    """Add the OncoTree code, lineage (tissue -> ... -> type codes), tissue and main type
    to the cancer nodes; EvidenceDb rolls subtypes up into broader types with the lineage."""
    for node, data in G.nodes(data=True):
        if data.get("category") == "Cancer":
            data.update(mapper.node_attributes(node))


def verified_associations(verdicts: dict[str, list[Verdict]], designs: dict[str, str]) -> pd.DataFrame:
    """``verified_associations.csv``: one row per verified (variant, cancer, treatment) or (variant, cancer).

    ``weight`` is the study-weighted number of papers reporting it; ``relation``
    the consensus of their labels; an example quote comes from the
    highest-weighted paper.
    """
    groups: dict[tuple[str, str, str], list[tuple[str, Verdict]]] = defaultdict(list)
    for pid, vs in verdicts.items():
        for v in vs:
            if v.verified:
                groups[v.candidate.key].append((pid, v))
    columns = ["variant", "gene", "cancer", "treatment", "relation", "weight", "papers", "label_counts",
               "paper_ids", "example_quote", "example_paper_id"]
    if not groups:
        return pd.DataFrame(columns=columns)
    votes = pd.DataFrame([("|".join(k), v.relation) for k, items in groups.items() for _, v in items],
                         columns=["Key", "Prediction"])
    relation = dict(compute_consensus(votes, key="Key").values)
    rows = []
    for key, items in groups.items():
        items = sorted(items, key=lambda x: (-study_weight(designs.get(x[0])), x[0]))
        labels: dict[str, int] = defaultdict(int)
        for _, v in items:
            labels[v.relation] += 1
        variant, cancer, treatment = key
        rows.append({
            "variant": variant, "gene": variant.rpartition("_")[2], "cancer": cancer, "treatment": treatment,
            "relation": relation["|".join(key)],
            "weight": round(sum(study_weight(designs.get(pid)) for pid, _ in items), 6),
            "papers": len(items), "label_counts": json.dumps(dict(labels)),
            "paper_ids": ";".join(pid for pid, _ in items[:50]),
            "example_quote": items[0][1].quote, "example_paper_id": items[0][0],
        })
    return pd.DataFrame(rows, columns=columns).sort_values(["weight", "variant"], ascending=[False, True])


def write_outputs(out_dir: Path, G: nx.Graph, consensus: pd.DataFrame, entities: dict, designs: dict,
                  votes: pd.DataFrame, cancer_synonyms: dict[str, list[str]],
                  summary: dict, tables: dict[str, pd.DataFrame] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (tables or {}).items():
        table.to_csv(out_dir / name, index=False)
    nx.write_gml(G, out_dir / "network_graph_weighted.gml")
    consensus.to_csv(out_dir / "final_variant_treatment_consensus.csv", index=False)
    pd.DataFrame(
        sorted(((n, d["category"]) for n, d in G.nodes(data=True)), key=lambda x: (x[1], x[0])),
        columns=["Entity", "Category"],
    ).to_csv(out_dir / "metadata_mapping_transposed.csv", index=False)

    # Names and synonyms per cancer node, for EvidenceDb's cancer search
    pd.DataFrame(
        [(name.capitalize(), str(syns)) for name, syns in sorted(cancer_synonyms.items())
         if name in G and G.nodes[name]["category"] == "Cancer"],
        columns=["name", "synonyms"],
    ).to_csv(out_dir / "cancer_synonyms.csv", index=False)

    # Supporting outputs (not read by EvidenceDb)
    pd.DataFrame(
        [(pid, designs.get(pid), study_weight(designs.get(pid)), cat, e)
         for pid, ents in entities.items() for cat in CATEGORIES for e in sorted(ents[cat])],
        columns=["PaperId", "Study_design", "Study_weight", "Category", "Entity"],
    ).to_csv(out_dir / "paper_entities.csv", index=False)
    votes.to_csv(out_dir / "variant_treatment_votes.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


def check_against_deployed(G: nx.Graph, deployed_graph: Path, max_drop: float = 0.2) -> str | None:
    """Return a problem description if the new graph is much smaller than the deployed one."""
    if not deployed_graph.exists():
        return None
    old = nx.read_gml(deployed_graph)
    for label, new_n, old_n in (("nodes", G.number_of_nodes(), old.number_of_nodes()),
                                ("edges", G.number_of_edges(), old.number_of_edges())):
        if old_n and new_n < (1 - max_drop) * old_n:
            return f"new graph has {new_n} {label} vs {old_n} deployed (> {max_drop:.0%} drop)"
    return None


def deploy(out_dir: Path, evidencedb_dir: Path, include_html: bool) -> list[Path]:
    """Copy artifacts into an EvidenceDb checkout, keeping timestamped backups."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    targets = [(out_dir / name, evidencedb_dir / "variantscape" / name) for name in ARTIFACTS]
    if include_html and (out_dir / "variantscape_network_graph.html").exists():
        targets.append((out_dir / "variantscape_network_graph.html",
                        evidencedb_dir / "static" / "variantscape_network_graph.html"))
    written = []
    for src, dst in targets:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            backup_dir = dst.parent / "backups"
            backup_dir.mkdir(exist_ok=True)
            shutil.copy2(dst, backup_dir / f"{dst.name}.{stamp}")
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copy2(src, tmp)
        tmp.replace(dst)
        written.append(dst)
    return written
