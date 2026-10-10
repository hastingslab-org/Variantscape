"""Stage orchestration. Every stage only processes papers it has not processed before.

This file can be run directly: ``python variantscape_pipeline/pipeline.py run --mode full``
(equivalent to ``python -m variantscape_pipeline run --mode full``).
"""

from __future__ import annotations

if __package__ in (None, ""):
    # Executed as a script: make the package importable so the relative imports below work
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "variantscape_pipeline"

import logging
import shutil
from datetime import date, timedelta
from functools import cached_property
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from . import build as build_mod
from .cancers import CancerExtractor, CancerMapper
from .clean import clean_batch, dedupe_key
from .coassociation import compute_consensus
from .config import STUDY_DESIGN_CATEGORIES, Settings
from .curated import add_curated_to_graph, build_curated_records, coverage_report, merge_consensus, records_frame
from .fetch import iter_openalex
from .genes import GeneSet, make_gene_matcher, oncology_genes, resolve_gene_set
from .llm import LLMClient, run_parallel
from .reference import Reference
from .store import Store
from .study_design import build_prompt as design_prompt
from .study_design import parse_label
from .treatments import TreatmentMatcher
from .variants import PROMPT_ID as VARIANT_PROMPT_ID
from .variants import VariantNormalizer, gene_alias_table
from .variants import build_prompt as variant_prompt
from .verification import PROMPT_ID as VERIFY_PROMPT_ID
from .verification import Candidate, candidates_from_json, candidates_to_json, make_candidates, max_tokens_for
from .verification import build_prompt as verify_prompt

log = logging.getLogger(__name__)

STAGES = ["fetch", "clean", "genes", "cancers", "treatments", "variants", "study_design", "verify", "build"]
LLM_STAGES = {"variants", "study_design", "verify"}


class Pipeline:
    def __init__(self, settings: Settings, store: Store, run_id: str,
                 limit: int | None = None, refresh_reference: bool = True):
        self.settings = settings
        self.store = store
        self.run_id = run_id
        self.limit = limit
        self.refresh_reference = refresh_reference
        self.summary: dict = {}
        self.fetched_through: str | None = None

    # ------------------------------------------------------------------ #
    # Lazily loaded resources
    # ------------------------------------------------------------------ #
    @cached_property
    def reference(self) -> Reference:
        return Reference.load(self.settings.reference_dir, refresh=self.refresh_reference)

    @cached_property
    def cancer_mapper(self) -> CancerMapper:
        return CancerMapper(self.reference, use_synonyms=self.settings.use_cancer_synonyms)

    @cached_property
    def normalizer(self) -> VariantNormalizer:
        return VariantNormalizer(self.reference.variants, hgnc=self.reference.hgnc)

    @cached_property
    def gene_set(self) -> GeneSet:
        gene_set = resolve_gene_set(self.settings.gene_set, self.reference)
        log.info("Gene set: %s (%d genes%s)", gene_set.name, len(gene_set.genes),
                 ", variants restricted to the set" if gene_set.restrict_variants else "")
        return gene_set

    @cached_property
    def llm(self) -> LLMClient:
        s = self.settings
        return LLMClient(s.llm_api_key, s.llm_base_url, s.llm_model, s.llm_temperature, s.llm_timeout)

    def _state(self) -> build_mod.PipelineState:
        return build_mod.compute_pipeline_state(self.store, self.cancer_mapper, self.normalizer, self.gene_set)

    def _limited(self, ids: Sequence[str]) -> list[str]:
        ids = sorted(ids)
        return ids[: self.limit] if self.limit else ids

    # ------------------------------------------------------------------ #
    # Stages
    # ------------------------------------------------------------------ #
    def fetch(self, from_date: str, to_date: str, batch_size: int = 1000) -> None:
        """Fetch works month by month; months completed earlier in this run are skipped on resume."""
        done = self.store.fetched_windows(self.run_id)
        windows = month_windows(from_date, to_date)
        total_seen = total_new = 0
        for n, (w_from, w_to) in enumerate(windows, start=1):
            if (w_from, w_to) in done:
                continue
            remaining = self.limit - total_seen if self.limit else None
            if remaining is not None and remaining <= 0:
                break
            batch, seen, new = [], 0, 0

            def flush():
                nonlocal new, batch
                known = self.store.existing_ids(r["paper_id"] for r in batch)
                new += self.store.insert_papers([r for r in batch if r["paper_id"] not in known], self.run_id)
                batch = []

            for record in iter_openalex(self.settings.search_term, w_from, w_to, self.settings.openalex_email,
                                        self.settings.openalex_api_key, max_records=remaining):
                batch.append(record)
                seen += 1
                if len(batch) >= batch_size:
                    flush()
            if batch:
                flush()
            if remaining is None or seen < remaining:  # a window cut short by --limit is not complete
                self.store.mark_window_fetched(self.run_id, w_from, w_to, seen, new)
            total_seen += seen
            total_new += new
            log.info("Fetch %s..%s (%d/%d): %d works, %d new, %d already in the database",
                     w_from, w_to, n, len(windows), seen, new, seen - new)
        # The run only counts as covering the months fetched completely, in order from the start;
        # a window cut short (e.g. by --limit) is fetched again by the next run
        done = self.store.fetched_windows(self.run_id)
        self.fetched_through = None
        for window in windows:
            if window not in done:
                break
            self.fetched_through = window[1]
        self.summary["fetch"] = {"from_date": from_date, "to_date": to_date, "fetched_through": self.fetched_through,
                                 "returned": total_seen, "new_papers": total_new}
        log.info("Fetch: %d works returned, %d new, %d already in the database",
                 total_seen, total_new, total_seen - total_new)

    def clean(self, batch_size: int = 5000) -> None:
        kept_total, rejected_total = 0, {}
        while True:
            df = self.store.papers_by_status("fetched", limit=batch_size)
            if df.empty:
                break
            known = self.store.existing_dedupe_keys(dedupe_key(t, a) for t, a in zip(df["title"], df["authors"]))
            kept, rejected = clean_batch(df, known)
            self.store.update_clean(kept, rejected)
            kept_total += len(kept)
            for reason in rejected.values():
                rejected_total[reason] = rejected_total.get(reason, 0) + 1
        self.summary["clean"] = {"kept": kept_total, "rejected": rejected_total}
        log.info("Clean: %d kept, %d rejected %s", kept_total, sum(rejected_total.values()), rejected_total)

    def _run_text_stage(self, stage: str, candidate_sql: str, process: Callable, batch_size: int) -> int:
        done = self.store.ids_done(stage)
        ids = [r[0] for r in self.store.conn.execute(candidate_sql) if r[0] not in done]
        log.info("%s: %d papers to process", stage, len(ids))
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            texts = self.store.texts(batch_ids)
            process(texts)
            self.store.mark_done(texts["paper_id"], stage)
            log.info("%s: %d/%d", stage, min(i + batch_size, len(ids)), len(ids))
        return len(ids)

    def genes(self) -> None:
        s = self.settings
        matcher = None

        def process(texts):
            nonlocal matcher
            # Detect the full oncology set (plus panel genes) so the gene set can change without reprocessing
            detectable = sorted(oncology_genes(self.reference) | self.gene_set.genes)
            matcher = matcher or make_gene_matcher(s.gene_filter_method, detectable, s.biobert_model, s.device)
            hits = matcher.extract(texts["title"].fillna("").tolist(), texts["abstract"].fillna("").tolist())
            self.store.insert_pairs("gene_hits", "gene", [(pid, g) for pid, gs in zip(texts["paper_id"], hits) for g in gs])

        n = self._run_text_stage("genes", "SELECT paper_id FROM papers WHERE status='clean'", process, 256)
        self.summary["genes"] = {"processed": n}

    def cancers(self) -> None:
        extractor = None

        def process(texts):
            nonlocal extractor
            extractor = extractor or CancerExtractor()
            terms = extractor.extract((texts["title"].fillna("") + " " + texts["abstract"].fillna("")).tolist())
            self.store.insert_pairs("cancer_terms", "term",
                                    [(pid, t) for pid, ts in zip(texts["paper_id"], terms) for t in ts])

        n = self._run_text_stage("cancers", "SELECT DISTINCT paper_id FROM gene_hits", process, 512)
        self.summary["cancers"] = {"processed": n}

    def treatments(self) -> None:
        matcher = TreatmentMatcher(self.reference.therapies)

        def process(texts):
            self.store.insert_pairs("treatment_hits", "treatment", [
                (pid, t) for pid, title, abstract in zip(texts["paper_id"], texts["title"].fillna(""), texts["abstract"].fillna(""))
                for t in matcher.match(title, abstract)
            ])

        n = self._run_text_stage("treatments", "SELECT DISTINCT paper_id FROM gene_hits", process, 2000)
        self.summary["treatments"] = {"processed": n}

    def _run_llm_stage(self, stage: str, ids: list[str], make_prompt: Callable, save: Callable,
                       client: LLMClient | None = None, max_tokens: Callable | None = None) -> None:
        texts = self.store.texts(ids).set_index("paper_id")
        client = client or self.llm
        ok = failed = 0

        def call(pid):
            row = texts.loc[pid]
            return client.complete(make_prompt(pid, row["title"] or "", row["abstract"] or ""),
                                   max_tokens=max_tokens(pid) if max_tokens else None)

        for n, (pid, response) in enumerate(run_parallel(ids, call, self.settings.llm_max_workers), start=1):
            if response is None:
                failed += 1
            else:
                save(pid, response)
                ok += 1
            if n % 25 == 0:
                self.store.commit()
                log.info("%s: %d/%d (%d failed)", stage, n, len(ids), failed)
        self.store.commit()
        self.summary[stage] = {"candidates": len(ids), "answered": ok, "failed": failed}
        log.info("%s: %d answered, %d failed (will be retried next run)", stage, ok, failed)

    def variants(self) -> None:
        # Papers answered with an older prompt version are extracted again
        done = {r[0] for r in self.store.conn.execute("SELECT paper_id FROM variant_llm WHERE prompt_id=?",
                                                      (VARIANT_PROMPT_ID,))}
        ids = self._limited(self._state().variant_candidates - done)
        model = self.settings.llm_model
        self._run_llm_stage(
            "variants", ids,
            lambda pid, title, abstract: variant_prompt(title, abstract),
            lambda pid, response: self.store.save_variant_response(pid, model, VARIANT_PROMPT_ID, response),
        )

    def study_design(self) -> None:
        done = {r[0] for r in self.store.conn.execute("SELECT paper_id FROM study_design")}
        ids = self._limited(set(self._state().included) - done)
        model = self.settings.llm_model
        self._run_llm_stage(
            "study_design", ids,
            lambda pid, title, abstract: design_prompt(title, abstract, STUDY_DESIGN_CATEGORIES),
            lambda pid, response: self.store.save_study_design(
                pid, model, parse_label(response, STUDY_DESIGN_CATEGORIES), response),
        )

    @cached_property
    def verify_llm(self) -> LLMClient:
        s = self.settings
        if s.llm_verify_model == s.llm_model and s.llm_verify_temperature == s.llm_temperature:
            return self.llm
        return LLMClient(s.llm_api_key, s.llm_base_url, s.llm_verify_model, s.llm_verify_temperature, s.llm_timeout)

    def verify(self) -> None:
        """Verify each included paper's mined associations against its abstract.

        A paper is (re)verified when it has no verification with the current prompt,
        or when its current candidates are not all among those verified before
        (e.g. after a normalization change or new CIViC aliases).
        """
        included = self._state().included
        verified_before = {
            pid: {Candidate(self.normalizer.canonical_node(c.variant), c.cancer, c.treatment).key
                  for c in candidates_from_json(cands)}
            for pid, prompt_id, cands in self.store.conn.execute(
                "SELECT paper_id, prompt_id, candidates_json FROM verify_llm")
            if prompt_id == VERIFY_PROMPT_ID
        }
        candidates, truncated = {}, {}
        for pid, ents in included.items():
            candidates[pid], truncated[pid] = make_candidates(ents["Variant"], ents["Cancer"], ents["Treatment"])
        todo = [pid for pid in included
                if pid not in verified_before or not {c.key for c in candidates[pid]} <= verified_before[pid]]
        ids = self._limited(todo)
        log.info("verify: %d included papers, %d to (re)verify (%d with capped candidates)",
                 len(included), len(todo), sum(truncated[pid] for pid in todo))
        model = self.settings.llm_verify_model
        self._run_llm_stage(
            "verify", ids,
            lambda pid, title, abstract: verify_prompt(title, abstract, candidates[pid]),
            lambda pid, response: self.store.save_verification(
                pid, model, VERIFY_PROMPT_ID, candidates_to_json(candidates[pid]), truncated[pid], response),
            client=self.verify_llm,
            max_tokens=lambda pid: max_tokens_for(candidates[pid]),
        )

    def build(self, deploy_to=None, html: bool = False, force: bool = False) -> None:
        entities = self._state().included
        designs = {pid: label for pid, label in self.store.conn.execute("SELECT paper_id, label FROM study_design")}
        verdicts = build_mod.collect_verdicts(self.store, self.normalizer, entities)
        G, graph_stats = build_mod.build_graph(entities, designs, verdicts)
        verified_table = build_mod.verified_associations(verdicts, designs)

        votes = build_mod.collect_votes(verdicts)
        consensus = compute_consensus(votes)

        # Curated CIViC associations: added with their own provenance; the coverage
        # report compares them with the literature-only graph
        literature_graph = G.copy()
        records, skipped = build_curated_records(self.reference, self.cancer_mapper, self.normalizer, self.gene_set)
        curated_stats = add_curated_to_graph(G, records)
        build_mod.annotate_variant_nodes(G, self.normalizer)
        build_mod.annotate_cancer_nodes(G, self.cancer_mapper)
        coverage_table, coverage = coverage_report(records, literature_graph, consensus, skipped)
        self.summary["build"] = {
            "gene_set": self.gene_set.name,
            "gene_set_size": len(self.gene_set.genes),
            "literature_graph": graph_stats,
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "consensus_pairs": len(consensus),
            "consensus_labels": consensus["Resolved_Prediction"].value_counts().to_dict(),
            "variants_dropped_not_in_text": self.normalizer.ungrounded,
            "verification": verification_stats(verdicts, entities),
            "curated": {"records": len(records), **curated_stats, **coverage},
            "cancer_types": cancer_type_stats(G, self.cancer_mapper),
            "reference_snapshot": str(self.reference.source_dir),
        }

        out_dir = self.settings.output_dir / self.run_id
        build_mod.write_outputs(
            out_dir, G, merge_consensus(consensus, records), entities, designs, votes,
            self.cancer_mapper.synonym_table(), self.summary,
            tables={"curated_associations.csv": records_frame(records, literature_graph, consensus),
                    "curated_coverage.csv": coverage_table,
                    "verified_associations.csv": verified_table,
                    "gene_aliases.csv": self.gene_alias_frame(G),
                    "cancer_mapping.csv": pd.DataFrame(self.cancer_mapper.mapping_table())},
        )
        if html:
            from .network_html import write_network_html
            write_network_html(G, out_dir / "variantscape_network_graph.html")
        latest = self.settings.output_dir / "latest"
        shutil.rmtree(latest, ignore_errors=True)
        shutil.copytree(out_dir, latest)
        log.info("Build: literature graph %s; with curated CIViC: %d nodes, %d edges -> %s",
                 graph_stats, G.number_of_nodes(), G.number_of_edges(), out_dir)
        log.info("Curated coverage: %s", {k: v for k, v in coverage.items() if k != "civic_records_skipped"})

        if deploy_to:
            if G.number_of_nodes() == 0:
                raise RuntimeError("Refusing to deploy an empty graph")
            problem = build_mod.check_against_deployed(G, deploy_to / "variantscape" / "network_graph_weighted.gml")
            if problem and not force:
                raise RuntimeError(f"Refusing to deploy: {problem}. Re-run with --force to deploy anyway.")
            written = build_mod.deploy(out_dir, deploy_to, include_html=html)
            self.summary["deployed"] = [str(p) for p in written]
            log.info("Deployed to %s", deploy_to)

    def gene_alias_frame(self, G):
        """Aliases of the graph's variant genes, for searching EvidenceDb by alias (MEK1 -> MAP2K1)."""
        self.normalizer  # configures the HGNC symbols
        genes = {n.rpartition("_")[2] for n, d in G.nodes(data=True) if d.get("category") == "Variant"}
        return pd.DataFrame(gene_alias_table(genes), columns=["alias", "symbol"])


def cancer_type_stats(G, mapper: CancerMapper) -> dict:
    """Cancer nodes and how the CIViC diseases were mapped to OncoTree (for summary.json)."""
    table = pd.DataFrame(mapper.mapping_table())
    nodes = [d for _, d in G.nodes(data=True) if d.get("category") == "Cancer"]
    return {
        "cancer_nodes": len(nodes),
        "cancer_nodes_without_oncotree": sum(1 for d in nodes if not d.get("oncotree_code")),
        "civic_diseases_by_method": table["method"].value_counts().to_dict() if len(table) else {},
        "civic_diseases_dropped": int((table["node"] == "").sum()) if len(table) else 0,
    }


def verification_stats(verdicts: dict, entities: dict) -> dict:
    """Counts of the verifier's verdicts (for summary.json)."""
    relations: dict[str, int] = {}
    verified = quote_rejected = 0
    for vs in verdicts.values():
        for v in vs:
            relations[v.relation] = relations.get(v.relation, 0) + 1
            verified += v.verified
            quote_rejected += (v.relation != "Not reported" and not v.quote_ok)
    return {
        "papers_included": len(entities),
        "papers_verified": len(verdicts),
        "verdicts": sum(relations.values()),
        "verified": verified,
        "rejected_quote_not_found": quote_rejected,
        "relations": relations,
    }


def check_requirements(settings: Settings, stages: Sequence[str], html: bool = False) -> list[str]:
    """Return problems (missing packages, models, API key) that would make the given stages fail."""
    import importlib.util

    needed: dict[str, str] = {}
    if "clean" in stages:
        needed.update({"langdetect": "langdetect", "cleantext": "clean-text"})
    if "genes" in stages and settings.gene_filter_method == "biobert":
        needed.update({"torch": "torch", "transformers": "transformers"})
    if "cancers" in stages:
        needed.update({"spacy": "scispacy", "scispacy": "scispacy",
                       "en_ner_bionlp13cg_md": "the en_ner_bionlp13cg_md model (see requirements-pipeline.txt)",
                       "en_ner_bc5cdr_md": "the en_ner_bc5cdr_md model (see requirements-pipeline.txt)"})
    if LLM_STAGES & set(stages):
        needed["openai"] = "openai"
    if html and "build" in stages:
        needed["plotly"] = "plotly"
    problems = [f"missing Python package '{module}' (install {package})"
                for module, package in needed.items() if importlib.util.find_spec(module) is None]
    if LLM_STAGES & set(stages) and not settings.llm_api_key:
        problems.append("no LLM API key (set LLM_API_KEY in .env, or use --skip-llm)")
    if settings.gene_set not in {"oncology", "civic"}:
        if not Path(settings.gene_set).is_file():
            problems.append(f"gene set panel file not found: {settings.gene_set}")
    if settings.gene_filter_method not in {"biobert", "string"}:
        problems.append(f"VARIANTSCAPE_GENE_FILTER must be 'biobert' or 'string', not {settings.gene_filter_method!r}")
    return problems


def month_windows(from_date: str, to_date: str) -> list[tuple[str, str]]:
    """Split an inclusive date range into calendar-month windows."""
    start, end = date.fromisoformat(from_date), date.fromisoformat(to_date)
    windows = []
    while start <= end:
        next_month = (start.replace(day=1) + timedelta(days=32)).replace(day=1)
        windows.append((start.isoformat(), min(next_month - timedelta(days=1), end).isoformat()))
        start = next_month
    return windows


if __name__ == "__main__":
    import sys

    from variantscape_pipeline.cli import main

    sys.exit(main())
