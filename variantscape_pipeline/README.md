# Variantscape pipeline (re-runnable)

`variantscape_pipeline` packages the core of the notebooks in `notebooks/01`–`06`
into one incremental command that can run unattended (for example monthly) and
produces the files EvidenceDb reads:

| File | Used by EvidenceDb for |
|---|---|
| `network_graph_weighted.gml` | variant / cancer / treatment co-occurrence graph, weighted by study design |
| `final_variant_treatment_consensus.csv` | Sensitive / Resistant / … label per variant–treatment pair |
| `metadata_mapping_transposed.csv` | entity → category (`Variant`, `Cancer`, `Treatment`) for autosuggest |
| `curated_associations.csv` | expert-curated CIViC associations (disease-specific), listed first in EvidenceDb |
| `verified_associations.csv` | literature associations verified against the abstracts (variant, cancer, treatment, relation, papers, example quote) |

## Setup

```bash
pip install -r requirements-pipeline.txt
cp pipeline.env.example .env   # fill in OPENALEX_EMAIL and LLM_API_KEY
```

## Usage

Run `variantscape_pipeline/pipeline.py` directly (from any directory), or use
the equivalent `python -m variantscape_pipeline` from the Variantscape folder.
The `.env` next to the package and the default `pipeline_data/` folder are
found relative to the Variantscape folder, wherever you start from.

The pipeline has two modes:

- **`--mode full`** archives the current database (to `pipeline_data/archive/`)
  and rebuilds everything from scratch for works published from
  `VARIANTSCAPE_FULL_START_DATE` (default 2014-01-01) until today.
- **`--mode incremental`** (the default) continues from the end of the last
  successful run, minus `VARIANTSCAPE_OVERLAP_DAYS` (default 30) to catch works
  that OpenAlex indexes late, up to today. Already-known works are skipped, and
  only new papers go through the stages and the LLM. The graph and consensus are
  then rebuilt over all papers.

```bash
# First run (or any later full refresh)
python variantscape_pipeline/pipeline.py run --mode full

# Monthly update, deployed into the EvidenceDb checkout
python variantscape_pipeline/pipeline.py run --deploy-to ../evidence-database

# Continue an interrupted or failed run where it stopped
python variantscape_pipeline/pipeline.py run --resume

# Cheap smoke test: explicit window, few papers, no LLM calls
python variantscape_pipeline/pipeline.py run --from-date 2026-09-01 --to-date 2026-09-07 --limit 500 --skip-llm

# Only rebuild the artifacts from stored results (e.g. after a rule change)
python variantscape_pipeline/pipeline.py build

python variantscape_pipeline/pipeline.py status
```

Works are fetched month by month, and each completed month is recorded. After
an interruption, `--resume` reuses the run's window, mode and stages, skips the
months it already fetched, and each stage picks up the papers it hasn't
processed yet. A full run is a few hundred thousand LLM calls (cheap on
DeepInfra), but BioBERT/SciSpaCy take days on a CPU, so a GPU
(`VARIANTSCAPE_DEVICE=0`) helps.

A run only counts as covering the months it fetched completely. If `--limit`
cuts a month short, the next incremental run fetches that month again.

### Gene set

The gene set decides which papers are analysed: a paper must mention one of
its genes. Choose it with `--gene-set` or `VARIANTSCAPE_GENE_SET`.

| Value | Genes | Variants in the graph |
|---|---|---|
| `oncology` (default) | all oncology-relevant genes: the OncoKB Cancer Gene List merged with all CIViC genes (~1,560) | any gene the LLM reports |
| `civic` | CIViC genes (~750), the notebooks' gene list | only CIViC genes |
| path to a panel file | e.g. the Oncomine panel: one symbol per line, or the first CSV column | only panel genes |

```bash
python variantscape_pipeline/pipeline.py build --gene-set path/to/oncomine_ngs_panel.csv
```

The genes stage always detects the full oncology set (plus any panel genes
outside it), and the gene set is applied when the graph is built. So you can
switch gene sets with a plain `build`, without reprocessing papers. Papers that
only become relevant under the new set still need the LLM stages, so run
`run --stages variants,study_design,verify,build --gene-set ...` when switching
to a broader set. The active gene set is recorded in `summary.json`.

Other options: `--from-date` / `--to-date` override the window, `--stages fetch,clean,...`
runs a subset, `--limit N` caps fetched works and LLM calls per stage,
`--reuse-reference` skips refreshing CIViC, and `--html` also renders
`variantscape_network_graph.html`.

### Cron example

```cron
# 03:00 on the 1st of every month
0 3 1 * * /path/to/venv/bin/python /path/to/Variantscape/variantscape_pipeline/pipeline.py run --mode incremental --deploy-to /path/to/evidence-database >> /path/to/Variantscape/pipeline_data/logs/cron.log 2>&1
```

A deploy is refused if the new graph has more than 20% fewer nodes or edges
than the deployed one (`--force` overrides this). Replaced files are kept in
`evidence-database/variantscape/backups/`. EvidenceDb loads the graph at
startup, so restart it after a deploy.

## How it works

State lives in `pipeline_data/variantscape.sqlite`. Each stage only handles
papers it has not processed yet. Failed LLM calls are not stored and are
retried on the next run. Reference data (CIViC genes, therapies, diseases and
variants; the OncoKB cancer gene list; MONDO synonyms; the Disease Ontology bulk `doid.json`) is snapshotted
under `pipeline_data/reference/<date>/`.

| Stage | Notebook | What it does |
|---|---|---|
| `fetch` | 01.1 | OpenAlex works for the search term in the date window |
| `clean` | 02.1 | duplicates, language, artifacts, non-research, length filters; text normalization |
| `genes` | 03.1 / 03.0 | BioBERT gene NER (or exact symbol matching) against the oncology gene set; gate for later stages |
| `cancers` | 04.1 | SciSpaCy cancer terms (raw terms stored) |
| `treatments` | 04.2 | CIViC therapy names/aliases string matching |
| `variants` | 05.1 | Llama-3.3-70B, prompt 3, for papers with gene + cancer + treatment |
| `study_design` | 04.3 | LLM study-design label, for papers that also have a variant |
| `verify` | replaces 06.02.5 | LLM checks each mined association against the abstract (see below) |
| `build` | 05.2, 06.01, 06.02, 06.04 | normalization, consensus, weighted graph, curated CIViC associations, artifacts |

Cancer mapping and harmonization, variant normalization, consensus and the graph
are recomputed from stored raw results on every build, using the current
CIViC snapshot, so rule changes apply to all papers without new LLM calls.

LLM calls are only made for papers that can reach the graph (gene, cancer and
treatment found), which costs less than the notebooks, where every cancer paper
was sent to the LLM. The graph content is the same, because it only ever used
papers with all four entity types.

Outputs go to `pipeline_data/outputs/<run-id>/` and `pipeline_data/outputs/latest/`.
Besides the three artifacts, each output folder contains `paper_entities.csv`
(per-paper entities and weights), `variant_treatment_votes.csv` (raw votes
behind the consensus), `cancer_synonyms.csv` (in the format of EvidenceDb's
`Network_cancer_synonyms.csv`; not deployed) and `summary.json`.

## Variant nodes and alteration classes

Variant nodes are named `<variant>_<GENE>`. Besides specific variants
(`v600e_BRAF`, `1799t>a_BRAF` → merged into `v600e_BRAF` via CIViC aliases,
`exon19del_EGFR`), the graph has alteration-class nodes for clinically
important alterations that are not a single change:

| Class | Node | Recognized from (literature and CIViC) |
|---|---|---|
| fusion | `fusion_ALK` | "EML4-ALK fusion", "ALK rearrangement", "BCR::ABL1", gene "BCR-ABL" |
| amplification | `amplification_ERBB2` | "amplification", "HER2 amplified", "copy number gain" |
| ITD | `itd_FLT3` | "ITD", "FLT3-ITD" |
| hotspot codon | `v600_BRAF`, `g12_KRAS`, `r132_IDH1`, … | a hotspot codon without the exact change ("V600", "V600X", "codon 12") |

- **Fusions** are named after the driver partner: a known driver kinase or
  transcription factor (`FUSION_DRIVERS`), otherwise the 3' partner. The
  observed partners are kept in the node attribute `fusion_partners`, e.g.
  `EML4::ALK;NPM1::ALK`. Point mutations reported on a fusion gene (BCR-ABL
  T315I) belong to the driver gene (`t315i_ABL1`).
- **Hotspot codons** come from a curated list (`HOTSPOT_CODONS` in `variants.py`).
  Other position-only mentions are still discarded as too unspecific.
- Every variant node has `variant_class` (specific, fusion, amplification,
  itd, codon).
- "Mutation", "expression", "loss", "overexpression" and similar remain
  excluded.
- The variant extraction prompt (prompt 4) is prompt 3 extended to ask for
  these alteration types. Responses are stored with their prompt version, so
  papers answered with an older prompt are extracted again. *The extension has
  not been evaluated yet; the published evaluation used prompt 3.*
- In EvidenceDb, searches like `ALK` + `fusion`, `EML4-ALK`, `ERBB2` +
  `amplification`, `FLT3` + `ITD` or `BRAF` + `V600` find these nodes. A
  specific hotspot variant (BRAF V600E) also lists curated and verified
  evidence of its codon class (BRAF V600), marked "via BRAF V600".

## Association verification

The co-occurrence graph of the notebooks links every pair of entities mentioned
in the same abstract. That is a major source of noise: e.g. a drug mentioned
only as the previous standard of care becomes associated with the variant. The
`verify` stage replaces the 06.02.5 co-association step and checks each
candidate association against the abstract:

- **Candidates:** for every included paper, all variant × cancer pairs and
  variant × cancer × treatment triples of its mined entities, capped at 80 per
  paper (the cap is recorded).
- **Verdicts:** the LLM decides per candidate whether the title/abstract
  *reports* the association: Sensitive, Resistant, No response, Diagnostic,
  Prognostic, Reported (direction unclear) or Not reported. It must give a
  verbatim supporting quote, and a verdict only counts if that quote is found in
  the title/abstract (`rejected_quote_not_found` in `summary.json`).
- **Model:** `LLM_MODEL`, or `LLM_VERIFY_MODEL` if set.
- **When papers are verified:** a paper is (re)verified when it has no
  verification with the current prompt version, or when its candidates changed
  (e.g. new CIViC aliases). Verdicts are parsed at build time and stored
  candidates are re-canonicalized, so normalization changes don't require new
  calls.

Resulting tiers in the graph:

| Tier | Edge attribute / `sources` | In EvidenceDb |
|---|---|---|
| curated | `curated_*`, `civic` | listed first, CIViC badge |
| verified literature | `weight` (study-weighted papers verifying it), `verified_papers`, `literature` | listed next, ranked by weight |
| co-occurrence only | `cooccurrence_weight`, `cooccurrence` | hidden unless "show unverified" is ticked; greyed out |

A verified triple also counts as verified support for its variant–cancer,
variant–treatment and cancer–treatment edges. Edges between entities of the
same type (variant–variant, …) are no longer built. `verified_associations.csv`
keeps the n-ary triples, so EvidenceDb scores a treatment for a variant in a
cancer by the papers reporting that exact triple. The variant–treatment
consensus file is now built from the verified relations. The coverage report
compares curated associations with both tiers (`supported_by_literature_pct`
= verified, `supported_by_cooccurrence_pct` = before verification).

Verified weights are much smaller than co-occurrence weights, so EvidenceDb
uses separate highlight floors (`EVIDENCE_DB_VERIFIED_TREATMENT_MIN_HIGHLIGHT`,
`EVIDENCE_DB_VERIFIED_CANCER_MIN_HIGHLIGHT`, default 3). Calibrate them after
the first full run. The first deploy with verification has far fewer edges than
the deployed co-occurrence graph, so it needs `--force`.

## Curated associations (CIViC)

Literature mining can miss well-known associations, so the build adds the
accepted CIViC knowledge to the graph with separate provenance:

- **Records:** accepted evidence items at levels A–D (E, inferential, is
  excluded) and assertions at AMP/ASCO/CAP tiers I–II.
  - Predictive records give variant–cancer, variant–treatment and
    cancer–treatment edges.
  - Diagnostic and prognostic records give variant–cancer edges.
  - "Does not support" records are kept as negative evidence, e.g. *No response*.
- **Mapping:** CIViC names go through the same variant normalization and cancer
  harmonization as the literature, so they land on the same nodes.
  - Molecular profiles combining several variants, non-specific variants
    ("Mutation", "Amplification", "Fusion", "Expression", codon-only such as
    "V600") and generic diseases ("Cancer") cannot be mapped to a single node.
    They are skipped and counted.
- **Provenance in the graph:**
  - Every node and edge has `sources` (`literature`, `civic` or
    `literature;civic`).
  - Edges keep `weight` as the literature weight (0 for curated-only edges) and
    gain `curated_score`, `curated_level`, `curated_count` and `curated_ids`.
  - Scores: assertion Tier I-A 6, Tier I-B 5.5, Tier II-C 3.5, Tier II-D 2.5;
    evidence level A 5, B 4, C 3, D 2, plus rating/10.
- **`curated_associations.csv`:** one row per record and treatment. It has the
  variant, cancer and treatment nodes, the prediction, level, score, CIViC IDs
  and PMID, and the literature weights and LLM label for the same association.
- **Consensus file:** `final_variant_treatment_consensus.csv` gains
  `Curated_Prediction` (or `Mixed` if it differs between cancers),
  `Curated_Level`, `Curated_Score`, `Curated_Detail` (per cancer),
  `Curated_IDs` and `Source`. `Resolved_Prediction` stays the literature (LLM)
  consensus and is empty for curated-only pairs.
- **Coverage report:** `curated_coverage.csv` and `summary.json` →
  `build.curated` report how many curated associations the literature-only
  graph recovers, per evidence type and level, and how often the LLM label
  agrees with the curated one.
- **In EvidenceDb:** curated associations for the exact variant, cancer and
  treatment are listed first, ordered by curated score and then literature
  weight. They show a badge linking to the CIViC record and are labelled as
  expert-curated in the LLM context. Literature-only associations follow,
  ranked as before.

## Differences from the notebooks

Bug fixes:

- **Variant parsing** (05.2): variants containing `>`, `+`, `(` or `)` (e.g.
  `c.2138C>G`, `IVS2+1G>A`, `p.(Gly12Cys)`) were silently dropped.
- **Variant alias merging** (05.2): CIViC variants were joined to molecular
  profiles on unrelated IDs, and raw aliases were compared with cleaned names.
  Aliases, HGVS descriptions and rsIDs now come straight from CIViC and get the
  same cleaning as the LLM output, and they only merge within the same gene.
- **Consensus keys** (06.02.5): punctuation was stripped from variant names, so
  pairs such as `r213*_TP53 + …` never matched a graph node. Verdicts are now
  matched to the candidates by number.
- **Study-design weights** (06.02/06.04): the weight lookup is now
  case-insensitive. The classifier's `undefined` label used to get the 0.5
  default instead of the intended 0.1. The 06.04 weights, which are the ones the
  graph used, apply everywhere.
- **Cancer synonyms** (04.1): synonym lists became strings after a CSV round
  trip and were ignored, so only disease names matched. Synonyms now match too,
  with names taking precedence; set `VARIANTSCAPE_USE_CANCER_SYNONYMS=false` for
  the old behavior. CIViC diseases are linked to MONDO by ID rather than by a
  name typeahead.
- **Spurious graph nodes**: merge-suffix columns such as `PaperTitle_Cancer`
  had ended up as cancer nodes in the deployed graph.
- **Edge counting**: the unweighted graph counted each edge twice; edges are
  now counted once per paper.
- **BioBERT windows** (03.1): the last partial window of long texts was dropped.
- **Alteration classes**: fusions, amplifications, ITD and hotspot codons were
  discarded on both the literature and CIViC side. They are now graph nodes
  (see above), which more than doubles the CIViC knowledge that can be mapped.
- **Exon-level variants** (05.2): exon events were matched by substring on the
  raw name. That missed "Exon 19 Deletion"/"exon 20 insertion" and would rename
  e.g. `c.1219del` to `exon19del`. Whole names are now matched, and MET exon 14
  skipping was added.
- **OpenAlex**: the polite pool is now requested with `mailto`.
- **Disease Ontology**: parents come from the bulk `doid.json` file (seconds)
  rather than per-term API calls (over an hour, rate-limited). Parent order is
  kept as the API returned it, which matters for choosing the final parent.
- **LLM errors**: failed calls are retried on the next run instead of being
  stored as `"ERROR"`.

Other changes:

- 04.3 called the `general-classifier` package. Its prompt is now sent directly
  through the same LLM client, and the answer is matched to the categories.
- The MyGene.info synonym expansion in 03.1 never had an effect (see
  `genes.py`), so it has been removed.
- Gene list: the notebooks used the CIViC genes (03) and the Oncomine panel
  (statistics only). The default is now all oncology-relevant genes (OncoKB +
  CIViC, fetched each run), with CIViC or a panel available as options.
- The Brown-corpus English pre-filter of 01.1 is dropped; the langdetect step of
  02.1 still applies.

## Tests

```bash
python -m pytest tests
```
