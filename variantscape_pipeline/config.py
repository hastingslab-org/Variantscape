"""Pipeline settings, read from environment variables (a ``.env`` file is supported)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Search string used in 01.1_literature_extraction_from_OpenAlex
DEFAULT_SEARCH_TERM = '("cancer" OR "carcinoma" OR "tumor")'

# Start of the publication window of a full run (the notebooks covered 2014 onwards)
DEFAULT_FULL_START_DATE = "2014-01-01"

DEFAULT_LLM_BASE_URL = "https://api.deepinfra.com/v1/openai"
DEFAULT_LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_BIOBERT_MODEL = "alvaroalon2/biobert_genetic_ner"

SYSTEM_MSG = (
    "You are a helpful medical question answering assistant. Please carefully "
    "follow the exact instructions and do not provide explanations."
)

# Study design categories produced by the general classifier (04.3 / 04.4)
STUDY_DESIGN_CATEGORIES = [
    "Clinical study",
    "Observational/RWE study",
    "Systematic review study",
    "Case report study",
    "In vivo/Animal study",
    "In vitro study",
    "In silico study",
    "Behavioral study",
    "undefined",
]

# Evidence-pyramid weights from 06.04_Network_analysis_with_synonyms (the graph
# used by EvidenceDb). Keys are matched case-insensitively; the notebooks
# matched case-sensitively, so the classifier's "undefined" label silently got
# the 0.5 default instead of the intended 0.1.
DEFAULT_STUDY_DESIGN_WEIGHTS = {
    "systematic review study": 1.0,
    "clinical study": 1.0,
    "observational/rwe study": 0.9,
    "case report study": 0.9,
    "in vivo/animal study": 0.8,
    "in vitro study": 0.7,
    "in silico study": 0.6,
    "undefined": 0.1,
    "other": 0.1,
}
DEFAULT_STUDY_WEIGHT = 0.5


def study_weight(design: str | None) -> float:
    if design is None:
        return DEFAULT_STUDY_WEIGHT
    return DEFAULT_STUDY_DESIGN_WEIGHTS.get(str(design).strip().lower(), DEFAULT_STUDY_WEIGHT)


# Relative paths in the settings are resolved against the Variantscape folder, so the
# pipeline behaves the same whichever directory it is started from
REPO_ROOT = Path(__file__).resolve().parent.parent


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def gene_set_spec(value: str) -> str:
    """'oncology', 'civic', or an absolute path to a panel file."""
    value = value.strip()
    return value.lower() if value.lower() in {"oncology", "civic"} else str(_path(value))


def _bool(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


@dataclass
class Settings:
    work_dir: Path
    openalex_email: str | None
    openalex_api_key: str | None
    search_term: str
    full_start_date: str
    overlap_days: int
    llm_api_key: str | None
    llm_base_url: str
    llm_model: str
    llm_verify_model: str
    llm_temperature: float | None
    llm_max_workers: int
    llm_timeout: float
    gene_set: str
    gene_filter_method: str
    biobert_model: str
    device: int
    use_cancer_synonyms: bool
    evidencedb_dir: Path | None

    @property
    def db_path(self) -> Path:
        return self.work_dir / "variantscape.sqlite"

    @property
    def reference_dir(self) -> Path:
        return self.work_dir / "reference"

    @property
    def output_dir(self) -> Path:
        return self.work_dir / "outputs"

    @property
    def log_dir(self) -> Path:
        return self.work_dir / "logs"

    @classmethod
    def from_env(cls, env_file: str | None = None) -> "Settings":
        # Default: the .env next to the package (Variantscape/.env), wherever we are started from
        load_dotenv(env_file or REPO_ROOT / ".env")
        evidencedb_dir = os.getenv("VARIANTSCAPE_EVIDENCEDB_DIR")
        return cls(
            work_dir=_path(os.getenv("VARIANTSCAPE_WORK_DIR", "pipeline_data")),
            openalex_email=os.getenv("OPENALEX_EMAIL") or os.getenv("EMAIL_ADDRESS"),
            openalex_api_key=os.getenv("OPENALEX_API_KEY"),
            search_term=os.getenv("VARIANTSCAPE_SEARCH_TERM", DEFAULT_SEARCH_TERM),
            full_start_date=os.getenv("VARIANTSCAPE_FULL_START_DATE", DEFAULT_FULL_START_DATE),
            # Incremental runs re-fetch this many days before the previous run's end date,
            # to catch works that OpenAlex indexed late (already-known works are skipped)
            overlap_days=int(os.getenv("VARIANTSCAPE_OVERLAP_DAYS", "30")),
            # The notebooks stored the DeepInfra key in OPENAI_API_KEY
            llm_api_key=os.getenv("LLM_API_KEY") or os.getenv("DEEPINFRA_API_KEY") or os.getenv("OPENAI_API_KEY"),
            llm_base_url=os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
            llm_model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
            # Optional separate model for the association verification step
            llm_verify_model=os.getenv("LLM_VERIFY_MODEL") or os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
            llm_temperature=_optional_float(os.getenv("LLM_TEMPERATURE")),
            llm_max_workers=int(os.getenv("LLM_MAX_WORKERS", "8")),
            llm_timeout=float(os.getenv("LLM_TIMEOUT", "120")),
            gene_set=gene_set_spec(os.getenv("VARIANTSCAPE_GENE_SET", "oncology")),
            gene_filter_method=os.getenv("VARIANTSCAPE_GENE_FILTER", "biobert").lower(),
            biobert_model=os.getenv("VARIANTSCAPE_BIOBERT_MODEL", DEFAULT_BIOBERT_MODEL),
            device=int(os.getenv("VARIANTSCAPE_DEVICE", "-1")),
            use_cancer_synonyms=_bool(os.getenv("VARIANTSCAPE_USE_CANCER_SYNONYMS"), True),
            evidencedb_dir=_path(evidencedb_dir) if evidencedb_dir else None,
        )

    def ensure_dirs(self) -> None:
        for d in (self.work_dir, self.reference_dir, self.output_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)
