"""Cleaning and normalization of fetched articles (02.1_Cleaning_of_acrticles_from_OpenAlex).

Filters are applied in the notebook's order; every rejected paper gets a reason.
"""

from __future__ import annotations

import re

import pandas as pd

# --------------------------------------------------------------------------- #
# Keyword lists (verbatim from 02.1)
# --------------------------------------------------------------------------- #
ARTIFACT_START_KEYWORDS = sorted([
    "Acknowledgments", "ADOBE PDF", "Addendum", "Advertisement", "Additional figure", "Additional Material",
    "Additional Table", "All Supplemental Data", "All Supplemental Figures",
    "All Supplemental Figures, Tables and Legends", "All Supplementary Data",
    "All Supplementary Figures", "All Supplementary Figures and Tables",
    "All Supplementary Tables, Figures and Methods", "Analysis Methods Supplement",
    "Appendix", "Article figures", "Author Correction", "Auxiliary Supplementary File",
    "Caption for Suppl Fig.", "Captions of supplementary", "Combined supplemental figures",
    "Contents Vol", "Contents:",
    "Corrigendum", "Correction", "Correction to", "Correction:", "Corrections to",
    "Data Augmentation", "Data associated with", "Data and code", "Data and analysis scripts", "Data File",
    "Data for", "Data from", "Data not shown", "Data and metadata",
    "Data on", "Data S", "Data Spreadsheet", "Data Supplement", "Data.zip",
    "Dataset", "Dataset and metadata", "Dataset for", "Dataset related to",
    "Description of Supplemental Figures", "Description of supplementary",
    "Download Supplementary", "eFigure", "Erratum", "Extended Data", "Expression of Concern",
    "Fig", "Fig.", "Fig Supp", "FigS", "Figure", "Figure S", "Figures", "Instructions for Authors",
    "File", "Funding statement", "http://", "https://",
    "Legend for", "Legend from", "Legend of", "Legend to",
    "Legends for", "Legends from", "Legends of", "Legends to",
    "Legends Supplemental Figures", "Link to Supplementary",
    "List of figures", "List of tables", "List of plates", "Manuscript Figures",
    "Mat & Met", "Material and Methods", "Materials and Methods",
    "Merged Supplementary", "Metadata and data", "Metadata record for",
    "Metadata supporting", "Methods References", "Methods and Materials from",
    "Methods and Supplementary", "Methods and Tables", "Methods file",
    "Methods from", "Methods, Figures", "Methods, Table", "MET supplementary methods",
    "methods_", "Movie", "Multimedia", "Methodology and Method", "Online supplement",
    "Online Supplementary Materials", "Online Supplementary Tables",
    "Online-only supplementary", "Online-Only Tables",
    "Original Western Blot Data", "Revised Supplementary", "S Figure", "S Table",
    "S-Figure", "S1 Legend", "S1 from", "S1.", "S1:", "S2 from", "S2.", "S2:",
    "S3 from", "S3.", "S3:", "S4 from", "S4.", "S4:", "S5 from", "S5.", "S5:",
    "S6 from", "S6.", "S6:", "S7 from", "S7.", "S7:", "S8 from", "S8.", "S8:",
    "S9 from", "S9.", "S9:", "SI Figure", "SI Methods", "SI Table",
    "SI materials", "SI tables and figures", "SFigure", "SF-", "SF1",
    "Sl Figure", "Supplement", "Supplement Material", "Supplement Table",
    "Supplemental", "Supplemental Data", "Supplemental Figure",
    "Supplemental Figures", "Supplemental Table", "Supplementary",
    "Supplementary Data", "Supplementary Figure", "Supplementary Figures",
    "Supplementary Legends", "Supplementary Methods", "Supplementary Movie",
    "Supplementary Table", "Supplementary Tables", "Supplementary Text",
    "Supplementary Video", "SupplementaryFigures", "Supplementary_Figure",
    "Supplementray Figures", "Supplementray Tables", "Supplymentary Figure",
    "Supplymentary Figures", "Supplymentary Table", "Supporting Data",
    "Supporting Document", "Supporting Figure", "Supporting Figures",
    "Supporting File", "Supporting Information", "Supporting Legend",
    "Supporting Legends", "Supporting Materials", "Supporting Methods",
    "Supporting Table", "Supporting Tables", "Supporting Text",
    "Supporting Video", "Suppl", "Suppl Figure", "Suppl Figures", "Suppl File",
    "Suppl Info", "Suppl Information", "Suppl Legends", "Suppl List of Videos, Tables and Methods",
    "Suppl Materials and Methods", "Suppl Methods", "Suppl Movie", "Suppl Table",
    "Suppl Video", "Supplimentary Figure", "Supplimentary Methods",
    "Supplimentary Tables", "Supplimentary_Figure", "Suplementary", "Supllementary Figure",
    "Suplplementary Figure Legend", "Suplpementary Table", "Supplrmentary Figures",
    "Supplrmentary Tables", "Suppltable", "Suppltext", "Supplvideo",
    "Suppl_Figure", "Suppl-Figure", "Suupplementary Tables", "Table", "Table of Content",
    "Validation Figures", "Video", "Western Blots from", "Whole Slide Image",
])

ARTIFACT_MIDDLE_KEYWORDS = sorted([
    "Combined Supplementary Materials", "Corrigendum", "The Case Files", "Additional file",
    "Figure S", "Figures S", "Manuscript Figures", "SI tables and figures",
    "Supplemental Data", "Supplemental figures", "Supplemental from",
    "Supplemental table", "Supplemental figure", "Supplementary Figure",
    "Supplementary Figures", "Supplementary Material", "Supplementary Methods",
    "Supplementary Table", "Supplementary Tables", "Supplementary data",
    "Supplementary document", "Supplementary information", "SupplementaryFigures",
    "Supplementary materials and methods",
    "Translation for the Article", "Video Dataset", "Operative Video",
])

PLACEHOLDER_TITLES = {
    "Title Not Available", "No Title", "Untitled", "N/A",
    "[No title available]", "No title available", "No title available - PubMed",
    "Error in Text", "Errors in Box", "Error in Author Name", "Error in Author Names",
    "Error in Author Surnames", "Errors in article text", "Error in Figure Label and Caption",
    "Error in Presentation of Author", "Errors in Author Name",
    "Error in Table Text", "Error in Corresponding Authorship",
}

NON_RESEARCH_KEYWORDS = [
    "A reply to Letter to the editor", "About the Author", "An Update from the Editor-in-Chief",
    "Announcement:", "Announcements", "Appeal", "ASO Author Reflection", "AUTHOR COPY ONLY",
    "Associate Editor", "Author comment:", "Author profile", "Author Reflections:", "Author reply",
    "Author response", "Author response to", "Author view", "Author's reply", "Author's respond",
    "Editor's Reflection", "Editors Reflection", "Editors' Reflection", "Editor' Reflection",
    "Author's response", "Author's view", "Authors reply", "Authors respond", "Authors response",
    "Authors view", "Authors' Reply", "Authors' response", "Authors's response", "Book Review", "EditorinChief",
    "Comment", "Comment on", "Commentary", "Conference Summary", "Conversation with the Editor",
    "Correspondence", "Editorial", "Editor Profile", "Editor response for", "Editor change", "New Editor",
    "Editor-in-Chief", "Editorinchief's introduction", "Editor's evaluation", "Editor's Introduction",
    "Editor's Message", "Editor's note", "Editors Introduction", "Editors Message", "Editors Note on",
    "Editors Spotlight", "Editors change", "Editors note", "Editors' Introduction", "Editors' Message",
    "Editors' note", "Foreword", "From the Editor", "From the Editor-in-Chief", "From the Editors...",
    "From the Editors Desk...", "From the Editor's desk", "From the editors desk", "From the Guest Editor", "Guest editor",
    "Guest editorial", "Introducing our new Editors", "Issue Editor Foreword", "Issue Information Editorial Board",
    "Letter to editor", "Letter to the Editor", "Letter:", "Letter Of The Editor", "Letters of the editor", "Letter-to-the-editor",
    "Letters to editor", "Letters to the editor", "Meet the First Author", "Meet the author", "Meet the editor",
    "Message from Editor", "Message from the Editor", "News", "Opinion", "our author", "our editor",
    "Reply:", "In reply:", "Reply to:", "In reply to:",
    "Perspective", "Preface", "Proceedings", "Publisher'?s? Note", "Reply by Author", "Reply by the author",
    "Reply to Editorial Comment", "Reply to Letter to", "Reply to correspondence", "Reply to the Letter to",
    "Response to Editorial", "Response to a letter", "Response to letter", "senior editor", "Special Section",
    "The Author Reply", "The Authors Reply", "To the Editor", "Transitioning Between Editor", "Viewpoint",
    "Welcome to our", "Executive Editor", "Assistant Editor",
]

_START_RE = re.compile(r"^(?:" + "|".join(map(re.escape, ARTIFACT_START_KEYWORDS)) + r")", re.IGNORECASE)
_MIDDLE_RE = re.compile(r"\b(?:{})\b".format("|".join(map(re.escape, ARTIFACT_MIDDLE_KEYWORDS))), re.IGNORECASE)
_SUPPLE_RE = re.compile(r"[\w\-_]*supple\w*", re.IGNORECASE)
_NUMERIC_RE = re.compile(r"^[^A-Za-z]+$")
_PLACEHOLDER_RE = re.compile(r"\b(?:{})\b".format("|".join(map(re.escape, PLACEHOLDER_TITLES))), re.IGNORECASE)
_WITHDRAWN_RE = re.compile(
    r"(?i)^(?:\[?)?(Withdrawn|Withdrawal|Retracted|Retraction|Errata|Erratum|Revoked|Removed|Correction Notice|Corrigendum)\b"
)
_NON_RESEARCH_RE = re.compile("|".join(re.escape(k) for k in NON_RESEARCH_KEYWORDS), re.IGNORECASE)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Extra cleaning applied in 03.1 after gene extraction
_TITLE_PREFIX_RE = re.compile("|".join([
    r"^\d{2,3}:\s*", r"^\d{4}:\s*", r"^\d{5}:\s*", r"^#\d{3,4}\s*", r"^<PHONE>:\s*",
]))
_ABSTRACT_PREFIX_RE = re.compile(
    "|".join([r"^\d{1,5}\^?\s+(?=Background)", r"^\d{1,5}\s+(?=Objectives[:]?|Abstract[:]?)"]),
    re.IGNORECASE,
)


def _real_words(text: str) -> int:
    return len(re.findall(r"\b[a-zA-Z]{2,}\b", str(text)))


def _is_english(title: str, abstract: str) -> bool:
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0

    def safe_detect(text: str) -> str:
        if len(text) <= 10:
            return "unknown"
        for _ in range(3):
            try:
                return detect(text)
            except Exception:  # noqa: BLE001 - langdetect raises on undetectable text
                pass
        return "error"

    return safe_detect(str(title).strip().lower()) == "en" or safe_detect(str(abstract).strip()) == "en"


def _to_sentence_case(text: str) -> str:
    return text.capitalize() if text.isupper() else text


def _clean_title_v1(title: str) -> str | None:
    from cleantext import clean

    if not isinstance(title, str) or title.strip() == "":
        return None
    title = clean(
        title, fix_unicode=True, to_ascii=True, lower=False, no_line_breaks=True,
        no_urls=True, no_emails=True, no_phone_numbers=True, no_numbers=False,
        no_digits=False, no_currency_symbols=True, no_punct=False,
    ).strip()
    if re.fullmatch(r"^\d+$", title):
        return None
    if title.isupper():
        title = title.title()
    if len(title) <= 3 or re.fullmatch(r"[^A-Za-z0-9]+", title):
        return None
    return title


def _clean_abstract_v1(abstract: str) -> str | None:
    from cleantext import clean

    if not isinstance(abstract, str) or abstract.strip() == "":
        return None
    abstract = clean(
        abstract, fix_unicode=True, to_ascii=True, lower=False, no_line_breaks=False,
        no_urls=True, no_emails=True, no_phone_numbers=True, no_numbers=False,
        no_digits=False, no_currency_symbols=True, no_punct=False, replace_with_url="<URL>",
    ).strip()
    if any(abstract.startswith(p) for p in ("Our website uses cookies to enhance your experience.", "(Cell Reports")):
        return None
    return abstract


def _normalize_title(title: str) -> str:
    title = re.sub(r"^\s*(\*?\[?(invited|keynote|winner|regular paper|blog)\]?\*?)\s*[:\-]*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"[_-]{5,}", "", title)
    title = re.sub(r"^\s*[_]*Abstract[_]*\s*[:\-]*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"^\s*[\.\,\-\:\;\//\=\s]+", "", title)
    title = re.sub(r"^\[([^\]]+)\]\.?\s*$", r"\1", title)
    title = re.sub(r"\s+", " ", title).strip()
    title = _to_sentence_case(title)
    return _TITLE_PREFIX_RE.sub("", title)


def _normalize_abstract(abstract: str) -> str:
    if abstract.lower().startswith("you have access"):
        m = re.search(r"\bintroduction\b", abstract, re.IGNORECASE)
        if m:
            abstract = abstract[m.start():]
    if abstract.startswith("//"):
        m = re.search(r"\babstracts?\b", abstract, re.IGNORECASE)
        if m:
            abstract = abstract[m.start():]
    abstract = re.sub(r"^\s*[:]+\s*", "", abstract)
    abstract = re.sub(r"[_-]{5,}", "", abstract)
    abstract = re.sub(r"^\s*[_]*Abstract[_]*\s*[:\-]*", "", abstract, flags=re.IGNORECASE)
    abstract = re.sub(r"^\s*[\.\,\-\:\;\=\s]+", "", abstract)
    abstract = re.sub(r"\s*<[^>]+>\s*", " ", abstract)
    abstract = re.sub(r"[,\.]{2,}", ".", abstract)
    abstract = re.sub(r"\s+", " ", abstract).strip()
    abstract = _to_sentence_case(abstract)
    return _ABSTRACT_PREFIX_RE.sub("", abstract)


def dedupe_key(title: str, authors: str) -> str:
    return f"{str(title).strip().lower()}||{str(authors).strip().lower()}"


def clean_batch(df: pd.DataFrame, known_dedupe_keys: set[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    """Apply all 02.1 cleaning steps.

    Returns the kept papers (with cleaned ``title``/``abstract`` and a ``dedupe_key``
    column) and a mapping ``paper_id -> rejection reason``.
    """
    rejected: dict[str, str] = {}
    df = df.copy()

    def reject(mask: pd.Series, reason: str) -> None:
        nonlocal df
        for pid in df.loc[mask, "paper_id"]:
            rejected[pid] = reason
        df = df.loc[~mask]

    def blank(col: str) -> pd.Series:
        return df[col].isna() | (df[col].astype(str).str.strip() == "")

    # 1) Missing metadata, invalid dates, duplicates
    for col in ("title", "abstract", "authors", "pub_date"):
        reject(blank(col), f"missing_{col}")
    reject(~df["pub_date"].astype(str).str.match(_DATE_RE), "invalid_pub_date")
    reject(df["paper_id"].duplicated(keep="first"), "duplicate_id")
    df["dedupe_key"] = [dedupe_key(t, a) for t, a in zip(df["title"], df["authors"])]
    reject(df["dedupe_key"].duplicated(keep="first") | df["dedupe_key"].isin(known_dedupe_keys),
           "duplicate_title_authors")

    # 2) Language
    if len(df):
        reject(~pd.Series([_is_english(t, a) for t, a in zip(df["title"], df["abstract"])], index=df.index),
               "non_english")

    # 3) Artifacts (supplementary material, figures, ...)
    titles = df["title"].astype(str)
    reject(
        titles.map(lambda t: bool(_START_RE.match(t)))
        | titles.map(lambda t: bool(_MIDDLE_RE.search(t)))
        | titles.str.contains("_", regex=False)
        | titles.map(lambda t: bool(_SUPPLE_RE.search(t))),
        "artifact",
    )

    # 4) Invalid titles, lengths, withdrawn and non-research articles
    reject(df["title"].astype(str).map(lambda t: bool(_NUMERIC_RE.match(t))), "numeric_title")
    reject(df["title"].astype(str).map(lambda t: bool(_PLACEHOLDER_RE.search(t))), "placeholder_title")
    reject(~df["title"].map(lambda t: 3 <= _real_words(t) <= 60), "title_length")
    reject(~df["abstract"].map(lambda a: 80 <= _real_words(a) <= 1000), "abstract_length")
    reject(df["title"].astype(str).map(lambda t: bool(_WITHDRAWN_RE.search(t))), "withdrawn")
    df["title"] = df["title"].fillna("").str.strip().str.replace(r"\s+", " ", regex=True)
    reject(df["title"].map(lambda t: bool(_NON_RESEARCH_RE.search(t))), "non_research")

    # 5) Text cleaning and normalization
    df["title"] = df["title"].map(_clean_title_v1)
    df["abstract"] = df["abstract"].map(_clean_abstract_v1)
    reject(df["title"].isna() | df["abstract"].isna(), "empty_after_cleaning")
    reject(df["title"].map(lambda t: len(t.split()) < 3), "title_too_short_after_cleaning")
    df["title"] = df["title"].map(_normalize_title)
    df["abstract"] = df["abstract"].map(_normalize_abstract)

    return df, rejected
