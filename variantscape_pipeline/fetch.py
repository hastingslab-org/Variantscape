"""Fetch titles and abstracts from OpenAlex (01.1_literature_extraction_from_OpenAlex)."""

from __future__ import annotations

import html
import logging
import re
import time
from typing import Iterator

import requests

log = logging.getLogger(__name__)

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
SELECT_FIELDS = "id,title,publication_date,publication_year,abstract_inverted_index,authorships"

_MOJIBAKE = [
    ("‚Äê", "—"), ("‚Äì", "-"), ("‚Äî", "–"), ("‚Äò", "'"), ("‚Äô", "'"),
    ("‚Ä¢", "•"), ("‚Äû", '"'), ("‚Äú", '"'), ("‚Ä¶", "…"), ("¬†", " "),
    ("√∫", "ú"), ("√©", "é"), ("√±", "ñ"), ("√≥", "ó"), ("√∂", "ö"),
    ("√", ""), ("‚Ä", ""), ("‚", ""),
]


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """Reconstruct an abstract from OpenAlex's inverted index."""
    if not inverted_index:
        return ""
    words = sorted(
        ((word, pos) for word, positions in inverted_index.items() for pos in positions),
        key=lambda x: x[1],
    )
    return " ".join(w for w, _ in words)


def clean_text(text: str | None) -> str:
    """Decode HTML entities, strip tags and encoding artifacts, drop non-ASCII."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    for bad, good in _MOJIBAKE:
        text = text.replace(bad, good)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"[^\x00-\x7F]+", "", text)


def _get_with_retries(session: requests.Session, params: dict, retries: int = 6) -> dict:
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(OPENALEX_WORKS_URL, params=params, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            log.warning("OpenAlex returned %s (attempt %d/%d)", resp.status_code, attempt, retries)
        except requests.RequestException as exc:
            log.warning("OpenAlex request failed: %s (attempt %d/%d)", exc, attempt, retries)
        time.sleep(delay)
        delay = min(delay * 2, 60)
    raise RuntimeError("OpenAlex request failed repeatedly; aborting fetch")


def iter_openalex(
    search_term: str,
    from_date: str,
    to_date: str,
    email: str | None = None,
    api_key: str | None = None,
    max_records: int | None = None,
    per_page: int = 200,
) -> Iterator[dict]:
    """Yield cleaned paper records published between ``from_date`` and ``to_date`` (inclusive)."""
    params = {
        "filter": (
            f"has_abstract:true,title_and_abstract.search:{search_term},"
            f"from_publication_date:{from_date},to_publication_date:{to_date}"
        ),
        "per_page": per_page,
        "select": SELECT_FIELDS,
        "cursor": "*",
    }
    # OpenAlex's polite pool is selected via the mailto parameter (the notebook sent an
    # 'email' header, which OpenAlex ignores)
    if email:
        params["mailto"] = email
    if api_key:
        params["api_key"] = api_key

    session = requests.Session()
    n = 0
    total = None
    while params["cursor"]:
        data = _get_with_retries(session, params)
        if total is None:
            total = data.get("meta", {}).get("count")
            log.info("OpenAlex reports %s works for %s .. %s", total, from_date, to_date)
        for work in data.get("results", []):
            try:
                yield {
                    "paper_id": work["id"].replace("https://openalex.org/W", ""),
                    "title": clean_text(work.get("title") or ""),
                    "abstract": clean_text(reconstruct_abstract(work.get("abstract_inverted_index"))),
                    "pub_date": work.get("publication_date") or "",
                    "pub_year": work.get("publication_year"),
                    "authors": ", ".join(
                        clean_text(a["author"]["display_name"])
                        for a in work.get("authorships", [])
                        if a.get("author") and a["author"].get("display_name")
                    ),
                }
            except Exception as exc:  # noqa: BLE001 - one malformed record must not stop the fetch
                log.error("Skipping malformed OpenAlex record: %s", exc)
                continue
            n += 1
            if max_records and n >= max_records:
                return
        params["cursor"] = data.get("meta", {}).get("next_cursor")
        time.sleep(0.2)
