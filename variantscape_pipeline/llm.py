"""OpenAI-compatible LLM client (DeepInfra by default) with retries and parallel execution."""

from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Iterator, TypeVar

from .config import SYSTEM_MSG

log = logging.getLogger(__name__)

T = TypeVar("T")


class LLMClient:
    def __init__(self, api_key: str | None, base_url: str, model: str,
                 temperature: float | None = None, timeout: float = 120, max_retries: int = 5):
        if not api_key:
            raise RuntimeError("No LLM API key configured (set LLM_API_KEY, DEEPINFRA_API_KEY or OPENAI_API_KEY)")
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

    def complete(self, prompt: str, system: str = SYSTEM_MSG, max_tokens: int | None = None) -> str | None:
        """Return the model's answer, or ``None`` if all retries failed.

        Failures are not stored, so the paper is retried on the next run (the
        notebooks stored the string "ERROR" permanently).
        """
        kwargs = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        delay = 2.0
        for attempt in range(1, self.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    **kwargs,
                )
                return completion.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - any API error is retried
                log.warning("LLM call failed (attempt %d/%d): %s", attempt, self.max_retries, exc)
                time.sleep(delay + random.random())
                delay = min(delay * 2, 60)
        return None


def run_parallel(items: Iterable[T], fn: Callable[[T], object], max_workers: int) -> Iterator[tuple[T, object]]:
    """Apply ``fn`` to ``items`` in a thread pool, yielding ``(item, result)`` as they finish.

    Results are yielded in the calling thread so callers can write to SQLite safely.
    """
    items = list(items)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                yield item, future.result()
            except Exception as exc:  # noqa: BLE001
                log.error("Worker failed for %r: %s", item, exc)
                yield item, None
