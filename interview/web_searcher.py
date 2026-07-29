"""
Optional web search for gap-filling during the interview.
Uses DuckDuckGo — no API key required.
"""

from __future__ import annotations
from typing import Optional


def search(query: str, max_results: int = 5) -> list[dict]:
    """Return list of {title, href, body} dicts."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return [{"title": "Search unavailable", "href": "", "body": str(e)}]


def summarise_for_llm(results: list[dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.get('title', '')}\n{r.get('body', '')}\n")
    return "\n".join(lines)
