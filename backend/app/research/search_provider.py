from __future__ import annotations

from typing import Protocol

import httpx

from backend.app.domain.reports import EvidenceTopic
from backend.app.research.types import SearchResult


class SearchProvider(Protocol):
    def search(self, query: str, limit: int) -> list[SearchResult]:
        ...


class FakeSearchProvider:
    def __init__(self, results: dict[str, list[SearchResult]] | None = None) -> None:
        self.results = results or {}

    def search(self, query: str, limit: int) -> list[SearchResult]:
        return self.results.get(query, [])[:limit]


class SearchProviderError(RuntimeError):
    pass


class TavilySearchProvider:
    endpoint = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: str,
        http_client: httpx.Client | None = None,
        search_depth: str = "basic",
        country: str = "argentina",
        timeout_seconds: int = 30,
    ) -> None:
        self.api_key = api_key
        self.http_client = http_client or httpx.Client(timeout=timeout_seconds)
        self.search_depth = search_depth
        self.country = country

    def search(self, query: str, limit: int) -> list[SearchResult]:
        if not self.api_key:
            raise SearchProviderError("TAVILY_API_KEY or SEARCH_PROVIDER_API_KEY is required.")

        response = self.http_client.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "max_results": limit,
                "search_depth": self.search_depth,
                "topic": "general",
                "country": self.country,
                "include_answer": False,
                "include_raw_content": False,
            },
        )
        if response.status_code >= 400:
            raise SearchProviderError(f"Tavily search failed with status {response.status_code}.")

        payload = response.json()
        results = payload.get("results", [])
        if not isinstance(results, list):
            raise SearchProviderError("Tavily response did not include a valid results list.")

        mapped: list[SearchResult] = []
        for index, item in enumerate(results[:limit], start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            content = str(item.get("content") or "").strip()
            if not title or not url:
                continue
            mapped.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=content,
                    rank=index,
                    query_topic=EvidenceTopic.general,
                )
            )
        return mapped
