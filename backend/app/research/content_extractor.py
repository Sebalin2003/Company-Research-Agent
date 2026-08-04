from __future__ import annotations

from bs4 import BeautifulSoup
import httpx
import trafilatura
from typing import Protocol

from backend.app.research.types import ExtractedContent


class ContentExtractor(Protocol):
    def extract(self, url: str) -> ExtractedContent | None:
        ...


class FakeContentExtractor:
    def __init__(self, content_by_url: dict[str, ExtractedContent] | None = None) -> None:
        self.content_by_url = content_by_url or {}

    def extract(self, url: str) -> ExtractedContent | None:
        return self.content_by_url.get(url)


class HttpContentExtractor:
    def __init__(
        self,
        http_client: httpx.Client | None = None,
        max_characters: int = 20_000,
        timeout_seconds: int = 20,
    ) -> None:
        self.http_client = http_client or httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "EnterpriseResearchAgent/0.1"},
        )
        self.max_characters = max_characters

    def extract(self, url: str) -> ExtractedContent | None:
        try:
            response = self.http_client.get(url)
        except httpx.HTTPError:
            return None
        if response.status_code >= 400:
            return None

        html = response.text
        extracted = trafilatura.extract(html) or fallback_text_from_html(html)
        text = " ".join((extracted or "").split())
        if not text:
            return None

        return ExtractedContent(
            url=url,
            title=fallback_title_from_html(html) or url,
            text=text[: self.max_characters],
        )


def fallback_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(" ")


def fallback_title_from_html(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return None
