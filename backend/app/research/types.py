from __future__ import annotations

from dataclasses import dataclass

from backend.app.domain.reports import ConfidenceLevel, EvidenceTopic, SourceType


@dataclass(frozen=True)
class SearchQuery:
    topic: EvidenceTopic
    text: str


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    rank: int
    query_topic: EvidenceTopic


@dataclass(frozen=True)
class ExtractedContent:
    url: str
    title: str
    text: str
    language: str | None = None
    published_at: str | None = None


@dataclass(frozen=True)
class ClassifiedEvidence:
    source_id: str
    topic: EvidenceTopic
    claim: str
    raw_text_excerpt: str
    confidence: ConfidenceLevel


@dataclass(frozen=True)
class ScoredSource:
    source_id: str
    title: str
    url: str
    domain: str
    source_type: SourceType
    reliability_score: int
    snippet: str
    is_current: bool
