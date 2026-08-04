from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from dataclasses import replace

from backend.app.domain.reports import ConfidenceLevel, EvidenceTopic, SourceType
from backend.app.research.content_extractor import ContentExtractor
from backend.app.research.evidence_classifier import classify_evidence
from backend.app.research.query_builder import (
    build_deepening_company_research_queries,
    build_fast_company_research_queries,
)
from backend.app.research.scoring import score_search_result
from backend.app.research.search_provider import SearchProvider
from backend.app.research.types import (
    ClassifiedEvidence,
    ExtractedContent,
    ScoredSource,
    SearchQuery,
    SearchResult,
)


TOPIC_GROUP_LIMITS: tuple[tuple[set[EvidenceTopic], int], ...] = (
    ({EvidenceTopic.business}, 2),
    ({EvidenceTopic.argentina_presence}, 2),
    ({EvidenceTopic.employees}, 2),
    ({EvidenceTopic.salary, EvidenceTopic.benefits}, 3),
    ({EvidenceTopic.culture}, 2),
    ({EvidenceTopic.interview_process, EvidenceTopic.interview_questions}, 2),
    ({EvidenceTopic.open_roles}, 2),
)

DEEPENING_TOPICS = {
    EvidenceTopic.argentina_presence,
    EvidenceTopic.employees,
    EvidenceTopic.salary,
    EvidenceTopic.culture,
    EvidenceTopic.interview_process,
    EvidenceTopic.interview_questions,
    EvidenceTopic.open_roles,
}

PROTECTED_DOMAIN_MARKERS = (
    "glassdoor.",
    "indeed.",
    "linkedin.",
    "openqube.",
)


@dataclass(frozen=True)
class ResearchPipelineResult:
    sources: list[ScoredSource]
    evidence: list[ClassifiedEvidence]
    research_duration_ms: int = 0
    search_duration_ms: int = 0
    extraction_duration_ms: int = 0
    search_count: int = 0
    extracted_url_count: int = 0
    extraction_skipped_count: int = 0
    snippet_only_count: int = 0


@dataclass
class SourceCandidate:
    source: ScoredSource
    discovered_topics: set[EvidenceTopic]
    evidence: list[ClassifiedEvidence]
    best_rank: int


class ResearchPipeline:
    def __init__(
        self,
        search_provider: SearchProvider,
        content_extractor: ContentExtractor,
        results_per_query: int = 5,
        search_concurrency: int = 4,
        extract_concurrency: int = 6,
        enable_deepening: bool = True,
        max_extract_urls: int = 12,
        extract_protected_domains: bool = False,
        max_extract_urls_per_topic: int = 2,
    ) -> None:
        self.search_provider = search_provider
        self.content_extractor = content_extractor
        self.results_per_query = results_per_query
        self.search_concurrency = search_concurrency
        self.extract_concurrency = extract_concurrency
        self.enable_deepening = enable_deepening
        self.max_extract_urls = max_extract_urls
        self.extract_protected_domains = extract_protected_domains
        self.max_extract_urls_per_topic = max_extract_urls_per_topic

    def run(self, company_name: str) -> ResearchPipelineResult:
        started = time.perf_counter()
        extracted_by_url: dict[str, ExtractedContent | None] = {}
        fast_queries = build_fast_company_research_queries(company_name)
        search_started = time.perf_counter()
        records = self.search_queries(fast_queries)
        search_duration = elapsed_ms(search_started)
        extraction_started = time.perf_counter()
        extracted_by_url.update(
            self.extract_urls(
                self.select_extraction_urls(company_name, records, extracted_by_url)
            )
        )
        extraction_duration = elapsed_ms(extraction_started)
        candidates_by_url = build_candidates(company_name, records, extracted_by_url)

        if self.enable_deepening:
            missing_topics = topics_needing_deepening(candidates_by_url.values())
            deep_queries = [
                query
                for query in build_deepening_company_research_queries(company_name)
                if query.topic in missing_topics
            ]
            if deep_queries:
                search_started = time.perf_counter()
                deep_records = self.search_queries(deep_queries)
                search_duration += elapsed_ms(search_started)
                new_urls = [
                    result.url
                    for _, result in deep_records
                    if result.url not in extracted_by_url
                ]
                extraction_started = time.perf_counter()
                extraction_records = [
                    (query, result)
                    for query, result in deep_records
                    if result.url in new_urls
                ]
                extracted_by_url.update(
                    self.extract_urls(
                        self.select_extraction_urls(
                            company_name,
                            extraction_records,
                            extracted_by_url,
                        )
                    )
                )
                extraction_duration += elapsed_ms(extraction_started)
                records.extend(deep_records)
                candidates_by_url = build_candidates(company_name, records, extracted_by_url)

        candidates = list(candidates_by_url.values())
        selected = select_balanced_candidates(candidates)
        selected_source_ids = {candidate.source.source_id for candidate in selected}

        return ResearchPipelineResult(
            sources=[candidate.source for candidate in selected],
            evidence=dedupe_evidence(
                item
                for candidate in selected
                for item in candidate.evidence
                if item.source_id in selected_source_ids
            ),
            research_duration_ms=elapsed_ms(started),
            search_duration_ms=search_duration,
            extraction_duration_ms=extraction_duration,
            search_count=len(fast_queries)
            + (
                len(deep_queries)
                if self.enable_deepening and "deep_queries" in locals()
                else 0
            ),
            extracted_url_count=len(extracted_by_url),
            extraction_skipped_count=count_extraction_skipped(records, extracted_by_url),
            snippet_only_count=count_snippet_only(records, extracted_by_url),
        )

    def search_queries(self, queries: list[SearchQuery]) -> list[tuple[SearchQuery, SearchResult]]:
        if not queries:
            return []
        records: list[tuple[int, SearchQuery, SearchResult]] = []
        with ThreadPoolExecutor(max_workers=max(1, self.search_concurrency)) as executor:
            futures = {
                executor.submit(self.search_provider.search, query.text, self.results_per_query): (
                    index,
                    query,
                )
                for index, query in enumerate(queries)
            }
            for future in as_completed(futures):
                index, query = futures[future]
                try:
                    results = future.result()
                except Exception:
                    continue
                for result in results:
                    records.append((index, query, replace(result, query_topic=query.topic)))
        records.sort(key=lambda item: (item[0], item[2].rank, item[2].url))
        return [(query, result) for _, query, result in records]

    def extract_urls(self, urls: list[str]) -> dict[str, ExtractedContent | None]:
        unique_urls = list(dict.fromkeys(urls))
        if not unique_urls:
            return {}
        extracted_by_url: dict[str, ExtractedContent | None] = {}
        with ThreadPoolExecutor(max_workers=max(1, self.extract_concurrency)) as executor:
            futures = {
                executor.submit(self.content_extractor.extract, url): url
                for url in unique_urls
            }
            for future in as_completed(futures):
                url = futures[future]
                try:
                    extracted_by_url[url] = future.result()
                except Exception:
                    extracted_by_url[url] = None
        return extracted_by_url

    def select_extraction_urls(
        self,
        company_name: str,
        records: list[tuple[SearchQuery, SearchResult]],
        extracted_by_url: dict[str, ExtractedContent | None],
    ) -> list[str]:
        remaining = self.max_extract_urls - len(extracted_by_url)
        if remaining <= 0:
            return []

        candidates = extraction_candidates(company_name, records, extracted_by_url)
        selected: list[str] = []
        selected_by_topic: dict[EvidenceTopic, int] = {}
        for query, result, source in candidates:
            if result.url in selected:
                continue
            if should_use_snippet_only(source, result, self.extract_protected_domains):
                continue
            topic_count = selected_by_topic.get(query.topic, 0)
            if topic_count >= self.max_extract_urls_per_topic:
                continue
            selected.append(result.url)
            selected_by_topic[query.topic] = topic_count + 1
            if len(selected) >= remaining:
                break
        return selected


def extraction_candidates(
    company_name: str,
    records: list[tuple[SearchQuery, SearchResult]],
    extracted_by_url: dict[str, ExtractedContent | None],
) -> list[tuple[SearchQuery, SearchResult, ScoredSource]]:
    candidates = []
    seen: set[str] = set()
    for query, result in records:
        if result.url in seen or result.url in extracted_by_url:
            continue
        seen.add(result.url)
        source = score_search_result(company_name, result)
        candidates.append((query, result, source))
    candidates.sort(key=lambda item: extraction_rank(item[0], item[1], item[2]), reverse=True)
    return candidates


def extraction_rank(
    query: SearchQuery,
    result: SearchResult,
    source: ScoredSource,
) -> tuple[int, int, int, int]:
    return (
        1 if source.source_type in {SourceType.official, SourceType.career_page} else 0,
        topic_source_preference(source.source_type, {query.topic}),
        source.reliability_score,
        -result.rank,
    )


def should_use_snippet_only(
    source: ScoredSource,
    result: SearchResult,
    extract_protected_domains: bool,
) -> bool:
    if extract_protected_domains:
        return False
    if not result.snippet.strip():
        return False
    if source.source_type in {
        SourceType.job_board,
        SourceType.linkedin,
        SourceType.salary_review_platform,
    }:
        return True
    return any(marker in source.domain for marker in PROTECTED_DOMAIN_MARKERS)


def build_candidates(
    company_name: str,
    records: list[tuple[SearchQuery, SearchResult]],
    extracted_by_url: dict[str, ExtractedContent | None],
) -> dict[str, SourceCandidate]:
    candidates_by_url: dict[str, SourceCandidate] = {}
    for query, result in records:
        candidate = candidates_by_url.get(result.url)
        if candidate is None:
            source = score_search_result(company_name, result)
            candidate = SourceCandidate(
                source=source,
                discovered_topics={query.topic},
                evidence=[],
                best_rank=result.rank,
            )
            candidates_by_url[result.url] = candidate
        else:
            candidate.discovered_topics.add(query.topic)
            candidate.best_rank = min(candidate.best_rank, result.rank)

        content = extracted_by_url.get(result.url) or snippet_content(result)
        if content is None:
            continue
        candidate.evidence.extend(
            ensure_topic_evidence(
                candidate.source,
                content,
                query.topic,
                classify_evidence(candidate.source, content),
            )
        )
    return candidates_by_url


def snippet_content(result) -> ExtractedContent | None:
    if not result.snippet.strip():
        return None
    return ExtractedContent(
        url=result.url,
        title=result.title,
        text=result.snippet,
    )


def ensure_topic_evidence(
    source: ScoredSource,
    content: ExtractedContent,
    query_topic: EvidenceTopic,
    evidence: list[ClassifiedEvidence],
) -> list[ClassifiedEvidence]:
    if any(item.topic == query_topic for item in evidence):
        return evidence
    return [
        *evidence,
        ClassifiedEvidence(
            source_id=source.source_id,
            topic=query_topic,
            claim=topic_fallback_claim(query_topic, source.title),
            raw_text_excerpt=content.text[:320],
            confidence=ConfidenceLevel.low,
        ),
    ]


def topic_fallback_claim(topic: EvidenceTopic, title: str) -> str:
    labels = {
        EvidenceTopic.business: "negocio principal",
        EvidenceTopic.argentina_presence: "presencia en Argentina",
        EvidenceTopic.employees: "cantidad de empleados",
        EvidenceTopic.salary: "sueldos",
        EvidenceTopic.benefits: "beneficios",
        EvidenceTopic.culture: "cultura laboral",
        EvidenceTopic.interview_process: "proceso de entrevista",
        EvidenceTopic.interview_questions: "preguntas de entrevista",
        EvidenceTopic.open_roles: "busquedas abiertas",
        EvidenceTopic.general: "informacion general",
    }
    return f"La fuente fue encontrada para investigar {labels[topic]} y requiere verificacion: {title}."


def select_balanced_candidates(candidates: list[SourceCandidate]) -> list[SourceCandidate]:
    selected_by_url: dict[str, SourceCandidate] = {}
    for topics, limit in TOPIC_GROUP_LIMITS:
        matching = [candidate for candidate in candidates if candidate_matches_topics(candidate, topics)]
        matching.sort(key=lambda candidate: candidate_rank(candidate, topics), reverse=True)
        added = 0
        for candidate in matching:
            if candidate.source.url in selected_by_url:
                continue
            selected_by_url[candidate.source.url] = candidate
            added += 1
            if added >= limit:
                break

    if not selected_by_url:
        return candidates
    return list(selected_by_url.values())


def topics_needing_deepening(candidates: list[SourceCandidate]) -> set[EvidenceTopic]:
    found = {
        item.topic
        for candidate in candidates
        for item in candidate.evidence
        if item.confidence != ConfidenceLevel.low
    }
    return DEEPENING_TOPICS - found


def candidate_matches_topics(candidate: SourceCandidate, topics: set[EvidenceTopic]) -> bool:
    evidence_topics = {item.topic for item in candidate.evidence}
    return bool(candidate.discovered_topics & topics or evidence_topics & topics)


def candidate_rank(candidate: SourceCandidate, topics: set[EvidenceTopic]) -> tuple[int, int, int, int]:
    evidence_topics = {item.topic for item in candidate.evidence}
    topic_match = 2 if evidence_topics & topics else 1
    return (
        topic_match,
        topic_source_preference(candidate.source.source_type, topics),
        candidate.source.reliability_score,
        -candidate.best_rank,
    )


def topic_source_preference(source_type: SourceType, topics: set[EvidenceTopic]) -> int:
    if topics & {EvidenceTopic.salary, EvidenceTopic.benefits, EvidenceTopic.culture}:
        if source_type == SourceType.salary_review_platform:
            return 3
        if source_type == SourceType.job_board:
            return 2
    if topics & {EvidenceTopic.interview_process, EvidenceTopic.interview_questions}:
        if source_type in {SourceType.salary_review_platform, SourceType.job_board}:
            return 2
    if topics & {EvidenceTopic.open_roles} and source_type in {
        SourceType.career_page,
        SourceType.job_board,
    }:
        return 2
    if topics & {EvidenceTopic.argentina_presence} and source_type in {
        SourceType.official,
        SourceType.career_page,
        SourceType.linkedin,
    }:
        return 2
    return 1


def dedupe_evidence(items) -> list[ClassifiedEvidence]:
    seen: set[tuple[str, EvidenceTopic, str, str]] = set()
    deduped = []
    for item in items:
        key = (item.source_id, item.topic, item.claim, item.raw_text_excerpt)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def count_extraction_skipped(
    records: list[tuple[SearchQuery, SearchResult]],
    extracted_by_url: dict[str, ExtractedContent | None],
) -> int:
    return len(unique_results_by_url(records)) - len(extracted_by_url)


def count_snippet_only(
    records: list[tuple[SearchQuery, SearchResult]],
    extracted_by_url: dict[str, ExtractedContent | None],
) -> int:
    return sum(
        1
        for url, result in unique_results_by_url(records).items()
        if url not in extracted_by_url and result.snippet.strip()
    )


def unique_results_by_url(records: list[tuple[SearchQuery, SearchResult]]) -> dict[str, SearchResult]:
    results_by_url = {}
    for _, result in records:
        results_by_url.setdefault(result.url, result)
    return results_by_url


def elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
