from __future__ import annotations

from urllib.parse import urlparse

from backend.app.domain.companies import normalize_company_name
from backend.app.domain.reports import EvidenceTopic, SourceType
from backend.app.research.types import ScoredSource, SearchResult


SOURCE_TYPE_HINTS: tuple[tuple[str, SourceType], ...] = (
    ("linkedin.com/company", SourceType.linkedin),
    ("linkedin.com/jobs", SourceType.job_board),
    ("glassdoor.", SourceType.salary_review_platform),
    ("indeed.", SourceType.salary_review_platform),
    ("openqube.io", SourceType.salary_review_platform),
    ("zoominfo.", SourceType.company_database),
    ("crunchbase.", SourceType.company_database),
    ("theorg.com", SourceType.company_database),
    ("computrabajo.", SourceType.job_board),
    ("bumeran.", SourceType.job_board),
    ("zonajobs.", SourceType.job_board),
    ("getonbrd.", SourceType.job_board),
    ("portalempleo.gob.ar", SourceType.job_board),
    ("buscojobs.com", SourceType.job_board),
    ("jooble.org", SourceType.job_board),
    ("talent.com", SourceType.job_board),
    ("greenhouse.io", SourceType.career_page),
    ("lever.co", SourceType.career_page),
    ("workable.com", SourceType.career_page),
    ("wikipedia.org", SourceType.company_database),
)
GEOGRAPHIC_SUFFIXES = {"argentina", "latam", "latinoamerica"}


def score_search_result(company_name: str, result: SearchResult) -> ScoredSource:
    domain = extract_domain(result.url)
    source_type = classify_source_type(company_name, result.url, result.query_topic)
    score = reliability_score(source_type, result.url, result.rank)
    return ScoredSource(
        source_id=stable_source_id(result.url),
        title=result.title,
        url=result.url,
        domain=domain,
        source_type=source_type,
        reliability_score=score,
        snippet=result.snippet,
        is_current=True,
    )


def classify_source_type(company_name: str, url: str, topic: EvidenceTopic) -> SourceType:
    normalized_url = url.lower()
    host = extract_domain(url)
    normalized_host = host.replace(".", "").replace("-", "")

    if any(token in normalized_host for token in company_domain_tokens(company_name)):
        if topic == EvidenceTopic.open_roles or "career" in normalized_url or "jobs" in normalized_url:
            return SourceType.career_page
        return SourceType.official

    for needle, source_type in SOURCE_TYPE_HINTS:
        if needle in normalized_url:
            return source_type

    return SourceType.secondary if host else SourceType.unknown


def company_domain_tokens(company_name: str) -> tuple[str, ...]:
    words = normalize_company_name(company_name).split()
    tokens = ["".join(words)] if words else []
    while words and words[-1] in GEOGRAPHIC_SUFFIXES:
        words.pop()
    if words:
        tokens.append("".join(words))
    return tuple(dict.fromkeys(token for token in tokens if len(token) >= 3))


def reliability_score(source_type: SourceType, url: str, rank: int) -> int:
    base = {
        SourceType.official: 5,
        SourceType.career_page: 5,
        SourceType.linkedin: 4,
        SourceType.job_board: 4,
        SourceType.salary_review_platform: 3,
        SourceType.company_database: 3,
        SourceType.news_media: 3,
        SourceType.secondary: 2,
        SourceType.unknown: 1,
    }[source_type]
    if not url.lower().startswith("https://"):
        base -= 1
    if rank > 5:
        base -= 1
    return max(1, min(5, base))


def extract_domain(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.lower().removeprefix("www.")


def stable_source_id(url: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in url.lower()).strip("_")
    return f"source_{safe[:80]}"
