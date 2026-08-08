from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = PROJECT_ROOT / ".env"


@dataclass(frozen=True)
class Settings:
    database_url: str = "sqlite:///./enterprise_research_agent.db"
    environment: str = "development"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.1-flash-lite"
    gemini_embedding_model: str = "gemini-embedding-2"
    gemini_embedding_dimensions: int = 768
    gemini_timeout_ms: int = 60_000
    search_provider: str = "mock"
    search_provider_api_key: str | None = None
    report_freshness_days: int = 30
    cv_text_max_characters: int = 50_000
    rag_top_k: int = 8
    research_search_concurrency: int = 4
    research_extract_concurrency: int = 6
    research_results_per_query: int = 3
    research_search_timeout_seconds: int = 12
    research_extract_timeout_seconds: int = 8
    research_enable_deepening: bool = True
    research_max_extract_urls: int = 12
    research_extract_protected_domains: bool = False
    research_max_extract_urls_per_topic: int = 2


def get_settings() -> Settings:
    load_env_file()
    return Settings(
        database_url=os.getenv("DATABASE_URL", Settings.database_url),
        environment=os.getenv("APP_ENV", Settings.environment),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", Settings.gemini_model),
        gemini_embedding_model=os.getenv(
            "GEMINI_EMBEDDING_MODEL", Settings.gemini_embedding_model
        ),
        gemini_embedding_dimensions=int(
            os.getenv(
                "GEMINI_EMBEDDING_DIMENSIONS",
                str(Settings.gemini_embedding_dimensions),
            )
        ),
        gemini_timeout_ms=int(
            os.getenv("GEMINI_TIMEOUT_MS", str(Settings.gemini_timeout_ms))
        ),
        search_provider=os.getenv("SEARCH_PROVIDER", Settings.search_provider),
        search_provider_api_key=os.getenv("SEARCH_PROVIDER_API_KEY") or os.getenv("TAVILY_API_KEY"),
        report_freshness_days=int(
            os.getenv("REPORT_FRESHNESS_DAYS", str(Settings.report_freshness_days))
        ),
        cv_text_max_characters=int(
            os.getenv("CV_TEXT_MAX_CHARACTERS", str(Settings.cv_text_max_characters))
        ),
        rag_top_k=int(os.getenv("RAG_TOP_K", str(Settings.rag_top_k))),
        research_search_concurrency=int(
            os.getenv("RESEARCH_SEARCH_CONCURRENCY", str(Settings.research_search_concurrency))
        ),
        research_extract_concurrency=int(
            os.getenv("RESEARCH_EXTRACT_CONCURRENCY", str(Settings.research_extract_concurrency))
        ),
        research_results_per_query=int(
            os.getenv("RESEARCH_RESULTS_PER_QUERY", str(Settings.research_results_per_query))
        ),
        research_search_timeout_seconds=int(
            os.getenv(
                "RESEARCH_SEARCH_TIMEOUT_SECONDS",
                str(Settings.research_search_timeout_seconds),
            )
        ),
        research_extract_timeout_seconds=int(
            os.getenv(
                "RESEARCH_EXTRACT_TIMEOUT_SECONDS",
                str(Settings.research_extract_timeout_seconds),
            )
        ),
        research_enable_deepening=os.getenv(
            "RESEARCH_ENABLE_DEEPENING",
            str(Settings.research_enable_deepening),
        ).lower()
        in {"1", "true", "yes", "on"},
        research_max_extract_urls=int(
            os.getenv("RESEARCH_MAX_EXTRACT_URLS", str(Settings.research_max_extract_urls))
        ),
        research_extract_protected_domains=os.getenv(
            "RESEARCH_EXTRACT_PROTECTED_DOMAINS",
            str(Settings.research_extract_protected_domains),
        ).lower()
        in {"1", "true", "yes", "on"},
        research_max_extract_urls_per_topic=int(
            os.getenv(
                "RESEARCH_MAX_EXTRACT_URLS_PER_TOPIC",
                str(Settings.research_max_extract_urls_per_topic),
            )
        ),
    )


def load_env_file(path: Path = ENV_FILE) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
