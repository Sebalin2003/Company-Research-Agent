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
    deepseek_api_key: str | None = None
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_timeout_seconds: int = 60
    gemini_api_key: str | None = None
    gemini_embedding_model: str = "gemini-embedding-2"
    gemini_embedding_dimensions: int = 768
    gemini_timeout_ms: int = 60_000
    search_provider: str = "mock"
    search_provider_api_key: str | None = None
    report_freshness_days: int = 30
    cv_text_max_characters: int = 50_000
    cv_file_max_bytes: int = 10 * 1024 * 1024
    cv_storage_dir: str = "./data/cvs"
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
    agent_max_model_turns: int = 12
    agent_max_searches: int = 6
    agent_max_inspections: int = 10
    agent_max_elapsed_seconds: int = 180
    agent_report_max_model_turns: int = 16
    agent_report_max_searches: int = 12
    agent_report_max_inspections: int = 20
    agent_report_max_elapsed_seconds: int = 300


def get_settings() -> Settings:
    load_env_file()
    return Settings(
        database_url=os.getenv("DATABASE_URL", Settings.database_url),
        environment=os.getenv("APP_ENV", Settings.environment),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", Settings.deepseek_model),
        deepseek_timeout_seconds=int(
            os.getenv("DEEPSEEK_TIMEOUT_SECONDS", str(Settings.deepseek_timeout_seconds))
        ),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
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
        cv_file_max_bytes=int(
            os.getenv("CV_FILE_MAX_BYTES", str(Settings.cv_file_max_bytes))
        ),
        cv_storage_dir=os.getenv("CV_STORAGE_DIR", Settings.cv_storage_dir),
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
        agent_max_model_turns=int(
            os.getenv("AGENT_MAX_MODEL_TURNS", str(Settings.agent_max_model_turns))
        ),
        agent_max_searches=int(
            os.getenv("AGENT_MAX_SEARCHES", str(Settings.agent_max_searches))
        ),
        agent_max_inspections=int(
            os.getenv("AGENT_MAX_INSPECTIONS", str(Settings.agent_max_inspections))
        ),
        agent_max_elapsed_seconds=int(
            os.getenv("AGENT_MAX_ELAPSED_SECONDS", str(Settings.agent_max_elapsed_seconds))
        ),
        agent_report_max_model_turns=int(
            os.getenv("AGENT_REPORT_MAX_MODEL_TURNS", str(Settings.agent_report_max_model_turns))
        ),
        agent_report_max_searches=int(
            os.getenv("AGENT_REPORT_MAX_SEARCHES", str(Settings.agent_report_max_searches))
        ),
        agent_report_max_inspections=int(
            os.getenv("AGENT_REPORT_MAX_INSPECTIONS", str(Settings.agent_report_max_inspections))
        ),
        agent_report_max_elapsed_seconds=int(
            os.getenv(
                "AGENT_REPORT_MAX_ELAPSED_SECONDS",
                str(Settings.agent_report_max_elapsed_seconds),
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
