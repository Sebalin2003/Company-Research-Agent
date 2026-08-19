from __future__ import annotations

from backend.app.core.config import Settings
from backend.app.domain.reports import (
    CompanySchema,
    ConfidenceLevel,
    EvidenceSchema,
    EvidenceTopic,
    ReportMetadataSchema,
    ReportStatus,
    SourceSchema,
    SourceType,
    StructuredReportSchema,
)
from backend.app.services.rag import RAGChatService, RAGChunk
from backend.app.services.report_chat import ReportChatService


class FakeDeepSeek:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = []

    def generate_json(self, system_prompt, user_prompt, schema, **kwargs):
        self.calls.append((system_prompt, user_prompt, schema, kwargs))
        return self.payload


class FakeEmbeddingService:
    def embed(self, texts):
        assert texts
        return [[1.0, 0.0] for _ in texts]


def settings() -> Settings:
    return Settings(deepseek_api_key="deepseek-key", gemini_api_key="embedding-key")


def test_report_chat_generates_with_deepseek_and_filters_unknown_sources() -> None:
    client = FakeDeepSeek(
        {"answer": "Acme desarrolla software.", "source_ids": ["source_1", "invented"]}
    )

    answer = ReportChatService(settings(), client=client).answer(sample_report(), "¿Qué hace?")

    assert answer.answer == "Acme desarrolla software."
    assert answer.source_ids == ["source_1"]
    assert client.calls[0][2].__name__ == "ChatAnswerPayload"


def test_rag_uses_google_embedding_boundary_and_deepseek_generation() -> None:
    client = FakeDeepSeek(
        {"answer": "Acme desarrolla software.", "citation_chunk_ids": ["chunk_1"]}
    )
    service = RAGChatService(settings(), client=client)
    service.embedding_service = FakeEmbeddingService()
    chunk = RAGChunk(
        id="chunk_1",
        report_id="report_1",
        company_id="company_1",
        company_name="Acme",
        chunk_key="business",
        chunk_type="evidence",
        chunk_title="Negocio",
        chunk_text="Acme desarrolla software.",
        source_ids=["source_1"],
        evidence_ids=["evidence_1"],
        embedding=[1.0, 0.0],
        embedding_provider="google",
        embedding_model="gemini-embedding-2",
        embedding_dimensions=2,
    )

    answer = service.answer(
        "¿Qué hace Acme?",
        [chunk],
        {"source_1": {"title": "Acme", "url": "https://acme.example"}},
    )

    assert answer.answer == "Acme desarrolla software."
    assert answer.citations[0].source_id == "source_1"
    assert client.calls[0][2].__name__ == "RAGChatPayload"


def sample_report() -> StructuredReportSchema:
    return StructuredReportSchema(
        report_id="report_1",
        company=CompanySchema(id="company_1", name="Acme", normalized_name="acme"),
        status=ReportStatus.completed,
        sections=[],
        sources=[
            SourceSchema(
                id="source_1",
                title="Acme",
                url="https://acme.example",
                domain="acme.example",
                source_type=SourceType.official,
                reliability_score=5,
                accessed_at="2026-08-18T00:00:00+00:00",
            )
        ],
        evidence=[
            EvidenceSchema(
                id="evidence_1",
                source_id="source_1",
                topic=EvidenceTopic.business,
                claim="Acme desarrolla software.",
                raw_text_excerpt="Acme desarrolla software.",
                confidence=ConfidenceLevel.medium,
            )
        ],
        metadata=ReportMetadataSchema(
            search_provider="tavily",
            llm_provider="deepseek",
            llm_model="deepseek-v4-flash",
        ),
    )
