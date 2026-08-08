from __future__ import annotations

from types import SimpleNamespace
from collections.abc import Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.api.errors import register_error_handlers
from backend.app.api.routes import router
from backend.app.core.config import Settings
from backend.app.db import models
from backend.app.db.models import Base
from backend.app.db.repositories import ReportRepository
from backend.app.db.session import get_db
from backend.app.domain.reports import ReportStatus
from backend.app.services.research_service import ResearchService
from backend.app.services.rag import RAGChunk, choose_scope, rank_chunks
from backend.app.services.rag_indexing import run_report_embedding_task
from backend.app.services.report_chat import ChatAnswerPayload


@pytest.fixture()
def db_session() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client(db_session: Session, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("SEARCH_PROVIDER", "mock")
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(router)

    def override_get_db() -> Generator[Session, None, None]:
        yield db_session

    def run_generation_in_test(
        report_id: str,
        include_cv: bool,
        include_cv_tailoring: bool,
        include_adapted_cv_draft: bool,
        cv_text: str | None = None,
        job_description: str | None = None,
    ) -> None:
        ResearchService(db_session).run_mock_generation(
            report_id=report_id,
            include_cv=include_cv,
            include_cv_tailoring=include_cv_tailoring,
            include_adapted_cv_draft=include_adapted_cv_draft,
            cv_text=cv_text,
            job_description=job_description,
        )

    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr("backend.app.api.routes.run_mock_generation_task", run_generation_in_test)
    monkeypatch.setattr(
        "backend.app.services.research_service.schedule_report_embedding_task",
        lambda report_id: None,
    )
    return TestClient(app)


class FakeEmbeddingService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings(gemini_api_key="fake")

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "salario" in lowered or "sueldo" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "entrevista" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


def test_report_completion_schedules_embedding_indexing_without_waiting(
    client: TestClient,
    db_session: Session,
    monkeypatch,
) -> None:
    scheduled_report_ids = []
    monkeypatch.setattr(
        "backend.app.services.research_service.schedule_report_embedding_task",
        scheduled_report_ids.append,
    )

    created = client.post("/api/research", json={"company_name": "Acme"}).json()
    report = client.get(created["status_url"]).json()["report"]

    chunks = db_session.query(models.ReportEmbeddingChunk).all()
    assert report["status"] == "completed"
    assert report["metadata"]["rag_index_status"] == "pending"
    assert report["metadata"]["rag_indexed_chunk_count"] == 0
    assert scheduled_report_ids == [created["report_id"]]
    assert chunks == []


def test_background_indexing_marks_report_ready_and_creates_chunks(
    client: TestClient,
    db_session: Session,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "backend.app.services.rag_indexing.GoogleEmbeddingService",
        FakeEmbeddingService,
    )
    patch_indexing_session(monkeypatch, db_session)
    created = client.post("/api/research", json={"company_name": "Acme"}).json()

    run_report_embedding_task(created["report_id"])

    report = client.get(created["status_url"]).json()["report"]
    chunks = db_session.query(models.ReportEmbeddingChunk).all()
    assert report["metadata"]["rag_index_status"] == "ready"
    assert report["metadata"]["rag_indexed_chunk_count"] == len(chunks)
    assert chunks
    assert all("cv" not in chunk.chunk_type for chunk in chunks)
    assert all(chunk.embedding_model == "gemini-embedding-2" for chunk in chunks)


def test_embedding_failure_leaves_report_completed(
    client: TestClient,
    db_session: Session,
    monkeypatch,
) -> None:
    class FailingEmbeddingService:
        def __init__(self, settings: Settings | None = None) -> None:
            self.settings = settings or Settings(gemini_api_key="fake")

        def embed(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embedding boom")

    monkeypatch.setattr(
        "backend.app.services.rag_indexing.GoogleEmbeddingService",
        FailingEmbeddingService,
    )
    patch_indexing_session(monkeypatch, db_session)
    created = client.post("/api/research", json={"company_name": "FailCo"}).json()

    run_report_embedding_task(created["report_id"])

    report = client.get(created["status_url"]).json()["report"]
    assert report["status"] == "completed"
    assert report["metadata"]["rag_index_status"] == "failed"
    assert "embedding boom" in report["metadata"]["rag_index_error"]


def test_reindex_backfills_completed_reports_missing_chunks(
    client: TestClient,
    db_session: Session,
    monkeypatch,
) -> None:
    created = client.post("/api/research", json={"company_name": "BackfillCo"}).json()
    report_id = created["report_id"]
    db_session.query(models.ReportEmbeddingChunk).delete()
    db_session.commit()
    monkeypatch.setattr(
        "backend.app.services.rag_indexing.GoogleEmbeddingService",
        FakeEmbeddingService,
    )

    response = client.post("/api/rag/reindex")

    assert response.status_code == 200
    assert response.json()["indexed_reports"] == 1
    report = client.get(f"/api/reports/{report_id}").json()["report"]
    assert report["metadata"]["rag_index_status"] == "ready"
    assert db_session.query(models.ReportEmbeddingChunk).filter_by(report_id=report_id).count() > 0


def test_vector_search_ranks_relevant_chunks_and_boosts_active_report() -> None:
    chunks = [
        chunk("report_a", "Acme", "salarios junior IT", [1.0, 0.0, 0.0]),
        chunk("report_b", "Globant", "proceso entrevista", [0.0, 1.0, 0.0]),
        chunk("report_c", "YPF", "salarios trainee IT", [0.9, 0.1, 0.0]),
    ]

    ranked = rank_chunks([1.0, 0.0, 0.0], chunks, top_k=2, active_report_id=None, scope_used="all_reports")
    active_ranked = rank_chunks(
        [1.0, 0.0, 0.0],
        chunks,
        top_k=2,
        active_report_id="report_c",
        scope_used="active_report",
    )

    assert ranked[0].report_id == "report_a"
    assert active_ranked[0].report_id == "report_c"


def test_scope_detection_uses_all_reports_for_comparative_questions() -> None:
    assert choose_scope("Compará salarios junior entre empresas", "report_a") == "all_reports"
    assert choose_scope("Que deberia preparar para la entrevista?", "report_a") == "active_report"


def test_global_chat_endpoint_saves_messages_and_citations(
    client: TestClient,
    db_session: Session,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "backend.app.services.rag_indexing.GoogleEmbeddingService",
        FakeEmbeddingService,
    )
    created = client.post("/api/research", json={"company_name": "Acme"}).json()
    report_id = created["report_id"]
    run_report_embedding_task_with_session(monkeypatch, db_session, report_id)

    class FakeRAGChatService:
        def __init__(self, settings) -> None:
            pass

        def answer(self, message, chunks, source_lookup, active_report_id=None):
            first_chunk = chunks[0]
            return SimpleNamespace(
                answer="Segun los informes guardados, Acme tiene evidencia disponible.",
                scope_used="active_report",
                citations=[
                    SimpleNamespace(
                        report_id=first_chunk.report_id,
                        company_name=first_chunk.company_name,
                        chunk_id=first_chunk.id,
                        chunk_title=first_chunk.chunk_title,
                        source_id=first_chunk.source_ids[0] if first_chunk.source_ids else None,
                        title="Fuente Acme",
                        url="https://example.com/acme",
                    )
                ],
            )

    monkeypatch.setattr("backend.app.api.routes.RAGChatService", FakeRAGChatService)

    response = client.post(
        "/api/chat",
        json={"message": "Que deberia preparar?", "active_report_id": report_id},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["scope_used"] == "active_report"
    assert payload["citations"][0]["report_id"] == report_id
    assert db_session.query(models.GlobalChatMessage).count() == 2


def test_global_chat_falls_back_to_active_report_before_indexing(
    client: TestClient,
    db_session: Session,
    monkeypatch,
) -> None:
    class FakeReportChatService:
        def __init__(self, settings) -> None:
            pass

        def answer(self, report, message: str) -> ChatAnswerPayload:
            return ChatAnswerPayload(
                answer="Respuesta desde el informe activo mientras se indexa.",
                source_ids=[report.sources[0].id],
            )

    monkeypatch.setattr("backend.app.api.routes.ReportChatService", FakeReportChatService)
    created = client.post("/api/research", json={"company_name": "FallbackCo"}).json()
    report_id = created["report_id"]

    response = client.post(
        "/api/chat",
        json={"message": "Que deberia preparar?", "active_report_id": report_id},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["scope_used"] == "active_report_pending_index"
    assert payload["answer"].startswith("Respuesta desde el informe activo")
    assert payload["citations"][0]["report_id"] is None
    assert db_session.query(models.GlobalChatMessage).count() == 2


def test_delete_report_removes_chunks_and_related_global_chat(
    client: TestClient,
    db_session: Session,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "backend.app.services.rag_indexing.GoogleEmbeddingService",
        FakeEmbeddingService,
    )
    created = client.post("/api/research", json={"company_name": "DeleteRagCo"}).json()
    report_id = created["report_id"]
    run_report_embedding_task_with_session(monkeypatch, db_session, report_id)
    repo = ReportRepository(db_session)
    repo.save_global_chat_exchange(
        user_message="Pregunta",
        answer=SimpleNamespace(answer="Respuesta", scope_used="active_report", citations=[]),
        citations=[],
        active_report_id=report_id,
    )
    db_session.commit()

    response = client.delete(f"/api/reports/{report_id}")

    assert response.status_code == 200
    assert db_session.query(models.ReportEmbeddingChunk).count() == 0
    assert db_session.query(models.GlobalChatMessage).count() == 0


def test_delete_all_reports_removes_chunks_and_global_chat(
    client: TestClient,
    db_session: Session,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "backend.app.services.rag_indexing.GoogleEmbeddingService",
        FakeEmbeddingService,
    )
    created = client.post("/api/research", json={"company_name": "BulkRagCo"}).json()
    report_id = created["report_id"]
    run_report_embedding_task_with_session(monkeypatch, db_session, report_id)
    ReportRepository(db_session).save_global_chat_exchange(
        user_message="Pregunta",
        answer=SimpleNamespace(answer="Respuesta", scope_used="all_reports", citations=[]),
        citations=[],
        active_report_id=report_id,
    )
    db_session.commit()

    response = client.delete("/api/reports")

    assert response.status_code == 200
    assert db_session.query(models.ReportEmbeddingChunk).count() == 0
    assert db_session.query(models.GlobalChatMessage).count() == 0


def test_cv_data_is_not_embedded(
    client: TestClient,
    db_session: Session,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "backend.app.services.rag_indexing.GoogleEmbeddingService",
        FakeEmbeddingService,
    )
    client.post(
        "/api/research",
        json={
            "company_name": "PrivacyCo",
            "cv_text": "CV privado con telefono y experiencia personal",
            "include_cv_tailoring": True,
        },
    )
    report_id = db_session.query(models.Report).one().id
    run_report_embedding_task_with_session(monkeypatch, db_session, report_id)

    indexed_text = "\n".join(
        chunk.chunk_text for chunk in db_session.query(models.ReportEmbeddingChunk).all()
    )
    assert "telefono" not in indexed_text.lower()
    assert "cv privado" not in indexed_text.lower()


def patch_indexing_session(monkeypatch: pytest.MonkeyPatch, db_session: Session) -> None:
    TestingSessionLocal = sessionmaker(
        bind=db_session.get_bind(),
        autoflush=False,
        autocommit=False,
    )
    monkeypatch.setattr("backend.app.services.rag_indexing.SessionLocal", TestingSessionLocal)


def run_report_embedding_task_with_session(
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
    report_id: str,
) -> None:
    patch_indexing_session(monkeypatch, db_session)
    run_report_embedding_task(report_id)


def chunk(report_id: str, company_name: str, text: str, embedding: list[float]) -> RAGChunk:
    return RAGChunk(
        id=f"{report_id}_chunk",
        report_id=report_id,
        company_id=f"{report_id}_company",
        company_name=company_name,
        chunk_key=text,
        chunk_type="section",
        chunk_title=text,
        chunk_text=text,
        source_ids=[],
        evidence_ids=[],
        embedding=embedding,
        embedding_provider="google",
        embedding_model="gemini-embedding-2",
        embedding_dimensions=len(embedding),
    )
