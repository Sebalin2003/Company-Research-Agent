from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.api.routes import router
from backend.app.api.routes import progress_message
from backend.app.api.errors import register_error_handlers
from backend.app.db.models import Base
from backend.app.db.models import CandidateProfile, Company, ReportChatMessage
from backend.app.db.repositories import CompanyRepository, ReportRepository
from backend.app.db.session import get_db
from backend.app.domain.reports import ReportStatus
from backend.app.services.report_chat import ChatAnswerPayload
from backend.app.services.research_service import ResearchService


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
    ) -> None:
        ResearchService(db_session).run_mock_generation(
            report_id=report_id,
            include_cv=include_cv,
            include_cv_tailoring=include_cv_tailoring,
            include_adapted_cv_draft=include_adapted_cv_draft,
            cv_text=cv_text,
        )

    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr("backend.app.api.routes.run_mock_generation_task", run_generation_in_test)
    monkeypatch.setattr(
        "backend.app.services.research_service.schedule_report_embedding_task",
        lambda report_id: None,
    )
    return TestClient(app)


def test_research_request_creates_completed_mock_report(client: TestClient) -> None:
    response = client.post("/api/research", json={"company_name": "Mercado Libre"})

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "pending"
    assert payload["company"]["normalized_name"] == "mercado libre"

    report_response = client.get(payload["status_url"])
    assert report_response.status_code == 200
    report_payload = report_response.json()["report"]
    assert report_payload["status"] == "completed"
    assert report_payload["metadata"]["llm_provider"] == "gemini"
    assert report_payload["metadata"]["source_count"] == 1
    assert "research_duration_ms" in report_payload["metadata"]
    assert "research_extraction_skipped_count" in report_payload["metadata"]
    assert "research_snippet_only_count" in report_payload["metadata"]
    assert report_payload["sources"][0]["source_type"] == "official"
    assert report_payload["sections"][0]["type"] == "executive_summary"


def test_research_with_cv_generates_personalization_and_tailoring(client: TestClient) -> None:
    response = client.post(
        "/api/research",
        json={
            "company_name": "Acme",
            "cv_text": "Analista de datos con experiencia en SQL y reportes.",
            "include_cv_tailoring": True,
        },
    )

    assert response.status_code == 202
    report_id = response.json()["report_id"]

    report_payload = client.get(f"/api/reports/{report_id}").json()["report"]
    assert report_payload["personalized_preparation"] is not None
    assert report_payload["cv_tailoring"] is not None
    assert report_payload["metadata"]["used_cv"] is True
    assert report_payload["metadata"]["used_cv_tailoring"] is True
    assert report_payload["cv_tailoring"]["change_suggestions"][0]["type"] == "emphasize"


def test_cv_tailoring_without_cv_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/research",
        json={"company_name": "Acme", "include_cv_tailoring": True},
    )

    assert response.status_code == 422
    assert "Para adaptar el CV" in response.text


def test_progress_message_does_not_claim_simulated_data() -> None:
    message = progress_message(ReportStatus.running)

    assert "simulado" not in message
    assert "recopilando fuentes" in message


def test_failed_generation_message_does_not_claim_simulated_data(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.app.db.repositories import CompanyRepository, ReportRepository

    company = CompanyRepository(db_session).get_or_create("Acme")
    report = ReportRepository(db_session).create_report(company.id)
    db_session.commit()

    class FailingBuilder:
        def build(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "backend.app.services.research_service.build_report_builder",
        lambda settings: FailingBuilder(),
    )

    ResearchService(db_session).run_mock_generation(report.id, False, False, False)

    failed_report = ReportRepository(db_session).get_by_id(report.id)
    assert failed_report.status == ReportStatus.failed
    assert failed_report.error_message is not None
    assert "simulado" not in failed_report.error_message


def test_report_lists_include_saved_report_summary(client: TestClient) -> None:
    created = client.post("/api/research", json={"company_name": "Globant"}).json()

    response = client.get("/api/reports")

    assert response.status_code == 200
    payload = response.json()
    assert payload["pagination"]["total"] == 1
    assert payload["items"][0]["report_id"] == created["report_id"]
    assert payload["items"][0]["company"]["normalized_name"] == "globant"


def test_repeated_reports_with_same_generated_ids_can_be_saved(client: TestClient) -> None:
    first = client.post("/api/research", json={"company_name": "Acme"}).json()
    second = client.post("/api/research", json={"company_name": "Acme"}).json()

    first_report = client.get(first["status_url"]).json()["report"]
    second_report = client.get(second["status_url"]).json()["report"]

    assert first_report["status"] == "completed"
    assert second_report["status"] == "completed"
    assert first_report["sources"][0]["id"] != second_report["sources"][0]["id"]
    assert first_report["evidence"][0]["source_id"] == first_report["sources"][0]["id"]
    assert second_report["evidence"][0]["source_id"] == second_report["sources"][0]["id"]


def test_delete_cv_data_removes_personalized_sections(client: TestClient) -> None:
    created = client.post(
        "/api/research",
        json={
            "company_name": "Acme",
            "cv_text": "Project manager con experiencia en banca.",
            "include_cv_tailoring": True,
        },
    ).json()
    report_id = created["report_id"]

    delete_response = client.delete(f"/api/reports/{report_id}/cv-data")

    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] == {
        "candidate_profile": True,
        "personalized_preparation": True,
        "cv_tailoring": True,
    }

    report_payload = client.get(f"/api/reports/{report_id}").json()["report"]
    assert report_payload["personalized_preparation"] is None
    assert report_payload["cv_tailoring"] is None


def test_cv_pdf_upload_extracts_text(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakePage:
        def extract_text(self) -> str:
            return "Analista de datos con SQL"

    class FakePdfReader:
        def __init__(self, stream) -> None:
            self.pages = [FakePage()]

    monkeypatch.setattr("backend.app.services.cv_file_extractor.PdfReader", FakePdfReader)

    response = client.post(
        "/api/cv/extract",
        files={"file": ("cv.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["cv_text"] == "Analista de datos con SQL"
    assert response.json()["character_count"] == len("Analista de datos con SQL")


def test_cv_docx_upload_extracts_text(client: TestClient) -> None:
    from io import BytesIO

    from docx import Document

    buffer = BytesIO()
    document = Document()
    document.add_paragraph("Project manager con experiencia regional")
    document.save(buffer)

    response = client.post(
        "/api/cv/extract",
        files={
            "file": (
                "cv.docx",
                buffer.getvalue(),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["cv_text"] == "Project manager con experiencia regional"


def test_cv_upload_rejects_unsupported_file(client: TestClient) -> None:
    response = client.post(
        "/api/cv/extract",
        files={"file": ("cv.txt", b"hola", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_cv_file"


def test_cv_upload_rejects_oversized_extracted_text(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakePage:
        def extract_text(self) -> str:
            return "texto demasiado largo"

    class FakePdfReader:
        def __init__(self, stream) -> None:
            self.pages = [FakePage()]

    monkeypatch.setenv("CV_TEXT_MAX_CHARACTERS", "5")
    monkeypatch.setattr("backend.app.services.cv_file_extractor.PdfReader", FakePdfReader)

    response = client.post(
        "/api/cv/extract",
        files={"file": ("cv.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert response.status_code == 400
    assert "limite" in response.json()["error"]["message"]


def test_report_chat_rejects_missing_running_and_failed_reports(
    client: TestClient,
    db_session: Session,
) -> None:
    missing = client.post("/api/reports/missing/chat", json={"message": "Que hace?"})
    assert missing.status_code == 404

    company = CompanyRepository(db_session).get_or_create("Acme")
    running_report = ReportRepository(db_session).create_report(company.id)
    failed_report = ReportRepository(db_session).create_report(company.id)
    ReportRepository(db_session).update_status(failed_report, ReportStatus.failed, "boom")
    db_session.commit()

    running = client.post(f"/api/reports/{running_report.id}/chat", json={"message": "Que hace?"})
    failed = client.post(f"/api/reports/{failed_report.id}/chat", json={"message": "Que hace?"})

    assert running.status_code == 409
    assert failed.status_code == 409


def test_report_chat_answers_completed_report_and_saves_messages(
    client: TestClient,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeChatService:
        def __init__(self, settings) -> None:
            pass

        def answer(self, report, message: str) -> ChatAnswerPayload:
            return ChatAnswerPayload(
                answer="Acme publica busquedas en Argentina segun su sitio de carreras.",
                source_ids=[report.sources[0].id],
            )

    monkeypatch.setattr("backend.app.api.routes.ReportChatService", FakeChatService)
    created = client.post("/api/research", json={"company_name": "Acme"}).json()
    report = client.get(created["status_url"]).json()["report"]

    response = client.post(
        f"/api/reports/{report['report_id']}/chat",
        json={"message": "Que puestos tiene?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"].startswith("Acme publica")
    assert payload["citations"][0]["url"].startswith("https://")

    messages = db_session.query(ReportChatMessage).all()
    assert [message.role for message in messages] == ["user", "assistant"]


def test_delete_report_removes_related_data_and_last_company(
    client: TestClient,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeChatService:
        def __init__(self, settings) -> None:
            pass

        def answer(self, report, message: str) -> ChatAnswerPayload:
            return ChatAnswerPayload(answer="Respuesta con fuente.", source_ids=[report.sources[0].id])

    monkeypatch.setattr("backend.app.api.routes.ReportChatService", FakeChatService)
    created = client.post(
        "/api/research",
        json={
            "company_name": "SoloCo",
            "cv_text": "Analista con SQL",
            "include_cv_tailoring": True,
        },
    ).json()
    report_id = created["report_id"]
    client.post(f"/api/reports/{report_id}/chat", json={"message": "Que hace?"})

    response = client.delete(f"/api/reports/{report_id}")

    assert response.status_code == 200
    assert response.json() == {
        "report_id": report_id,
        "deleted_report": True,
        "deleted_company": True,
    }
    assert client.get(f"/api/reports/{report_id}").status_code == 404
    assert db_session.query(ReportChatMessage).count() == 0
    assert db_session.query(Company).count() == 0


def test_delete_one_report_keeps_company_with_remaining_reports(
    client: TestClient,
    db_session: Session,
) -> None:
    first = client.post("/api/research", json={"company_name": "MultiCo"}).json()
    second = client.post("/api/research", json={"company_name": "MultiCo"}).json()

    response = client.delete(f"/api/reports/{first['report_id']}")

    assert response.status_code == 200
    assert response.json()["deleted_company"] is False
    assert client.get(f"/api/reports/{second['report_id']}").status_code == 200
    assert db_session.query(Company).count() == 1


def test_delete_all_reports_returns_zero_counts_when_history_is_empty(client: TestClient) -> None:
    response = client.delete("/api/reports")

    assert response.status_code == 200
    assert response.json() == {"deleted_reports": 0, "deleted_companies": 0}


def test_delete_all_reports_removes_related_data_and_orphan_companies(
    client: TestClient,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeChatService:
        def __init__(self, settings) -> None:
            pass

        def answer(self, report, message: str) -> ChatAnswerPayload:
            return ChatAnswerPayload(answer="Respuesta con fuente.", source_ids=[report.sources[0].id])

    monkeypatch.setattr("backend.app.api.routes.ReportChatService", FakeChatService)
    first = client.post(
        "/api/research",
        json={
            "company_name": "BulkCo",
            "cv_text": "Analista con SQL",
            "include_cv_tailoring": True,
        },
    ).json()
    second = client.post("/api/research", json={"company_name": "BulkCo"}).json()
    third = client.post("/api/research", json={"company_name": "OtherCo"}).json()
    client.post(f"/api/reports/{first['report_id']}/chat", json={"message": "Que hace?"})

    response = client.delete("/api/reports")

    assert response.status_code == 200
    assert response.json() == {"deleted_reports": 3, "deleted_companies": 2}
    assert client.get(f"/api/reports/{first['report_id']}").status_code == 404
    assert client.get(f"/api/reports/{second['report_id']}").status_code == 404
    assert client.get(f"/api/reports/{third['report_id']}").status_code == 404
    assert client.get("/api/reports").json()["pagination"]["total"] == 0
    assert db_session.query(ReportChatMessage).count() == 0
    assert db_session.query(CandidateProfile).count() == 0
    assert db_session.query(Company).count() == 0
