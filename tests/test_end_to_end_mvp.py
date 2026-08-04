from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.db.models import Base
from backend.app.db.session import get_db
from backend.app.main import create_app
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
    app = create_app()

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


def test_mvp_frontend_api_cv_report_history_and_privacy_flow(client: TestClient) -> None:
    page = client.get("/")
    script = client.get("/static/app.js")
    styles = client.get("/static/styles.css")

    assert page.status_code == 200
    assert "Radar Laboral" in page.text
    assert "viewport" in page.text
    assert script.status_code == 200
    assert "include_adapted_cv_draft" in script.text
    assert styles.status_code == 200
    assert "@media (max-width: 760px)" in styles.text

    created = client.post(
        "/api/research",
        json={
            "company_name": "Acme Argentina",
            "cv_text": (
                "Analista de datos semi senior con SQL, Python y Power BI.\n"
                "Optimice reportes comerciales."
            ),
            "include_cv_tailoring": True,
            "include_adapted_cv_draft": True,
        },
    )

    assert created.status_code == 202
    status_url = created.json()["status_url"]
    report_payload = client.get(status_url).json()["report"]

    assert report_payload["status"] == "completed"
    assert report_payload["language"] == "es-AR"
    assert report_payload["metadata"]["used_cv"] is True
    assert report_payload["metadata"]["used_cv_tailoring"] is True
    assert report_payload["sources"][0]["url"].startswith("https://")
    assert report_payload["evidence"][0]["source_id"] == report_payload["sources"][0]["id"]
    assert report_payload["personalized_preparation"]["cv_profile"]["hard_skills"] == [
        "python",
        "sql",
        "power bi",
    ]
    assert report_payload["cv_tailoring"]["adapted_cv_draft"]["content_markdown"]
    assert "Optimice reportes comerciales." in report_payload["cv_tailoring"]["adapted_cv_draft"]["content_markdown"]

    history = client.get("/api/reports").json()
    assert history["pagination"]["total"] == 1
    assert history["items"][0]["used_cv"] is True
    assert history["items"][0]["used_cv_tailoring"] is True

    deleted = client.delete(f"/api/reports/{report_payload['report_id']}/cv-data")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"]["candidate_profile"] is True

    scrubbed_report = client.get(status_url).json()["report"]
    assert scrubbed_report["personalized_preparation"] is None
    assert scrubbed_report["cv_tailoring"] is None
