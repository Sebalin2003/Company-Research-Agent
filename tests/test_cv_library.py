from __future__ import annotations

from collections.abc import Generator
from io import BytesIO
from pathlib import Path

import pytest
from docx import Document
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.api.errors import register_error_handlers
from backend.app.api.routes import router
from backend.app.db import models
from backend.app.db.models import Base
from backend.app.db.session import get_db
from backend.app.services.agent_orchestrator import AgentOrchestrator, AgentPaused, AgentState
from backend.app.services.cv_library import CVLibraryError, CVLibraryService


@pytest.fixture()
def db_session() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client(
    db_session: Session, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> TestClient:
    monkeypatch.setenv("CV_STORAGE_DIR", str(tmp_path / "cvs"))
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(router)

    def override_get_db() -> Generator[Session, None, None]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


def docx_bytes(text: str) -> bytes:
    document = Document()
    document.add_paragraph(text)
    output = BytesIO()
    document.save(output)
    return output.getvalue()


def upload_cv(client: TestClient, name: str = "CV Junior") -> dict:
    response = client.post(
        "/api/cvs",
        data={"display_name": name},
        files={
            "file": (
                "cv.docx",
                docx_bytes("Desarrollador junior con Python, SQL e ingles avanzado."),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_cv_upload_list_text_edit_and_file_lifecycle(client: TestClient, tmp_path: Path) -> None:
    created = upload_cv(client)
    cv_id = created["cv_id"]
    upload_version = created["current_version_id"]
    assert created["is_default"] is True
    assert created["structured_profile"]["hard_skills"] == ["python", "sql"]

    listing = client.get("/api/cvs").json()["items"]
    assert "extracted_text" not in listing[0]
    assert listing[0]["display_name"] == "CV Junior"

    edited = client.patch(
        f"/api/cvs/{cv_id}",
        json={
            "display_name": "CV Backend",
            "extracted_text": "Desarrollador backend junior con Python y SQL.",
            "source_version_id": upload_version,
        },
    )
    assert edited.status_code == 200
    assert edited.json()["current_version"]["created_from"] == "text_edit"
    assert len(edited.json()["versions"]) == 2

    file_response = client.get(f"/api/cvs/{cv_id}/versions/{upload_version}/file")
    assert file_response.status_code == 200
    assert file_response.content

    deleted_version = client.delete(
        f"/api/cvs/{cv_id}/versions/{edited.json()['current_version_id']}"
    )
    assert deleted_version.status_code == 204
    assert client.get(f"/api/cvs/{cv_id}").json()["current_version_id"] == upload_version

    stored_files = list((tmp_path / "cvs").rglob("*.docx"))
    assert len(stored_files) == 1
    assert client.delete(f"/api/cvs/{cv_id}").status_code == 204
    assert not stored_files[0].exists()


def test_first_cv_default_and_switching_default_is_atomic(client: TestClient) -> None:
    first = upload_cv(client, "Primero")
    second = upload_cv(client, "Segundo")
    assert second["is_default"] is False

    response = client.patch(f"/api/cvs/{second['cv_id']}", json={"is_default": True})
    assert response.status_code == 200
    defaults = [item for item in client.get("/api/cvs").json()["items"] if item["is_default"]]
    assert [item["cv_id"] for item in defaults] == [second["cv_id"]]
    assert first["cv_id"] != second["cv_id"]


def test_job_description_and_cv_attachments_are_persisted(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.app.llm.deepseek import DeepSeekResponse, DeepSeekToolCall

    class DirectClient:
        def select_tool(self, *_args, **_kwargs):
            call = DeepSeekToolCall(
                id="call",
                name="respond",
                arguments={"answer": "Contexto recibido.", "answer_type": "guidance", "claims": [], "warnings": []},
            )
            return DeepSeekResponse(tool_calls=[call], assistant_message={"role": "assistant", "content": None, "tool_calls": []})

    monkeypatch.setattr(
        "backend.app.services.agent_orchestrator.AgentOrchestrator.get_client",
        lambda self: DirectClient(),
    )
    cv = upload_cv(client)
    conversation = client.post("/api/conversations", json={"title": "Postulación"}).json()
    response = client.post(
        f"/api/conversations/{conversation['conversation_id']}/messages",
        json={
            "content": "Usá mi CV para esta vacante.",
            "attachments": [
                {"type": "cv", "artifact_id": cv["cv_id"]},
                {
                    "type": "job_description",
                    "title": "Backend Junior",
                    "content": "Requisito: Python y SQL. Responsabilidad: desarrollar APIs.",
                },
            ],
        },
    )
    assert response.status_code == 202, response.text
    artifacts = list(db_session.scalars(select(models.ConversationArtifact)))
    assert {item.artifact_type for item in artifacts} >= {"cv", "job_description"}
    job = db_session.scalars(select(models.JobDescription)).one()
    assert "Python" in job.raw_text


def test_upload_rejects_unsupported_and_oversized_files(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    unsupported = client.post(
        "/api/cvs", files={"file": ("cv.txt", b"private", "text/plain")}
    )
    assert unsupported.status_code == 400
    assert unsupported.json()["error"]["code"] == "invalid_cv_file"

    monkeypatch.setenv("CV_FILE_MAX_BYTES", "3")
    oversized = client.post(
        "/api/cvs", files={"file": ("cv.docx", b"1234", "application/octet-stream")}
    )
    assert oversized.status_code == 400
    assert oversized.json()["error"]["code"] == "cv_file_too_large"


def test_agent_cv_tools_create_review_pause(
    db_session: Session, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CV_STORAGE_DIR", str(tmp_path / "cvs"))
    library = CVLibraryService(db_session)
    cv, version = library.create_upload(
        filename="cv.docx",
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        content=docx_bytes("Desarrollador junior con Python y SQL."),
        display_name="CV Junior",
    )
    from backend.app.db.conversation_repository import ConversationRepository

    repo = ConversationRepository(db_session)
    conversation = repo.create("Vacante")
    message = repo.add_message(
        conversation, role="user", content="Adaptá mi CV a esta vacante.", status="completed"
    )
    repo.add_artifact_link(
        conversation,
        artifact_type="cv",
        artifact_id=cv.id,
        relationship_type="attached",
        message_id=message.id,
    )
    job = library.create_job_description(
        conversation,
        message,
        "Backend",
        "Requisito: Python y SQL. Responsabilidad: desarrollar APIs.",
    )
    repo.add_artifact_link(
        conversation,
        artifact_type="job_description",
        artifact_id=job.id,
        relationship_type="attached",
        message_id=message.id,
    )
    task = repo.create_task(conversation, message, budget={})
    repo.update_task(task, "running")
    db_session.commit()

    class TailoringClient:
        def generate_json(self, *_args, **_kwargs):
            return {
                "positioning_summary": "Priorizá experiencia backend.",
                "change_suggestions": [
                    {
                        "id": "rewrite_1",
                        "type": "rewrite",
                        "original_text": "Desarrollador junior con Python y SQL.",
                        "suggested_text": "Desarrollador backend junior con Python y SQL.",
                        "reason": "Coincide con la vacante.",
                        "evidence_ids": ["cv_line_1"],
                        "requires_user_confirmation": False,
                        "confidence": "high",
                    }
                ],
                "adapted_cv_draft": None,
                "warnings": [],
            }

    state = AgentState(goal=message.content)
    orchestrator = AgentOrchestrator(db_session, client=TailoringClient())
    orchestrator.tool_get_cv_profile(task, state, {}, {})
    orchestrator.tool_analyze_job_description(task, state, {}, {})
    with pytest.raises(AgentPaused):
        orchestrator.tool_prepare_cv_recommendations(task, state, {}, {})

    db_session.refresh(task)
    artifact = db_session.scalars(select(models.CVRecommendationArtifact)).one()
    events = repo.list_events(conversation.id)
    assert task.status == "awaiting_review"
    assert artifact.cv_version_id == version.id
    assert [item.event_type for item in events[-2:]] == ["artifact.created", "review.required"]
    assert version.extracted_text not in task.working_state_json


def test_review_requires_truth_confirmation_and_can_save_text_version(
    db_session: Session, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CV_STORAGE_DIR", str(tmp_path / "cvs"))
    library = CVLibraryService(db_session)
    cv, version = library.create_upload(
        filename="cv.docx",
        content_type=None,
        content=docx_bytes("Desarrollador junior con Python."),
    )
    conversation = models.Conversation(
        id="conversation",
        title="CV",
        summary=None,
        active_context_json="{}",
        created_at=version.created_at,
        updated_at=version.created_at,
    )
    db_session.add(conversation)
    artifact = models.CVRecommendationArtifact(
        id="recommendation",
        conversation_id=conversation.id,
        task_run_id="task",
        cv_id=cv.id,
        cv_version_id=version.id,
        job_description_id=None,
        status="awaiting_review",
        payload_json='{"change_suggestions":[{"id":"add_1","type":"add_only_if_true","suggested_text":"AWS","original_text":null}]}',
        review_json="{}",
        draft_text=version.extracted_text,
        created_at=version.created_at,
        updated_at=version.created_at,
    )
    db_session.add(artifact)
    db_session.flush()

    with pytest.raises(CVLibraryError) as error:
        library.review_recommendation(
            artifact,
            [{"suggestion_id": "add_1", "decision": "accepted", "truth_confirmed": False}],
            None,
            False,
        )
    assert error.value.code == "cv_truth_confirmation_required"

    saved = library.review_recommendation(
        artifact,
        [{"suggestion_id": "add_1", "decision": "accepted", "truth_confirmed": True}],
        "Desarrollador junior con Python.\nAWS.",
        True,
    )
    assert saved is not None
    assert saved.created_from == "text_edit"
    assert artifact.status == "reviewed"


def test_conversation_and_cv_deletion_keep_ownership_boundaries(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "backend.app.api.routes.run_local_conversation_task", lambda *_args, **_kwargs: None
    )
    cv = upload_cv(client)
    conversation_id = client.post("/api/conversations", json={"title": "Postulación"}).json()[
        "conversation_id"
    ]
    sent = client.post(
        f"/api/conversations/{conversation_id}/messages",
        json={
            "content": "Revisá mi CV para esta vacante.",
            "attachments": [
                {"type": "cv", "artifact_id": cv["cv_id"]},
                {"type": "job_description", "content": "Requisito: Python y SQL."},
            ],
        },
    )
    assert sent.status_code == 202
    task = db_session.get(models.TaskRun, sent.json()["task_run_id"])
    artifact = models.CVRecommendationArtifact(
        id="owned-recommendation",
        conversation_id=conversation_id,
        task_run_id=task.id,
        cv_id=cv["cv_id"],
        cv_version_id=cv["current_version_id"],
        job_description_id=None,
        status="awaiting_review",
        payload_json="{}",
        review_json="{}",
        draft_text="Borrador local",
        created_at=task.created_at,
        updated_at=task.created_at,
    )
    db_session.add(artifact)
    db_session.commit()

    assert client.delete(f"/api/conversations/{conversation_id}").status_code == 204
    assert db_session.get(models.StoredCV, cv["cv_id"]) is not None
    assert db_session.get(models.CVRecommendationArtifact, artifact.id) is None
    assert db_session.scalars(select(models.JobDescription)).all() == []


def test_deleting_cv_removes_links_and_stale_active_context(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "backend.app.api.routes.run_local_conversation_task", lambda *_args, **_kwargs: None
    )
    cv = upload_cv(client)
    conversation_id = client.post("/api/conversations", json={"title": "CV"}).json()[
        "conversation_id"
    ]
    response = client.post(
        f"/api/conversations/{conversation_id}/messages",
        json={"content": "Usá mi CV.", "attachments": [{"type": "cv", "artifact_id": cv["cv_id"]}]},
    )
    assert response.status_code == 202
    assert client.delete(f"/api/cvs/{cv['cv_id']}").status_code == 204

    conversation = db_session.get(models.Conversation, conversation_id)
    db_session.refresh(conversation)
    assert "active_cv_id" not in conversation.active_context_json
    assert db_session.scalars(
        select(models.ConversationArtifact).where(models.ConversationArtifact.artifact_type == "cv")
    ).all() == []
    assert db_session.get(models.Conversation, conversation_id) is not None


def test_instruction_like_rewrite_is_kept_manual(
    db_session: Session, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CV_STORAGE_DIR", str(tmp_path / "cvs"))
    library = CVLibraryService(db_session)
    cv, version = library.create_upload(
        filename="cv.docx",
        content_type=None,
        content=docx_bytes("Desarrollador junior con Python y SQL."),
    )
    conversation = models.Conversation(
        id="manual-conversation",
        title="CV",
        summary=None,
        active_context_json="{}",
        created_at=version.created_at,
        updated_at=version.created_at,
    )
    db_session.add(conversation)
    artifact = models.CVRecommendationArtifact(
        id="manual-recommendation",
        conversation_id=conversation.id,
        task_run_id="manual-task",
        cv_id=cv.id,
        cv_version_id=version.id,
        job_description_id=None,
        status="awaiting_review",
        payload_json='{"change_suggestions":[{"id":"rewrite_1","type":"rewrite","original_text":"Desarrollador junior con Python y SQL.","suggested_text":"Reescribir el resumen para destacar Python."}]}',
        review_json="{}",
        draft_text=version.extracted_text,
        created_at=version.created_at,
        updated_at=version.created_at,
    )
    db_session.add(artifact)
    db_session.flush()

    library.review_recommendation(
        artifact,
        [{"suggestion_id": "rewrite_1", "decision": "accepted"}],
        None,
        False,
    )
    review = artifact.review_json
    assert artifact.draft_text == version.extracted_text
    assert '"manual_suggestion_ids": ["rewrite_1"]' in review
