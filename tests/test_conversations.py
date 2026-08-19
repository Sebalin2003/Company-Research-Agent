from __future__ import annotations

import json
from collections.abc import Generator
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.db import models
from backend.app.db.conversation_repository import ConversationRepository
from backend.app.db.models import Base
from backend.app.db.repositories import CompanyRepository, ReportRepository
from backend.app.db.session import get_db
from backend.app.llm.deepseek import DeepSeekResponse, DeepSeekToolCall
from backend.app.main import create_app


class DirectResponseClient:
    def select_tool(self, *_args, **_kwargs):
        call = DeepSeekToolCall(
            id="call_1",
            name="respond",
            arguments={
                "answer": "La conversación quedó guardada y DeepSeek respondió mediante el flujo persistente.",
                "answer_type": "guidance",
                "claims": [],
                "warnings": [],
            },
        )
        return DeepSeekResponse(
            tool_calls=[call],
            assistant_message={"role": "assistant", "content": None, "tool_calls": []},
        )


@pytest.fixture()
def db_session() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    testing_session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = testing_session()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client(db_session: Session, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    app = create_app()

    def override_get_db() -> Generator[Session, None, None]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(
        "backend.app.services.agent_orchestrator.AgentOrchestrator.get_client",
        lambda self: DirectResponseClient(),
    )
    return TestClient(app)


def create_conversation(client: TestClient, title: str | None = None) -> dict:
    response = client.post("/api/conversations", json={"title": title})
    assert response.status_code == 201
    return response.json()


def test_conversation_crud_search_and_pagination(client: TestClient) -> None:
    first = create_conversation(client)
    second = create_conversation(client, "Buscar empleo en datos")

    listing = client.get("/api/conversations", params={"q": "datos", "limit": 1}).json()
    assert listing["pagination"] == {"limit": 1, "offset": 0, "total": 1}
    assert listing["items"][0]["conversation_id"] == second["conversation_id"]

    renamed = client.patch(
        f"/api/conversations/{first['conversation_id']}",
        json={"title": "Preparación junior"},
    )
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Preparación junior"

    detail = client.get(f"/api/conversations/{first['conversation_id']}")
    assert detail.status_code == 200
    assert detail.json()["messages"] == []

    deleted = client.delete(f"/api/conversations/{first['conversation_id']}")
    assert deleted.status_code == 204
    missing = client.get(f"/api/conversations/{first['conversation_id']}")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "conversation_not_found"


def test_message_persists_response_events_and_deterministic_title(client: TestClient) -> None:
    conversation = create_conversation(client)
    conversation_id = conversation["conversation_id"]
    content = "Dame un consejo general para organizar mi búsqueda laboral junior"

    accepted = client.post(
        f"/api/conversations/{conversation_id}/messages",
        json={"content": content, "attachments": []},
    )
    assert accepted.status_code == 202
    assert accepted.json()["events_url"] == (
        f"/api/conversations/{conversation_id}/events?after_event_id=0"
    )

    detail = client.get(f"/api/conversations/{conversation_id}").json()
    assert detail["title"].endswith("…")
    assert len(detail["title"]) == 42
    assert [message["role"] for message in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][1]["content"].startswith("La conversación quedó guardada")
    assert detail["current_task"]["status"] == "completed"

    stream = client.get(f"/api/conversations/{conversation_id}/events")
    assert stream.status_code == 200
    assert stream.headers["content-type"].startswith("text/event-stream")
    for event_type in (
        "task.started",
        "task.progress",
        "message.started",
        "message.delta",
        "message.completed",
        "task.completed",
    ):
        assert f"event: {event_type}" in stream.text

    first_event_id = int(stream.text.splitlines()[0].removeprefix("id: "))
    replay = client.get(
        f"/api/conversations/{conversation_id}/events",
        headers={"Last-Event-ID": str(first_event_id)},
    )
    assert f"id: {first_event_id}\n" not in replay.text
    assert "event: task.completed" in replay.text


def test_second_message_stream_starts_after_prior_task_events(client: TestClient) -> None:
    conversation_id = create_conversation(client)["conversation_id"]
    first = client.post(
        f"/api/conversations/{conversation_id}/messages",
        json={"content": "Dame un consejo general para entrevistas", "attachments": []},
    )
    assert first.status_code == 202

    first_stream = client.get(first.json()["events_url"])
    prior_terminal_id = max(
        int(line.removeprefix("id: "))
        for line in first_stream.text.splitlines()
        if line.startswith("id: ")
    )

    second = client.post(
        f"/api/conversations/{conversation_id}/messages",
        json={"content": "Dame otro consejo general para entrevistas", "attachments": []},
    )
    assert second.status_code == 202
    assert second.json()["events_url"].endswith(f"after_event_id={prior_terminal_id}")

    second_stream = client.get(second.json()["events_url"])
    assert f"id: {prior_terminal_id}\n" not in second_stream.text
    assert second_stream.text.count("event: task.completed") == 1


def test_pending_task_conflict_and_cancellation(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("backend.app.api.routes.run_local_conversation_task", lambda *_: None)
    conversation_id = create_conversation(client)["conversation_id"]

    first = client.post(
        f"/api/conversations/{conversation_id}/messages",
        json={"content": "Primer mensaje", "attachments": []},
    )
    assert first.status_code == 202

    conflict = client.post(
        f"/api/conversations/{conversation_id}/messages",
        json={"content": "Segundo mensaje", "attachments": []},
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "conversation_busy"

    task_id = first.json()["task_run_id"]
    cancelled = client.post(f"/api/task-runs/{task_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"

    repeated = client.post(f"/api/task-runs/{task_id}/cancel")
    assert repeated.status_code == 409
    assert repeated.json()["error"]["code"] == "task_not_cancellable"

    stream = client.get(f"/api/conversations/{conversation_id}/events")
    assert "event: task.cancelled" in stream.text


def test_failed_task_retry_reuses_task_without_duplicating_user_message(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("backend.app.api.routes.run_local_conversation_task", lambda *_: None)
    conversation_id = create_conversation(client)["conversation_id"]
    accepted = client.post(
        f"/api/conversations/{conversation_id}/messages",
        json={"content": "¿Google tiene pasantías?", "attachments": []},
    ).json()
    task = db_session.get(models.TaskRun, accepted["task_run_id"])
    task.status = "failed"
    task.pause_reason_json = json.dumps(
        {
            "type": "provider_unavailable",
            "message": "DeepSeek está temporalmente no disponible.",
            "retry_at": "2999-01-01T00:00:00+00:00",
        }
    )
    db_session.commit()
    messages_before = len(ConversationRepository(db_session).list_messages(conversation_id))

    cooldown = client.post(f"/api/task-runs/{task.id}/retry")
    assert cooldown.status_code == 409
    assert cooldown.json()["error"]["code"] == "retry_cooldown_active"

    other_conversation_id = create_conversation(client)["conversation_id"]
    blocked = client.post(
        f"/api/conversations/{other_conversation_id}/messages",
        json={"content": "Otro mensaje", "attachments": []},
    )
    assert blocked.status_code == 429
    assert blocked.json()["error"]["code"] == "provider_cooldown_active"
    assert ConversationRepository(db_session).list_messages(other_conversation_id) == []

    task.pause_reason_json = json.dumps({"type": "provider_unavailable", "retry_at": "2000-01-01T00:00:00+00:00"})
    db_session.commit()
    retried = client.post(f"/api/task-runs/{task.id}/retry")

    assert retried.status_code == 202
    assert retried.json()["task_run_id"] == task.id
    assert retried.json()["status"] == "pending"
    assert len(ConversationRepository(db_session).list_messages(conversation_id)) == messages_before


def test_paused_task_resume_accepts_clarification_and_one_budget_extension(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("backend.app.api.routes.run_local_conversation_task", lambda *_: None)

    clarification_conversation = create_conversation(client)["conversation_id"]
    accepted = client.post(
        f"/api/conversations/{clarification_conversation}/messages",
        json={"content": "Necesito ayuda", "attachments": []},
    ).json()
    clarification_task = db_session.get(models.TaskRun, accepted["task_run_id"])
    clarification_task.status = "needs_clarification"
    clarification_task.pause_reason_json = json.dumps(
        {"type": "clarification", "prompt": "¿Qué rol?", "options": []}
    )
    db_session.commit()

    resumed = client.post(
        f"/api/task-runs/{clarification_task.id}/resume",
        json={
            "response_type": "clarification",
            "decision": None,
            "content": "Backend junior",
            "selected_option_ids": [],
        },
    )
    assert resumed.status_code == 202
    assert resumed.json()["status"] == "pending"
    assert ConversationRepository(db_session).list_messages(clarification_conversation)[-1].content == "Backend junior"

    approval_conversation = create_conversation(client)["conversation_id"]
    approved_task_response = client.post(
        f"/api/conversations/{approval_conversation}/messages",
        json={"content": "Investigá Acme", "attachments": []},
    ).json()
    approval_task = db_session.get(models.TaskRun, approved_task_response["task_run_id"])
    approval_task.status = "awaiting_approval"
    approval_task.pause_reason_json = json.dumps({"type": "budget_extension", "prompt": "¿Continuar?"})
    db_session.commit()

    approved = client.post(
        f"/api/task-runs/{approval_task.id}/resume",
        json={
            "response_type": "approval",
            "decision": "approved",
            "content": None,
            "selected_option_ids": [],
        },
    )
    assert approved.status_code == 202
    db_session.refresh(approval_task)
    budget = json.loads(approval_task.budget_json)
    assert budget["extension_used"] is True
    assert budget["model_turns"] == 18

    approval_task.status = "awaiting_approval"
    db_session.commit()
    repeated = client.post(
        f"/api/task-runs/{approval_task.id}/resume",
        json={"response_type": "approval", "decision": "approved"},
    )
    assert repeated.status_code == 409
    assert repeated.json()["error"]["code"] == "budget_extension_already_used"


def test_report_attachment_survives_conversation_deletion(
    client: TestClient, db_session: Session
) -> None:
    company = CompanyRepository(db_session).get_or_create("Acme")
    report = ReportRepository(db_session).create_report(company.id)
    report_id = report.id
    db_session.commit()
    conversation_id = create_conversation(client)["conversation_id"]

    accepted = client.post(
        f"/api/conversations/{conversation_id}/messages",
        json={
            "content": "Usá este informe como contexto.",
            "attachments": [{"type": "report", "artifact_id": report_id}],
        },
    )
    assert accepted.status_code == 202
    detail = client.get(f"/api/conversations/{conversation_id}").json()
    assert detail["artifacts"][0]["artifact_id"] == report_id

    assert client.delete(f"/api/conversations/{conversation_id}").status_code == 204
    assert db_session.get(models.Report, report_id) is not None


def test_invalid_and_cv_attachments_are_rejected(client: TestClient) -> None:
    conversation_id = create_conversation(client)["conversation_id"]

    missing_report = client.post(
        f"/api/conversations/{conversation_id}/messages",
        json={
            "content": "Usá este informe.",
            "attachments": [{"type": "report", "artifact_id": "missing"}],
        },
    )
    assert missing_report.status_code == 400
    assert missing_report.json()["error"]["code"] == "invalid_attachment"

    cv = client.post(
        f"/api/conversations/{conversation_id}/messages",
        json={
            "content": "Usá mi CV.",
            "attachments": [{"type": "cv", "artifact_id": "cv-1"}],
        },
    )
    assert cv.status_code == 400
    assert cv.json()["error"]["code"] == "cv_library_unavailable"


def test_invalid_event_cursor_and_interrupted_task_recovery(
    client: TestClient,
    db_session: Session,
) -> None:
    conversation_id = create_conversation(client)["conversation_id"]
    invalid = client.get(
        f"/api/conversations/{conversation_id}/events",
        headers={"Last-Event-ID": "not-a-number"},
    )
    assert invalid.status_code == 400
    assert invalid.json()["error"]["code"] == "invalid_event_cursor"

    repo = ConversationRepository(db_session)
    conversation = repo.get(conversation_id)
    message = repo.add_message(
        conversation,
        role="user",
        content="Mensaje pendiente",
        status="completed",
    )
    task = repo.create_task(conversation, message)
    db_session.commit()

    assert repo.fail_interrupted_tasks("Reinicio local") == 1
    db_session.commit()
    db_session.refresh(task)
    assert task.status == "failed"
    event = db_session.scalar(
        select(models.TaskEvent).where(models.TaskEvent.task_run_id == task.id)
    )
    assert event.event_type == "task.failed"
    assert json.loads(event.result_summary_json)["message"] == "Reinicio local"
