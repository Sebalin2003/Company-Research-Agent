from __future__ import annotations

import json
from collections.abc import Generator
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.core.config import Settings
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


def test_paused_task_resume_accepts_clarification_but_not_budget_approval(
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
            "content": "Backend junior",
            "selected_option_ids": [],
        },
    )
    assert resumed.status_code == 202
    assert resumed.json()["status"] == "pending"
    assert ConversationRepository(db_session).list_messages(clarification_conversation)[-1].content == "Backend junior"

    removed = client.post(
        f"/api/task-runs/{clarification_task.id}/resume",
        json={"response_type": "approval", "decision": "approved"},
    )
    assert removed.status_code == 422


def test_affirmative_report_clarification_resumes_full_report(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("backend.app.api.routes.run_local_conversation_task", lambda *_: None)
    monkeypatch.setattr(
        "backend.app.api.routes.get_settings",
        lambda: Settings(
            agent_report_max_model_turns=16,
            agent_report_max_searches=12,
            agent_report_max_inspections=20,
            agent_report_max_elapsed_seconds=300,
        ),
    )
    conversation_id = create_conversation(client)["conversation_id"]
    accepted = client.post(
        f"/api/conversations/{conversation_id}/messages",
        json={"content": "Accenture", "attachments": []},
    ).json()
    task = db_session.get(models.TaskRun, accepted["task_run_id"])
    task.status = "needs_clarification"
    task.pause_reason_json = json.dumps(
        {
            "type": "clarification",
            "prompt": "¿Querés que genere un informe completo sobre Accenture como empleador en Argentina?",
            "options": [],
        }
    )
    task.working_state_json = json.dumps(
        {
            "goal": "Accenture",
            "request_intent": "standard",
            "pending_clarification": {
                "question": "¿Querés que genere un informe completo sobre Accenture como empleador en Argentina?",
                "options": [{"id": "accenture_ar", "label": "Sí, informe sobre Accenture Argentina"}],
            },
        }
    )
    db_session.commit()

    resumed = client.post(
        f"/api/task-runs/{task.id}/resume",
        json={
            "response_type": "clarification",
            "content": None,
            "selected_option_ids": ["accenture_ar"],
        },
    )

    assert resumed.status_code == 202
    db_session.refresh(task)
    assert json.loads(task.budget_json)["profile"] == "full_report"
    state = json.loads(task.working_state_json)
    assert state["request_intent"] == "full_report"
    assert state["requires_grounding"] is True
    assert state["company_name"] == "Accenture"
    assert "informe completo" in state["goal"]


def test_company_clarification_option_starts_grounded_standard_research(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("backend.app.api.routes.run_local_conversation_task", lambda *_: None)
    conversation_id = create_conversation(client)["conversation_id"]
    accepted = client.post(
        f"/api/conversations/{conversation_id}/messages",
        json={"content": "stanley", "attachments": []},
    ).json()
    task = db_session.get(models.TaskRun, accepted["task_run_id"])
    task.status = "needs_clarification"
    task.pause_reason_json = json.dumps(
        {"type": "clarification", "prompt": "¿Huawei o Stanley?", "options": []}
    )
    task.working_state_json = json.dumps(
        {
            "goal": "stanley",
            "request_intent": "standard",
            "pending_clarification": {
                "question": "¿Querés investigar una de estas empresas?",
                "options": [{"id": "stanley", "label": "Investigar Stanley en Argentina"}],
            },
        }
    )
    db_session.commit()

    resumed = client.post(
        f"/api/task-runs/{task.id}/resume",
        json={
            "response_type": "clarification",
            "content": None,
            "selected_option_ids": ["stanley"],
        },
    )

    assert resumed.status_code == 202
    db_session.refresh(task)
    state = json.loads(task.working_state_json)
    assert state["company_name"] == "Stanley"
    assert state["requires_grounding"] is True
    assert state["request_intent"] == "standard"
    assert state["pending_clarification"] is None
    assert state["goal"] == "Investigá a Stanley como empleador en Argentina."


def test_message_tasks_use_standard_and_full_report_budget_profiles(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("backend.app.api.routes.run_local_conversation_task", lambda *_: None)
    monkeypatch.setattr(
        "backend.app.api.routes.get_settings",
        lambda: Settings(
            agent_max_model_turns=12,
            agent_max_searches=6,
            agent_max_inspections=10,
            agent_max_elapsed_seconds=180,
            agent_report_max_model_turns=16,
            agent_report_max_searches=12,
            agent_report_max_inspections=20,
            agent_report_max_elapsed_seconds=300,
        ),
    )

    standard_conversation = create_conversation(client)["conversation_id"]
    standard = client.post(
        f"/api/conversations/{standard_conversation}/messages",
        json={"content": "Investigá Acme", "attachments": []},
    ).json()
    standard_task = db_session.get(models.TaskRun, standard["task_run_id"])
    assert json.loads(standard_task.budget_json) == {
        "profile": "standard",
        "model_turns": 12,
        "searches": 6,
        "inspections": 10,
        "elapsed_seconds": 180,
    }

    report_conversation = create_conversation(client)["conversation_id"]
    report = client.post(
        f"/api/conversations/{report_conversation}/messages",
        json={"content": "Generá un informe de Acme", "attachments": []},
    ).json()
    report_task = db_session.get(models.TaskRun, report["task_run_id"])
    assert json.loads(report_task.budget_json) == {
        "profile": "full_report",
        "model_turns": 16,
        "searches": 12,
        "inspections": 20,
        "elapsed_seconds": 300,
    }


def test_startup_retires_legacy_budget_approval_tasks(db_session: Session) -> None:
    repo = ConversationRepository(db_session)
    conversation = repo.create()
    message = repo.add_message(conversation, role="user", content="Investigá Acme", status="completed")
    task = repo.create_task(conversation, message)
    task.status = "awaiting_approval"
    db_session.commit()

    assert repo.fail_interrupted_tasks("Tarea interrumpida.") == 1
    db_session.commit()

    assert task.status == "failed"
    assert task.stopping_reason == "budget_extension_removed"
    assert "ampliación de presupuesto ya no está disponible" in (task.pause_reason_json or "")


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


def test_generated_artifact_without_legacy_message_link_is_ordered_after_prompt(
    client: TestClient, db_session: Session
) -> None:
    conversation_id = create_conversation(client)["conversation_id"]
    conversation = db_session.get(models.Conversation, conversation_id)
    message = ConversationRepository(db_session).add_message(
        conversation,
        role="user",
        content="Generá un informe sobre Acme.",
        status="completed",
    )
    company = CompanyRepository(db_session).get_or_create("Acme")
    report = ReportRepository(db_session).create_report(company.id)
    ConversationRepository(db_session).add_artifact_link(
        conversation,
        artifact_type="report",
        artifact_id=report.id,
        relationship_type="generated",
    )
    db_session.commit()

    detail = client.get(f"/api/conversations/{conversation_id}").json()

    assert detail["artifacts"][0]["message_id"] == message.id


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
    assert cv.json()["error"]["code"] == "invalid_attachment"


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
    task.working_state_json = json.dumps({"stage": "research", "source_ids": ["source-1"]})
    task.usage_json = json.dumps({"provider_requests": 2, "total_tokens": 120})
    db_session.commit()

    assert repo.fail_interrupted_tasks("Reinicio local") == 1
    db_session.commit()
    db_session.refresh(task)
    assert task.status == "failed"
    assert json.loads(task.pause_reason_json)["type"] == "server_restart"
    assert json.loads(task.working_state_json)["source_ids"] == ["source-1"]
    assert json.loads(task.usage_json)["total_tokens"] == 120
    event = db_session.scalar(
        select(models.TaskEvent).where(models.TaskEvent.task_run_id == task.id)
    )
    assert event.event_type == "task.failed"
    assert json.loads(event.result_summary_json)["message"] == "Reinicio local"
