from __future__ import annotations

import json
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.core.time import utc_now
from backend.app.db import models


ACTIVE_TASK_STATUSES = {
    "pending",
    "running",
    "needs_clarification",
    "awaiting_approval",
    "awaiting_review",
}
TERMINAL_TASK_STATUSES = {"completed", "cancelled", "failed"}


class ConversationRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, title: str | None = None) -> models.Conversation:
        now = utc_now()
        conversation = models.Conversation(
            id=str(uuid4()),
            title=(title or "Nueva conversación").strip(),
            summary=None,
            active_context_json="{}",
            created_at=now,
            updated_at=now,
        )
        self.db.add(conversation)
        self.db.flush()
        return conversation

    def get(self, conversation_id: str) -> models.Conversation | None:
        return self.db.get(models.Conversation, conversation_id)

    def list(self, *, query: str | None, limit: int, offset: int) -> list[models.Conversation]:
        statement = (
            select(models.Conversation)
            .where(models.Conversation.archived_at.is_(None))
            .order_by(models.Conversation.updated_at.desc())
        )
        if query:
            statement = statement.where(
                func.lower(models.Conversation.title).contains(query.strip().lower())
            )
        return list(self.db.scalars(statement.limit(limit).offset(offset)))

    def count(self, query: str | None = None) -> int:
        statement = select(func.count(models.Conversation.id)).where(
            models.Conversation.archived_at.is_(None)
        )
        if query:
            statement = statement.where(
                func.lower(models.Conversation.title).contains(query.strip().lower())
            )
        return int(self.db.scalar(statement) or 0)

    def rename(self, conversation: models.Conversation, title: str) -> models.Conversation:
        conversation.title = title.strip()
        conversation.updated_at = utc_now()
        self.db.flush()
        return conversation

    def update_summary(self, conversation: models.Conversation, summary: str) -> None:
        conversation.summary = summary.strip()[:4000]
        conversation.updated_at = utc_now()
        self.db.flush()

    def update_active_context(self, conversation: models.Conversation, context: dict) -> None:
        conversation.active_context_json = json.dumps(context, ensure_ascii=False)
        conversation.updated_at = utc_now()
        self.db.flush()

    def delete(self, conversation: models.Conversation) -> None:
        self.db.delete(conversation)
        self.db.flush()

    def list_messages(self, conversation_id: str) -> list[models.ConversationMessage]:
        statement = (
            select(models.ConversationMessage)
            .where(models.ConversationMessage.conversation_id == conversation_id)
            .order_by(models.ConversationMessage.created_at, models.ConversationMessage.id)
        )
        return list(self.db.scalars(statement))

    def add_message(
        self,
        conversation: models.Conversation,
        *,
        role: str,
        content: str,
        status: str,
        citations: list[dict] | None = None,
    ) -> models.ConversationMessage:
        now = utc_now()
        message = models.ConversationMessage(
            id=str(uuid4()),
            conversation_id=conversation.id,
            role=role,
            content=content,
            status=status,
            citations_json=json.dumps(citations or [], ensure_ascii=False),
            created_at=now,
            completed_at=now if status == "completed" else None,
        )
        self.db.add(message)
        conversation.updated_at = now
        self.db.flush()
        return message

    def complete_message(self, message: models.ConversationMessage, content: str) -> None:
        message.content = content
        message.status = "completed"
        message.completed_at = utc_now()
        self.db.flush()

    def update_message_content(self, message: models.ConversationMessage, content: str) -> None:
        message.content = content
        self.db.flush()

    def add_report_attachment(
        self,
        conversation: models.Conversation,
        message: models.ConversationMessage,
        report_id: str,
    ) -> models.ConversationArtifact:
        link = models.ConversationArtifact(
            id=str(uuid4()),
            conversation_id=conversation.id,
            message_id=message.id,
            artifact_type="report",
            artifact_id=report_id,
            relationship_type="attached",
            created_at=utc_now(),
        )
        self.db.add(link)
        self.db.flush()
        return link

    def add_artifact_link(
        self,
        conversation: models.Conversation,
        *,
        artifact_type: str,
        artifact_id: str,
        relationship_type: str,
        message_id: str | None = None,
    ) -> models.ConversationArtifact:
        link = models.ConversationArtifact(
            id=str(uuid4()),
            conversation_id=conversation.id,
            message_id=message_id,
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            relationship_type=relationship_type,
            created_at=utc_now(),
        )
        self.db.add(link)
        self.db.flush()
        return link

    def list_artifacts(self, conversation_id: str) -> list[models.ConversationArtifact]:
        statement = (
            select(models.ConversationArtifact)
            .where(models.ConversationArtifact.conversation_id == conversation_id)
            .order_by(models.ConversationArtifact.created_at, models.ConversationArtifact.id)
        )
        return list(self.db.scalars(statement))

    def create_task(
        self,
        conversation: models.Conversation,
        message: models.ConversationMessage,
        budget: dict | None = None,
    ) -> models.TaskRun:
        now = utc_now()
        task = models.TaskRun(
            id=str(uuid4()),
            conversation_id=conversation.id,
            trigger_message_id=message.id,
            task_type="agent_turn",
            status="pending",
            working_state_json="{}",
            budget_json=json.dumps(budget or {}, ensure_ascii=False),
            usage_json="{}",
            created_at=now,
            updated_at=now,
        )
        self.db.add(task)
        self.db.flush()
        return task

    def get_task(self, task_id: str) -> models.TaskRun | None:
        return self.db.get(models.TaskRun, task_id)

    def active_task(self, conversation_id: str) -> models.TaskRun | None:
        statement = (
            select(models.TaskRun)
            .where(
                models.TaskRun.conversation_id == conversation_id,
                models.TaskRun.status.in_(ACTIVE_TASK_STATUSES),
            )
            .order_by(models.TaskRun.created_at.desc())
        )
        return self.db.scalars(statement).first()

    def latest_task(self, conversation_id: str) -> models.TaskRun | None:
        statement = (
            select(models.TaskRun)
            .where(models.TaskRun.conversation_id == conversation_id)
            .order_by(models.TaskRun.created_at.desc())
        )
        return self.db.scalars(statement).first()

    def provider_retry_at(self) -> str | None:
        tasks = self.db.scalars(
            select(models.TaskRun)
            .where(models.TaskRun.status == "failed")
            .order_by(models.TaskRun.updated_at.desc())
            .limit(20)
        )
        retry_values = [
            str(safe.get("retry_at"))
            for task in tasks
            if (safe := json.loads(task.pause_reason_json or "{}"))
            and safe.get("retry_at")
        ]
        active = [value for value in retry_values if value > utc_now().isoformat()]
        return max(active, default=None)

    def update_task(self, task: models.TaskRun, status: str, reason: str | None = None) -> None:
        now = utc_now()
        task.status = status
        task.stopping_reason = reason
        task.updated_at = now
        if status in TERMINAL_TASK_STATUSES:
            task.completed_at = now
        else:
            task.completed_at = None
        self.db.flush()

    def add_event(
        self,
        task: models.TaskRun,
        event_type: str,
        payload: dict | None = None,
        *,
        tool_name: str | None = None,
        arguments: dict | None = None,
        artifact_refs: list[dict] | None = None,
    ) -> models.TaskEvent:
        event = models.TaskEvent(
            task_run_id=task.id,
            conversation_id=task.conversation_id,
            event_type=event_type,
            tool_name=tool_name,
            arguments_json=(
                json.dumps(arguments, ensure_ascii=False) if arguments is not None else None
            ),
            result_summary_json=json.dumps(payload or {}, ensure_ascii=False),
            artifact_refs_json=json.dumps(artifact_refs or [], ensure_ascii=False),
            created_at=utc_now(),
        )
        self.db.add(event)
        self.db.flush()
        return event

    def get_event(self, event_id: int) -> models.TaskEvent | None:
        return self.db.get(models.TaskEvent, event_id)

    def list_events(self, conversation_id: str, after_id: int = 0) -> list[models.TaskEvent]:
        statement = (
            select(models.TaskEvent)
            .where(
                models.TaskEvent.conversation_id == conversation_id,
                models.TaskEvent.id > after_id,
            )
            .order_by(models.TaskEvent.id)
        )
        return list(self.db.scalars(statement))

    def latest_event(self, conversation_id: str) -> models.TaskEvent | None:
        statement = (
            select(models.TaskEvent)
            .where(models.TaskEvent.conversation_id == conversation_id)
            .order_by(models.TaskEvent.id.desc())
        )
        return self.db.scalars(statement).first()

    def create_comparison(
        self,
        conversation: models.Conversation,
        *,
        title: str,
        report_ids: list[str],
        dimensions: list[str],
        payload: dict,
        citations: list[dict],
        warnings: list[str],
    ) -> models.ComparisonArtifact:
        now = utc_now()
        comparison = models.ComparisonArtifact(
            id=str(uuid4()),
            conversation_id=conversation.id,
            title=title,
            report_ids_json=json.dumps(report_ids, ensure_ascii=False),
            dimensions_json=json.dumps(dimensions, ensure_ascii=False),
            payload_json=json.dumps(payload, ensure_ascii=False),
            citations_json=json.dumps(citations, ensure_ascii=False),
            warnings_json=json.dumps(warnings, ensure_ascii=False),
            created_at=now,
            updated_at=now,
        )
        self.db.add(comparison)
        self.db.flush()
        return comparison

    def get_comparison(self, comparison_id: str) -> models.ComparisonArtifact | None:
        return self.db.get(models.ComparisonArtifact, comparison_id)

    def fail_interrupted_tasks(self, message: str) -> int:
        tasks = list(
            self.db.scalars(
                select(models.TaskRun).where(models.TaskRun.status.in_({"pending", "running"}))
            )
        )
        for task in tasks:
            task.pause_reason_json = json.dumps(
                {"type": "server_restart", "message": message}, ensure_ascii=False
            )
            self.update_task(task, "failed", "server_restarted")
            self.add_event(
                task,
                "task.failed",
                {"task_run_id": task.id, "status": "failed", "message": message},
            )
        return len(tasks)
