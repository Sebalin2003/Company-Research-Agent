from __future__ import annotations

from sqlalchemy.engine import Engine

from backend.app.db.session import engine
from backend.app.services.agent_orchestrator import run_agent_task


def run_local_conversation_task(task_id: str, bind: Engine | None = None) -> None:
    """Compatibility entry point retained for the Phase 1 background-task wiring."""
    run_agent_task(task_id, bind or engine)
