from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.errors import register_error_handlers
from backend.app.api.routes import router
from backend.app.db.conversation_repository import ConversationRepository
from backend.app.db.repositories import ReportRepository
from backend.app.db.session import SessionLocal
from backend.app.db.session import init_db

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
INTERRUPTED_REPORT_MESSAGE = (
    "La investigaci\u00f3n fue interrumpida por un reinicio. "
    "Volv\u00e9 a intentarlo."
)
INTERRUPTED_TASK_MESSAGE = (
    "La tarea fue interrumpida por un reinicio. Volvé a enviar el mensaje para continuar."
)


def recover_interrupted_reports() -> int:
    db = SessionLocal()
    try:
        count = ReportRepository(db).fail_interrupted_reports(INTERRUPTED_REPORT_MESSAGE)
        db.commit()
        return count
    finally:
        db.close()


def recover_interrupted_conversation_tasks() -> int:
    db = SessionLocal()
    try:
        count = ConversationRepository(db).fail_interrupted_tasks(INTERRUPTED_TASK_MESSAGE)
        db.commit()
        return count
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_db()
    recover_interrupted_reports()
    recover_interrupted_conversation_tasks()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Enterprise Research Agent API",
        version="0.1.0",
        description="API para investigar empresas y preparar entrevistas laborales en espanol.",
        lifespan=lifespan,
    )
    register_error_handlers(app)
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    return app


app = create_app()
