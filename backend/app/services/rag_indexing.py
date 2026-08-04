from __future__ import annotations

import threading

from sqlalchemy.orm import Session

from backend.app.core.config import Settings, get_settings
from backend.app.db import models
from backend.app.db.repositories import ReportRepository
from backend.app.db.session import SessionLocal
from backend.app.llm.synthesizer import SynthesisError
from backend.app.services.rag import GoogleEmbeddingService, ReportEmbeddingIndexer


def index_report_embeddings(
    db: Session,
    report: models.Report,
    settings: Settings,
    embedding_service: GoogleEmbeddingService | None = None,
) -> int:
    report_repo = ReportRepository(db)
    structured_report = report_repo.to_structured_report(report)
    indexer = ReportEmbeddingIndexer(embedding_service or GoogleEmbeddingService(settings))
    chunks = indexer.build_chunks(structured_report)
    return report_repo.replace_embedding_chunks(report, chunks)


def schedule_report_embedding_task(report_id: str) -> None:
    thread = threading.Thread(
        target=run_report_embedding_task,
        args=(report_id,),
        daemon=True,
    )
    thread.start()


def run_report_embedding_task(report_id: str) -> None:
    db = SessionLocal()
    try:
        settings = get_settings()
        report_repo = ReportRepository(db)
        report = report_repo.get_by_id(report_id)
        if not report or report.status != "completed":
            return

        report_repo.update_rag_index_status(report, "indexing", chunk_count=0, error=None)
        db.commit()

        chunk_count = index_report_embeddings(db, report, settings)
        report_repo.update_rag_index_status(report, "ready", chunk_count=chunk_count, error=None)
        db.commit()
    except Exception as exc:
        db.rollback()
        report_repo = ReportRepository(db)
        report = report_repo.get_by_id(report_id)
        if report:
            report_repo.update_rag_index_status(
                report,
                "failed",
                chunk_count=report_repo.count_embedding_chunks(report_id),
                error=str(exc),
            )
            db.commit()
    finally:
        db.close()


def backfill_missing_report_embeddings(db: Session, settings: Settings) -> dict[str, int]:
    report_repo = ReportRepository(db)
    indexed_reports = 0
    indexed_chunks = 0
    for report in report_repo.list_completed_reports_missing_embeddings():
        report_repo.update_rag_index_status(report, "indexing", chunk_count=0, error=None)
        chunk_count = index_report_embeddings(db, report, settings)
        report_repo.update_rag_index_status(report, "ready", chunk_count=chunk_count, error=None)
        indexed_chunks += chunk_count
        indexed_reports += 1
    return {"indexed_reports": indexed_reports, "indexed_chunks": indexed_chunks}
