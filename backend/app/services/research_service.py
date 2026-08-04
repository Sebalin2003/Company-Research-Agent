from __future__ import annotations

from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.db.repositories import ReportRepository
from backend.app.db.session import SessionLocal
from backend.app.domain.reports import ReportStatus
from backend.app.services.rag_indexing import schedule_report_embedding_task
from backend.app.services.report_builder import build_report_builder


class ResearchService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.report_repo = ReportRepository(db)

    def run_mock_generation(
        self,
        report_id: str,
        include_cv: bool,
        include_cv_tailoring: bool,
        include_adapted_cv_draft: bool,
        cv_text: str | None = None,
    ) -> None:
        report = self.report_repo.get_by_id(report_id)
        if not report:
            return

        try:
            self.report_repo.update_status(report, ReportStatus.running)
            self.db.commit()

            settings = get_settings()
            structured_report = build_report_builder(settings).build(
                report=report,
                include_cv=include_cv,
                include_cv_tailoring=include_cv_tailoring,
                include_adapted_cv_draft=include_adapted_cv_draft,
                cv_text=cv_text,
            )
            self.report_repo.save_structured_report(report, structured_report)
            self.db.commit()
        except Exception:
            self.db.rollback()
            report = self.report_repo.get_by_id(report_id)
            if report:
                self.report_repo.update_status(
                    report,
                    ReportStatus.failed,
                    error_message="No se pudo generar el informe. Revisa la configuracion de busqueda y modelo.",
                )
                self.db.commit()
        else:
            try:
                schedule_report_embedding_task(report.id)
            except Exception:
                pass


def run_mock_generation_task(
    report_id: str,
    include_cv: bool,
    include_cv_tailoring: bool,
    include_adapted_cv_draft: bool,
    cv_text: str | None = None,
) -> None:
    db = SessionLocal()
    try:
        ResearchService(db).run_mock_generation(
            report_id=report_id,
            include_cv=include_cv,
            include_cv_tailoring=include_cv_tailoring,
            include_adapted_cv_draft=include_adapted_cv_draft,
            cv_text=cv_text,
        )
    finally:
        db.close()
