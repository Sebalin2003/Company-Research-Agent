from __future__ import annotations

import json

from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, UploadFile, status
from sqlalchemy.orm import Session

from backend.app.api.errors import api_error
from backend.app.api.schemas import (
    ChatCitation,
    ChatRequest,
    ChatResponse,
    CompanyReportsResponse,
    CompanySummary,
    CompletedReportResponse,
    CVExtractResponse,
    DeleteAllReportsResponse,
    DeleteCVDataResponse,
    DeleteReportResponse,
    FailedReportError,
    FailedReportResponse,
    GlobalChatRequest,
    GlobalChatResponse,
    Pagination,
    RAGReindexResponse,
    ReportListItem,
    ReportListResponse,
    ReportProgress,
    ResearchAcceptedResponse,
    ResearchRequest,
    RunningReportResponse,
)
from backend.app.core.config import get_settings
from backend.app.core.time import utc_now
from backend.app.db import models
from backend.app.db.repositories import CompanyRepository, ReportRepository
from backend.app.db.session import get_db
from backend.app.domain.reports import ReportStatus
from backend.app.llm.synthesizer import SynthesisError
from backend.app.services.cv_file_extractor import CVExtractionError, extract_cv_text
from backend.app.services.rag import RAGChatCitation, RAGChatResult, RAGChatService, choose_scope
from backend.app.services.rag_indexing import backfill_missing_report_embeddings
from backend.app.services.report_chat import ReportChatService
from backend.app.services.research_service import run_mock_generation_task


router = APIRouter(prefix="/api")


@router.post("/cv/extract", response_model=CVExtractResponse)
async def extract_cv_file(file: UploadFile = File(...)):
    settings = get_settings()
    content = await file.read()
    try:
        extracted = extract_cv_text(
            filename=file.filename or "cv",
            content_type=file.content_type,
            content=content,
            max_characters=settings.cv_text_max_characters,
        )
    except CVExtractionError as exc:
        raise api_error(
            status.HTTP_400_BAD_REQUEST,
            "invalid_cv_file",
            str(exc),
            {"filename": file.filename},
        ) from exc
    return CVExtractResponse(
        filename=extracted.filename,
        content_type=extracted.content_type,
        character_count=extracted.character_count,
        cv_text=extracted.text,
    )


@router.post(
    "/research",
    response_model=ResearchAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_research_request(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    company_repo = CompanyRepository(db)
    report_repo = ReportRepository(db)

    company = company_repo.get_or_create(request.company_name)
    report = report_repo.create_report(company.id)
    report_repo.update_status(report, ReportStatus.pending)
    db.commit()

    background_tasks.add_task(
        run_mock_generation_task,
        report.id,
        request.cv_text is not None,
        request.include_cv_tailoring,
        request.include_adapted_cv_draft,
        request.cv_text,
    )

    return ResearchAcceptedResponse(
        report_id=report.id,
        status=ReportStatus.pending,
        company=company_summary(company),
        status_url=f"/api/reports/{report.id}",
        reused_existing_report=False,
    )


@router.get("/reports/{report_id}")
def get_report(report_id: str, db: Session = Depends(get_db)):
    report = ReportRepository(db).get_by_id(report_id)
    if not report:
        raise api_error(
            status.HTTP_404_NOT_FOUND,
            "report_not_found",
            "No se encontro el informe solicitado.",
            {"report_id": report_id},
        )

    if report.status == ReportStatus.failed.value:
        return FailedReportResponse(
            report_id=report.id,
            status=ReportStatus.failed,
            company=company_summary(report.company),
            error=FailedReportError(
                code="report_generation_failed",
                message=report.error_message or "No se pudo generar el informe.",
            ),
        )

    if report.status == ReportStatus.completed.value:
        return CompletedReportResponse(report=ReportRepository(db).to_structured_report(report))

    return RunningReportResponse(
        report_id=report.id,
        status=ReportStatus(report.status),
        company=company_summary(report.company),
        progress=ReportProgress(
            stage=report.status,
            message=progress_message(ReportStatus(report.status)),
        ),
    )


@router.post("/reports/{report_id}/chat", response_model=ChatResponse)
def answer_report_chat(
    report_id: str,
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    report_repo = ReportRepository(db)
    report = report_repo.get_by_id(report_id)
    if not report:
        raise api_error(
            status.HTTP_404_NOT_FOUND,
            "report_not_found",
            "No se encontro el informe solicitado.",
            {"report_id": report_id},
        )
    if report.status != ReportStatus.completed.value:
        raise api_error(
            status.HTTP_409_CONFLICT,
            "report_not_ready",
            "Primero genera un informe completo para poder responder con fuentes.",
            {"report_id": report_id, "status": report.status},
        )

    structured_report = report_repo.to_structured_report(report)
    try:
        answer = ReportChatService(get_settings()).answer(structured_report, request.message)
    except SynthesisError as exc:
        raise api_error(
            status.HTTP_502_BAD_GATEWAY,
            "chat_provider_failed",
            "No se pudo responder la pregunta con Gemini. Intentalo nuevamente.",
            {"reason": str(exc)[:300]},
        ) from exc

    citations = chat_citations(structured_report.sources, answer.source_ids)
    report_repo.save_chat_exchange(
        report,
        user_message=request.message,
        answer=answer,
        citations=[citation.model_dump(mode="json") for citation in citations],
    )
    db.commit()
    return ChatResponse(
        report_id=report.id,
        answer=answer.answer,
        citations=citations,
        created_at=utc_now().isoformat(),
    )


@router.post("/chat", response_model=GlobalChatResponse)
def answer_global_chat(
    request: GlobalChatRequest,
    db: Session = Depends(get_db),
):
    report_repo = ReportRepository(db)
    active_report = None
    if request.active_report_id:
        active_report = report_repo.get_by_id(request.active_report_id)
        if not active_report:
            raise api_error(
                status.HTTP_404_NOT_FOUND,
                "report_not_found",
                "No se encontro el informe activo.",
                {"report_id": request.active_report_id},
            )
        if active_report.status != ReportStatus.completed.value:
            raise api_error(
                status.HTTP_409_CONFLICT,
                "report_not_ready",
                "Primero genera un informe completo para poder responder con fuentes.",
                {"report_id": request.active_report_id, "status": active_report.status},
            )

    requested_scope = choose_scope(request.message, request.active_report_id)
    if (
        active_report
        and requested_scope == "active_report"
        and report_repo.count_embedding_chunks(active_report.id) == 0
    ):
        return answer_global_chat_from_active_report(
            report_repo=report_repo,
            report=active_report,
            message=request.message,
            db=db,
        )

    chunks = report_repo.list_embedding_chunks()
    source_lookup = report_repo.source_lookup_for_chunks(chunks)
    try:
        answer = RAGChatService(get_settings()).answer(
            message=request.message,
            chunks=chunks,
            source_lookup=source_lookup,
            active_report_id=request.active_report_id,
        )
    except SynthesisError as exc:
        raise api_error(
            status.HTTP_502_BAD_GATEWAY,
            "chat_provider_failed",
            "No se pudo responder la pregunta con Gemini. Intentalo nuevamente.",
            {"reason": str(exc)[:300]},
        ) from exc

    if active_report and answer.scope_used == "all_reports":
        metadata = safe_json_loads(active_report.metadata_json)
        if metadata.get("rag_index_status", "pending") in {"pending", "indexing"}:
            answer = RAGChatResult(
                answer=(
                    f"{answer.answer}\n\n"
                    "Nota: el informe actual todavia se esta indexando para el chat comparativo."
                ),
                scope_used=answer.scope_used,
                citations=answer.citations,
            )

    citations = rag_chat_citations(answer.citations)
    report_repo.save_global_chat_exchange(
        user_message=request.message,
        answer=answer,
        citations=[citation.model_dump(mode="json") for citation in citations],
        active_report_id=request.active_report_id,
    )
    db.commit()
    return GlobalChatResponse(
        answer=answer.answer,
        scope_used=answer.scope_used,
        citations=citations,
        created_at=utc_now().isoformat(),
    )


def answer_global_chat_from_active_report(
    report_repo: ReportRepository,
    report: models.Report,
    message: str,
    db: Session,
) -> GlobalChatResponse:
    structured_report = report_repo.to_structured_report(report)
    try:
        answer = ReportChatService(get_settings()).answer(structured_report, message)
    except SynthesisError as exc:
        raise api_error(
            status.HTTP_502_BAD_GATEWAY,
            "chat_provider_failed",
            "No se pudo responder la pregunta con Gemini. Intentalo nuevamente.",
            {"reason": str(exc)[:300]},
        ) from exc

    citations = chat_citations(structured_report.sources, answer.source_ids)
    rag_result = RAGChatResult(
        answer=answer.answer,
        scope_used="active_report_pending_index",
        citations=[],
    )
    report_repo.save_global_chat_exchange(
        user_message=message,
        answer=rag_result,
        citations=[citation.model_dump(mode="json") for citation in citations],
        active_report_id=report.id,
    )
    db.commit()
    return GlobalChatResponse(
        answer=answer.answer,
        scope_used="active_report_pending_index",
        citations=citations,
        created_at=utc_now().isoformat(),
    )


@router.post("/rag/reindex", response_model=RAGReindexResponse)
def reindex_rag_embeddings(db: Session = Depends(get_db)):
    try:
        result = backfill_missing_report_embeddings(db, get_settings())
    except SynthesisError as exc:
        raise api_error(
            status.HTTP_502_BAD_GATEWAY,
            "embedding_provider_failed",
            "No se pudieron generar embeddings con Gemini.",
            {"reason": str(exc)[:300]},
        ) from exc
    db.commit()
    return RAGReindexResponse(**result)


@router.get("/reports", response_model=ReportListResponse)
def list_reports(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status_filter: str | None = Query(default=None, alias="status"),
    company_name: str | None = None,
    db: Session = Depends(get_db),
):
    reports = ReportRepository(db).list_reports(
        limit=limit, offset=offset, status=status_filter, company_name=company_name
    )
    items = [report_list_item(report) for report in reports]
    return ReportListResponse(
        items=items,
        pagination=Pagination(limit=limit, offset=offset, total=len(items)),
    )


@router.delete("/reports", response_model=DeleteAllReportsResponse)
def delete_all_reports(db: Session = Depends(get_db)):
    deleted = ReportRepository(db).delete_all_reports()
    db.commit()
    return DeleteAllReportsResponse(**deleted)


@router.get("/companies/{company_id}/reports", response_model=CompanyReportsResponse)
def list_company_reports(
    company_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    company_repo = CompanyRepository(db)
    company = company_repo.get_by_id(company_id)
    if not company:
        raise api_error(
            status.HTTP_404_NOT_FOUND,
            "company_not_found",
            "No se encontro la empresa solicitada.",
            {"company_id": company_id},
        )

    reports = ReportRepository(db).list_company_reports(company_id, limit=limit, offset=offset)
    items = [report_list_item(report) for report in reports]
    return CompanyReportsResponse(
        company=company_summary(company),
        items=items,
        pagination=Pagination(limit=limit, offset=offset, total=len(items)),
    )


@router.delete("/reports/{report_id}/cv-data", response_model=DeleteCVDataResponse)
def delete_report_cv_data(report_id: str, db: Session = Depends(get_db)):
    report_repo = ReportRepository(db)
    report = report_repo.get_by_id(report_id)
    if not report:
        raise api_error(
            status.HTTP_404_NOT_FOUND,
            "report_not_found",
            "No se encontro el informe solicitado.",
            {"report_id": report_id},
        )

    deleted = report_repo.delete_cv_data(report)
    db.commit()
    return DeleteCVDataResponse(report_id=report.id, deleted=deleted)


@router.delete("/reports/{report_id}", response_model=DeleteReportResponse)
def delete_report(report_id: str, db: Session = Depends(get_db)):
    report_repo = ReportRepository(db)
    report = report_repo.get_by_id(report_id)
    if not report:
        raise api_error(
            status.HTTP_404_NOT_FOUND,
            "report_not_found",
            "No se encontro el informe solicitado.",
            {"report_id": report_id},
        )

    deleted_company = report_repo.delete_report(report)
    db.commit()
    return DeleteReportResponse(
        report_id=report_id,
        deleted_report=True,
        deleted_company=deleted_company,
    )


def company_summary(company: models.Company) -> CompanySummary:
    return CompanySummary(
        id=company.id,
        name=company.name,
        normalized_name=company.normalized_name,
    )


def report_list_item(report: models.Report) -> ReportListItem:
    metadata = safe_json_loads(report.metadata_json)
    return ReportListItem(
        report_id=report.id,
        company=company_summary(report.company),
        status=ReportStatus(report.status),
        summary=report.summary,
        generated_at=report.generated_at.isoformat() if report.generated_at else None,
        valid_until=report.valid_until.isoformat() if report.valid_until else None,
        used_cv=bool(metadata.get("used_cv", False)),
        used_cv_tailoring=bool(metadata.get("used_cv_tailoring", False)),
    )


def chat_citations(sources: list, source_ids: list[str]) -> list[ChatCitation]:
    sources_by_id = {source.id: source for source in sources}
    citations = []
    for source_id in source_ids:
        source = sources_by_id.get(source_id)
        if not source:
            continue
        citations.append(
            ChatCitation(
                source_id=source.id,
                title=source.title,
                url=str(source.url),
            )
        )
    return citations


def rag_chat_citations(citations: list[RAGChatCitation]) -> list[ChatCitation]:
    return [
        ChatCitation(
            source_id=citation.source_id or citation.chunk_id,
            title=citation.title,
            url=citation.url or "",
            report_id=citation.report_id,
            company_name=citation.company_name,
            chunk_title=citation.chunk_title,
        )
        for citation in citations
    ]


def safe_json_loads(value: str | None) -> dict:
    if not value:
        return {}
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def progress_message(status_value: ReportStatus) -> str:
    if status_value == ReportStatus.running:
        return "El sistema esta recopilando fuentes y preparando el informe."
    return "El informe esta pendiente de generacion."
