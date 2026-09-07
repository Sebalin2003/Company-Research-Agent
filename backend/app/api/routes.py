from __future__ import annotations

import asyncio
import json
import re
import time

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Header, Query, Response, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session, sessionmaker

from backend.app.api.errors import api_error
from backend.app.api.schemas import (
    ChatCitation,
    ChatRequest,
    ChatResponse,
    ComparisonArtifactResponse,
    CompanyReportsResponse,
    CompanySummary,
    CompletedReportResponse,
    ConversationCreatedResponse,
    ConversationCreateRequest,
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationMessageAcceptedResponse,
    ConversationMessageRequest,
    ConversationMessageResponse,
    ConversationArtifactResponse,
    ConversationSummaryResponse,
    ConversationUpdateRequest,
    CVExtractResponse,
    CVDetailResponse,
    CVListResponse,
    CVRecommendationResponse,
    CVSummaryResponse,
    CVUpdateRequest,
    CVVersionDetailResponse,
    CVVersionSummaryResponse,
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
    TaskCancelResponse,
    TaskResumeRequest,
    TaskResumeResponse,
    TaskRunResponse,
)
from backend.app.core.config import get_settings
from backend.app.core.time import utc_now
from backend.app.db import models
from backend.app.db.conversation_repository import (
    ACTIVE_TASK_STATUSES,
    TERMINAL_TASK_STATUSES,
    ConversationRepository,
)
from backend.app.db.repositories import (
    CompanyRepository,
    ReportRepository,
    safe_json_dict,
    safe_json_list,
)
from backend.app.db.session import get_db
from backend.app.domain.reports import ReportStatus
from backend.app.llm.synthesizer import SynthesisError
from backend.app.services.cv_file_extractor import CVExtractionError, extract_cv_text
from backend.app.services.cv_library import (
    CVLibraryError,
    CVLibraryService,
    profile_payload,
    remove_managed_files,
)
from backend.app.services.conversation_tasks import run_local_conversation_task
from backend.app.services.rag import RAGChatCitation, RAGChatResult, RAGChatService, choose_scope
from backend.app.services.rag_indexing import backfill_missing_report_embeddings
from backend.app.services.report_chat import ReportChatService
from backend.app.services.performance import sanitize_usage
from backend.app.services.research_service import run_mock_generation_task


router = APIRouter(prefix="/api")
EXPLICIT_REPORT_RE = re.compile(r"\b(?:informe|reporte|report|an[aá]lisis\s+completo)\b", re.IGNORECASE)
AFFIRMATIVE_CLARIFICATION_RE = re.compile(
    r"^(?:s[ií]|si,? claro|confirmo|de acuerdo|adelante|correcto)[!. ]*$",
    re.IGNORECASE,
)


def clarification_option_company(label: str) -> str | None:
    match = re.search(
        r"\b(?:sobre|investigar(?:\s+a)?|investigá(?:\s+a)?|investiga(?:\s+a)?)\s+(.+?)"
        r"(?=\s+(?:en|como|del|de la|para)\b|[.!?,;:]?$)",
        label.strip(),
        re.IGNORECASE,
    )
    if not match:
        return None
    company = re.sub(r"\s+", " ", match.group(1)).strip(" .,!?:;")
    company = re.sub(r"\s+Argentina$", "", company, flags=re.IGNORECASE)
    return company or None


@router.post(
    "/conversations",
    response_model=ConversationCreatedResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_conversation(
    request: ConversationCreateRequest,
    db: Session = Depends(get_db),
):
    conversation = ConversationRepository(db).create(request.title)
    db.commit()
    return ConversationCreatedResponse(
        conversation_id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at.isoformat(),
    )


@router.get("/conversations", response_model=ConversationListResponse)
def list_conversations(
    q: str | None = Query(default=None, max_length=120),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    repo = ConversationRepository(db)
    conversations = repo.list(query=q, limit=limit, offset=offset)
    return ConversationListResponse(
        items=[conversation_summary(repo, conversation) for conversation in conversations],
        pagination=Pagination(limit=limit, offset=offset, total=repo.count(q)),
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
def get_conversation(conversation_id: str, db: Session = Depends(get_db)):
    repo = ConversationRepository(db)
    conversation = require_conversation(repo, conversation_id)
    return conversation_detail(repo, conversation)


@router.patch("/conversations/{conversation_id}", response_model=ConversationSummaryResponse)
def update_conversation(
    conversation_id: str,
    request: ConversationUpdateRequest,
    db: Session = Depends(get_db),
):
    repo = ConversationRepository(db)
    conversation = require_conversation(repo, conversation_id)
    repo.rename(conversation, request.title)
    db.commit()
    return conversation_summary(repo, conversation)


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)) -> Response:
    repo = ConversationRepository(db)
    conversation = require_conversation(repo, conversation_id)
    repo.delete(conversation)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=ConversationMessageAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_conversation_message(
    conversation_id: str,
    request: ConversationMessageRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    request_started = time.monotonic()
    repo = ConversationRepository(db)
    conversation = require_conversation(repo, conversation_id)
    if repo.active_task(conversation.id):
        raise api_error(
            status.HTTP_409_CONFLICT,
            "conversation_busy",
            "La conversación ya tiene una tarea en curso.",
            {"conversation_id": conversation.id},
        )
    retry_at = repo.provider_retry_at()
    if retry_at:
        raise api_error(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "provider_cooldown_active",
            "DeepSeek alcanzó temporalmente su límite. Esperá a que termine la cuenta regresiva.",
            {"retry_at": retry_at},
        )

    report_repo = ReportRepository(db)
    cv_service = CVLibraryService(db)
    for attachment in request.attachments:
        if attachment.type == "cv" and not cv_service.get_cv(attachment.artifact_id or ""):
            raise api_error(
                status.HTTP_400_BAD_REQUEST,
                "invalid_attachment",
                "El CV adjunto no existe o ya no está disponible.",
                {"artifact_id": attachment.artifact_id},
            )
        if attachment.type == "report" and not report_repo.get_by_id(attachment.artifact_id or ""):
            raise api_error(
                status.HTTP_400_BAD_REQUEST,
                "invalid_attachment",
                "El informe adjunto no existe o ya no está disponible.",
                {"artifact_id": attachment.artifact_id},
            )

    event_cursor = repo.latest_event(conversation.id)
    message = repo.add_message(
        conversation,
        role="user",
        content=request.content,
        status="completed",
    )
    if conversation.title == "Nueva conversación":
        repo.rename(conversation, title_from_message(request.content))
    for attachment in request.attachments:
        if attachment.type == "report":
            repo.add_report_attachment(conversation, message, attachment.artifact_id or "")
        elif attachment.type == "cv":
            repo.add_artifact_link(
                conversation,
                artifact_type="cv",
                artifact_id=attachment.artifact_id or "",
                relationship_type="attached",
                message_id=message.id,
            )
        else:
            job = cv_service.create_job_description(
                conversation, message, attachment.title, attachment.content or ""
            )
            repo.add_artifact_link(
                conversation,
                artifact_type="job_description",
                artifact_id=job.id,
                relationship_type="attached",
                message_id=message.id,
            )
    attached_report_ids = [
        attachment.artifact_id for attachment in request.attachments if attachment.type == "report"
    ]
    if attached_report_ids:
        active_context = safe_json_dict(conversation.active_context_json)
        active_context["active_report_ids"] = list(dict.fromkeys(attached_report_ids))
        repo.update_active_context(conversation, active_context)
    attached_cv_ids = [
        attachment.artifact_id for attachment in request.attachments if attachment.type == "cv"
    ]
    attached_job_ids = [
        item.artifact_id
        for item in repo.list_artifacts(conversation.id)
        if item.message_id == message.id and item.artifact_type == "job_description"
    ]
    if attached_cv_ids or attached_job_ids:
        active_context = safe_json_dict(conversation.active_context_json)
        if attached_cv_ids:
            active_context["active_cv_id"] = attached_cv_ids[-1]
        if attached_job_ids:
            active_context["active_job_description_id"] = attached_job_ids[-1]
        repo.update_active_context(conversation, active_context)
    settings = get_settings()
    is_full_report = bool(EXPLICIT_REPORT_RE.search(request.content))
    task = repo.create_task(
        conversation,
        message,
        budget={
            "profile": "full_report" if is_full_report else "standard",
            "model_turns": (
                settings.agent_report_max_model_turns
                if is_full_report
                else settings.agent_max_model_turns
            ),
            "searches": (
                settings.agent_report_max_searches if is_full_report else settings.agent_max_searches
            ),
            "inspections": (
                settings.agent_report_max_inspections
                if is_full_report
                else settings.agent_max_inspections
            ),
            "elapsed_seconds": (
                settings.agent_report_max_elapsed_seconds
                if is_full_report
                else settings.agent_max_elapsed_seconds
            ),
        },
    )
    task.working_state_json = json.dumps(
        {"request_intent": "full_report" if is_full_report else "standard"},
        ensure_ascii=False,
    )
    task.usage_json = json.dumps(
        {"acceptance_ms": int((time.monotonic() - request_started) * 1000)},
        ensure_ascii=False,
    )
    db.commit()

    background_tasks.add_task(run_local_conversation_task, task.id, db.get_bind())
    return ConversationMessageAcceptedResponse(
        message_id=message.id,
        task_run_id=task.id,
        status=task.status,
        events_url=(
            f"/api/conversations/{conversation.id}/events"
            f"?after_event_id={event_cursor.id if event_cursor else 0}"
        ),
    )


@router.get("/conversations/{conversation_id}/events")
async def stream_conversation_events(
    conversation_id: str,
    after_event_id: int | None = Query(default=None, ge=0),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    db: Session = Depends(get_db),
):
    repo = ConversationRepository(db)
    require_conversation(repo, conversation_id)
    cursor = parse_event_cursor(after_event_id, last_event_id)
    if cursor:
        event = repo.get_event(cursor)
        if not event or event.conversation_id != conversation_id:
            raise api_error(
                status.HTTP_400_BAD_REQUEST,
                "invalid_event_cursor",
                "El punto de reanudación de eventos no es válido para esta conversación.",
                {"event_id": cursor},
            )

    stream_session = sessionmaker(bind=db.get_bind(), autoflush=False, autocommit=False)

    async def event_stream():
        last_sent = cursor
        last_activity = time.monotonic()
        while True:
            with stream_session() as stream_db:
                stream_repo = ConversationRepository(stream_db)
                events = stream_repo.list_events(conversation_id, last_sent)
                for event in events:
                    last_sent = event.id
                    last_activity = time.monotonic()
                    yield format_sse_event(event)

                active_task = stream_repo.active_task(conversation_id)
                if not active_task and not events:
                    break
                if (
                    active_task
                    and active_task.status
                    in {"needs_clarification", "awaiting_review"}
                    and not events
                ):
                    break
                if active_task and time.monotonic() - last_activity >= 15:
                    heartbeat = stream_repo.add_event(
                        active_task,
                        "heartbeat",
                        {"task_run_id": active_task.id, "status": active_task.status},
                    )
                    stream_db.commit()
                    last_sent = heartbeat.id
                    last_activity = time.monotonic()
                    yield format_sse_event(heartbeat)
            await asyncio.sleep(0.25)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/task-runs/{task_run_id}/cancel", response_model=TaskCancelResponse)
def cancel_conversation_task(task_run_id: str, db: Session = Depends(get_db)):
    repo = ConversationRepository(db)
    task = repo.get_task(task_run_id)
    if not task:
        raise api_error(
            status.HTTP_404_NOT_FOUND,
            "task_not_found",
            "No se encontró la tarea solicitada.",
            {"task_run_id": task_run_id},
        )
    if task.status in TERMINAL_TASK_STATUSES:
        raise api_error(
            status.HTTP_409_CONFLICT,
            "task_not_cancellable",
            "La tarea ya terminó y no puede cancelarse.",
            {"task_run_id": task.id, "status": task.status},
        )

    repo.update_task(task, "cancelled", "user_cancelled")
    response_message_id = safe_json_dict(task.working_state_json).get("response_message_id")
    response_message = (
        db.get(models.ConversationMessage, response_message_id) if response_message_id else None
    )
    if response_message and response_message.status == "streaming":
        response_message.status = "cancelled"
        response_message.content = "Cancelaste la respuesta del agente antes de que terminara."
        response_message.completed_at = utc_now()
    repo.add_event(task, "task.cancelled", {"task_run_id": task.id, "status": "cancelled"})
    db.commit()
    return TaskCancelResponse(task_run_id=task.id, status=task.status)


@router.post(
    "/task-runs/{task_run_id}/resume",
    response_model=TaskResumeResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def resume_conversation_task(
    task_run_id: str,
    request: TaskResumeRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    repo = ConversationRepository(db)
    task = repo.get_task(task_run_id)
    if not task:
        raise api_error(
            status.HTTP_404_NOT_FOUND,
            "task_not_found",
            "No se encontró la tarea solicitada.",
            {"task_run_id": task_run_id},
        )
    expected_status = {
        "clarification": "needs_clarification",
        "review": "awaiting_review",
    }[request.response_type]
    if task.status != expected_status:
        raise api_error(
            status.HTTP_409_CONFLICT,
            "task_not_resumable",
            "La tarea no está esperando ese tipo de respuesta.",
            {"task_run_id": task.id, "status": task.status},
        )

    cursor = repo.latest_event(task.conversation_id)
    conversation = repo.get(task.conversation_id)
    if not conversation:
        raise api_error(status.HTTP_404_NOT_FOUND, "conversation_not_found", "La conversación ya no existe.")
    working_state = safe_json_dict(task.working_state_json)
    if request.response_type == "clarification":
        response_text = request.content or ", ".join(request.selected_option_ids)
        repo.add_message(conversation, role="user", content=response_text, status="completed")
        working_state["clarification_response"] = {
            "content": request.content,
            "selected_option_ids": request.selected_option_ids,
        }
        pending = working_state.get("pending_clarification") or {}
        pending_question = str(pending.get("question") or "")
        original_goal = str(working_state.get("goal") or "").strip()
        selected_option_ids = set(request.selected_option_ids)
        selected_option_labels = [
            str(option.get("label") or "")
            for option in pending.get("options", [])
            if str(option.get("id") or "") in selected_option_ids
        ]
        report_option_selected = any(
            EXPLICIT_REPORT_RE.search(label) for label in selected_option_labels
        )
        selected_company = next(
            (
                clarification_option_company(label)
                for label in selected_option_labels
                if clarification_option_company(label)
            ),
            None,
        )
        research_option_selected = bool(
            selected_company
            and any(
                re.search(r"\b(?:investig|vacante|perfil|sueld|salari|cultur|entrevist)", label, re.IGNORECASE)
                for label in selected_option_labels
            )
        )
        report_confirmation = (
            AFFIRMATIVE_CLARIFICATION_RE.fullmatch(response_text.strip())
            or report_option_selected
        )
        if (
            (report_confirmation or research_option_selected)
            and (EXPLICIT_REPORT_RE.search(pending_question) or research_option_selected)
            and original_goal
            and len(re.findall(r"\w+", original_goal, re.UNICODE)) <= 3
        ):
            settings = get_settings()
            company_name = str(selected_company or working_state.get("company_name") or original_goal).strip()
            full_report_selected = bool(
                report_confirmation and EXPLICIT_REPORT_RE.search(pending_question)
            )
            working_state.update(
                {
                    "goal": (
                        f"Generá un informe completo sobre {company_name} como empleador en Argentina."
                        if full_report_selected
                        else f"Investigá a {company_name} como empleador en Argentina."
                    ),
                    "company_name": company_name,
                    "request_intent": "full_report" if full_report_selected else "standard",
                    "requires_grounding": True,
                    "pending_clarification": None,
                }
            )
            if full_report_selected:
                task.budget_json = json.dumps(
                    {
                        "profile": "full_report",
                        "model_turns": settings.agent_report_max_model_turns,
                        "searches": settings.agent_report_max_searches,
                        "inspections": settings.agent_report_max_inspections,
                        "elapsed_seconds": settings.agent_report_max_elapsed_seconds,
                    },
                    ensure_ascii=False,
                )
            active_context = safe_json_dict(conversation.active_context_json)
            active_context["active_company_name"] = company_name
            repo.update_active_context(conversation, active_context)
    elif request.response_type == "review":
        artifact = db.get(models.CVRecommendationArtifact, request.artifact_id)
        if not artifact or artifact.task_run_id != task.id:
            raise api_error(
                status.HTTP_400_BAD_REQUEST,
                "invalid_cv_review",
                "Las recomendaciones no pertenecen a esta tarea.",
            )
        try:
            saved = CVLibraryService(db).review_recommendation(
                artifact,
                request.review_decisions,
                request.draft_text,
                request.save_as_cv_version,
            )
        except CVLibraryError as exc:
            raise api_error(status.HTTP_400_BAD_REQUEST, exc.code, str(exc)) from exc
        working_state["review_response"] = {
            "artifact_id": artifact.id,
            "status": artifact.status,
            "saved_version_id": saved.id if saved else None,
        }
        working_state["force_finalize"] = True
    task.working_state_json = json.dumps(working_state, ensure_ascii=False)
    task.pause_reason_json = None
    repo.update_task(task, "pending")
    db.commit()
    background_tasks.add_task(run_local_conversation_task, task.id, db.get_bind())
    return TaskResumeResponse(
        task_run_id=task.id,
        status=task.status,
        events_url=(
            f"/api/conversations/{task.conversation_id}/events"
            f"?after_event_id={cursor.id if cursor else 0}"
        ),
    )


@router.post(
    "/task-runs/{task_run_id}/retry",
    response_model=TaskResumeResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def retry_conversation_task(
    task_run_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    repo = ConversationRepository(db)
    task = repo.get_task(task_run_id)
    if not task:
        raise api_error(
            status.HTTP_404_NOT_FOUND,
            "task_not_found",
            "No se encontró la tarea solicitada.",
            {"task_run_id": task_run_id},
        )
    if task.status != "failed":
        raise api_error(
            status.HTTP_409_CONFLICT,
            "task_not_retryable",
            "Solo se pueden reintentar tareas interrumpidas.",
            {"task_run_id": task.id, "status": task.status},
        )
    pause = safe_json_dict(task.pause_reason_json)
    retry_at = str(pause.get("retry_at") or "")
    if retry_at and utc_now().isoformat() < retry_at:
        raise api_error(
            status.HTTP_409_CONFLICT,
            "retry_cooldown_active",
            "DeepSeek todavía está en espera. Reintentá cuando termine la cuenta regresiva.",
            {"task_run_id": task.id, "retry_at": retry_at},
        )
    cursor = repo.latest_event(task.conversation_id)
    task.pause_reason_json = None
    repo.update_task(task, "pending")
    db.commit()
    background_tasks.add_task(run_local_conversation_task, task.id, db.get_bind())
    return TaskResumeResponse(
        task_run_id=task.id,
        status=task.status,
        events_url=(
            f"/api/conversations/{task.conversation_id}/events"
            f"?after_event_id={cursor.id if cursor else 0}"
        ),
    )


@router.get("/comparisons/{comparison_id}", response_model=ComparisonArtifactResponse)
def get_comparison_artifact(comparison_id: str, db: Session = Depends(get_db)):
    comparison = ConversationRepository(db).get_comparison(comparison_id)
    if not comparison:
        raise api_error(
            status.HTTP_404_NOT_FOUND,
            "comparison_not_found",
            "La comparación solicitada no existe o ya fue eliminada.",
        )
    return ComparisonArtifactResponse(
        comparison_id=comparison.id,
        title=comparison.title,
        report_ids=safe_json_list(comparison.report_ids_json),
        dimensions=safe_json_list(comparison.dimensions_json),
        payload=safe_json_dict(comparison.payload_json),
        citations=safe_json_list(comparison.citations_json),
        warnings=safe_json_list(comparison.warnings_json),
        created_at=comparison.created_at.isoformat(),
    )


@router.post("/cvs", response_model=CVDetailResponse, status_code=status.HTTP_201_CREATED)
async def create_stored_cv(
    file: UploadFile = File(...),
    display_name: str | None = Form(default=None),
    db: Session = Depends(get_db),
):
    service = CVLibraryService(db)
    try:
        cv, _ = service.create_upload(
            filename=file.filename or "cv",
            content_type=file.content_type,
            content=await file.read(),
            display_name=display_name,
        )
        db.commit()
        return cv_detail_payload(service, cv)
    except CVLibraryError as exc:
        db.rollback()
        raise api_error(status.HTTP_400_BAD_REQUEST, exc.code, str(exc)) from exc


@router.get("/cvs", response_model=CVListResponse)
def list_stored_cvs(db: Session = Depends(get_db)):
    service = CVLibraryService(db)
    return CVListResponse(items=[cv_summary_payload(service, cv) for cv in service.list_cvs()])


@router.get("/cvs/{cv_id}", response_model=CVDetailResponse)
def get_stored_cv(cv_id: str, db: Session = Depends(get_db)):
    service = CVLibraryService(db)
    cv = require_cv(service, cv_id)
    return cv_detail_payload(service, cv)


@router.patch("/cvs/{cv_id}", response_model=CVDetailResponse)
def update_stored_cv(cv_id: str, request: CVUpdateRequest, db: Session = Depends(get_db)):
    service = CVLibraryService(db)
    cv = require_cv(service, cv_id)
    try:
        if request.display_name is not None:
            cv.display_name = request.display_name
            cv.updated_at = utc_now()
        if request.is_default is True:
            service.set_default(cv)
        elif request.is_default is False:
            cv.is_default = False
            cv.updated_at = utc_now()
        if request.extracted_text is not None:
            if request.source_version_id and not service.get_version(cv.id, request.source_version_id):
                raise CVLibraryError("cv_version_not_found", "La versión base del CV no existe.")
            service.create_text_version(cv, request.extracted_text, request.source_version_id)
        db.commit()
        return cv_detail_payload(service, cv)
    except CVLibraryError as exc:
        db.rollback()
        raise api_error(status.HTTP_400_BAD_REQUEST, exc.code, str(exc)) from exc


@router.post("/cvs/{cv_id}/versions", response_model=CVDetailResponse, status_code=status.HTTP_201_CREATED)
async def create_stored_cv_version(
    cv_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    service = CVLibraryService(db)
    cv = require_cv(service, cv_id)
    try:
        service.create_upload(
            filename=file.filename or "cv",
            content_type=file.content_type,
            content=await file.read(),
            cv=cv,
        )
        db.commit()
        return cv_detail_payload(service, cv)
    except CVLibraryError as exc:
        db.rollback()
        raise api_error(status.HTTP_400_BAD_REQUEST, exc.code, str(exc)) from exc


@router.get("/cvs/{cv_id}/versions/{version_id}", response_model=CVVersionDetailResponse)
def get_stored_cv_version(cv_id: str, version_id: str, db: Session = Depends(get_db)):
    service = CVLibraryService(db)
    require_cv(service, cv_id)
    version = service.get_version(cv_id, version_id)
    if not version:
        raise api_error(status.HTTP_404_NOT_FOUND, "cv_version_not_found", "La versión del CV no existe.")
    return cv_version_detail_payload(version)


@router.get("/cvs/{cv_id}/versions/{version_id}/file")
def get_stored_cv_file(cv_id: str, version_id: str, db: Session = Depends(get_db)):
    service = CVLibraryService(db)
    require_cv(service, cv_id)
    version = service.get_version(cv_id, version_id)
    if not version or not version.storage_key:
        raise api_error(status.HTTP_404_NOT_FOUND, "cv_file_not_found", "Esta versión no tiene un archivo original.")
    path = service.resolve_storage_key(version.storage_key)
    if not path.is_file():
        raise api_error(status.HTTP_404_NOT_FOUND, "cv_file_not_found", "El archivo original ya no está disponible.")
    return FileResponse(path, media_type=version.content_type, filename=version.original_filename)


@router.delete("/cvs/{cv_id}/versions/{version_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_stored_cv_version(cv_id: str, version_id: str, db: Session = Depends(get_db)):
    service = CVLibraryService(db)
    cv = require_cv(service, cv_id)
    version = service.get_version(cv_id, version_id)
    if not version:
        raise api_error(status.HTTP_404_NOT_FOUND, "cv_version_not_found", "La versión del CV no existe.")
    try:
        paths = service.delete_version(cv, version)
        db.commit()
        remove_managed_files(paths)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except CVLibraryError as exc:
        db.rollback()
        raise api_error(status.HTTP_409_CONFLICT, exc.code, str(exc)) from exc


@router.delete("/cvs/{cv_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_stored_cv(cv_id: str, db: Session = Depends(get_db)):
    service = CVLibraryService(db)
    cv = require_cv(service, cv_id)
    paths = service.delete_cv(cv)
    db.commit()
    remove_managed_files(paths)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/cv-recommendations/{artifact_id}", response_model=CVRecommendationResponse)
def get_cv_recommendation(artifact_id: str, db: Session = Depends(get_db)):
    artifact = db.get(models.CVRecommendationArtifact, artifact_id)
    if not artifact:
        raise api_error(status.HTTP_404_NOT_FOUND, "cv_recommendation_not_found", "Las recomendaciones ya no están disponibles.")
    return cv_recommendation_payload(artifact)


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
        request.job_description,
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
            "No se pudo responder la pregunta con DeepSeek. Intentalo nuevamente.",
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
            "No se pudo responder la pregunta con DeepSeek. Intentalo nuevamente.",
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
            "No se pudo responder la pregunta con DeepSeek. Intentalo nuevamente.",
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
        used_job_description=bool(metadata.get("used_job_description", False)),
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


def require_conversation(
    repo: ConversationRepository, conversation_id: str
) -> models.Conversation:
    conversation = repo.get(conversation_id)
    if not conversation:
        raise api_error(
            status.HTTP_404_NOT_FOUND,
            "conversation_not_found",
            "No se encontró la conversación solicitada.",
            {"conversation_id": conversation_id},
        )
    return conversation


def title_from_message(content: str) -> str:
    cleaned = " ".join(content.split())
    return cleaned if len(cleaned) <= 42 else f"{cleaned[:41].rstrip()}…"


def require_cv(service: CVLibraryService, cv_id: str) -> models.StoredCV:
    cv = service.get_cv(cv_id)
    if not cv:
        raise api_error(status.HTTP_404_NOT_FOUND, "cv_not_found", "El CV no existe o fue eliminado.")
    return cv


def cv_version_summary_payload(version: models.CVVersion) -> CVVersionSummaryResponse:
    return CVVersionSummaryResponse(
        version_id=version.id,
        version_number=version.version_number,
        created_from=version.created_from,
        original_filename=version.original_filename,
        content_type=version.content_type,
        file_size=version.file_size,
        has_file=bool(version.storage_key),
        created_at=version.created_at.isoformat(),
    )


def cv_version_detail_payload(version: models.CVVersion) -> CVVersionDetailResponse:
    return CVVersionDetailResponse(
        **cv_version_summary_payload(version).model_dump(),
        extracted_text=version.extracted_text,
        structured_profile=profile_payload(version),
    )


def cv_summary_payload(service: CVLibraryService, cv: models.StoredCV) -> CVSummaryResponse:
    current = service.current_version(cv)
    return CVSummaryResponse(
        cv_id=cv.id,
        display_name=cv.display_name,
        is_default=cv.is_default,
        current_version_id=current.id,
        current_version=cv_version_summary_payload(current),
        created_at=cv.created_at.isoformat(),
        updated_at=cv.updated_at.isoformat(),
    )


def cv_detail_payload(service: CVLibraryService, cv: models.StoredCV) -> CVDetailResponse:
    current = service.current_version(cv)
    return CVDetailResponse(
        **cv_summary_payload(service, cv).model_dump(),
        extracted_text=current.extracted_text,
        structured_profile=profile_payload(current),
        versions=[cv_version_summary_payload(item) for item in service.versions(cv.id)],
    )


def cv_recommendation_payload(artifact: models.CVRecommendationArtifact) -> CVRecommendationResponse:
    return CVRecommendationResponse(
        artifact_id=artifact.id,
        conversation_id=artifact.conversation_id,
        task_run_id=artifact.task_run_id,
        cv_id=artifact.cv_id,
        cv_version_id=artifact.cv_version_id,
        job_description_id=artifact.job_description_id,
        status=artifact.status,
        payload=safe_json_dict(artifact.payload_json),
        review=safe_json_dict(artifact.review_json),
        draft_text=artifact.draft_text,
        created_at=artifact.created_at.isoformat(),
        updated_at=artifact.updated_at.isoformat(),
    )


def conversation_summary(
    repo: ConversationRepository, conversation: models.Conversation
) -> ConversationSummaryResponse:
    messages = repo.list_messages(conversation.id)
    latest_task = repo.latest_task(conversation.id)
    preview = messages[-1].content[:120] if messages else None
    return ConversationSummaryResponse(
        conversation_id=conversation.id,
        title=conversation.title,
        status=latest_task.status if latest_task else "ready",
        last_message_preview=preview,
        created_at=conversation.created_at.isoformat(),
        updated_at=conversation.updated_at.isoformat(),
    )


def conversation_detail(
    repo: ConversationRepository, conversation: models.Conversation
) -> ConversationDetailResponse:
    messages = repo.list_messages(conversation.id)
    artifacts = repo.list_artifacts(conversation.id)
    generated_message_ids = {
        artifact.id: (
            artifact.message_id
            or next(
                (
                    message.id
                    for message in reversed(messages)
                    if message.role == "user" and message.created_at <= artifact.created_at
                ),
                None,
            )
        )
        for artifact in artifacts
        if artifact.relationship_type == "generated"
    }
    latest_task = repo.latest_task(conversation.id)
    latest_event = repo.latest_event(conversation.id)
    return ConversationDetailResponse(
        conversation_id=conversation.id,
        title=conversation.title,
        summary=conversation.summary,
        active_context=safe_json_dict(conversation.active_context_json),
        messages=[
            ConversationMessageResponse(
                message_id=message.id,
                role=message.role,
                content=message.content,
                status=message.status,
                citations=safe_json_list(message.citations_json),
                created_at=message.created_at.isoformat(),
                completed_at=message.completed_at.isoformat()
                if message.completed_at
                else None,
            )
            for message in messages
        ],
        artifacts=[
            ConversationArtifactResponse(
                artifact_link_id=artifact.id,
                message_id=generated_message_ids.get(artifact.id, artifact.message_id),
                type=artifact.artifact_type,
                artifact_id=artifact.artifact_id,
                relationship_type=artifact.relationship_type,
            )
            for artifact in artifacts
        ],
        current_task=TaskRunResponse(
            task_run_id=latest_task.id,
            task_type=latest_task.task_type,
            status=latest_task.status,
            stopping_reason=latest_task.stopping_reason,
            pause=(
                safe_json_dict(latest_task.pause_reason_json)
                if latest_task.pause_reason_json
                else None
            ),
            usage=sanitize_usage(safe_json_dict(latest_task.usage_json)),
            created_at=latest_task.created_at.isoformat(),
            updated_at=latest_task.updated_at.isoformat(),
            completed_at=latest_task.completed_at.isoformat()
            if latest_task.completed_at
            else None,
        )
        if latest_task
        else None,
        last_event_id=latest_event.id if latest_event else None,
        created_at=conversation.created_at.isoformat(),
        updated_at=conversation.updated_at.isoformat(),
    )


def parse_event_cursor(after_event_id: int | None, last_event_id: str | None) -> int:
    if after_event_id is not None:
        return after_event_id
    if not last_event_id:
        return 0
    try:
        cursor = int(last_event_id)
    except ValueError as exc:
        raise api_error(
            status.HTTP_400_BAD_REQUEST,
            "invalid_event_cursor",
            "El identificador del último evento no es válido.",
            {"event_id": last_event_id},
        ) from exc
    if cursor < 0:
        raise api_error(
            status.HTTP_400_BAD_REQUEST,
            "invalid_event_cursor",
            "El identificador del último evento no es válido.",
            {"event_id": last_event_id},
        )
    return cursor


def format_sse_event(event: models.TaskEvent) -> str:
    payload = safe_json_dict(event.result_summary_json)
    return (
        f"id: {event.id}\n"
        f"event: {event.event_type}\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    )


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
