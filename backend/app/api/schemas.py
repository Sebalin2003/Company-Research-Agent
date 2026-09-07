from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.app.domain.reports import ReportStatus, StructuredReportSchema


class CompanySummary(BaseModel):
    id: str
    name: str
    normalized_name: str


class ResearchRequest(BaseModel):
    company_name: str = Field(min_length=1, max_length=200)
    force_refresh: bool = False
    cv_text: str | None = None
    job_description: str | None = Field(default=None, max_length=20_000)
    include_cv_tailoring: bool = False
    include_adapted_cv_draft: bool = False

    @field_validator("company_name")
    @classmethod
    def company_name_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("El nombre de la empresa es obligatorio.")
        return cleaned

    @field_validator("cv_text")
    @classmethod
    def cv_text_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("El CV no puede estar vacio.")
        return cleaned

    @field_validator("job_description")
    @classmethod
    def job_description_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("La descripcion del puesto no puede estar vacia.")
        return cleaned

    @model_validator(mode="after")
    def validate_cv_options(self) -> ResearchRequest:
        if self.include_cv_tailoring and not self.cv_text:
            raise ValueError("Para adaptar el CV, primero tenes que pegar o subir un CV.")
        if self.job_description and not self.cv_text:
            raise ValueError("Para usar la descripcion del puesto, primero agrega tu CV.")
        if self.include_adapted_cv_draft and not self.include_cv_tailoring:
            raise ValueError("Para generar un borrador adaptado, activa la adaptacion del CV.")
        return self


class ResearchAcceptedResponse(BaseModel):
    report_id: str
    status: ReportStatus
    company: CompanySummary
    status_url: str
    reused_existing_report: bool = False


class ResearchCompletedResponse(BaseModel):
    report_id: str
    status: ReportStatus
    company: CompanySummary
    reused_existing_report: bool
    report: StructuredReportSchema


class ReportProgress(BaseModel):
    stage: str
    message: str


class RunningReportResponse(BaseModel):
    report_id: str
    status: ReportStatus
    company: CompanySummary
    progress: ReportProgress


class FailedReportError(BaseModel):
    code: str
    message: str


class FailedReportResponse(BaseModel):
    report_id: str
    status: ReportStatus
    company: CompanySummary
    error: FailedReportError


class CompletedReportResponse(BaseModel):
    report: StructuredReportSchema


class ReportListItem(BaseModel):
    report_id: str
    company: CompanySummary
    status: ReportStatus
    summary: str | None = None
    generated_at: str | None = None
    valid_until: str | None = None
    used_cv: bool = False
    used_cv_tailoring: bool = False
    used_job_description: bool = False


class Pagination(BaseModel):
    limit: int
    offset: int
    total: int


class ReportListResponse(BaseModel):
    items: list[ReportListItem]
    pagination: Pagination


class CompanyReportsResponse(BaseModel):
    company: CompanySummary
    items: list[ReportListItem]
    pagination: Pagination


class CVDataDeletedFlags(BaseModel):
    candidate_profile: bool
    personalized_preparation: bool
    cv_tailoring: bool


class DeleteCVDataResponse(BaseModel):
    report_id: str
    deleted: CVDataDeletedFlags


class DeleteReportResponse(BaseModel):
    report_id: str
    deleted_report: bool
    deleted_company: bool


class DeleteAllReportsResponse(BaseModel):
    deleted_reports: int
    deleted_companies: int


class CVExtractResponse(BaseModel):
    filename: str
    content_type: str | None = None
    character_count: int
    cv_text: str


class ChatCitation(BaseModel):
    source_id: str
    title: str
    url: str
    report_id: str | None = None
    company_name: str | None = None
    chunk_title: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("La pregunta no puede estar vacia.")
        return cleaned


class ChatResponse(BaseModel):
    report_id: str
    answer: str
    citations: list[ChatCitation] = Field(default_factory=list)
    created_at: str


class GlobalChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    active_report_id: str | None = None

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("La pregunta no puede estar vacia.")
        return cleaned


class GlobalChatResponse(BaseModel):
    answer: str
    scope_used: str
    citations: list[ChatCitation] = Field(default_factory=list)
    created_at: str


class RAGReindexResponse(BaseModel):
    indexed_reports: int
    indexed_chunks: int


JsonObject = dict[str, Any]


class CVVersionSummaryResponse(BaseModel):
    version_id: str
    version_number: int
    created_from: str
    original_filename: str | None = None
    content_type: str | None = None
    file_size: int
    has_file: bool
    created_at: str


class CVSummaryResponse(BaseModel):
    cv_id: str
    display_name: str
    is_default: bool
    current_version_id: str
    current_version: CVVersionSummaryResponse
    created_at: str
    updated_at: str


class CVListResponse(BaseModel):
    items: list[CVSummaryResponse]


class CVVersionDetailResponse(CVVersionSummaryResponse):
    extracted_text: str
    structured_profile: JsonObject


class CVDetailResponse(CVSummaryResponse):
    extracted_text: str
    structured_profile: JsonObject
    versions: list[CVVersionSummaryResponse]


class CVUpdateRequest(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=200)
    is_default: bool | None = None
    extracted_text: str | None = Field(default=None, max_length=50_000)
    source_version_id: str | None = None

    @model_validator(mode="after")
    def validate_update(self):
        if self.display_name is None and self.is_default is None and self.extracted_text is None:
            raise ValueError("La actualización del CV no contiene cambios.")
        if self.display_name is not None:
            self.display_name = self.display_name.strip()
        if self.extracted_text is not None:
            self.extracted_text = self.extracted_text.strip()
            if not self.extracted_text:
                raise ValueError("El CV no puede estar vacío.")
        return self


class CVRecommendationResponse(BaseModel):
    artifact_id: str
    conversation_id: str
    task_run_id: str
    cv_id: str
    cv_version_id: str
    job_description_id: str | None = None
    status: str
    payload: JsonObject
    review: JsonObject
    draft_text: str
    created_at: str
    updated_at: str


class ConversationCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=120)

    @field_validator("title")
    @classmethod
    def title_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("El título no puede estar vacío.")
        return cleaned


class ConversationUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)

    @field_validator("title")
    @classmethod
    def title_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("El título no puede estar vacío.")
        return cleaned


class ConversationCreatedResponse(BaseModel):
    conversation_id: str
    title: str
    created_at: str


class ConversationSummaryResponse(BaseModel):
    conversation_id: str
    title: str
    status: str
    last_message_preview: str | None = None
    created_at: str
    updated_at: str


class ConversationListResponse(BaseModel):
    items: list[ConversationSummaryResponse]
    pagination: Pagination


class ConversationMessageResponse(BaseModel):
    message_id: str
    role: str
    content: str
    status: str
    citations: list[JsonObject] = Field(default_factory=list)
    created_at: str
    completed_at: str | None = None


class ConversationArtifactResponse(BaseModel):
    artifact_link_id: str
    message_id: str | None = None
    type: str
    artifact_id: str
    relationship_type: str


class TaskRunResponse(BaseModel):
    task_run_id: str
    task_type: str
    status: str
    stopping_reason: str | None = None
    pause: JsonObject | None = None
    usage: JsonObject = Field(default_factory=dict)
    created_at: str
    updated_at: str
    completed_at: str | None = None


class ConversationDetailResponse(BaseModel):
    conversation_id: str
    title: str
    summary: str | None = None
    active_context: JsonObject = Field(default_factory=dict)
    messages: list[ConversationMessageResponse]
    artifacts: list[ConversationArtifactResponse]
    current_task: TaskRunResponse | None = None
    last_event_id: int | None = None
    created_at: str
    updated_at: str


class ConversationAttachmentRequest(BaseModel):
    type: Literal["report", "cv", "job_description"]
    artifact_id: str | None = Field(default=None, max_length=200)
    content: str | None = Field(default=None, max_length=20_000)
    title: str | None = Field(default=None, max_length=200)

    @field_validator("artifact_id")
    @classmethod
    def artifact_id_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return value
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("El identificador del adjunto no puede estar vacío.")
        return cleaned

    @model_validator(mode="after")
    def validate_attachment(self):
        if self.type in {"report", "cv"} and not self.artifact_id:
            raise ValueError("El identificador del adjunto no puede estar vacío.")
        if self.type == "job_description":
            self.content = (self.content or "").strip()
            if not self.content:
                raise ValueError("La descripción del puesto no puede estar vacía.")
            self.title = (self.title or "Descripción del puesto").strip()
        return self


class ConversationMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=20_000)
    attachments: list[ConversationAttachmentRequest] = Field(default_factory=list, max_length=10)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("El mensaje no puede estar vacío.")
        return cleaned


class ConversationMessageAcceptedResponse(BaseModel):
    message_id: str
    task_run_id: str
    status: str
    events_url: str


class TaskCancelResponse(BaseModel):
    task_run_id: str
    status: str


class TaskResumeRequest(BaseModel):
    response_type: Literal["clarification", "review"]
    content: str | None = Field(default=None, max_length=20_000)
    selected_option_ids: list[str] = Field(default_factory=list, max_length=10)
    artifact_id: str | None = None
    review_decisions: list[JsonObject] = Field(default_factory=list, max_length=100)
    draft_text: str | None = Field(default=None, max_length=50_000)
    save_as_cv_version: bool = False

    @model_validator(mode="after")
    def validate_response(self):
        if self.response_type == "clarification":
            content = (self.content or "").strip()
            if not content and not self.selected_option_ids:
                raise ValueError("La aclaración requiere una respuesta.")
            self.content = content or None
        if self.response_type == "review" and not (self.artifact_id or "").strip():
            raise ValueError("La revisión requiere un artefacto de recomendaciones.")
        return self


class TaskResumeResponse(BaseModel):
    task_run_id: str
    status: str
    events_url: str


class ComparisonArtifactResponse(BaseModel):
    comparison_id: str
    title: str
    report_ids: list[str]
    dimensions: list[str]
    payload: JsonObject
    citations: list[JsonObject]
    warnings: list[str]
    created_at: str
