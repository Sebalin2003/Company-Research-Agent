from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl


class ReportStatus(StrEnum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class SectionType(StrEnum):
    executive_summary = "executive_summary"
    business = "business"
    argentina_presence = "argentina_presence"
    employees = "employees"
    salary_benefits = "salary_benefits"
    culture = "culture"
    interview_process = "interview_process"
    interview_questions = "interview_questions"
    open_roles = "open_roles"
    personalized_preparation = "personalized_preparation"
    cv_tailoring = "cv_tailoring"
    sources = "sources"
    warnings = "warnings"


class SourceType(StrEnum):
    official = "official"
    career_page = "career_page"
    linkedin = "linkedin"
    job_board = "job_board"
    salary_review_platform = "salary_review_platform"
    news_media = "news_media"
    company_database = "company_database"
    secondary = "secondary"
    unknown = "unknown"


class EvidenceTopic(StrEnum):
    business = "business"
    argentina_presence = "argentina_presence"
    employees = "employees"
    salary = "salary"
    benefits = "benefits"
    culture = "culture"
    interview_process = "interview_process"
    interview_questions = "interview_questions"
    open_roles = "open_roles"
    general = "general"


class ClaimType(StrEnum):
    fact = "fact"
    inference = "inference"
    recommendation = "recommendation"
    missing_evidence = "missing_evidence"


class ConfidenceLevel(StrEnum):
    high = "high"
    medium = "medium"
    low = "low"
    unknown = "unknown"


class WarningSeverity(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"


class CompanySchema(BaseModel):
    id: str
    name: str
    normalized_name: str
    possible_aliases: list[str] = Field(default_factory=list)
    ambiguity_warning: str | None = None


class ClaimSchema(BaseModel):
    id: str
    type: ClaimType
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel


class SectionSchema(BaseModel):
    type: SectionType
    title: str
    summary: str
    claims: list[ClaimSchema] = Field(default_factory=list)
    confidence: ConfidenceLevel
    missing_evidence: bool = False


class SourceSchema(BaseModel):
    id: str
    title: str
    url: HttpUrl | str
    domain: str
    source_type: SourceType
    reliability_score: int = Field(ge=1, le=5)
    accessed_at: str
    published_at: str | None = None
    snippet: str | None = None
    language: str | None = None
    is_current: bool = True


class EvidenceSchema(BaseModel):
    id: str
    source_id: str
    topic: EvidenceTopic
    claim: str
    raw_text_excerpt: str | None = None
    confidence: ConfidenceLevel


class WarningSchema(BaseModel):
    id: str
    type: str
    message: str
    severity: WarningSeverity
    related_section: SectionType | None = None


class CVProfileSchema(BaseModel):
    roles: list[str] = Field(default_factory=list)
    seniority: str | None = None
    industries: list[str] = Field(default_factory=list)
    hard_skills: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.unknown


class PersonalizedPreparationItemSchema(BaseModel):
    text: str
    reason: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel


class FitSummarySchema(BaseModel):
    summary: str
    confidence: ConfidenceLevel
    evidence_ids: list[str] = Field(default_factory=list)


class StarAnswerOutlineSchema(BaseModel):
    question: str
    situation: str
    task: str
    action: str
    result: str
    cv_basis: str | None = None
    confidence: ConfidenceLevel


class PersonalizedPreparationSchema(BaseModel):
    cv_profile: CVProfileSchema
    fit_summary: FitSummarySchema
    strengths_to_highlight: list[PersonalizedPreparationItemSchema] = Field(default_factory=list)
    gaps_to_prepare: list[PersonalizedPreparationItemSchema] = Field(default_factory=list)
    suggested_pitch: PersonalizedPreparationItemSchema | None = None
    personalized_questions: list[PersonalizedPreparationItemSchema] = Field(default_factory=list)
    star_answer_outlines: list[StarAnswerOutlineSchema] = Field(default_factory=list)
    questions_for_company: list[PersonalizedPreparationItemSchema] = Field(default_factory=list)
    warnings: list[WarningSchema] = Field(default_factory=list)


class CVTailoringSuggestionType(StrEnum):
    rewrite = "rewrite"
    reorder = "reorder"
    emphasize = "emphasize"
    add_only_if_true = "add_only_if_true"


class CVTailoringSuggestionSchema(BaseModel):
    id: str
    type: CVTailoringSuggestionType
    original_text: str | None = None
    suggested_text: str
    reason: str
    evidence_ids: list[str] = Field(default_factory=list)
    requires_user_confirmation: bool
    confidence: ConfidenceLevel


class AdaptedCVDraftSchema(BaseModel):
    title: str
    content_markdown: str
    included_suggestion_ids: list[str] = Field(default_factory=list)
    excluded_suggestion_ids: list[str] = Field(default_factory=list)
    warnings: list[WarningSchema] = Field(default_factory=list)


class CVTailoringSchema(BaseModel):
    positioning_summary: str
    change_suggestions: list[CVTailoringSuggestionSchema] = Field(default_factory=list)
    adapted_cv_draft: AdaptedCVDraftSchema | None = None
    warnings: list[WarningSchema] = Field(default_factory=list)


class ReportMetadataSchema(BaseModel):
    search_provider: str
    llm_provider: str = "gemini"
    llm_model: str
    source_count: int = 0
    evidence_count: int = 0
    used_cv: bool = False
    used_cv_tailoring: bool = False
    used_job_description: bool = False
    generation_duration_ms: int | None = None
    rag_index_status: str = "pending"
    rag_indexed_chunk_count: int = 0
    rag_index_error: str | None = None
    research_duration_ms: int | None = None
    research_search_duration_ms: int | None = None
    research_extraction_duration_ms: int | None = None
    synthesis_duration_ms: int | None = None
    research_search_count: int = 0
    research_extracted_url_count: int = 0
    research_extraction_skipped_count: int = 0
    research_snippet_only_count: int = 0


class StructuredReportSchema(BaseModel):
    schema_version: str = "1.0"
    report_id: str
    company: CompanySchema
    status: ReportStatus
    language: str = "es-AR"
    generated_at: str | None = None
    valid_until: str | None = None
    sections: list[SectionSchema] = Field(default_factory=list)
    personalized_preparation: PersonalizedPreparationSchema | None = None
    cv_tailoring: CVTailoringSchema | None = None
    sources: list[SourceSchema] = Field(default_factory=list)
    evidence: list[EvidenceSchema] = Field(default_factory=list)
    warnings: list[WarningSchema] = Field(default_factory=list)
    metadata: ReportMetadataSchema
