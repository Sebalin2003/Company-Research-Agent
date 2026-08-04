from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    normalized_name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    possible_aliases_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    ambiguity_warning: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    reports: Mapped[list[Report]] = relationship(back_populates="company")


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    company_id: Mapped[str] = mapped_column(ForeignKey("companies.id"), nullable=False, index=True)
    schema_version: Mapped[str] = mapped_column(String(20), nullable=False, default="1.0")
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    language: Mapped[str] = mapped_column(String(10), nullable=False, default="es-AR")
    generated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    summary: Mapped[str | None] = mapped_column(Text)
    warnings_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    company: Mapped[Company] = relationship(back_populates="reports")
    sections: Mapped[list[ReportSection]] = relationship(
        back_populates="report", cascade="all, delete-orphan"
    )
    sources: Mapped[list[Source]] = relationship(back_populates="report", cascade="all, delete-orphan")
    evidence_items: Mapped[list[EvidenceItem]] = relationship(
        back_populates="report", cascade="all, delete-orphan"
    )
    candidate_profile: Mapped[CandidateProfile | None] = relationship(
        back_populates="report", cascade="all, delete-orphan"
    )
    chat_messages: Mapped[list[ReportChatMessage]] = relationship(
        back_populates="report", cascade="all, delete-orphan"
    )
    embedding_chunks: Mapped[list[ReportEmbeddingChunk]] = relationship(
        back_populates="report", cascade="all, delete-orphan"
    )


class ReportSection(Base):
    __tablename__ = "report_sections"
    __table_args__ = (UniqueConstraint("report_id", "section_type"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    report_id: Mapped[str] = mapped_column(ForeignKey("reports.id"), nullable=False, index=True)
    section_type: Mapped[str] = mapped_column(String(50), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[str] = mapped_column(String(20), nullable=False)
    missing_evidence: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    display_order: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    report: Mapped[Report] = relationship(back_populates="sections")
    claims: Mapped[list[ReportClaim]] = relationship(
        back_populates="section", cascade="all, delete-orphan"
    )


class ReportClaim(Base):
    __tablename__ = "report_claims"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    section_id: Mapped[str] = mapped_column(
        ForeignKey("report_sections.id"), nullable=False, index=True
    )
    claim_type: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[str] = mapped_column(String(20), nullable=False)
    display_order: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    section: Mapped[ReportSection] = relationship(back_populates="claims")
    evidence_links: Mapped[list[ClaimEvidence]] = relationship(
        back_populates="claim", cascade="all, delete-orphan"
    )


class Source(Base):
    __tablename__ = "sources"
    __table_args__ = (UniqueConstraint("report_id", "url"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    report_id: Mapped[str] = mapped_column(ForeignKey("reports.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    reliability_score: Mapped[int] = mapped_column(Integer, nullable=False)
    accessed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    snippet: Mapped[str | None] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(String(20))
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    report: Mapped[Report] = relationship(back_populates="sources")
    evidence_items: Mapped[list[EvidenceItem]] = relationship(back_populates="source")


class EvidenceItem(Base):
    __tablename__ = "evidence_items"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    report_id: Mapped[str] = mapped_column(ForeignKey("reports.id"), nullable=False, index=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("sources.id"), nullable=False, index=True)
    topic: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    claim: Mapped[str] = mapped_column(Text, nullable=False)
    raw_text_excerpt: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    report: Mapped[Report] = relationship(back_populates="evidence_items")
    source: Mapped[Source] = relationship(back_populates="evidence_items")
    claim_links: Mapped[list[ClaimEvidence]] = relationship(back_populates="evidence")


class ClaimEvidence(Base):
    __tablename__ = "claim_evidence"
    __table_args__ = (UniqueConstraint("claim_id", "evidence_id"),)

    claim_id: Mapped[str] = mapped_column(ForeignKey("report_claims.id"), primary_key=True)
    evidence_id: Mapped[str] = mapped_column(ForeignKey("evidence_items.id"), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    claim: Mapped[ReportClaim] = relationship(back_populates="evidence_links")
    evidence: Mapped[EvidenceItem] = relationship(back_populates="claim_links")


class CandidateProfile(Base):
    __tablename__ = "candidate_profiles"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    report_id: Mapped[str] = mapped_column(
        ForeignKey("reports.id"), nullable=False, unique=True, index=True
    )
    raw_cv_text: Mapped[str | None] = mapped_column(Text)
    roles_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    seniority: Mapped[str | None] = mapped_column(String(100))
    industries_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    hard_skills_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    soft_skills_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    tools_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    education_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    languages_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    achievements_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    confidence: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    report: Mapped[Report] = relationship(back_populates="candidate_profile")
    personalized_preparation: Mapped[PersonalizedPreparation | None] = relationship(
        back_populates="candidate_profile", cascade="all, delete-orphan"
    )
    cv_tailoring: Mapped[CVTailoring | None] = relationship(
        back_populates="candidate_profile", cascade="all, delete-orphan"
    )


class PersonalizedPreparation(Base):
    __tablename__ = "personalized_preparations"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    candidate_profile_id: Mapped[str] = mapped_column(
        ForeignKey("candidate_profiles.id"), nullable=False, unique=True, index=True
    )
    fit_summary: Mapped[str] = mapped_column(Text, nullable=False)
    fit_confidence: Mapped[str] = mapped_column(String(20), nullable=False)
    suggested_pitch: Mapped[str | None] = mapped_column(Text)
    warnings_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    candidate_profile: Mapped[CandidateProfile] = relationship(
        back_populates="personalized_preparation"
    )
    items: Mapped[list[PreparationItem]] = relationship(
        back_populates="personalized_preparation", cascade="all, delete-orphan"
    )


class PreparationItem(Base):
    __tablename__ = "preparation_items"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    personalized_preparation_id: Mapped[str] = mapped_column(
        ForeignKey("personalized_preparations.id"), nullable=False, index=True
    )
    item_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[str] = mapped_column(String(20), nullable=False)
    payload_json: Mapped[str | None] = mapped_column(Text)
    display_order: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    personalized_preparation: Mapped[PersonalizedPreparation] = relationship(back_populates="items")


class CVTailoring(Base):
    __tablename__ = "cv_tailorings"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    candidate_profile_id: Mapped[str] = mapped_column(
        ForeignKey("candidate_profiles.id"), nullable=False, unique=True, index=True
    )
    positioning_summary: Mapped[str] = mapped_column(Text, nullable=False)
    adapted_cv_draft_markdown: Mapped[str | None] = mapped_column(Text)
    warnings_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    candidate_profile: Mapped[CandidateProfile] = relationship(back_populates="cv_tailoring")
    suggestions: Mapped[list[CVTailoringSuggestion]] = relationship(
        back_populates="cv_tailoring", cascade="all, delete-orphan"
    )


class CVTailoringSuggestion(Base):
    __tablename__ = "cv_tailoring_suggestions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    cv_tailoring_id: Mapped[str] = mapped_column(
        ForeignKey("cv_tailorings.id"), nullable=False, index=True
    )
    suggestion_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    original_text: Mapped[str | None] = mapped_column(Text)
    suggested_text: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_ids_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    requires_user_confirmation: Mapped[bool] = mapped_column(Boolean, nullable=False)
    confidence: Mapped[str] = mapped_column(String(20), nullable=False)
    included_in_draft: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    display_order: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    cv_tailoring: Mapped[CVTailoring] = relationship(back_populates="suggestions")


class ReportChatMessage(Base):
    __tablename__ = "report_chat_messages"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    report_id: Mapped[str] = mapped_column(ForeignKey("reports.id"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    report: Mapped[Report] = relationship(back_populates="chat_messages")


class ReportEmbeddingChunk(Base):
    __tablename__ = "report_embedding_chunks"
    __table_args__ = (UniqueConstraint("report_id", "chunk_key"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    report_id: Mapped[str] = mapped_column(ForeignKey("reports.id"), nullable=False, index=True)
    company_id: Mapped[str] = mapped_column(ForeignKey("companies.id"), nullable=False, index=True)
    chunk_key: Mapped[str] = mapped_column(String(120), nullable=False)
    chunk_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    chunk_title: Mapped[str] = mapped_column(String(200), nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_ids_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    evidence_ids_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    embedding_provider: Mapped[str] = mapped_column(String(50), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(120), nullable=False)
    embedding_dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    report: Mapped[Report] = relationship(back_populates="embedding_chunks")
    company: Mapped[Company] = relationship()


class GlobalChatMessage(Base):
    __tablename__ = "global_chat_messages"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    scope_used: Mapped[str | None] = mapped_column(String(50), index=True)
    active_report_id: Mapped[str | None] = mapped_column(String, index=True)
    cited_report_ids_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    citations_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
