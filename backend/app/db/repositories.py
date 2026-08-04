from __future__ import annotations

import json
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.core.time import utc_now
from backend.app.db import models
from backend.app.domain.companies import normalize_company_name
from backend.app.domain.reports import (
    AdaptedCVDraftSchema,
    ClaimSchema,
    CompanySchema,
    CVProfileSchema,
    CVTailoringSchema,
    CVTailoringSuggestionSchema,
    EvidenceSchema,
    FitSummarySchema,
    PersonalizedPreparationItemSchema,
    PersonalizedPreparationSchema,
    ReportMetadataSchema,
    ReportStatus,
    SectionSchema,
    SourceSchema,
    StarAnswerOutlineSchema,
    StructuredReportSchema,
    WarningSchema,
)
from backend.app.services.report_chat import ChatAnswerPayload
from backend.app.services.rag import RAGChatResult, RAGChunk


class CompanyRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_by_id(self, company_id: str) -> models.Company | None:
        return self.db.get(models.Company, company_id)

    def get_or_create(self, name: str) -> models.Company:
        normalized_name = normalize_company_name(name)
        existing = self.db.scalar(
            select(models.Company).where(models.Company.normalized_name == normalized_name)
        )
        if existing:
            return existing

        now = utc_now()
        company = models.Company(
            id=str(uuid4()),
            name=name.strip(),
            normalized_name=normalized_name,
            possible_aliases_json="[]",
            created_at=now,
            updated_at=now,
        )
        self.db.add(company)
        self.db.flush()
        return company


class ReportRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_report(self, company_id: str) -> models.Report:
        now = utc_now()
        report = models.Report(
            id=str(uuid4()),
            company_id=company_id,
            schema_version="1.0",
            status=ReportStatus.pending.value,
            language="es-AR",
            warnings_json="[]",
            metadata_json=json.dumps({"llm_provider": "gemini"}),
            created_at=now,
            updated_at=now,
        )
        self.db.add(report)
        self.db.flush()
        return report

    def get_by_id(self, report_id: str) -> models.Report | None:
        return self.db.get(models.Report, report_id)

    def list_reports(
        self,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
        company_name: str | None = None,
    ) -> list[models.Report]:
        statement = select(models.Report).join(models.Company).order_by(models.Report.created_at.desc())
        if status:
            statement = statement.where(models.Report.status == status)
        if company_name:
            statement = statement.where(models.Company.normalized_name.contains(normalize_company_name(company_name)))
        return list(self.db.scalars(statement.limit(limit).offset(offset)))

    def list_company_reports(
        self, company_id: str, limit: int = 20, offset: int = 0
    ) -> list[models.Report]:
        statement = (
            select(models.Report)
            .where(models.Report.company_id == company_id)
            .order_by(models.Report.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.db.scalars(statement))

    def count_company_reports(self, company_id: str) -> int:
        return len(list(self.db.scalars(select(models.Report).where(models.Report.company_id == company_id))))

    def update_status(
        self, report: models.Report, status: ReportStatus, error_message: str | None = None
    ) -> models.Report:
        report.status = status.value
        report.error_message = error_message
        report.updated_at = utc_now()
        self.db.flush()
        return report

    def save_structured_report(
        self, report: models.Report, structured_report: StructuredReportSchema
    ) -> models.Report:
        now = utc_now()
        report.status = ReportStatus.completed.value
        report.generated_at = now
        report.valid_until = parse_datetime_or_none(structured_report.valid_until)
        report.summary = (
            structured_report.sections[0].summary if structured_report.sections else None
        )
        report.warnings_json = json.dumps(
            [warning.model_dump(mode="json") for warning in structured_report.warnings]
        )
        metadata = structured_report.metadata.model_dump(mode="json")
        metadata.setdefault("rag_index_status", "pending")
        metadata.setdefault("rag_indexed_chunk_count", 0)
        metadata.setdefault("rag_index_error", None)
        report.metadata_json = json.dumps(metadata)
        report.error_message = None
        report.updated_at = now

        claim_evidence_links: list[tuple[str, str]] = []
        for section_index, section in enumerate(structured_report.sections):
            section_model = models.ReportSection(
                id=str(uuid4()),
                report_id=report.id,
                section_type=section.type.value,
                title=section.title,
                summary=section.summary,
                confidence=section.confidence.value,
                missing_evidence=section.missing_evidence,
                display_order=section_index,
                created_at=now,
            )
            self.db.add(section_model)
            self.db.flush()
            for claim_index, claim in enumerate(section.claims):
                persisted_claim_id = scoped_report_id(report.id, claim.id)
                claim_model = models.ReportClaim(
                    id=persisted_claim_id,
                    section_id=section_model.id,
                    claim_type=claim.type.value,
                    text=claim.text,
                    confidence=claim.confidence.value,
                    display_order=claim_index,
                    created_at=now,
                )
                self.db.add(claim_model)
                claim_evidence_links.extend(
                    (persisted_claim_id, evidence_id) for evidence_id in claim.evidence_ids
                )

        source_id_map = {}
        for source in structured_report.sources:
            persisted_source_id = scoped_report_id(report.id, source.id)
            source_id_map[source.id] = persisted_source_id
            source_model = models.Source(
                id=persisted_source_id,
                report_id=report.id,
                title=source.title,
                url=str(source.url),
                domain=source.domain,
                source_type=source.source_type.value,
                reliability_score=source.reliability_score,
                accessed_at=parse_datetime_or_none(source.accessed_at) or now,
                published_at=parse_datetime_or_none(source.published_at),
                snippet=source.snippet,
                language=source.language,
                is_current=source.is_current,
                created_at=now,
            )
            self.db.add(source_model)

        evidence_id_map = {}
        for evidence in structured_report.evidence:
            persisted_evidence_id = scoped_report_id(report.id, evidence.id)
            evidence_model = models.EvidenceItem(
                id=persisted_evidence_id,
                report_id=report.id,
                source_id=source_id_map.get(evidence.source_id, evidence.source_id),
                topic=evidence.topic.value,
                claim=evidence.claim,
                raw_text_excerpt=evidence.raw_text_excerpt,
                confidence=evidence.confidence.value,
                created_at=now,
            )
            self.db.add(evidence_model)
            evidence_id_map[evidence.id] = persisted_evidence_id

        self.db.flush()
        for claim_id, original_evidence_id in claim_evidence_links:
            persisted_evidence_id = evidence_id_map.get(original_evidence_id)
            if persisted_evidence_id:
                self.db.add(
                    models.ClaimEvidence(
                        claim_id=claim_id,
                        evidence_id=persisted_evidence_id,
                        created_at=now,
                    )
                )

        profile = None
        if structured_report.personalized_preparation:
            profile = self._save_personalized_preparation(
                report, structured_report.personalized_preparation, now
            )
        if structured_report.cv_tailoring:
            profile = profile or report.candidate_profile or self._ensure_candidate_profile(report, now)
            self._save_cv_tailoring(profile, structured_report.cv_tailoring, now)

        self.db.flush()
        return report

    def to_structured_report(self, report: models.Report) -> StructuredReportSchema:
        evidence_by_claim = {
            link.claim_id: [] for section in report.sections for claim in section.claims for link in claim.evidence_links
        }
        for section in report.sections:
            for claim in section.claims:
                evidence_by_claim[claim.id] = [link.evidence_id for link in claim.evidence_links]

        return StructuredReportSchema(
            schema_version=report.schema_version,
            report_id=report.id,
            company=CompanySchema(
                id=report.company.id,
                name=report.company.name,
                normalized_name=report.company.normalized_name,
                possible_aliases=safe_json_list(report.company.possible_aliases_json),
                ambiguity_warning=report.company.ambiguity_warning,
            ),
            status=ReportStatus(report.status),
            language=report.language,
            generated_at=report.generated_at.isoformat() if report.generated_at else None,
            valid_until=report.valid_until.isoformat() if report.valid_until else None,
            sections=[
                SectionSchema(
                    type=section.section_type,
                    title=section.title,
                    summary=section.summary,
                    claims=[
                        ClaimSchema(
                            id=claim.id,
                            type=claim.claim_type,
                            text=claim.text,
                            evidence_ids=evidence_by_claim.get(claim.id, []),
                            confidence=claim.confidence,
                        )
                        for claim in sorted(section.claims, key=lambda item: item.display_order)
                    ],
                    confidence=section.confidence,
                    missing_evidence=section.missing_evidence,
                )
                for section in sorted(report.sections, key=lambda item: item.display_order)
            ],
            personalized_preparation=self._to_personalized_preparation(report),
            cv_tailoring=self._to_cv_tailoring(report),
            sources=[
                SourceSchema(
                    id=source.id,
                    title=source.title,
                    url=source.url,
                    domain=source.domain,
                    source_type=source.source_type,
                    reliability_score=source.reliability_score,
                    accessed_at=source.accessed_at.isoformat(),
                    published_at=source.published_at.isoformat() if source.published_at else None,
                    snippet=source.snippet,
                    language=source.language,
                    is_current=source.is_current,
                )
                for source in report.sources
            ],
            evidence=[
                EvidenceSchema(
                    id=evidence.id,
                    source_id=evidence.source_id,
                    topic=evidence.topic,
                    claim=evidence.claim,
                    raw_text_excerpt=evidence.raw_text_excerpt,
                    confidence=evidence.confidence,
                )
                for evidence in report.evidence_items
            ],
            warnings=[WarningSchema(**warning) for warning in safe_json_list(report.warnings_json)],
            metadata=ReportMetadataSchema(**safe_json_dict(report.metadata_json)),
        )

    def delete_cv_data(self, report: models.Report) -> dict[str, bool]:
        profile = report.candidate_profile
        deleted = {
            "candidate_profile": profile is not None,
            "personalized_preparation": bool(profile and profile.personalized_preparation),
            "cv_tailoring": bool(profile and profile.cv_tailoring),
        }
        if profile:
            self.db.delete(profile)
            self.db.flush()
        return deleted

    def delete_report(self, report: models.Report) -> bool:
        company = report.company
        company_id = company.id
        self.delete_global_chat_messages_for_report(report.id)
        self.db.delete(report)
        self.db.flush()
        deleted_company = self.count_company_reports(company_id) == 0
        if deleted_company:
            self.db.delete(company)
            self.db.flush()
        return deleted_company

    def delete_all_reports(self) -> dict[str, int]:
        reports = list(self.db.scalars(select(models.Report)))
        deleted_companies = 0
        for report in reports:
            if self.delete_report(report):
                deleted_companies += 1
        self.db.execute(delete(models.GlobalChatMessage))
        self.db.flush()
        return {"deleted_reports": len(reports), "deleted_companies": deleted_companies}

    def save_chat_exchange(
        self,
        report: models.Report,
        user_message: str,
        answer: ChatAnswerPayload,
        citations: list[dict],
    ) -> None:
        now = utc_now()
        self.db.add(
            models.ReportChatMessage(
                id=str(uuid4()),
                report_id=report.id,
                role="user",
                content=user_message,
                citations_json="[]",
                created_at=now,
            )
        )
        self.db.add(
            models.ReportChatMessage(
                id=str(uuid4()),
                report_id=report.id,
                role="assistant",
                content=answer.answer,
                citations_json=json.dumps(citations),
                created_at=now,
            )
        )
        self.db.flush()

    def replace_embedding_chunks(self, report: models.Report, chunks: list[RAGChunk]) -> int:
        now = utc_now()
        self.db.execute(
            delete(models.ReportEmbeddingChunk).where(
                models.ReportEmbeddingChunk.report_id == report.id
            )
        )
        for chunk in chunks:
            self.db.add(
                models.ReportEmbeddingChunk(
                    id=chunk.id,
                    report_id=report.id,
                    company_id=report.company_id,
                    chunk_key=chunk.chunk_key,
                    chunk_type=chunk.chunk_type,
                    chunk_title=chunk.chunk_title,
                    chunk_text=chunk.chunk_text,
                    source_ids_json=json.dumps(chunk.source_ids),
                    evidence_ids_json=json.dumps(chunk.evidence_ids),
                    embedding_provider=chunk.embedding_provider,
                    embedding_model=chunk.embedding_model,
                    embedding_dimensions=chunk.embedding_dimensions,
                    embedding_json=json.dumps(chunk.embedding),
                    created_at=now,
                )
            )
        self.db.flush()
        return len(chunks)

    def count_embedding_chunks(self, report_id: str) -> int:
        return int(
            self.db.scalar(
                select(func.count(models.ReportEmbeddingChunk.id)).where(
                    models.ReportEmbeddingChunk.report_id == report_id
                )
            )
            or 0
        )

    def update_rag_index_status(
        self,
        report: models.Report,
        status: str,
        chunk_count: int | None = None,
        error: str | None = None,
    ) -> models.Report:
        metadata = safe_json_dict(report.metadata_json)
        metadata["rag_index_status"] = status
        if chunk_count is not None:
            metadata["rag_indexed_chunk_count"] = chunk_count
        metadata["rag_index_error"] = error[:300] if error else None
        report.metadata_json = json.dumps(metadata)
        report.updated_at = utc_now()
        self.db.flush()
        return report

    def list_completed_reports_missing_embeddings(self) -> list[models.Report]:
        statement = (
            select(models.Report)
            .where(models.Report.status == ReportStatus.completed.value)
            .outerjoin(models.ReportEmbeddingChunk)
            .where(models.ReportEmbeddingChunk.id.is_(None))
        )
        return list(self.db.scalars(statement))

    def list_embedding_chunks(self) -> list[RAGChunk]:
        statement = select(models.ReportEmbeddingChunk).join(models.Report).join(models.Company)
        return [
            RAGChunk(
                id=chunk.id,
                report_id=chunk.report_id,
                company_id=chunk.company_id,
                company_name=chunk.report.company.name,
                chunk_key=chunk.chunk_key,
                chunk_type=chunk.chunk_type,
                chunk_title=chunk.chunk_title,
                chunk_text=chunk.chunk_text,
                source_ids=safe_json_list(chunk.source_ids_json),
                evidence_ids=safe_json_list(chunk.evidence_ids_json),
                embedding=safe_float_list(chunk.embedding_json),
                embedding_provider=chunk.embedding_provider,
                embedding_model=chunk.embedding_model,
                embedding_dimensions=chunk.embedding_dimensions,
            )
            for chunk in self.db.scalars(statement)
        ]

    def source_lookup_for_chunks(self, chunks: list[RAGChunk]) -> dict[str, dict[str, str]]:
        source_ids = {source_id for chunk in chunks for source_id in chunk.source_ids}
        if not source_ids:
            return {}
        sources = self.db.scalars(select(models.Source).where(models.Source.id.in_(source_ids)))
        return {
            source.id: {
                "title": source.title,
                "url": source.url,
            }
            for source in sources
        }

    def save_global_chat_exchange(
        self,
        user_message: str,
        answer: RAGChatResult,
        citations: list[dict],
        active_report_id: str | None,
    ) -> None:
        now = utc_now()
        cited_report_ids = sorted({citation.report_id for citation in answer.citations})
        self.db.add(
            models.GlobalChatMessage(
                id=str(uuid4()),
                role="user",
                content=user_message,
                scope_used=answer.scope_used,
                active_report_id=active_report_id,
                cited_report_ids_json="[]",
                citations_json="[]",
                created_at=now,
            )
        )
        self.db.add(
            models.GlobalChatMessage(
                id=str(uuid4()),
                role="assistant",
                content=answer.answer,
                scope_used=answer.scope_used,
                active_report_id=active_report_id,
                cited_report_ids_json=json.dumps(cited_report_ids),
                citations_json=json.dumps(citations),
                created_at=now,
            )
        )
        self.db.flush()

    def delete_global_chat_messages_for_report(self, report_id: str) -> None:
        messages = list(self.db.scalars(select(models.GlobalChatMessage)))
        for message in messages:
            cited_report_ids = set(safe_json_list(message.cited_report_ids_json))
            if message.active_report_id == report_id or report_id in cited_report_ids:
                self.db.delete(message)
        self.db.flush()

    def _ensure_candidate_profile(
        self, report: models.Report, now
    ) -> models.CandidateProfile:
        if report.candidate_profile:
            return report.candidate_profile
        profile = models.CandidateProfile(
            id=str(uuid4()),
            report_id=report.id,
            raw_cv_text=None,
            roles_json="[]",
            industries_json="[]",
            hard_skills_json="[]",
            soft_skills_json="[]",
            tools_json="[]",
            education_json="[]",
            languages_json="[]",
            achievements_json="[]",
            confidence="unknown",
            created_at=now,
        )
        self.db.add(profile)
        self.db.flush()
        return profile

    def _save_personalized_preparation(
        self, report: models.Report, preparation: PersonalizedPreparationSchema, now
    ) -> models.CandidateProfile:
        profile = self._ensure_candidate_profile(report, now)
        cv_profile = preparation.cv_profile
        profile.roles_json = json.dumps(cv_profile.roles)
        profile.seniority = cv_profile.seniority
        profile.industries_json = json.dumps(cv_profile.industries)
        profile.hard_skills_json = json.dumps(cv_profile.hard_skills)
        profile.soft_skills_json = json.dumps(cv_profile.soft_skills)
        profile.tools_json = json.dumps(cv_profile.tools)
        profile.education_json = json.dumps(cv_profile.education)
        profile.languages_json = json.dumps(cv_profile.languages)
        profile.achievements_json = json.dumps(cv_profile.achievements)
        profile.confidence = cv_profile.confidence.value

        preparation_model = models.PersonalizedPreparation(
            id=str(uuid4()),
            candidate_profile_id=profile.id,
            fit_summary=preparation.fit_summary.summary,
            fit_confidence=preparation.fit_summary.confidence.value,
            suggested_pitch=preparation.suggested_pitch.text
            if preparation.suggested_pitch
            else None,
            warnings_json=json.dumps(
                [warning.model_dump(mode="json") for warning in preparation.warnings]
            ),
            created_at=now,
        )
        self.db.add(preparation_model)
        self.db.flush()

        items: list[tuple[str, PersonalizedPreparationItemSchema | StarAnswerOutlineSchema]] = []
        items.extend(("strength", item) for item in preparation.strengths_to_highlight)
        items.extend(("gap", item) for item in preparation.gaps_to_prepare)
        items.extend(("question", item) for item in preparation.personalized_questions)
        items.extend(("question_for_company", item) for item in preparation.questions_for_company)
        items.extend(("star_outline", item) for item in preparation.star_answer_outlines)
        for index, (item_type, item) in enumerate(items):
            text = item.question if isinstance(item, StarAnswerOutlineSchema) else item.text
            confidence = item.confidence.value
            self.db.add(
                models.PreparationItem(
                    id=str(uuid4()),
                    personalized_preparation_id=preparation_model.id,
                    item_type=item_type,
                    text=text,
                    reason=item.reason if isinstance(item, PersonalizedPreparationItemSchema) else None,
                    confidence=confidence,
                    payload_json=item.model_dump_json()
                    if isinstance(item, StarAnswerOutlineSchema)
                    else None,
                    display_order=index,
                    created_at=now,
                )
            )
        return profile

    def _save_cv_tailoring(
        self, profile: models.CandidateProfile, tailoring: CVTailoringSchema, now
    ) -> None:
        tailoring_model = models.CVTailoring(
            id=str(uuid4()),
            candidate_profile_id=profile.id,
            positioning_summary=tailoring.positioning_summary,
            adapted_cv_draft_markdown=tailoring.adapted_cv_draft.content_markdown
            if tailoring.adapted_cv_draft
            else None,
            warnings_json=json.dumps(
                [warning.model_dump(mode="json") for warning in tailoring.warnings]
            ),
            created_at=now,
        )
        self.db.add(tailoring_model)
        self.db.flush()
        included_ids = (
            set(tailoring.adapted_cv_draft.included_suggestion_ids)
            if tailoring.adapted_cv_draft
            else set()
        )
        for index, suggestion in enumerate(tailoring.change_suggestions):
            self.db.add(
                models.CVTailoringSuggestion(
                    id=suggestion.id,
                    cv_tailoring_id=tailoring_model.id,
                    suggestion_type=suggestion.type.value,
                    original_text=suggestion.original_text,
                    suggested_text=suggestion.suggested_text,
                    reason=suggestion.reason,
                    evidence_ids_json=json.dumps(suggestion.evidence_ids),
                    requires_user_confirmation=suggestion.requires_user_confirmation,
                    confidence=suggestion.confidence.value,
                    included_in_draft=suggestion.id in included_ids,
                    display_order=index,
                    created_at=now,
                )
            )

    def _to_personalized_preparation(
        self, report: models.Report
    ) -> PersonalizedPreparationSchema | None:
        profile = report.candidate_profile
        if not profile or not profile.personalized_preparation:
            return None
        preparation = profile.personalized_preparation
        cv_profile = CVProfileSchema(
            roles=safe_json_list(profile.roles_json),
            seniority=profile.seniority,
            industries=safe_json_list(profile.industries_json),
            hard_skills=safe_json_list(profile.hard_skills_json),
            soft_skills=safe_json_list(profile.soft_skills_json),
            tools=safe_json_list(profile.tools_json),
            education=safe_json_list(profile.education_json),
            languages=safe_json_list(profile.languages_json),
            achievements=safe_json_list(profile.achievements_json),
            confidence=profile.confidence,
        )
        items = sorted(preparation.items, key=lambda item: item.display_order)
        return PersonalizedPreparationSchema(
            cv_profile=cv_profile,
            fit_summary=FitSummarySchema(
                summary=preparation.fit_summary,
                confidence=preparation.fit_confidence,
                evidence_ids=[],
            ),
            strengths_to_highlight=[
                preparation_item_schema(item) for item in items if item.item_type == "strength"
            ],
            gaps_to_prepare=[
                preparation_item_schema(item) for item in items if item.item_type == "gap"
            ],
            personalized_questions=[
                preparation_item_schema(item) for item in items if item.item_type == "question"
            ],
            questions_for_company=[
                preparation_item_schema(item)
                for item in items
                if item.item_type == "question_for_company"
            ],
            star_answer_outlines=[
                StarAnswerOutlineSchema(**safe_json_dict(item.payload_json))
                for item in items
                if item.item_type == "star_outline" and item.payload_json
            ],
            warnings=[WarningSchema(**warning) for warning in safe_json_list(preparation.warnings_json)],
        )

    def _to_cv_tailoring(self, report: models.Report) -> CVTailoringSchema | None:
        profile = report.candidate_profile
        if not profile or not profile.cv_tailoring:
            return None
        tailoring = profile.cv_tailoring
        suggestions = sorted(tailoring.suggestions, key=lambda item: item.display_order)
        return CVTailoringSchema(
            positioning_summary=tailoring.positioning_summary,
            change_suggestions=[
                CVTailoringSuggestionSchema(
                    id=suggestion.id,
                    type=suggestion.suggestion_type,
                    original_text=suggestion.original_text,
                    suggested_text=suggestion.suggested_text,
                    reason=suggestion.reason,
                    evidence_ids=safe_json_list(suggestion.evidence_ids_json),
                    requires_user_confirmation=suggestion.requires_user_confirmation,
                    confidence=suggestion.confidence,
                )
                for suggestion in suggestions
            ],
            adapted_cv_draft=AdaptedCVDraftSchema(
                title="CV adaptado",
                content_markdown=tailoring.adapted_cv_draft_markdown,
                included_suggestion_ids=[
                    suggestion.id for suggestion in suggestions if suggestion.included_in_draft
                ],
                excluded_suggestion_ids=[
                    suggestion.id for suggestion in suggestions if not suggestion.included_in_draft
                ],
            )
            if tailoring.adapted_cv_draft_markdown
            else None,
            warnings=[WarningSchema(**warning) for warning in safe_json_list(tailoring.warnings_json)],
        )


def preparation_item_schema(item: models.PreparationItem) -> PersonalizedPreparationItemSchema:
    return PersonalizedPreparationItemSchema(
        text=item.text,
        reason=item.reason,
        evidence_ids=[],
        confidence=item.confidence,
    )


def parse_datetime_or_none(value: str | None):
    if not value:
        return None
    from datetime import datetime

    return datetime.fromisoformat(value)


def safe_json_list(value: str | None) -> list:
    if not value:
        return []
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return []
    return loaded if isinstance(loaded, list) else []


def safe_json_dict(value: str | None) -> dict:
    if not value:
        return {}
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def safe_float_list(value: str | None) -> list[float]:
    if not value:
        return []
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(loaded, list):
        return []
    return [float(item) for item in loaded if isinstance(item, int | float)]


def scoped_report_id(report_id: str, item_id: str) -> str:
    prefix = f"{report_id}_"
    return item_id if item_id.startswith(prefix) else f"{prefix}{item_id}"
