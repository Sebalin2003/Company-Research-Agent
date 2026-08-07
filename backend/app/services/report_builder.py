from __future__ import annotations

import re
import time
from datetime import timedelta
from typing import Protocol

from backend.app.core.config import Settings
from backend.app.core.time import utc_now
from backend.app.db import models
from backend.app.domain.cv import CandidateSignals, extract_candidate_signals, select_cv_evidence_lines
from backend.app.domain.reports import (
    ClaimSchema,
    ClaimType,
    CompanySchema,
    ConfidenceLevel,
    EvidenceSchema,
    EvidenceTopic,
    ReportMetadataSchema,
    ReportStatus,
    SectionSchema,
    SectionType,
    SourceSchema,
    SourceType,
    StructuredReportSchema,
    WarningSchema,
    WarningSeverity,
)
from backend.app.llm.cv_tailoring import GeminiCVTailoringService, fallback_warning
from backend.app.llm.gemini import GeminiReportSynthesizer
from backend.app.llm.synthesizer import ReportSynthesizer, SynthesisError, SynthesisRequest
from backend.app.research.content_extractor import HttpContentExtractor
from backend.app.research.pipeline import ResearchPipeline, ResearchPipelineResult
from backend.app.research.search_provider import TavilySearchProvider
from backend.app.services.cv_preparation import build_cv_tailoring, build_personalized_preparation


REQUIRED_REPORT_SECTIONS: tuple[tuple[SectionType, str, str], ...] = (
    (
        SectionType.executive_summary,
        "Resumen ejecutivo",
        "No hay evidencia suficiente para generar un resumen ejecutivo completo.",
    ),
    (
        SectionType.business,
        "Negocio principal",
        "No hay evidencia suficiente sobre el negocio principal, productos, servicios o modelo de negocio.",
    ),
    (
        SectionType.argentina_presence,
        "Presencia en Argentina",
        "No hay evidencia suficiente sobre oficinas, contratacion, presencia local o modalidad remota/hibrida.",
    ),
    (
        SectionType.employees,
        "Cantidad de empleados",
        "No hay evidencia suficiente para estimar la cantidad de empleados con confianza.",
    ),
    (
        SectionType.salary_benefits,
        "Sueldos y beneficios",
        "No hay evidencia suficiente sobre rangos salariales IT para trainee, pasantia/internship o junior; cualquier estimacion seria incierta.",
    ),
    (
        SectionType.culture,
        "Cultura laboral",
        "No hay evidencia suficiente sobre cultura laboral o reviews de empleados.",
    ),
    (
        SectionType.interview_process,
        "Proceso de entrevista",
        "No hay evidencia suficiente sobre etapas, evaluaciones o dificultad de entrevista.",
    ),
    (
        SectionType.interview_questions,
        "Posibles preguntas de entrevista",
        "No hay evidencia suficiente para generar preguntas especificas respaldadas por fuentes.",
    ),
    (
        SectionType.open_roles,
        "Busquedas abiertas",
        "No hay evidencia suficiente sobre busquedas abiertas actuales.",
    ),
)

SALARY_RANGE_RE = re.compile(
    r"((\$|ars|usd|pesos?)\s*\d[\d\.\,]*|\d[\d\.\,]*\s*(ars|usd|pesos?|k|mil|millones?))"
    r".{0,40}(a|hasta|-|/|entre)"
    r".{0,40}((\$|ars|usd|pesos?)\s*\d[\d\.\,]*|\d[\d\.\,]*\s*(ars|usd|pesos?|k|mil|millones?))|"
    r"entre.{0,40}((\$|ars|usd|pesos?)\s*\d[\d\.\,]*|\d[\d\.\,]*\s*(ars|usd|pesos?|k|mil|millones?))"
    r".{0,40}(y|-|a|hasta)"
    r".{0,40}((\$|ars|usd|pesos?)\s*\d[\d\.\,]*|\d[\d\.\,]*\s*(ars|usd|pesos?|k|mil|millones?))",
    re.IGNORECASE,
)
EMPLOYEE_COUNT_RE = re.compile(
    r"\d[\d\.\,]*\s*(empleados|employees|personas|trabajadores)|"
    r"(empleados|employees|personas|trabajadores)\s*[:\-]?\s*\d",
    re.IGNORECASE,
)
EXACT_ADDRESS_RE = re.compile(
    r"\b(av\.?|avenida|calle|ruta|boulevard|bouchard|alem|corrientes|cordoba|"
    r"caba|piso|oficina|domicilio|direccion|sede)\b.{0,80}\d|"
    r"\d{2,5}.{0,80}\b(buenos aires|caba|piso|oficina)\b",
    re.IGNORECASE,
)
INTERVIEW_STAGE_RE = re.compile(
    r"\b(etapa|etapas|ronda|rondas|recruiter|rrhh|tecnica|manager|"
    r"assessment|psicotecnico|entrevista final)\b",
    re.IGNORECASE,
)
IT_ENTRY_LEVEL_SALARY_CONTEXT_RE = re.compile(
    r"\b(it|tecnologia|tecnologia|sistemas|software|developer|desarrollador|programador|"
    r"trainee|junior|jr|pasantia|pasantia|pasante|internship|intern)\b",
    re.IGNORECASE,
)

EXACT_DETAIL_RULES = (
    (
        SectionType.argentina_presence,
        (EvidenceTopic.argentina_presence,),
        EXACT_ADDRESS_RE,
        "Direccion u oficina exacta: no disponible en las fuentes consultadas.",
    ),
    (
        SectionType.employees,
        (EvidenceTopic.employees,),
        EMPLOYEE_COUNT_RE,
        "Cantidad total de empleados: no disponible en las fuentes consultadas.",
    ),
    (
        SectionType.salary_benefits,
        (EvidenceTopic.salary,),
        None,
        "Rango salarial IT trainee/pasantia/junior: no disponible en las fuentes consultadas.",
    ),
    (
        SectionType.interview_process,
        (EvidenceTopic.interview_process,),
        INTERVIEW_STAGE_RE,
        "Etapas exactas del proceso: no disponible en las fuentes consultadas.",
    ),
)


class ReportBuilder(Protocol):
    def build(
        self,
        report: models.Report,
        include_cv: bool,
        include_cv_tailoring: bool,
        include_adapted_cv_draft: bool,
        cv_text: str | None = None,
    ) -> StructuredReportSchema:
        ...


class MockReportBuilder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build(
        self,
        report: models.Report,
        include_cv: bool,
        include_cv_tailoring: bool,
        include_adapted_cv_draft: bool,
        cv_text: str | None = None,
    ) -> StructuredReportSchema:
        now = utc_now()
        generated_at = now.isoformat()
        valid_until = (now + timedelta(days=self.settings.report_freshness_days)).isoformat()
        company = CompanySchema(
            id=report.company.id,
            name=report.company.name,
            normalized_name=report.company.normalized_name,
        )
        sources = [
            SourceSchema(
                id="source_mock_001",
                title=f"Sitio oficial de {report.company.name}",
                url=f"https://example.com/{report.company.normalized_name.replace(' ', '-')}",
                domain="example.com",
                source_type=SourceType.official,
                reliability_score=3,
                accessed_at=generated_at,
                snippet="Fuente simulada para validar el flujo del MVP.",
                language="es",
                is_current=True,
            )
        ]
        evidence = [
            EvidenceSchema(
                id="evidence_mock_001",
                source_id="source_mock_001",
                topic=EvidenceTopic.business,
                claim="Este es un dato simulado para validar el flujo de generacion.",
                raw_text_excerpt="Fuente simulada para validar el flujo del MVP.",
                confidence=ConfidenceLevel.low,
            )
        ]
        sections = [
            SectionSchema(
                type=SectionType.executive_summary,
                title="Resumen ejecutivo",
                summary=(
                    f"Informe simulado para {report.company.name}. "
                    "Este contenido valida el flujo tecnico antes de integrar busqueda real y Gemini."
                ),
                claims=[
                    ClaimSchema(
                        id="claim_mock_001",
                        type=ClaimType.fact,
                        text="El informe fue generado con datos simulados para validar el flujo del MVP.",
                        evidence_ids=["evidence_mock_001"],
                        confidence=ConfidenceLevel.low,
                    )
                ],
                confidence=ConfidenceLevel.low,
                missing_evidence=False,
            ),
        ]
        warnings = [
            WarningSchema(
                id="warning_mock_001",
                type="mock_report",
                message="Este informe usa datos simulados. No debe usarse para tomar decisiones laborales.",
                severity=WarningSeverity.medium,
                related_section=SectionType.warnings,
            )
        ]

        personalized_preparation = None
        if include_cv:
            signals = extract_candidate_signals(cv_text or "")
            personalized_preparation = build_personalized_preparation(
                signals=signals,
                evidence_ids=["evidence_mock_001"],
            )

        cv_tailoring = None
        if include_cv_tailoring:
            signals = extract_candidate_signals(cv_text or "")
            cv_tailoring = build_cv_tailoring(
                signals=signals,
                company_name=report.company.name,
                evidence_ids=["evidence_mock_001"],
                include_adapted_cv_draft=include_adapted_cv_draft,
            )

        return complete_report_sections(StructuredReportSchema(
            report_id=report.id,
            company=company,
            status=ReportStatus.completed,
            generated_at=generated_at,
            valid_until=valid_until,
            sections=sections,
            personalized_preparation=personalized_preparation,
            cv_tailoring=cv_tailoring,
            sources=sources,
            evidence=evidence,
            warnings=warnings,
            metadata=ReportMetadataSchema(
                search_provider=self.settings.search_provider,
                llm_provider="gemini",
                llm_model=self.settings.gemini_model,
                source_count=len(sources),
                evidence_count=len(evidence),
                used_cv=include_cv,
                used_cv_tailoring=include_cv_tailoring,
                generation_duration_ms=0,
            ),
        ))


class RealReportBuilder:
    def __init__(
        self,
        settings: Settings,
        research_pipeline: ResearchPipeline,
        synthesizer: ReportSynthesizer,
        cv_tailoring_service: GeminiCVTailoringService | None = None,
    ) -> None:
        self.settings = settings
        self.research_pipeline = research_pipeline
        self.synthesizer = synthesizer
        self.cv_tailoring_service = cv_tailoring_service

    def build(
        self,
        report: models.Report,
        include_cv: bool,
        include_cv_tailoring: bool,
        include_adapted_cv_draft: bool,
        cv_text: str | None = None,
    ) -> StructuredReportSchema:
        build_started = time.perf_counter()
        research_result = self.research_pipeline.run(report.company.name)
        synthesis_request = SynthesisRequest(
            report_id=report.id,
            company_id=report.company.id,
            company_name=report.company.name,
            normalized_company_name=report.company.normalized_name,
            sources=research_result.sources,
            evidence=research_result.evidence,
            include_cv=False,
            include_cv_tailoring=False,
            include_adapted_cv_draft=False,
            cv_text=None,
        )
        synthesis_started = time.perf_counter()
        if not research_result.sources or not research_result.evidence:
            synthesized = build_fallback_report(
                report,
                research_result,
                self.settings,
                synthesis_error="No se encontraron fuentes suficientes para generar el informe.",
            )
        else:
            try:
                synthesized = self.synthesizer.synthesize(synthesis_request)
            except SynthesisError as exc:
                synthesized = build_fallback_report(
                    report,
                    research_result,
                    self.settings,
                    synthesis_error=str(exc),
                )
        synthesis_duration_ms = elapsed_ms(synthesis_started)

        personalized_preparation = None
        cv_tailoring = None
        if include_cv or include_cv_tailoring:
            signals = extract_candidate_signals(cv_text or "")
            evidence_ids = first_evidence_ids(synthesized)
            if include_cv and personalized_preparation is None:
                personalized_preparation = build_personalized_preparation(signals, evidence_ids)
            if include_cv_tailoring:
                cv_tailoring = self.build_ai_or_rule_cv_tailoring(
                    company_name=report.company.name,
                    signals=signals,
                    cv_text=cv_text or "",
                    report=synthesized,
                    evidence_ids=evidence_ids,
                    include_adapted_cv_draft=include_adapted_cv_draft,
                )
                cv_tailoring = warn_if_cv_signals_are_weak(cv_tailoring, signals)

        completed = synthesized.model_copy(
            update={
                "report_id": report.id,
                "company": CompanySchema(
                    id=report.company.id,
                    name=report.company.name,
                    normalized_name=report.company.normalized_name,
                ),
                "status": ReportStatus.completed,
                "personalized_preparation": personalized_preparation,
                "cv_tailoring": cv_tailoring,
                "metadata": synthesized.metadata.model_copy(
                    update={
                        "search_provider": self.settings.search_provider,
                        "llm_provider": "gemini",
                        "llm_model": self.settings.gemini_model,
                        "source_count": len(synthesized.sources),
                        "evidence_count": len(synthesized.evidence),
                        "used_cv": include_cv,
                        "used_cv_tailoring": include_cv_tailoring,
                        "generation_duration_ms": elapsed_ms(build_started),
                        "research_duration_ms": research_result.research_duration_ms,
                        "research_search_duration_ms": research_result.search_duration_ms,
                        "research_extraction_duration_ms": research_result.extraction_duration_ms,
                        "synthesis_duration_ms": synthesis_duration_ms,
                        "research_search_count": research_result.search_count,
                        "research_extracted_url_count": research_result.extracted_url_count,
                        "research_extraction_skipped_count": research_result.extraction_skipped_count,
                        "research_snippet_only_count": research_result.snippet_only_count,
                    }
                ),
            }
        )
        return complete_report_sections(completed)

    def build_ai_or_rule_cv_tailoring(
        self,
        company_name: str,
        signals: CandidateSignals,
        cv_text: str,
        report: StructuredReportSchema,
        evidence_ids: list[str],
        include_adapted_cv_draft: bool,
    ):
        fallback = build_cv_tailoring(
            signals=signals,
            company_name=company_name,
            evidence_ids=evidence_ids,
            include_adapted_cv_draft=include_adapted_cv_draft,
        )
        service = self.cv_tailoring_service or GeminiCVTailoringService(
            api_key=self.settings.gemini_api_key or "",
            model=self.settings.gemini_model,
        )
        try:
            return service.build(
                company_name=company_name,
                signals=signals,
                cv_evidence_lines=select_cv_evidence_lines(cv_text, signals),
                report=report,
                include_adapted_cv_draft=include_adapted_cv_draft,
            )
        except SynthesisError as exc:
            return fallback.model_copy(
                update={"warnings": [*fallback.warnings, fallback_warning(str(exc))]}
            )


def build_report_builder(settings: Settings) -> ReportBuilder:
    if settings.search_provider == "mock":
        return MockReportBuilder(settings)
    if settings.search_provider == "tavily":
        search_provider = TavilySearchProvider(
            api_key=settings.search_provider_api_key or "",
            timeout_seconds=settings.research_search_timeout_seconds,
        )
        research_pipeline = ResearchPipeline(
            search_provider,
            HttpContentExtractor(timeout_seconds=settings.research_extract_timeout_seconds),
            results_per_query=settings.research_results_per_query,
            search_concurrency=settings.research_search_concurrency,
            extract_concurrency=settings.research_extract_concurrency,
            enable_deepening=settings.research_enable_deepening,
            max_extract_urls=settings.research_max_extract_urls,
            extract_protected_domains=settings.research_extract_protected_domains,
            max_extract_urls_per_topic=settings.research_max_extract_urls_per_topic,
        )
        synthesizer = GeminiReportSynthesizer(
            api_key=settings.gemini_api_key or "",
            model=settings.gemini_model,
        )
        cv_tailoring_service = GeminiCVTailoringService(
            api_key=settings.gemini_api_key or "",
            model=settings.gemini_model,
        )
        return RealReportBuilder(settings, research_pipeline, synthesizer, cv_tailoring_service)
    raise ValueError(f"Search provider no soportado: {settings.search_provider}")


def first_evidence_ids(report: StructuredReportSchema) -> list[str]:
    return [item.id for item in report.evidence[:1]]


def warn_if_cv_signals_are_weak(cv_tailoring, signals: CandidateSignals):
    if signals.confidence not in {"low", "unknown"}:
        return cv_tailoring
    if any(warning.type == "insufficient_cv_detail" for warning in cv_tailoring.warnings):
        return cv_tailoring
    warning = WarningSchema(
        id="warning_cv_tailoring_detail",
        type="insufficient_cv_detail",
        message="La adaptacion del CV es limitada porque faltan logros, herramientas o roles claros.",
        severity=WarningSeverity.medium,
    )
    return cv_tailoring.model_copy(update={"warnings": [*cv_tailoring.warnings, warning]})


def build_fallback_report(
    report: models.Report,
    research_result: ResearchPipelineResult,
    settings: Settings,
    synthesis_error: str | None = None,
) -> StructuredReportSchema:
    now = utc_now().isoformat()
    sources = [
        SourceSchema(
            id=source.source_id,
            title=source.title,
            url=source.url,
            domain=source.domain,
            source_type=source.source_type,
            reliability_score=source.reliability_score,
            accessed_at=now,
            snippet=source.snippet,
            language="es",
            is_current=source.is_current,
        )
        for source in research_result.sources
    ]
    evidence = [
        EvidenceSchema(
            id=f"evidence_{index + 1}",
            source_id=item.source_id,
            topic=item.topic,
            claim=item.claim,
            raw_text_excerpt=item.raw_text_excerpt,
            confidence=item.confidence,
        )
        for index, item in enumerate(research_result.evidence)
    ]
    claim_ids = [item.id for item in evidence[:5]]
    claims = [
        ClaimSchema(
            id=f"claim_fallback_{index + 1}",
            type=ClaimType.fact,
            text=item.claim,
            evidence_ids=[item.id],
            confidence=item.confidence,
        )
        for index, item in enumerate(evidence[:5])
    ]
    if not claims:
        claims = [
            ClaimSchema(
                id="claim_fallback_missing_evidence",
                type=ClaimType.missing_evidence,
                text="No se pudo extraer evidencia suficiente para generar afirmaciones factuales.",
                evidence_ids=[],
                confidence=ConfidenceLevel.high,
            )
        ]

    has_evidence = bool(claim_ids)
    summary = (
        f"Informe con informacion limitada para {report.company.name}. "
        "La busqueda real no encontro evidencia suficiente para generar afirmaciones factuales."
    )
    if has_evidence:
        summary = (
            f"Informe de respaldo para {report.company.name}. "
            "La busqueda real encontro fuentes, pero la sintesis con Gemini no devolvio JSON valido."
        )

    sections = [
        SectionSchema(
            type=SectionType.executive_summary,
            title="Resumen ejecutivo",
            summary=summary,
            claims=claims,
            confidence=ConfidenceLevel.medium if has_evidence else ConfidenceLevel.low,
            missing_evidence=not has_evidence,
        ),
    ]
    warning_type = "llm_synthesis_failed" if has_evidence else "insufficient_research_evidence"
    warnings = [
        WarningSchema(
            id=f"warning_{warning_type}",
            type=warning_type,
            message=fallback_warning_message(synthesis_error),
            severity=WarningSeverity.medium,
            related_section=SectionType.warnings,
        )
    ]
    return complete_report_sections(StructuredReportSchema(
        report_id=report.id,
        company=CompanySchema(
            id=report.company.id,
            name=report.company.name,
            normalized_name=report.company.normalized_name,
        ),
        status=ReportStatus.completed,
        generated_at=now,
        sections=sections,
        sources=sources,
        evidence=evidence,
        warnings=warnings,
        metadata=ReportMetadataSchema(
            search_provider=settings.search_provider,
            llm_provider="gemini",
            llm_model=settings.gemini_model,
            source_count=len(sources),
            evidence_count=len(evidence),
            generation_duration_ms=0,
        ),
    ))


def fallback_warning_message(synthesis_error: str | None) -> str:
    if synthesis_error and "No se encontraron fuentes suficientes" in synthesis_error:
        return (
            "No se encontraron fuentes suficientes para generar un informe factual. "
            "Se muestra un informe limitado sin inventar datos."
        )
    message = (
        "Gemini no genero un informe estructurado valido. "
        "Se muestra un informe de respaldo basado en evidencia recolectada."
    )
    if not synthesis_error:
        return message
    return f"{message} Motivo tecnico: {synthesis_error[:300]}"


def elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def complete_report_sections(report: StructuredReportSchema) -> StructuredReportSchema:
    sections_by_type = {section.type: section for section in report.sections}
    ordered_sections = []
    for section_type, title, fallback_summary in REQUIRED_REPORT_SECTIONS:
        section = sections_by_type.pop(section_type, None)
        if section is None:
            section = missing_section(section_type, title, fallback_summary)
        ordered_sections.append(section)
    ordered_sections.extend(sections_by_type.values())
    return clarify_unavailable_exact_details(report.model_copy(update={"sections": ordered_sections}))


def clarify_unavailable_exact_details(report: StructuredReportSchema) -> StructuredReportSchema:
    evidence_text_by_topic = build_evidence_text_by_topic(report)
    updated_sections = []
    for section in report.sections:
        unavailable_messages = exact_detail_messages_for_section(
            section.type,
            evidence_text_by_topic,
        )
        summary = section.summary
        for message in unavailable_messages:
            if message not in summary:
                summary = append_exact_detail_message(section.type, summary, message)
        updated_sections.append(section.model_copy(update={"summary": summary}))
    return report.model_copy(update={"sections": updated_sections})


def append_exact_detail_message(section_type: SectionType, summary: str, message: str) -> str:
    if section_type == SectionType.salary_benefits:
        return f"{message} {summary.strip()}"
    return f"{summary.rstrip()} {message}"


def build_evidence_text_by_topic(report: StructuredReportSchema) -> dict[EvidenceTopic, str]:
    source_snippets_by_id = {source.id: source.snippet or "" for source in report.sources}
    evidence_text_by_topic: dict[EvidenceTopic, list[str]] = {}
    for item in report.evidence:
        text = " ".join(
            part
            for part in (
                item.claim,
                item.raw_text_excerpt or "",
                source_snippets_by_id.get(item.source_id, ""),
            )
            if part
        )
        evidence_text_by_topic.setdefault(item.topic, []).append(text)
    return {
        topic: " ".join(texts)
        for topic, texts in evidence_text_by_topic.items()
    }


def exact_detail_messages_for_section(
    section_type: SectionType,
    evidence_text_by_topic: dict[EvidenceTopic, str],
) -> list[str]:
    messages = []
    for rule_section, topics, pattern, unavailable_message in EXACT_DETAIL_RULES:
        if section_type != rule_section:
            continue
        topic_text = " ".join(evidence_text_by_topic.get(topic, "") for topic in topics)
        if section_type == SectionType.salary_benefits:
            messages.append(salary_range_message(topic_text, unavailable_message))
            continue
        if not pattern.search(topic_text):
            messages.append(unavailable_message)
    return messages


def salary_range_message(text: str, unavailable_message: str) -> str:
    range_match = SALARY_RANGE_RE.search(text)
    if range_match and IT_ENTRY_LEVEL_SALARY_CONTEXT_RE.search(text):
        range_text = " ".join(range_match.group(0).split())
        return f"Rango salarial IT trainee/pasantia/junior en fuentes: {range_text}."
    return unavailable_message


def missing_section(section_type: SectionType, title: str, summary: str) -> SectionSchema:
    return SectionSchema(
        type=section_type,
        title=title,
        summary=summary,
        claims=[
            ClaimSchema(
                id=f"claim_missing_{section_type.value}",
                type=ClaimType.missing_evidence,
                text=summary,
                evidence_ids=[],
                confidence=ConfidenceLevel.high,
            )
        ],
        confidence=ConfidenceLevel.low,
        missing_evidence=True,
    )
