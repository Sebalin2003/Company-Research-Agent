from __future__ import annotations

from types import SimpleNamespace

import httpx

from backend.app.core.config import Settings
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
)
from backend.app.llm.synthesizer import SynthesisRequest
from backend.app.llm.synthesizer import SynthesisError
from backend.app.research.content_extractor import HttpContentExtractor
from backend.app.research.pipeline import ResearchPipelineResult
from backend.app.research.types import ClassifiedEvidence, ExtractedContent, ScoredSource
from backend.app.services.report_builder import (
    RealReportBuilder,
    build_report_builder,
    clarify_unavailable_exact_details,
)
from backend.app.services.cv_preparation import build_cv_tailoring


class RecordingPipeline:
    def __init__(self) -> None:
        self.company_name = None
        self.result = ResearchPipelineResult(
            sources=[
                ScoredSource(
                    source_id="source_1",
                    title="Acme Careers",
                    url="https://careers.acme.com/jobs",
                    domain="careers.acme.com",
                    source_type=SourceType.career_page,
                    reliability_score=5,
                    snippet="Empleos en Argentina.",
                    is_current=True,
                )
            ],
            evidence=[
                ClassifiedEvidence(
                    source_id="source_1",
                    topic=EvidenceTopic.open_roles,
                    claim="Acme publica empleos en Argentina.",
                    raw_text_excerpt="Empleos en Argentina.",
                    confidence=ConfidenceLevel.medium,
                )
            ],
            research_duration_ms=120,
            search_duration_ms=70,
            extraction_duration_ms=40,
            search_count=8,
            extracted_url_count=3,
            extraction_skipped_count=5,
            snippet_only_count=4,
        )

    def run(self, company_name: str) -> ResearchPipelineResult:
        self.company_name = company_name
        return self.result


class RecordingSynthesizer:
    def __init__(self) -> None:
        self.request: SynthesisRequest | None = None

    def synthesize(self, request: SynthesisRequest) -> StructuredReportSchema:
        self.request = request
        return StructuredReportSchema(
            report_id="wrong_report_id",
            company=CompanySchema(id="wrong_company_id", name="Wrong", normalized_name="wrong"),
            status=ReportStatus.completed,
            generated_at="2026-07-25T00:00:00+00:00",
            sections=[
                SectionSchema(
                    type=SectionType.executive_summary,
                    title="Resumen ejecutivo",
                    summary="Acme publica empleos en Argentina.",
                    claims=[
                        ClaimSchema(
                            id="claim_1",
                            type=ClaimType.fact,
                            text="Acme publica empleos en Argentina.",
                            evidence_ids=["evidence_1"],
                            confidence=ConfidenceLevel.medium,
                        )
                    ],
                    confidence=ConfidenceLevel.medium,
                    missing_evidence=False,
                )
            ],
            sources=[
                SourceSchema(
                    id="source_1",
                    title="Acme Careers",
                    url="https://careers.acme.com/jobs",
                    domain="careers.acme.com",
                    source_type=SourceType.career_page,
                    reliability_score=5,
                    accessed_at="2026-07-25T00:00:00+00:00",
                    snippet="Empleos en Argentina.",
                    is_current=True,
                )
            ],
            evidence=[
                EvidenceSchema(
                    id="evidence_1",
                    source_id="source_1",
                    topic=EvidenceTopic.open_roles,
                    claim="Acme publica empleos en Argentina.",
                    raw_text_excerpt="Empleos en Argentina.",
                    confidence=ConfidenceLevel.medium,
                )
            ],
            warnings=[],
            metadata=ReportMetadataSchema(
                search_provider="unknown",
                llm_model="wrong-model",
            ),
        )


class FailingSynthesizer:
    def synthesize(self, request: SynthesisRequest) -> StructuredReportSchema:
        raise SynthesisError("bad model output")


class RecordingCVTailoringService:
    def __init__(self) -> None:
        self.calls = []

    def build(
        self,
        company_name,
        signals,
        cv_evidence_lines,
        report,
        include_adapted_cv_draft,
    ):
        self.calls.append(
            {
                "company_name": company_name,
                "signals": signals,
                "cv_evidence_lines": cv_evidence_lines,
                "report": report,
                "include_adapted_cv_draft": include_adapted_cv_draft,
            }
        )
        return build_cv_tailoring(
            signals=signals,
            company_name=company_name,
            evidence_ids=["evidence_1"],
            include_adapted_cv_draft=include_adapted_cv_draft,
        )


class FailingCVTailoringService:
    def build(self, **kwargs):
        raise SynthesisError("tailoring unavailable")


class EmptyPipeline:
    def run(self, company_name: str) -> ResearchPipelineResult:
        return ResearchPipelineResult(sources=[], evidence=[], search_count=8)


def test_real_report_builder_runs_research_and_synthesis_with_cv_flags() -> None:
    pipeline = RecordingPipeline()
    synthesizer = RecordingSynthesizer()
    cv_tailoring_service = RecordingCVTailoringService()
    builder = RealReportBuilder(
        Settings(search_provider="tavily", gemini_model="gemini-test-model"),
        pipeline,  # type: ignore[arg-type]
        synthesizer,
        cv_tailoring_service,  # type: ignore[arg-type]
    )
    company = SimpleNamespace(id="company_1", name="Acme", normalized_name="acme")
    report = SimpleNamespace(id="report_1", company=company)

    structured_report = builder.build(
        report,  # type: ignore[arg-type]
        include_cv=True,
        include_cv_tailoring=True,
        include_adapted_cv_draft=True,
        cv_text="Analista de datos con SQL. Mejore reportes internos.",
    )

    assert pipeline.company_name == "Acme"
    assert synthesizer.request is not None
    assert synthesizer.request.include_cv is False
    assert synthesizer.request.include_cv_tailoring is False
    assert synthesizer.request.include_adapted_cv_draft is False
    assert synthesizer.request.cv_text is None
    assert cv_tailoring_service.calls[0]["cv_evidence_lines"]
    assert structured_report.report_id == "report_1"
    assert structured_report.company.id == "company_1"
    assert structured_report.metadata.search_provider == "tavily"
    assert structured_report.metadata.llm_model == "gemini-test-model"
    assert structured_report.metadata.generation_duration_ms is not None
    assert structured_report.metadata.research_duration_ms == 120
    assert structured_report.metadata.research_search_duration_ms == 70
    assert structured_report.metadata.research_extraction_duration_ms == 40
    assert structured_report.metadata.synthesis_duration_ms is not None
    assert structured_report.metadata.research_search_count == 8
    assert structured_report.metadata.research_extracted_url_count == 3
    assert structured_report.metadata.research_extraction_skipped_count == 5
    assert structured_report.metadata.research_snippet_only_count == 4
    assert structured_report.personalized_preparation is not None
    assert structured_report.cv_tailoring is not None
    assert structured_report.cv_tailoring.adapted_cv_draft is not None
    assert [section.type for section in structured_report.sections[:9]] == [
        SectionType.executive_summary,
        SectionType.business,
        SectionType.argentina_presence,
        SectionType.employees,
        SectionType.salary_benefits,
        SectionType.culture,
        SectionType.interview_process,
        SectionType.interview_questions,
        SectionType.open_roles,
    ]


def test_real_report_builder_returns_fallback_report_when_synthesis_fails() -> None:
    pipeline = RecordingPipeline()
    builder = RealReportBuilder(
        Settings(search_provider="tavily", gemini_model="gemini-test-model"),
        pipeline,  # type: ignore[arg-type]
        FailingSynthesizer(),
    )
    company = SimpleNamespace(id="company_1", name="Acme", normalized_name="acme")
    report = SimpleNamespace(id="report_1", company=company)

    structured_report = builder.build(
        report,  # type: ignore[arg-type]
        include_cv=False,
        include_cv_tailoring=False,
        include_adapted_cv_draft=False,
    )

    assert structured_report.status == ReportStatus.completed
    assert structured_report.sources[0].id == "source_1"
    assert structured_report.evidence[0].source_id == "source_1"
    assert structured_report.warnings[0].type == "llm_synthesis_failed"
    assert "bad model output" in structured_report.warnings[0].message
    assert len(structured_report.sections) == 9
    assert structured_report.sections[1].missing_evidence is True


def test_real_report_builder_falls_back_when_ai_cv_tailoring_fails() -> None:
    builder = RealReportBuilder(
        Settings(search_provider="tavily", gemini_model="gemini-test-model"),
        RecordingPipeline(),  # type: ignore[arg-type]
        RecordingSynthesizer(),
        FailingCVTailoringService(),  # type: ignore[arg-type]
    )
    company = SimpleNamespace(id="company_1", name="Acme", normalized_name="acme")
    report = SimpleNamespace(id="report_1", company=company)

    structured_report = builder.build(
        report,  # type: ignore[arg-type]
        include_cv=True,
        include_cv_tailoring=True,
        include_adapted_cv_draft=False,
        cv_text="Python y SQL.",
    )

    assert structured_report.cv_tailoring is not None
    assert structured_report.cv_tailoring.warnings[-1].type == "ai_cv_tailoring_fallback"


def test_real_report_builder_returns_limited_report_when_real_search_finds_no_sources() -> None:
    builder = RealReportBuilder(
        Settings(search_provider="tavily", gemini_model="gemini-test-model"),
        EmptyPipeline(),  # type: ignore[arg-type]
        RecordingSynthesizer(),
    )
    company = SimpleNamespace(id="company_1", name="Acme", normalized_name="acme")
    report = SimpleNamespace(id="report_1", company=company)

    structured_report = builder.build(
        report,  # type: ignore[arg-type]
        include_cv=False,
        include_cv_tailoring=False,
        include_adapted_cv_draft=False,
    )

    assert structured_report.status == ReportStatus.completed
    assert structured_report.sources == []
    assert structured_report.evidence == []
    assert structured_report.warnings[0].type == "insufficient_research_evidence"
    assert structured_report.sections[0].missing_evidence is True
    assert "informacion limitada" in structured_report.sections[0].summary


def test_build_report_builder_keeps_mock_as_default() -> None:
    builder = build_report_builder(Settings())

    assert builder.__class__.__name__ == "MockReportBuilder"


def test_report_builder_marks_unavailable_exact_details() -> None:
    report = StructuredReportSchema(
        report_id="report_1",
        company=CompanySchema(id="company_1", name="Acme", normalized_name="acme"),
        status=ReportStatus.completed,
        sections=[
            SectionSchema(
                type=SectionType.argentina_presence,
                title="Presencia en Argentina",
                summary="Hay senales de presencia local.",
                confidence=ConfidenceLevel.medium,
            ),
            SectionSchema(
                type=SectionType.employees,
                title="Cantidad de empleados",
                summary="La empresa parece grande.",
                confidence=ConfidenceLevel.low,
            ),
            SectionSchema(
                type=SectionType.salary_benefits,
                title="Sueldos y beneficios",
                summary="La fuente menciona beneficios y compensacion.",
                confidence=ConfidenceLevel.low,
            ),
            SectionSchema(
                type=SectionType.interview_process,
                title="Proceso de entrevista",
                summary="Hay menciones generales sobre entrevistas.",
                confidence=ConfidenceLevel.low,
            ),
        ],
        sources=[
            SourceSchema(
                id="source_1",
                title="Acme Reviews",
                url="https://example.com/acme",
                domain="example.com",
                source_type=SourceType.salary_review_platform,
                reliability_score=3,
                accessed_at="2026-07-25T00:00:00+00:00",
                snippet="Reviews en Buenos Aires con beneficios.",
            )
        ],
        evidence=[
            EvidenceSchema(
                id="evidence_1",
                source_id="source_1",
                topic=EvidenceTopic.argentina_presence,
                claim="La fuente menciona Buenos Aires.",
                raw_text_excerpt="Reviews en Buenos Aires.",
                confidence=ConfidenceLevel.low,
            ),
            EvidenceSchema(
                id="evidence_2",
                source_id="source_1",
                topic=EvidenceTopic.employees,
                claim="La fuente menciona una empresa grande.",
                raw_text_excerpt="Acme es una empresa grande.",
                confidence=ConfidenceLevel.low,
            ),
            EvidenceSchema(
                id="evidence_3",
                source_id="source_1",
                topic=EvidenceTopic.salary,
                claim="La fuente menciona compensacion.",
                raw_text_excerpt="Los empleados valoran beneficios y compensacion.",
                confidence=ConfidenceLevel.low,
            ),
            EvidenceSchema(
                id="evidence_4",
                source_id="source_1",
                topic=EvidenceTopic.interview_process,
                claim="La fuente menciona entrevistas.",
                raw_text_excerpt="Hay entrevistas.",
                confidence=ConfidenceLevel.low,
            ),
        ],
        metadata=ReportMetadataSchema(search_provider="tavily", llm_model="gemini-test-model"),
    )

    clarified = clarify_unavailable_exact_details(report)
    sections = {section.type: section.summary for section in clarified.sections}

    assert "Direccion u oficina exacta: no disponible" in sections[SectionType.argentina_presence]
    assert "Cantidad total de empleados: no disponible" in sections[SectionType.employees]
    assert "Rango salarial IT trainee/pasantia/junior: no disponible" in sections[SectionType.salary_benefits]
    assert sections[SectionType.salary_benefits].startswith("Rango salarial IT trainee/pasantia/junior")
    assert "Etapas exactas del proceso: no disponible" in sections[SectionType.interview_process]


def test_report_builder_does_not_mark_available_exact_details_unavailable() -> None:
    report = StructuredReportSchema(
        report_id="report_1",
        company=CompanySchema(id="company_1", name="Acme", normalized_name="acme"),
        status=ReportStatus.completed,
        sections=[
            SectionSchema(
                type=SectionType.argentina_presence,
                title="Presencia en Argentina",
                summary="Tiene oficina local.",
                confidence=ConfidenceLevel.medium,
            ),
            SectionSchema(
                type=SectionType.employees,
                title="Cantidad de empleados",
                summary="Informa dotacion global.",
                confidence=ConfidenceLevel.medium,
            ),
            SectionSchema(
                type=SectionType.salary_benefits,
                title="Sueldos y beneficios",
                summary="Informa salarios.",
                confidence=ConfidenceLevel.medium,
            ),
            SectionSchema(
                type=SectionType.interview_process,
                title="Proceso de entrevista",
                summary="Informa etapas.",
                confidence=ConfidenceLevel.medium,
            ),
        ],
        sources=[
            SourceSchema(
                id="source_1",
                title="Acme Details",
                url="https://example.com/acme",
                domain="example.com",
                source_type=SourceType.secondary,
                reliability_score=3,
                accessed_at="2026-07-25T00:00:00+00:00",
                snippet="Detalle de empresa.",
            )
        ],
        evidence=[
            EvidenceSchema(
                id="evidence_1",
                source_id="source_1",
                topic=EvidenceTopic.argentina_presence,
                claim="Direccion local.",
                raw_text_excerpt="Av. Corrientes 1234, Buenos Aires.",
                confidence=ConfidenceLevel.high,
            ),
            EvidenceSchema(
                id="evidence_2",
                source_id="source_1",
                topic=EvidenceTopic.employees,
                claim="Dotacion global.",
                raw_text_excerpt="La empresa tiene 12.500 empleados.",
                confidence=ConfidenceLevel.high,
            ),
            EvidenceSchema(
                id="evidence_3",
                source_id="source_1",
                topic=EvidenceTopic.salary,
                claim="Rango salarial mensual IT junior.",
                raw_text_excerpt="Desarrollador junior IT: rango mensual entre ARS 900.000 y ARS 1.200.000.",
                confidence=ConfidenceLevel.high,
            ),
            EvidenceSchema(
                id="evidence_4",
                source_id="source_1",
                topic=EvidenceTopic.interview_process,
                claim="Etapas de entrevista.",
                raw_text_excerpt="Proceso con recruiter, tecnica y manager.",
                confidence=ConfidenceLevel.high,
            ),
        ],
        metadata=ReportMetadataSchema(search_provider="tavily", llm_model="gemini-test-model"),
    )

    clarified = clarify_unavailable_exact_details(report)
    salary_summary = {
        section.type: section.summary for section in clarified.sections
    }[SectionType.salary_benefits]

    assert "no disponible en las fuentes consultadas" not in " ".join(
        section.summary for section in clarified.sections
    )
    assert salary_summary.startswith("Rango salarial IT trainee/pasantia/junior en fuentes:")
    assert "ARS 900.000" in salary_summary
    assert "ARS 1.200.000" in salary_summary


def test_report_builder_requires_it_entry_level_context_for_salary_range() -> None:
    report = StructuredReportSchema(
        report_id="report_1",
        company=CompanySchema(id="company_1", name="Acme", normalized_name="acme"),
        status=ReportStatus.completed,
        sections=[
            SectionSchema(
                type=SectionType.salary_benefits,
                title="Sueldos y beneficios",
                summary="La fuente informa salarios generales.",
                confidence=ConfidenceLevel.medium,
            ),
        ],
        sources=[
            SourceSchema(
                id="source_1",
                title="Acme Salary",
                url="https://example.com/acme",
                domain="example.com",
                source_type=SourceType.salary_review_platform,
                reliability_score=3,
                accessed_at="2026-07-25T00:00:00+00:00",
                snippet="Salarios generales.",
            )
        ],
        evidence=[
            EvidenceSchema(
                id="evidence_1",
                source_id="source_1",
                topic=EvidenceTopic.salary,
                claim="Rango salarial general.",
                raw_text_excerpt="Rango salarial mensual entre ARS 2.000.000 y ARS 3.000.000.",
                confidence=ConfidenceLevel.high,
            ),
        ],
        metadata=ReportMetadataSchema(search_provider="tavily", llm_model="gemini-test-model"),
    )

    clarified = clarify_unavailable_exact_details(report)

    assert clarified.sections[0].summary.startswith("Rango salarial IT trainee/pasantia/junior")


def test_http_content_extractor_returns_readable_page_text() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                html="<html><head><title>Acme</title></head><body><script>x</script><main>Trabajos en Argentina</main></body></html>",
            )
        )
    )

    content = HttpContentExtractor(http_client=client).extract("https://acme.com")

    assert content == ExtractedContent(
        url="https://acme.com",
        title="Acme",
        text="Trabajos en Argentina",
    )
