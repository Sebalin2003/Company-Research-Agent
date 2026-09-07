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
from backend.app.services.report_quality import report_completion_issues, validate_report_grounding


def build_report(summary: str, evidence_ids: list[str]) -> StructuredReportSchema:
    return StructuredReportSchema(
        report_id="report_1",
        company=CompanySchema(id="company_1", name="Acme", normalized_name="acme"),
        status=ReportStatus.completed,
        sections=[
            SectionSchema(
                type=SectionType.salary_benefits,
                title="Sueldos y beneficios",
                summary=summary,
                claims=[
                    ClaimSchema(
                        id="claim_1",
                        type=ClaimType.fact,
                        text=summary,
                        evidence_ids=evidence_ids,
                        confidence=ConfidenceLevel.high,
                    )
                ],
                confidence=ConfidenceLevel.high,
            )
        ],
        sources=[
            SourceSchema(
                id="source_1",
                title="Acme salarios",
                url="https://example.com/salarios",
                domain="example.com",
                source_type=SourceType.salary_review_platform,
                reliability_score=3,
                accessed_at="2026-08-07T00:00:00+00:00",
            )
        ],
        evidence=[
            EvidenceSchema(
                id="evidence_1",
                source_id="source_1",
                topic=EvidenceTopic.salary,
                claim="Rango mensual para junior.",
                raw_text_excerpt="Junior: ARS 900.000 - ARS 1.200.000 por mes.",
                confidence=ConfidenceLevel.medium,
            )
        ],
        metadata=ReportMetadataSchema(search_provider="tavily", llm_model="gemini-test"),
    )


def build_completion_ready_report() -> StructuredReportSchema:
    sources = [
        SourceSchema(
            id="source_official",
            title="Acme Argentina",
            url="https://acme.example/argentina",
            domain="acme.example",
            source_type=SourceType.official,
            reliability_score=5,
            accessed_at="2026-08-28T00:00:00+00:00",
        ),
        SourceSchema(
            id="source_jobs",
            title="Acme jobs in Argentina",
            url="https://linkedin.com/jobs/acme-argentina",
            domain="linkedin.com",
            source_type=SourceType.linkedin,
            reliability_score=4,
            accessed_at="2026-08-28T00:00:00+00:00",
        ),
    ]
    evidence = [
        EvidenceSchema(
            id="evidence_business",
            source_id="source_official",
            topic=EvidenceTopic.business,
            claim="Acme desarrolla software.",
            raw_text_excerpt="Acme desarrolla software en Argentina.",
            confidence=ConfidenceLevel.high,
        ),
        EvidenceSchema(
            id="evidence_presence",
            source_id="source_official",
            topic=EvidenceTopic.argentina_presence,
            claim="Acme opera en Buenos Aires.",
            raw_text_excerpt="Acme opera en Buenos Aires, Argentina.",
            confidence=ConfidenceLevel.high,
        ),
        EvidenceSchema(
            id="evidence_roles",
            source_id="source_jobs",
            topic=EvidenceTopic.open_roles,
            claim="Hay 2 vacantes activas.",
            raw_text_excerpt="2 vacantes activas de Acme en Argentina.",
            confidence=ConfidenceLevel.high,
        ),
    ]

    def section(section_type, title, summary, evidence_id):
        return SectionSchema(
            type=section_type,
            title=title,
            summary=summary,
            claims=[
                ClaimSchema(
                    id=f"claim_{section_type.value}",
                    type=ClaimType.fact,
                    text=summary,
                    evidence_ids=[evidence_id],
                    confidence=ConfidenceLevel.high,
                )
            ],
            confidence=ConfidenceLevel.high,
        )

    return StructuredReportSchema(
        report_id="report_ready",
        company=CompanySchema(id="company_1", name="Acme", normalized_name="acme"),
        status=ReportStatus.completed,
        sections=[
            section(SectionType.executive_summary, "Resumen", "Acme desarrolla software.", "evidence_business"),
            section(SectionType.business, "Negocio", "Acme desarrolla software.", "evidence_business"),
            section(SectionType.argentina_presence, "Argentina", "Acme opera en Buenos Aires.", "evidence_presence"),
            section(SectionType.open_roles, "Vacantes", "Hay 2 vacantes activas.", "evidence_roles"),
        ],
        sources=sources,
        evidence=evidence,
        metadata=ReportMetadataSchema(search_provider="tavily", llm_model="deepseek-test"),
    )


def test_invalid_evidence_and_unsupported_numbers_reduce_confidence() -> None:
    validated = validate_report_grounding(
        build_report("El rango es ARS 2.000.000 por mes.", ["missing_evidence"])
    )

    section = validated.sections[0]
    assert section.confidence == ConfidenceLevel.low
    assert section.missing_evidence is True
    assert section.claims[0].confidence == ConfidenceLevel.low
    assert section.claims[0].evidence_ids == []
    assert any(warning.type == "limited_section_grounding" for warning in validated.warnings)


def test_malformed_salary_range_is_kept_but_marked_low_confidence() -> None:
    validated = validate_report_grounding(
        build_report("Rango salarial: $1 M - $4.", ["evidence_1"])
    )

    assert validated.sections[0].summary == "Rango salarial: $1 M - $4."
    assert validated.sections[0].confidence == ConfidenceLevel.low
    assert validated.sections[0].claims[0].confidence == ConfidenceLevel.low
    assert validated.sections[0].missing_evidence is True


def test_missing_evidence_caps_section_confidence() -> None:
    report = build_report(
        "Junior: ARS 900.000 - ARS 1.200.000 por mes.",
        ["evidence_1"],
    )
    report.sections[0].missing_evidence = True

    validated = validate_report_grounding(report)

    assert validated.sections[0].confidence == ConfidenceLevel.low


def test_single_domain_adds_source_diversity_warning() -> None:
    validated = validate_report_grounding(
        build_report(
            "Junior: ARS 900.000 - ARS 1.200.000 por mes.",
            ["evidence_1"],
        )
    )

    assert any(warning.type == "limited_source_diversity" for warning in validated.warnings)


def test_report_completion_requires_verified_core_sections_and_sources() -> None:
    assert report_completion_issues(build_completion_ready_report()) == []

    issues = report_completion_issues(build_report("Sin datos", []))

    assert any("fuente oficial" in issue for issue in issues)
    assert any("vacantes actuales en Argentina" in issue for issue in issues)


def test_report_completion_allows_only_exhausted_open_roles_gap() -> None:
    report = build_completion_ready_report()
    report.sections = [
        section for section in report.sections if section.type != SectionType.open_roles
    ]
    report.evidence = [
        evidence for evidence in report.evidence if evidence.topic != EvidenceTopic.open_roles
    ]
    report.sources[1] = report.sources[1].model_copy(
        update={"source_type": SourceType.secondary}
    )

    assert any(
        "vacantes actuales en Argentina" in issue
        for issue in report_completion_issues(report)
    )
    assert report_completion_issues(report, allow_unverified_open_roles=True) == []


def test_conflicting_open_role_counts_reduce_report_confidence() -> None:
    report = build_completion_ready_report()
    source = next(item for item in report.sources if item.id == "source_jobs")
    source.title = "(52 vacantes) empleos de Acme en Argentina"
    role_evidence = next(item for item in report.evidence if item.id == "evidence_roles")
    role_evidence.raw_text_excerpt = "702 empleos de Acme en Argentina."
    role_section = next(item for item in report.sections if item.type == SectionType.open_roles)
    role_section.summary = "Hay 702 vacantes activas."
    role_section.claims[0].text = "Hay 702 vacantes activas."

    validated = validate_report_grounding(report)

    section = next(item for item in validated.sections if item.type == SectionType.open_roles)
    assert section.missing_evidence is True
    assert section.confidence == ConfidenceLevel.low
