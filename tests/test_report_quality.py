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
from backend.app.services.report_quality import validate_report_grounding


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
