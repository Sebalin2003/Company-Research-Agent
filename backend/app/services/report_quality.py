from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

from backend.app.domain.reports import (
    ClaimType,
    ConfidenceLevel,
    EvidenceTopic,
    SectionType,
    StructuredReportSchema,
    WarningSchema,
    WarningSeverity,
)

SECTION_TOPICS = {
    SectionType.business: {EvidenceTopic.business, EvidenceTopic.general},
    SectionType.argentina_presence: {
        EvidenceTopic.argentina_presence,
        EvidenceTopic.general,
    },
    SectionType.employees: {EvidenceTopic.employees, EvidenceTopic.general},
    SectionType.salary_benefits: {
        EvidenceTopic.salary,
        EvidenceTopic.benefits,
        EvidenceTopic.general,
    },
    SectionType.culture: {EvidenceTopic.culture, EvidenceTopic.general},
    SectionType.interview_process: {
        EvidenceTopic.interview_process,
        EvidenceTopic.general,
    },
    SectionType.interview_questions: {
        EvidenceTopic.interview_questions,
        EvidenceTopic.general,
    },
    SectionType.open_roles: {EvidenceTopic.open_roles, EvidenceTopic.general},
}
NUMBER_RE = re.compile(r"\d[\d.,]*")
SALARY_RANGE_RE = re.compile(
    r"(?:\$|ars|usd)?\s*(\d[\d.,]*)\s*([km])?\s*[-–]\s*"
    r"(?:\$|ars|usd)?\s*(\d[\d.,]*)\s*([km])?",
    re.IGNORECASE,
)
MAX_SUPPORTING_QUOTE_CHARACTERS = 450
QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)
logger = logging.getLogger(__name__)


def validate_and_strip_supporting_quotes(raw_report: dict[str, Any]) -> dict[str, Any]:
    evidence_by_id = {
        item.get("id"): item
        for item in raw_report.get("evidence", [])
        if isinstance(item, dict) and item.get("id")
    }
    warnings = raw_report.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
        raw_report["warnings"] = warnings

    for section in raw_report.get("sections", []):
        if not isinstance(section, dict):
            continue
        section_type = section.get("type")
        allowed_topics = allowed_topics_for(section_type)
        has_factual_claim = False
        verified_factual_claims = 0
        section_failed = False

        for claim in section.get("claims", []):
            if not isinstance(claim, dict):
                continue
            quote = claim.pop("supporting_quote", None)
            if claim.get("type") not in {ClaimType.fact.value, ClaimType.inference.value}:
                continue
            has_factual_claim = True
            failure = supporting_quote_failure(
                claim=claim,
                quote=quote,
                evidence_by_id=evidence_by_id,
                allowed_topics=allowed_topics,
            )
            if failure:
                section_failed = True
                claim["confidence"] = ConfidenceLevel.low.value
                logger.warning(
                    "Supporting quote validation failed",
                    extra={
                        "report_id": raw_report.get("report_id"),
                        "section_type": section_type,
                        "claim_id": claim.get("id"),
                        "validation_status": "failed",
                        "failure_reason": failure,
                    },
                )
            else:
                verified_factual_claims += 1

        if has_factual_claim and verified_factual_claims == 0:
            section_failed = True
        if section_failed:
            section["confidence"] = ConfidenceLevel.low.value
            section["missing_evidence"] = True
            warnings.append(
                {
                    "id": f"warning_supporting_quote_{section_type}",
                    "type": "supporting_quote_validation_failed",
                    "message": (
                        f"{section.get('title', 'Seccion')}: una afirmacion no pudo "
                        "vincularse con un fragmento exacto de la evidencia citada."
                    ),
                    "severity": WarningSeverity.medium.value,
                    "related_section": section_type,
                }
            )
    return raw_report


def supporting_quote_failure(
    claim: dict[str, Any],
    quote: Any,
    evidence_by_id: dict[str, dict[str, Any]],
    allowed_topics: set[EvidenceTopic] | None,
) -> str | None:
    if not isinstance(quote, str) or not quote.strip():
        return "missing_quote"
    if len(quote) > MAX_SUPPORTING_QUOTE_CHARACTERS:
        return "quote_too_long"

    cited_evidence = []
    for evidence_id in claim.get("evidence_ids", []):
        evidence = evidence_by_id.get(evidence_id)
        if not evidence:
            continue
        try:
            topic = EvidenceTopic(evidence.get("topic"))
        except ValueError:
            continue
        if allowed_topics is None or topic in allowed_topics:
            cited_evidence.append(evidence)
    if not cited_evidence:
        return "invalid_evidence"

    normalized_quote = normalize_quote_text(quote)
    matching_evidence = next(
        (
            evidence
            for evidence in cited_evidence
            if normalized_quote
            and normalized_quote
            in normalize_quote_text(evidence.get("raw_text_excerpt") or "")
        ),
        None,
    )
    if not matching_evidence:
        return "quote_not_found"

    claim_numbers = canonical_numbers(str(claim.get("text") or ""))
    if not claim_numbers.issubset(canonical_numbers(quote)):
        return "unsupported_numbers"
    if not claim_numbers.issubset(
        canonical_numbers(str(matching_evidence.get("raw_text_excerpt") or ""))
    ):
        return "unsupported_numbers"
    return None


def allowed_topics_for(section_type: Any) -> set[EvidenceTopic] | None:
    try:
        return SECTION_TOPICS.get(SectionType(section_type))
    except ValueError:
        return set()


def normalize_quote_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).translate(QUOTE_TRANSLATION)
    return " ".join(normalized.split())


def validate_report_grounding(report: StructuredReportSchema) -> StructuredReportSchema:
    evidence_by_id = {item.id: item for item in report.evidence}
    warnings = list(report.warnings)
    sections = []

    for section in report.sections:
        section_topics = SECTION_TOPICS.get(section.type)
        claims = []
        cited_evidence = []
        section_is_weak = section.missing_evidence
        reasons = []

        for claim in section.claims:
            valid_ids = [
                evidence_id
                for evidence_id in claim.evidence_ids
                if evidence_id in evidence_by_id
                and (
                    section_topics is None
                    or evidence_by_id[evidence_id].topic in section_topics
                )
            ]
            claim_evidence = [evidence_by_id[evidence_id] for evidence_id in valid_ids]
            cited_evidence.extend(claim_evidence)
            claim_is_weak = (
                claim.type in {ClaimType.fact, ClaimType.inference}
                and (
                    not valid_ids
                    or has_unsupported_numbers(claim.text, claim_evidence)
                )
            )
            if claim_is_weak:
                section_is_weak = True
                reasons.append("afirmaciones sin respaldo suficiente")
            claims.append(
                claim.model_copy(
                    update={
                        "evidence_ids": valid_ids,
                        "confidence": (
                            ConfidenceLevel.low
                            if claim_is_weak or section.missing_evidence
                            else claim.confidence
                        ),
                    }
                )
            )

        if has_unsupported_numbers(section.summary, cited_evidence):
            section_is_weak = True
            reasons.append("valores numericos no respaldados")
        if section.type == SectionType.salary_benefits and has_malformed_salary_range(
            section.summary
        ):
            section_is_weak = True
            reasons.append("rango salarial incompleto")

        if section_is_weak and reasons:
            warnings.append(
                WarningSchema(
                    id=f"warning_grounding_{section.type.value}",
                    type="limited_section_grounding",
                    message=(
                        f"{section.title}: confianza reducida por "
                        f"{', '.join(dict.fromkeys(reasons))}."
                    ),
                    severity=WarningSeverity.medium,
                    related_section=section.type,
                )
            )

        sections.append(
            section.model_copy(
                update={
                    "claims": claims,
                    "confidence": (
                        ConfidenceLevel.low if section_is_weak else section.confidence
                    ),
                    "missing_evidence": section_is_weak,
                }
            )
        )

    domains = {source.domain.lower() for source in report.sources if source.domain}
    if report.sources and len(domains) == 1:
        warnings.append(
            WarningSchema(
                id="warning_limited_source_diversity",
                type="limited_source_diversity",
                message="El informe depende de un unico dominio de origen.",
                severity=WarningSeverity.low,
                related_section=SectionType.sources,
            )
        )

    return report.model_copy(
        update={
            "sections": sections,
            "warnings": unique_warnings(warnings),
        }
    )


def has_unsupported_numbers(text: str, evidence) -> bool:
    numbers = canonical_numbers(text)
    if not numbers:
        return False
    evidence_text = " ".join(
        f"{item.claim} {item.raw_text_excerpt or ''}" for item in evidence
    )
    return not numbers.issubset(canonical_numbers(evidence_text))


def canonical_numbers(text: str) -> set[str]:
    return {
        re.sub(r"[.,]", "", match.group(0))
        for match in NUMBER_RE.finditer(text)
    }


def has_malformed_salary_range(text: str) -> bool:
    for match in SALARY_RANGE_RE.finditer(text):
        first_suffix, second_suffix = match.group(2), match.group(4)
        if bool(first_suffix) != bool(second_suffix):
            return True
    return False


def unique_warnings(warnings: list[WarningSchema]) -> list[WarningSchema]:
    return list({warning.id: warning for warning in warnings}.values())
