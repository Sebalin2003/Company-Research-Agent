from __future__ import annotations

import json
from typing import Any

from google import genai
from google.genai import types
from pydantic import ValidationError

from backend.app.domain.cv import CandidateSignals
from backend.app.domain.jobs import JobSignals, job_signal_payload
from backend.app.domain.reports import (
    AdaptedCVDraftSchema,
    CVTailoringSchema,
    CVTailoringSuggestionType,
    StructuredReportSchema,
    WarningSchema,
    WarningSeverity,
)
from backend.app.llm.gemini import extract_json_object, short_error
from backend.app.llm.synthesizer import SynthesisError


CV_TAILORING_SYSTEM_PROMPT = """\
Sos un asistente de empleabilidad para Argentina.
Devolve JSON valido compatible con el schema indicado.
No inventes experiencia, empleadores, fechas, herramientas, certificaciones, idiomas, metricas ni logros.
Cada sugerencia debe basarse en senales del CV, lineas seleccionadas del CV, requisitos seleccionados del puesto o evidencia del informe.
Un requisito del puesto que no aparece en el CV es una brecha a confirmar, no experiencia del candidato.
Si algo requiere datos que el CV no prueba, usa add_only_if_true y requires_user_confirmation=true.
El borrador adaptado no debe incluir sugerencias add_only_if_true ni hechos no verificados.
Todo texto visible debe estar en espanol.
"""


class GeminiCVTailoringService:
    def __init__(
        self,
        api_key: str,
        model: str,
        client: Any | None = None,
        max_output_tokens: int = 4000,
        timeout_ms: int = 60_000,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.client = client
        self.max_output_tokens = max_output_tokens
        self.timeout_ms = timeout_ms

    def build(
        self,
        company_name: str,
        signals: CandidateSignals,
        cv_evidence_lines: list[str],
        job_signals: JobSignals | None,
        job_evidence_lines: list[str],
        report: StructuredReportSchema,
        include_adapted_cv_draft: bool,
    ) -> CVTailoringSchema:
        if not self.api_key:
            raise SynthesisError("GEMINI_API_KEY is required.")

        client = self.client or genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=self.timeout_ms),
        )
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=build_cv_tailoring_prompt(
                    company_name,
                    signals,
                    cv_evidence_lines,
                    job_signals,
                    job_evidence_lines,
                    report,
                    include_adapted_cv_draft,
                ),
                config=types.GenerateContentConfig(
                    system_instruction=CV_TAILORING_SYSTEM_PROMPT,
                    temperature=0.2,
                    max_output_tokens=self.max_output_tokens,
                    response_mime_type="application/json",
                    response_schema=CVTailoringSchema,
                ),
            )
        except Exception as exc:
            raise SynthesisError("Gemini CV tailoring request failed.") from exc

        try:
            tailoring = parse_cv_tailoring_response(response)
        except SynthesisError as exc:
            raise SynthesisError(f"Gemini CV tailoring output was invalid: {short_error(exc)}") from exc
        return sanitize_cv_tailoring(tailoring, include_adapted_cv_draft)


def build_cv_tailoring_prompt(
    company_name: str,
    signals: CandidateSignals,
    cv_evidence_lines: list[str],
    job_signals: JobSignals | None,
    job_evidence_lines: list[str],
    report: StructuredReportSchema,
    include_adapted_cv_draft: bool,
) -> str:
    payload = {
        "company_name": company_name,
        "cv_signals": {
            "roles": signals.roles,
            "seniority": signals.seniority,
            "industries": signals.industries,
            "hard_skills": signals.hard_skills,
            "soft_skills": signals.soft_skills,
            "tools": signals.tools,
            "education": signals.education,
            "languages": signals.languages,
            "achievements": signals.achievements,
            "confidence": signals.confidence,
        },
        "selected_cv_lines": cv_evidence_lines,
        "job_signals": job_signal_payload(job_signals) if job_signals else None,
        "selected_job_lines": job_evidence_lines,
        "report_evidence": [
            {
                "id": item.id,
                "topic": item.topic.value,
                "claim": item.claim,
                "excerpt": item.raw_text_excerpt,
                "confidence": item.confidence.value,
            }
            for item in report.evidence[:10]
        ],
        "adapted_draft_requested": include_adapted_cv_draft,
    }
    return (
        "Genera sugerencias de adaptacion de CV seguras y accionables para la empresa y el puesto. "
        "Usa solo este contexto estructurado. "
        "Si no hay evidencia suficiente, se conservador y explica la limitacion en warnings. "
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def parse_cv_tailoring_response(response: Any) -> CVTailoringSchema:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, CVTailoringSchema):
        return parsed
    if isinstance(parsed, dict):
        raw = parsed
    else:
        text = getattr(response, "text", None)
        if not text:
            raise SynthesisError("Gemini response did not include text content.")
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            try:
                raw = json.loads(extract_json_object(text))
            except json.JSONDecodeError as exc:
                raise SynthesisError("Gemini response content was not valid JSON.") from exc
    try:
        return CVTailoringSchema.model_validate(raw)
    except ValidationError as exc:
        raise SynthesisError("Gemini CV tailoring response did not match schema.") from exc


def sanitize_cv_tailoring(
    tailoring: CVTailoringSchema,
    include_adapted_cv_draft: bool,
) -> CVTailoringSchema:
    suggestions = []
    add_only_ids = set()
    for suggestion in tailoring.change_suggestions:
        if suggestion.type == CVTailoringSuggestionType.add_only_if_true:
            add_only_ids.add(suggestion.id)
            suggestion = suggestion.model_copy(update={"requires_user_confirmation": True})
        suggestions.append(suggestion)

    draft = tailoring.adapted_cv_draft if include_adapted_cv_draft else None
    if draft:
        included = [
            suggestion_id
            for suggestion_id in draft.included_suggestion_ids
            if suggestion_id not in add_only_ids
        ]
        excluded = list(dict.fromkeys([*draft.excluded_suggestion_ids, *add_only_ids]))
        draft = AdaptedCVDraftSchema(
            title=draft.title,
            content_markdown=draft.content_markdown,
            included_suggestion_ids=included,
            excluded_suggestion_ids=excluded,
            warnings=draft.warnings,
        )

    return tailoring.model_copy(
        update={
            "change_suggestions": suggestions,
            "adapted_cv_draft": draft,
        }
    )


def fallback_warning(reason: str) -> WarningSchema:
    return WarningSchema(
        id="warning_ai_cv_tailoring_fallback",
        type="ai_cv_tailoring_fallback",
        message=(
            "No se pudo generar la adaptacion del CV con IA. "
            f"Se muestran sugerencias conservadoras. Motivo tecnico: {reason[:180]}"
        ),
        severity=WarningSeverity.medium,
    )
