from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from backend.app.domain.cv import extract_candidate_signals, select_cv_evidence_lines
from backend.app.domain.jobs import extract_job_signals, select_job_evidence_lines
from backend.app.domain.reports import (
    CompanySchema,
    ConfidenceLevel,
    CVTailoringSuggestionType,
    EvidenceSchema,
    EvidenceTopic,
    ReportMetadataSchema,
    ReportStatus,
    SourceSchema,
    SourceType,
    StructuredReportSchema,
)
from backend.app.llm.cv_tailoring import (
    DeepSeekCVTailoringService as GeminiCVTailoringService,
    build_cv_tailoring_prompt,
)
from backend.app.llm.synthesizer import SynthesisError


class FakeClient:
    def __init__(self, response=None) -> None:
        self.response = response
        self.calls = []

    def generate_json(self, system_prompt, contents, schema, **kwargs):
        self.calls.append(
            {"system_prompt": system_prompt, "contents": contents, "schema": schema, **kwargs}
        )
        text = getattr(self.response, "text", None)
        if text is not None:
            return json.loads(text)
        return self.response


def test_gemini_cv_tailoring_returns_structured_suggestions() -> None:
    client = FakeClient(response=SimpleNamespace(text=json.dumps(valid_tailoring_payload())))
    service = GeminiCVTailoringService(
        api_key="test-key",
        model="gemini-test-model",
        client=client,
    )

    tailoring = service.build(
        company_name="Acme",
        signals=extract_candidate_signals("Desarrollador con Python y SQL. Optimice APIs."),
        cv_evidence_lines=["Desarrollador con Python y SQL.", "Optimice APIs."],
        job_signals=None,
        job_evidence_lines=[],
        report=sample_report(),
        include_adapted_cv_draft=True,
    )

    call = client.calls[0]
    assert call["max_tokens"] == 4000
    assert "change_suggestions" in call["schema"].model_json_schema()["properties"]
    assert tailoring.change_suggestions[0].type == CVTailoringSuggestionType.emphasize
    assert tailoring.change_suggestions[1].requires_user_confirmation is True
    assert "suggestion_add_metric" not in tailoring.adapted_cv_draft.included_suggestion_ids
    assert "suggestion_add_metric" in tailoring.adapted_cv_draft.excluded_suggestion_ids


def test_gemini_cv_tailoring_rejects_invalid_output() -> None:
    service = GeminiCVTailoringService(
        api_key="test-key",
        model="gemini-test-model",
        client=FakeClient(response=SimpleNamespace(text="not json")),
    )

    with pytest.raises(SynthesisError):
        service.build(
            company_name="Acme",
            signals=extract_candidate_signals("Python y SQL."),
            cv_evidence_lines=["Python y SQL."],
            job_signals=None,
            job_evidence_lines=[],
            report=sample_report(),
            include_adapted_cv_draft=False,
        )


def test_cv_tailoring_prompt_uses_selected_cv_lines_not_full_cv_text() -> None:
    cv_text = """
    Desarrollador con Python y SQL.
    Experiencia privada que no debe enviarse completa.
    Optimice APIs internas.
    """
    signals = extract_candidate_signals(cv_text)
    selected_lines = select_cv_evidence_lines(cv_text, signals)

    prompt = build_cv_tailoring_prompt(
        company_name="Acme",
        signals=signals,
        cv_evidence_lines=selected_lines,
        job_signals=None,
        job_evidence_lines=[],
        report=sample_report(),
        include_adapted_cv_draft=False,
    )

    assert "Desarrollador con Python y SQL" in prompt
    assert "Optimice APIs internas" in prompt
    assert "Experiencia privada que no debe enviarse completa" not in prompt


def test_cv_tailoring_prompt_uses_selected_job_lines_not_full_description() -> None:
    job_description = """
    Desarrollador backend.
    Requisito: Python y SQL.
    Informacion interna que no debe enviarse completa.
    Responsabilidades: optimizar APIs.
    """
    job_signals = extract_job_signals(job_description)
    selected_lines = select_job_evidence_lines(job_description, job_signals)

    prompt = build_cv_tailoring_prompt(
        company_name="Acme",
        signals=extract_candidate_signals("Desarrollador con Python."),
        cv_evidence_lines=["Desarrollador con Python."],
        job_signals=job_signals,
        job_evidence_lines=selected_lines,
        report=sample_report(),
        include_adapted_cv_draft=False,
    )

    assert "Requisito: Python y SQL" in prompt
    assert "Responsabilidades: optimizar APIs" in prompt
    assert "Informacion interna que no debe enviarse completa" not in prompt


def sample_report() -> StructuredReportSchema:
    return StructuredReportSchema(
        report_id="report_1",
        company=CompanySchema(id="company_1", name="Acme", normalized_name="acme"),
        status=ReportStatus.completed,
        sections=[],
        sources=[
            SourceSchema(
                id="source_1",
                title="Acme Careers",
                url="https://careers.acme.com",
                domain="careers.acme.com",
                source_type=SourceType.career_page,
                reliability_score=5,
                accessed_at="2026-08-06T00:00:00+00:00",
            )
        ],
        evidence=[
            EvidenceSchema(
                id="evidence_1",
                source_id="source_1",
                topic=EvidenceTopic.open_roles,
                claim="Acme publica roles backend.",
                raw_text_excerpt="Roles backend con Python.",
                confidence=ConfidenceLevel.medium,
            )
        ],
        metadata=ReportMetadataSchema(search_provider="tavily", llm_model="gemini-test-model"),
    )


def valid_tailoring_payload() -> dict:
    return {
        "positioning_summary": "Prioriza backend con Python y SQL.",
        "change_suggestions": [
            {
                "id": "suggestion_emphasize_backend",
                "type": "emphasize",
                "original_text": None,
                "suggested_text": "Enfatiza Python y SQL en proyectos backend.",
                "reason": "El CV menciona Python y SQL y el informe muestra roles backend.",
                "evidence_ids": ["evidence_1"],
                "requires_user_confirmation": False,
                "confidence": "medium",
            },
            {
                "id": "suggestion_add_metric",
                "type": "add_only_if_true",
                "original_text": None,
                "suggested_text": "Agrega metricas de impacto solo si son reales.",
                "reason": "El CV no prueba metricas.",
                "evidence_ids": [],
                "requires_user_confirmation": False,
                "confidence": "low",
            },
        ],
        "adapted_cv_draft": {
            "title": "CV adaptado para Acme",
            "content_markdown": "# Perfil\nDesarrollador con Python y SQL.",
            "included_suggestion_ids": ["suggestion_emphasize_backend", "suggestion_add_metric"],
            "excluded_suggestion_ids": [],
            "warnings": [],
        },
        "warnings": [],
    }
