from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from backend.app.domain.reports import (
    ClaimType,
    ConfidenceLevel,
    EvidenceTopic,
    ReportStatus,
    SectionType,
    SourceType,
)
from backend.app.llm.gemini import GeminiReportSynthesizer
from backend.app.llm.prompts import SYSTEM_PROMPT
from backend.app.llm.prompts import build_compact_report_payload
from backend.app.llm.prompts import build_report_user_prompt
from backend.app.llm.synthesizer import SynthesisError, SynthesisRequest
from backend.app.research.types import ClassifiedEvidence, ScoredSource


class RecordingModels:
    def __init__(self, response=None, responses=None, error: Exception | None = None) -> None:
        self.responses = list(responses) if responses is not None else None
        self.response = response
        self.error = error
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        if self.responses is not None:
            return self.responses.pop(0)
        return self.response


class FakeClient:
    def __init__(self, response=None, responses=None, error: Exception | None = None) -> None:
        self.models = RecordingModels(response=response, responses=responses, error=error)


def test_gemini_synthesizer_sends_structured_output_request() -> None:
    client = FakeClient(response=SimpleNamespace(text=json.dumps(valid_report_payload())))
    synthesizer = GeminiReportSynthesizer(
        api_key="test-key",
        model="gemini-test-model",
        client=client,
    )

    report = synthesizer.synthesize(synthesis_request())

    call = client.models.calls[0]
    assert call["model"] == "gemini-test-model"
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].max_output_tokens == 8000
    assert report.status == ReportStatus.completed
    assert report.sections[0].type == SectionType.executive_summary
    assert report.sections[0].claims[0].type == ClaimType.fact
    assert report.metadata.llm_provider == "gemini"
    assert report.metadata.llm_model == "gemini-test-model"


def test_gemini_synthesizer_accepts_fenced_json_content() -> None:
    client = FakeClient(
        response=SimpleNamespace(text=f"```json\n{json.dumps(valid_report_payload())}\n```")
    )
    synthesizer = GeminiReportSynthesizer(
        api_key="test-key",
        model="gemini-test-model",
        client=client,
    )

    report = synthesizer.synthesize(synthesis_request())

    assert report.report_id == "report_1"


def test_gemini_synthesizer_normalizes_missing_source_metadata() -> None:
    payload = valid_report_payload()
    del payload["sources"][0]["accessed_at"]
    payload["metadata"] = {}
    client = FakeClient(response=SimpleNamespace(text=json.dumps(payload)))
    synthesizer = GeminiReportSynthesizer(
        api_key="test-key",
        model="gemini-test-model",
        client=client,
    )

    report = synthesizer.synthesize(synthesis_request())

    assert report.sources[0].accessed_at is not None
    assert report.metadata.llm_provider == "gemini"
    assert report.metadata.source_count == 1


def test_gemini_synthesizer_raises_for_provider_error() -> None:
    client = FakeClient(error=RuntimeError("provider down"))
    synthesizer = GeminiReportSynthesizer(
        api_key="test-key",
        model="gemini-test-model",
        client=client,
    )

    with pytest.raises(SynthesisError):
        synthesizer.synthesize(synthesis_request())


def test_gemini_synthesizer_raises_for_invalid_json_content() -> None:
    client = FakeClient(responses=[SimpleNamespace(text="not json"), SimpleNamespace(text="still not json")])
    synthesizer = GeminiReportSynthesizer(
        api_key="test-key",
        model="gemini-test-model",
        client=client,
    )

    with pytest.raises(SynthesisError):
        synthesizer.synthesize(synthesis_request())


def test_compact_prompt_limits_evidence_per_topic_and_sources() -> None:
    request = large_synthesis_request()

    payload = build_compact_report_payload(request)

    assert len([item for item in payload["evidence"] if item["topic"] == "salary"]) == 3
    assert all(len(item["raw_text_excerpt"]) <= 450 for item in payload["evidence"])
    assert all("snippet" not in source for source in payload["sources"])
    selected_source_ids = {item["source_id"] for item in payload["evidence"]}
    assert {source["id"] for source in payload["sources"]} == selected_source_ids


def test_compact_prompt_limits_total_evidence() -> None:
    request = multi_topic_synthesis_request()

    payload = build_compact_report_payload(request)

    assert len(payload["evidence"]) == 18
    assert len([item for item in payload["evidence"] if item["topic"] == "general"]) <= 1


def test_report_prompt_requires_exact_details_or_unavailable_language() -> None:
    prompt = build_report_user_prompt(synthesis_request())

    assert "valores exactos" in prompt
    assert "No disponible en las fuentes consultadas" in prompt
    assert "IT trainee, pasantia/internship y junior" in prompt
    assert "no infieras rangos salariales" in SYSTEM_PROMPT
    assert "trainee, pasantia/internship y junior" in SYSTEM_PROMPT


def test_gemini_synthesizer_repairs_invalid_first_response() -> None:
    client = FakeClient(
        responses=[
            SimpleNamespace(text=json.dumps({"sections": [{"type": "not_valid"}]})),
            SimpleNamespace(text=json.dumps(valid_report_payload())),
        ]
    )
    synthesizer = GeminiReportSynthesizer(
        api_key="test-key",
        model="gemini-test-model",
        client=client,
    )

    report = synthesizer.synthesize(synthesis_request())

    assert report.status == ReportStatus.completed
    assert len(client.models.calls) == 2
    assert "respuesta anterior no cumplio" in client.models.calls[1]["contents"].lower()


def test_gemini_synthesizer_raises_diagnostic_after_failed_repair() -> None:
    client = FakeClient(
        responses=[
            SimpleNamespace(text=json.dumps({"sections": [{"type": "not_valid"}]})),
            SimpleNamespace(text=json.dumps({"sections": [{"type": "still_not_valid"}]})),
        ]
    )
    synthesizer = GeminiReportSynthesizer(
        api_key="test-key",
        model="gemini-test-model",
        client=client,
    )

    with pytest.raises(SynthesisError, match="after repair"):
        synthesizer.synthesize(synthesis_request())


def test_gemini_synthesizer_raises_for_missing_api_key() -> None:
    synthesizer = GeminiReportSynthesizer(
        api_key="",
        model="gemini-test-model",
        client=FakeClient(response=SimpleNamespace(text=json.dumps(valid_report_payload()))),
    )

    with pytest.raises(SynthesisError):
        synthesizer.synthesize(synthesis_request())


def synthesis_request() -> SynthesisRequest:
    source = ScoredSource(
        source_id="source_1",
        title="Acme",
        url="https://acme.com",
        domain="acme.com",
        source_type=SourceType.official,
        reliability_score=5,
        snippet="Acme fabrica productos industriales.",
        is_current=True,
    )
    evidence = ClassifiedEvidence(
        source_id="source_1",
        topic=EvidenceTopic.business,
        claim="Acme fabrica productos industriales.",
        raw_text_excerpt="Acme fabrica productos industriales.",
        confidence=ConfidenceLevel.medium,
    )
    return SynthesisRequest(
        report_id="report_1",
        company_id="company_1",
        company_name="Acme",
        normalized_company_name="acme",
        sources=[source],
        evidence=[evidence],
    )


def large_synthesis_request() -> SynthesisRequest:
    sources = [
        ScoredSource(
            source_id=f"source_{index}",
            title=f"Acme Salary {index}",
            url=f"https://glassdoor.com/acme/{index}",
            domain="glassdoor.com",
            source_type=SourceType.salary_review_platform,
            reliability_score=3,
            snippet="Sueldos y salarios en Argentina.",
            is_current=True,
        )
        for index in range(6)
    ]
    sources.append(
        ScoredSource(
            source_id="source_unused",
            title="Unused",
            url="https://unused.example.com",
            domain="unused.example.com",
            source_type=SourceType.secondary,
            reliability_score=1,
            snippet="No debe incluirse.",
            is_current=True,
        )
    )
    evidence = [
        ClassifiedEvidence(
            source_id=f"source_{index}",
            topic=EvidenceTopic.salary,
            claim=f"Acme tiene senales salariales {index}.",
            raw_text_excerpt="x" * 700,
            confidence=ConfidenceLevel.medium,
        )
        for index in range(6)
    ]
    return SynthesisRequest(
        report_id="report_large",
        company_id="company_1",
        company_name="Acme",
        normalized_company_name="acme",
        sources=sources,
        evidence=evidence,
    )


def multi_topic_synthesis_request() -> SynthesisRequest:
    sources = []
    evidence = []
    index = 0
    for topic in EvidenceTopic:
        for _ in range(4):
            sources.append(
                ScoredSource(
                    source_id=f"source_{index}",
                    title=f"Acme {topic.value} {index}",
                    url=f"https://example.com/{topic.value}/{index}",
                    domain="example.com",
                    source_type=SourceType.official,
                    reliability_score=4,
                    snippet=f"Snippet {topic.value}",
                    is_current=True,
                )
            )
            evidence.append(
                ClassifiedEvidence(
                    source_id=f"source_{index}",
                    topic=topic,
                    claim=f"Claim {topic.value} {index}.",
                    raw_text_excerpt=f"Evidence {topic.value} {index}.",
                    confidence=ConfidenceLevel.medium,
                )
            )
            index += 1
    return SynthesisRequest(
        report_id="report_multi",
        company_id="company_1",
        company_name="Acme",
        normalized_company_name="acme",
        sources=sources,
        evidence=evidence,
    )


def valid_report_payload() -> dict:
    return {
        "schema_version": "1.0",
        "report_id": "report_1",
        "company": {
            "id": "company_1",
            "name": "Acme",
            "normalized_name": "acme",
            "possible_aliases": [],
            "ambiguity_warning": None,
        },
        "status": "completed",
        "language": "es-AR",
        "generated_at": "2026-07-25T00:00:00Z",
        "valid_until": None,
        "sections": [
            {
                "type": "executive_summary",
                "title": "Resumen ejecutivo",
                "summary": "Acme fabrica productos industriales.",
                "claims": [
                    {
                        "id": "claim_1",
                        "type": "fact",
                        "text": "Acme fabrica productos industriales.",
                        "evidence_ids": ["evidence_1"],
                        "confidence": "medium",
                    }
                ],
                "confidence": "medium",
                "missing_evidence": False,
            }
        ],
        "personalized_preparation": None,
        "cv_tailoring": None,
        "sources": [
            {
                "id": "source_1",
                "title": "Acme",
                "url": "https://acme.com",
                "domain": "acme.com",
                "source_type": "official",
                "reliability_score": 5,
                "accessed_at": "2026-07-25T00:00:00Z",
                "published_at": None,
                "snippet": "Acme fabrica productos industriales.",
                "language": "es",
                "is_current": True,
            }
        ],
        "evidence": [
            {
                "id": "evidence_1",
                "source_id": "source_1",
                "topic": "business",
                "claim": "Acme fabrica productos industriales.",
                "raw_text_excerpt": "Acme fabrica productos industriales.",
                "confidence": "medium",
            }
        ],
        "warnings": [],
        "metadata": {
            "search_provider": "mock",
            "llm_provider": "gemini",
            "llm_model": "gemini-test-model",
            "source_count": 1,
            "evidence_count": 1,
            "used_cv": False,
            "used_cv_tailoring": False,
            "generation_duration_ms": 10,
        },
    }
