from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from backend.app.domain.reports import (
    ClaimType,
    ConfidenceLevel,
    EvidenceTopic,
    ReportStatus,
    SectionType,
    SourceType,
    StructuredReportSchema,
)
from backend.app.llm.deepseek import (
    DeepSeekAPIError,
    DeepSeekReportSynthesizer as GeminiReportSynthesizer,
    extract_json_object,
)
from backend.app.llm.prompts import SYSTEM_PROMPT
from backend.app.llm.prompts import build_compact_report_payload
from backend.app.llm.prompts import build_report_user_prompt
from backend.app.llm.synthesizer import SynthesisError, SynthesisRequest
from backend.app.research.types import ClassifiedEvidence, ScoredSource


class FakeClient:
    def __init__(self, response=None, responses=None, error: Exception | None = None) -> None:
        self.responses = list(responses) if responses is not None else None
        self.response = response
        self.error = error
        self.calls = []
        self.models = self

    def generate_json(self, system_prompt, contents, schema, **kwargs):
        self.calls.append(
            {"system_prompt": system_prompt, "contents": contents, "schema": schema, **kwargs}
        )
        if self.error:
            raise self.error
        if self.responses is not None:
            response = self.responses.pop(0)
        else:
            response = self.response
        if isinstance(response, dict):
            return response
        text = getattr(response, "text", None)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return json.loads(extract_json_object(text))
            except (json.JSONDecodeError, SynthesisError) as exc:
                raise DeepSeekAPIError("DeepSeek returned invalid JSON.", raw_content=text) from exc


def test_deepseek_synthesizer_sends_structured_output_request() -> None:
    client = FakeClient(response=valid_report_payload())
    synthesizer = GeminiReportSynthesizer(
        api_key="test-key",
        model="deepseek-test-model",
        client=client,
    )

    report = synthesizer.synthesize(synthesis_request())

    call = client.calls[0]
    assert call["max_tokens"] == 8000
    assert "supporting_quote" in json.dumps(call["schema"].model_json_schema())
    assert report.status == ReportStatus.completed
    assert report.sections[0].type == SectionType.executive_summary
    assert report.sections[0].claims[0].type == ClaimType.fact
    assert report.metadata.llm_provider == "deepseek"
    assert report.metadata.llm_model == "deepseek-test-model"
    assert "supporting_quote" not in report.model_dump_json()
    assert "supporting_quote" not in json.dumps(StructuredReportSchema.model_json_schema())


def test_generation_contract_omits_backend_owned_evidence_and_metadata() -> None:
    payload = valid_report_payload()
    payload = {key: payload[key] for key in ("sections", "warnings")}
    client = FakeClient(response=payload)
    report = GeminiReportSynthesizer(
        api_key="test", model="deepseek-test", client=client,
    ).synthesize(synthesis_request())
    properties = client.calls[0]["schema"].model_json_schema()["properties"]
    assert not {"sources", "evidence", "metadata", "company"} & properties.keys()
    assert report.evidence[0].id == "evidence_1"
    assert report.sources[0].id == "source_1"
    mapping = build_compact_report_payload(synthesis_request())["allowed_evidence_ids_by_section"]
    assert mapping["business"] == ["evidence_1"]
    assert mapping["argentina_presence"] == []
    definitions = client.calls[0]["schema"].model_json_schema()["$defs"]
    allowed = definitions["Claim_business"]["properties"]["evidence_ids"]["items"]
    assert allowed.get("const") == "evidence_1"
    assert definitions["Claim_argentina_presence"]["properties"]["evidence_ids"]["items"]["type"] == "null"


def test_gemini_synthesizer_accepts_normalized_supporting_quote() -> None:
    payload = valid_report_payload()
    payload["sections"][0]["claims"][0]["supporting_quote"] = (
        "Acme\u00a0 fabrica   productos industriales."
    )
    client = FakeClient(response=SimpleNamespace(text=json.dumps(payload)))

    report = GeminiReportSynthesizer(
        api_key="test-key", model="gemini-test-model", client=client
    ).synthesize(synthesis_request())

    assert report.sections[0].confidence == ConfidenceLevel.medium
    assert report.sections[0].missing_evidence is False


def test_quote_id_resolves_original_text_without_mutating_model_response() -> None:
    payload = valid_report_payload()
    claim = payload["sections"][0]["claims"][0]
    claim.pop("supporting_quote")
    claim["supporting_quote_id"] = "evidence_1"
    payload["evidence"][0]["raw_text_excerpt"] = "Alterado por el modelo."
    client = FakeClient(response=payload)
    report = GeminiReportSynthesizer(
        api_key="test-key", model="deepseek-test", client=client,
    ).synthesize(synthesis_request())
    assert not report.sections[0].missing_evidence
    assert report.evidence[0].raw_text_excerpt == "Acme fabrica productos industriales."
    assert "supporting_quote" not in report.model_dump_json()
    assert "supporting_quote" not in claim
    assert len(client.calls) == 1


@pytest.mark.parametrize("failure", ["unknown", "uncited", "topic", "numbers"])
def test_quote_id_failure_repairs_with_authorized_context(failure: str) -> None:
    payload = valid_report_payload()
    claim = payload["sections"][0]["claims"][0]
    claim["supporting_quote_id"] = "evidence_1"
    if failure == "unknown":
        claim["supporting_quote_id"] = "invented"
    elif failure == "uncited":
        claim["evidence_ids"] = []
    elif failure == "topic":
        payload["sections"][0]["type"] = "argentina_presence"
    else:
        claim["text"] = "Acme fabrica 999 productos."
    repaired = valid_report_payload()
    repaired["sections"][0]["claims"][0].pop("supporting_quote")
    repaired["sections"][0]["claims"][0]["supporting_quote_id"] = "evidence_1"
    client = FakeClient(responses=[payload, repaired])
    report = GeminiReportSynthesizer(
        api_key="test-key", model="deepseek-test", client=client,
    ).synthesize(synthesis_request())
    assert len(client.calls) == 2
    repair_prompt = client.calls[1]["contents"]
    assert "Acme fabrica productos industriales." in repair_prompt
    assert "allowed_quote_ids" in repair_prompt
    assert "claim_1" in repair_prompt
    assert not report.sections[0].missing_evidence


@pytest.mark.parametrize(
    "supporting_quote",
    [None, "Acme presta servicios financieros.", "x" * 451],
)
def test_invalid_required_supporting_quote_uses_existing_repair(supporting_quote) -> None:
    payload = valid_report_payload()
    claim = payload["sections"][0]["claims"][0]
    if supporting_quote is None:
        claim.pop("supporting_quote")
    else:
        claim["supporting_quote"] = supporting_quote
    client = FakeClient(
        responses=[
            SimpleNamespace(text=json.dumps(payload)),
            SimpleNamespace(text=json.dumps(valid_report_payload())),
        ]
    )

    report = GeminiReportSynthesizer(
        api_key="test-key", model="gemini-test-model", client=client
    ).synthesize(synthesis_request())

    assert len(client.models.calls) == 2
    assert report.sections[0].confidence == ConfidenceLevel.medium
    assert report.sections[0].missing_evidence is False
    assert "supporting_quote" in client.models.calls[1]["contents"]


def test_quote_validation_uses_original_evidence_and_checks_topic() -> None:
    payload = valid_report_payload()
    payload["evidence"][0]["raw_text_excerpt"] = "Texto inventado por el modelo."
    payload["sections"][0]["type"] = "employees"

    report = GeminiReportSynthesizer(
        api_key="test-key",
        model="gemini-test-model",
        client=FakeClient(response=SimpleNamespace(text=json.dumps(payload))),
    ).synthesize(synthesis_request())

    assert report.evidence[0].raw_text_excerpt == "Acme fabrica productos industriales."
    assert report.sections[0].confidence == ConfidenceLevel.low
    assert report.sections[0].missing_evidence is True


def test_quote_found_only_in_uncited_evidence_is_rejected() -> None:
    request = synthesis_request()
    uncited_quote = "Acme tambien presta servicios de consultoria especializada global."
    request.evidence.append(
        ClassifiedEvidence(
            source_id="source_1",
            topic=EvidenceTopic.business,
            claim=uncited_quote,
            raw_text_excerpt=uncited_quote,
            confidence=ConfidenceLevel.medium,
        )
    )
    payload = valid_report_payload()
    payload["sections"][0]["type"] = "employees"
    payload["sections"][0]["claims"][0]["supporting_quote"] = uncited_quote

    report = GeminiReportSynthesizer(
        api_key="test-key",
        model="gemini-test-model",
        client=FakeClient(response=SimpleNamespace(text=json.dumps(payload))),
    ).synthesize(request)

    assert report.sections[0].claims[0].confidence == ConfidenceLevel.low
    assert report.sections[0].missing_evidence is True


def test_invalid_quote_is_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    payload = valid_report_payload()
    private_quote = "Texto privado que no pertenece a la evidencia."
    payload["sections"][0]["claims"][0]["supporting_quote"] = private_quote
    payload["sections"][0]["type"] = "culture"

    with caplog.at_level(logging.WARNING):
        GeminiReportSynthesizer(
            api_key="test-key",
            model="gemini-test-model",
            client=FakeClient(response=SimpleNamespace(text=json.dumps(payload))),
        ).synthesize(synthesis_request())

    assert "Supporting quote validation failed" in caplog.text
    assert private_quote not in caplog.text


def test_quote_must_contain_claim_numbers() -> None:
    payload = valid_report_payload()
    payload["sections"][0]["type"] = "salary_benefits"
    payload["sections"][0]["claims"][0]["text"] = (
        "Acme fabrica 20 productos industriales."
    )

    report = GeminiReportSynthesizer(
        api_key="test-key",
        model="gemini-test-model",
        client=FakeClient(response=SimpleNamespace(text=json.dumps(payload))),
    ).synthesize(synthesis_request())

    assert report.sections[0].claims[0].confidence == ConfidenceLevel.low
    assert report.sections[0].missing_evidence is True


def test_required_quote_failure_after_repair_remains_an_error() -> None:
    payload = valid_report_payload()
    payload["sections"][0]["claims"][0]["supporting_quote"] = "Texto inventado."
    client = FakeClient(
        responses=[
            SimpleNamespace(text=json.dumps(payload)),
            SimpleNamespace(text=json.dumps(payload)),
        ]
    )

    with pytest.raises(SynthesisError, match="after repair"):
        GeminiReportSynthesizer(
            api_key="test-key",
            model="gemini-test-model",
            client=client,
        ).synthesize(synthesis_request())

    assert len(client.models.calls) == 2


def test_recommendation_claim_does_not_require_supporting_quote() -> None:
    payload = valid_report_payload()
    claim = payload["sections"][0]["claims"][0]
    claim["type"] = "recommendation"
    claim.pop("supporting_quote")

    report = GeminiReportSynthesizer(
        api_key="test-key",
        model="gemini-test-model",
        client=FakeClient(response=SimpleNamespace(text=json.dumps(payload))),
    ).synthesize(synthesis_request())

    assert report.sections[0].confidence == ConfidenceLevel.medium
    assert report.sections[0].missing_evidence is False


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
    assert report.metadata.llm_provider == "deepseek"
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
    assert "supporting_quote" in prompt
    assert "supporting_quote" in SYSTEM_PROMPT


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
                        "supporting_quote": "Acme fabrica productos industriales.",
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
