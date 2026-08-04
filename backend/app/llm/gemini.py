from __future__ import annotations

import json
from typing import Any

from google import genai
from google.genai import types
from pydantic import ValidationError

from backend.app.core.time import utc_now
from backend.app.domain.reports import StructuredReportSchema
from backend.app.llm.prompts import SYSTEM_PROMPT, build_repair_prompt, build_report_user_prompt
from backend.app.llm.synthesizer import SynthesisError, SynthesisRequest


class GeminiReportSynthesizer:
    def __init__(
        self,
        api_key: str,
        model: str,
        client: Any | None = None,
        max_output_tokens: int = 8000,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.client = client
        self.max_output_tokens = max_output_tokens

    def synthesize(self, request: SynthesisRequest) -> StructuredReportSchema:
        if not self.api_key:
            raise SynthesisError("GEMINI_API_KEY is required.")

        client = self.client or genai.Client(api_key=self.api_key)
        try:
            response = self.generate_content(client, build_report_user_prompt(request))
        except Exception as exc:
            raise SynthesisError("Gemini request failed before receiving a valid response.") from exc

        try:
            return validate_response(response, request, self.model)
        except SynthesisError as first_error:
            first_error_text = short_error(first_error)
            invalid_output = response_text(response)

        try:
            repair_response = self.generate_content(
                client,
                build_repair_prompt(invalid_output, first_error_text),
            )
        except Exception as exc:
            raise SynthesisError(
                f"Gemini repair request failed after invalid response: {first_error_text}"
            ) from exc

        try:
            return validate_response(repair_response, request, self.model)
        except SynthesisError as repair_error:
            raise SynthesisError(
                "Gemini response did not match report schema after repair: "
                f"{short_error(repair_error)}"
            ) from repair_error

    def generate_content(self, client: Any, contents: str) -> Any:
        return client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=self.max_output_tokens,
                response_mime_type="application/json",
                response_schema=StructuredReportSchema,
            ),
        )


def validate_response(
    response: Any,
    request: SynthesisRequest,
    model: str,
) -> StructuredReportSchema:
    raw_report = parse_gemini_response(response)
    raw_report = normalize_report_payload(raw_report, request, model)
    try:
        return StructuredReportSchema.model_validate(raw_report)
    except ValidationError as exc:
        raise SynthesisError("Gemini response did not match report schema.") from exc


def parse_gemini_response(response: Any) -> dict[str, Any]:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, StructuredReportSchema):
        return parsed.model_dump(mode="json")
    if isinstance(parsed, dict):
        return parsed

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
    if not isinstance(raw, dict):
        raise SynthesisError("Gemini response JSON was not an object.")
    return raw


def response_text(response: Any) -> str:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, StructuredReportSchema):
        return parsed.model_dump_json()
    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False)
    text = getattr(response, "text", None)
    if text:
        return text
    return "<empty Gemini response>"


def short_error(error: BaseException, max_length: int = 500) -> str:
    text = str(error)
    return text if len(text) <= max_length else text[: max_length - 3] + "..."


def extract_json_object(content: str) -> str:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise SynthesisError("Gemini response content was not valid JSON.")
    return cleaned[start : end + 1]


def normalize_report_payload(
    raw_report: dict[str, Any],
    request: SynthesisRequest,
    model: str,
) -> dict[str, Any]:
    now = utc_now().isoformat()
    raw_report["report_id"] = request.report_id
    raw_report["status"] = "completed"
    raw_report.setdefault("schema_version", "1.0")
    raw_report.setdefault("language", "es-AR")
    raw_report["company"] = {
        "id": request.company_id,
        "name": request.company_name,
        "normalized_name": request.normalized_company_name,
        **raw_report.get("company", {}),
    }
    raw_report.setdefault("generated_at", now)
    raw_report.setdefault("valid_until", None)
    raw_report.setdefault("personalized_preparation", None)
    raw_report.setdefault("cv_tailoring", None)
    raw_report.setdefault("warnings", [])
    raw_report["sources"] = normalize_sources(raw_report.get("sources"), request, now)
    raw_report["evidence"] = normalize_evidence(raw_report.get("evidence"), request)
    raw_report["metadata"] = normalize_metadata(raw_report.get("metadata"), request, raw_report, model)
    return raw_report


def normalize_sources(raw_sources: Any, request: SynthesisRequest, accessed_at: str) -> list[dict[str, Any]]:
    raw_items = raw_sources if isinstance(raw_sources, list) else []
    sources_by_id = {
        source.source_id: {
            "id": source.source_id,
            "title": source.title,
            "url": source.url,
            "domain": source.domain,
            "source_type": source.source_type.value,
            "reliability_score": source.reliability_score,
            "accessed_at": accessed_at,
            "published_at": None,
            "snippet": source.snippet,
            "language": "es",
            "is_current": source.is_current,
        }
        for source in request.sources
    }
    normalized = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        source_id = item.get("id")
        fallback = sources_by_id.get(source_id, {})
        merged = {**fallback, **item}
        merged.setdefault("accessed_at", accessed_at)
        merged.setdefault("published_at", None)
        merged.setdefault("language", "es")
        normalized.append(merged)
    return normalized or list(sources_by_id.values())


def normalize_evidence(raw_evidence: Any, request: SynthesisRequest) -> list[dict[str, Any]]:
    if isinstance(raw_evidence, list) and raw_evidence:
        return [item for item in raw_evidence if isinstance(item, dict)]
    return [
        {
            "id": f"evidence_{index + 1}",
            "source_id": item.source_id,
            "topic": item.topic.value,
            "claim": item.claim,
            "raw_text_excerpt": item.raw_text_excerpt,
            "confidence": item.confidence.value,
        }
        for index, item in enumerate(request.evidence)
    ]


def normalize_metadata(
    raw_metadata: Any,
    request: SynthesisRequest,
    raw_report: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    metadata.setdefault("search_provider", "unknown")
    metadata["llm_provider"] = "gemini"
    metadata["llm_model"] = model
    metadata["source_count"] = len(raw_report.get("sources", []))
    metadata["evidence_count"] = len(raw_report.get("evidence", []))
    metadata["used_cv"] = request.include_cv
    metadata["used_cv_tailoring"] = request.include_cv_tailoring
    metadata.setdefault("generation_duration_ms", None)
    return metadata
