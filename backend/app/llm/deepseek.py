from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from backend.app.core.time import utc_now
from backend.app.domain.reports import ClaimSchema, SectionSchema, StructuredReportSchema
from backend.app.llm.prompts import (
    SYSTEM_PROMPT,
    build_compact_report_payload,
    build_repair_prompt,
    build_report_user_prompt,
)
from backend.app.llm.synthesizer import SynthesisError, SynthesisRequest
from backend.app.services.report_quality import validate_and_strip_supporting_quotes

DEEPSEEK_CHAT_URL = "https://api.deepseek.com/chat/completions"


class DeepSeekAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: str | None = None,
        raw_content: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.raw_content = raw_content


@dataclass(frozen=True)
class DeepSeekToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class DeepSeekResponse:
    content: str = ""
    finish_reason: str = ""
    tool_calls: list[DeepSeekToolCall] = field(default_factory=list)
    assistant_message: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: int = 60,
        http_client: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.http_client = http_client
        self.last_usage: dict[str, int] = {}

    def select_tool(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str | dict[str, Any] = "required",
        max_tokens: int = 4000,
        temperature: float = 0.1,
    ) -> DeepSeekResponse:
        return self.complete(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[BaseModel],
        *,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        response = self.complete(
            [
                {
                    "role": "system",
                    "content": (
                        f"{system_prompt}\nDevolv\u00e9 solamente un objeto JSON v\u00e1lido compatible "
                        f"con este JSON Schema:\n{schema_json}"
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            temperature=temperature,
        )
        try:
            payload = json.loads(response.content)
        except json.JSONDecodeError:
            try:
                payload = json.loads(extract_json_object(response.content))
            except (json.JSONDecodeError, SynthesisError) as exc:
                raise DeepSeekAPIError(
                    "DeepSeek returned invalid JSON.", raw_content=response.content
                ) from exc
        if not isinstance(payload, dict):
            raise DeepSeekAPIError("DeepSeek returned JSON that was not an object.")
        return payload

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> DeepSeekResponse:
        return self.complete(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, str] | None = None,
        max_tokens: int,
        temperature: float,
    ) -> DeepSeekResponse:
        if not self.api_key:
            raise DeepSeekAPIError("DEEPSEEK_API_KEY is required.", status_code=401)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "thinking": {"type": "disabled"},
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if tools:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if response_format:
            body["response_format"] = response_format
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            requester = self.http_client or httpx
            response = requester.post(
                DEEPSEEK_CHAT_URL,
                headers=headers,
                json=body,
                timeout=self.timeout_seconds,
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise DeepSeekAPIError("DeepSeek request failed.") from exc
        if response.status_code >= 400:
            raise DeepSeekAPIError(
                "DeepSeek API request failed.",
                status_code=response.status_code,
                retry_after=response.headers.get("Retry-After"),
            )
        try:
            raw = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise DeepSeekAPIError("DeepSeek returned an invalid response.") from exc
        parsed = parse_deepseek_response(raw)
        self.last_usage = parsed.usage
        return parsed


def parse_deepseek_response(raw: Any) -> DeepSeekResponse:
    if not isinstance(raw, dict) or not isinstance(raw.get("choices"), list) or not raw["choices"]:
        raise DeepSeekAPIError("DeepSeek response did not include a choice.")
    choice = raw["choices"][0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise DeepSeekAPIError("DeepSeek response did not include a message.")
    finish_reason = str(choice.get("finish_reason") or "")
    if finish_reason == "length":
        raise DeepSeekAPIError("DeepSeek response exceeded the output limit.")
    if finish_reason == "content_filter":
        raise DeepSeekAPIError("DeepSeek blocked the response.")
    if finish_reason == "insufficient_system_resource":
        raise DeepSeekAPIError("DeepSeek request failed.", status_code=503)
    message = choice["message"]
    calls = []
    for item in message.get("tool_calls") or []:
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict):
            raise DeepSeekAPIError("DeepSeek returned an invalid tool call.")
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError as exc:
            raise DeepSeekAPIError("DeepSeek returned invalid tool arguments.") from exc
        if not isinstance(arguments, dict):
            raise DeepSeekAPIError("DeepSeek tool arguments were not an object.")
        call_id = str(item.get("id") or "").strip()
        name = str(function.get("name") or "").strip()
        if not call_id or not name:
            raise DeepSeekAPIError("DeepSeek returned an incomplete tool call.")
        calls.append(DeepSeekToolCall(id=call_id, name=name, arguments=arguments))
    content = str(message.get("content") or "").strip()
    if not content and not calls:
        raise DeepSeekAPIError("DeepSeek response was empty.")
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    normalized_usage = {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "cache_hit_tokens": int(usage.get("prompt_cache_hit_tokens") or 0),
        "cache_miss_tokens": int(usage.get("prompt_cache_miss_tokens") or 0),
    }
    return DeepSeekResponse(
        content=content,
        finish_reason=finish_reason,
        tool_calls=calls,
        assistant_message={
            "role": "assistant",
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls") or [],
        },
        usage=normalized_usage,
    )


class DeepSeekClaimSchema(ClaimSchema):
    supporting_quote: str | None = None


class DeepSeekSectionSchema(SectionSchema):
    claims: list[DeepSeekClaimSchema] = Field(default_factory=list)


class DeepSeekReportSchema(StructuredReportSchema):
    sections: list[DeepSeekSectionSchema] = Field(default_factory=list)


class DeepSeekReportSynthesizer:
    def __init__(
        self,
        api_key: str,
        model: str,
        client: Any | None = None,
        max_output_tokens: int = 8000,
        timeout_seconds: int = 60,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.client = client
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds

    def synthesize(self, request: SynthesisRequest) -> StructuredReportSchema:
        if not self.api_key:
            raise SynthesisError("DEEPSEEK_API_KEY is required.")
        client = self.client or DeepSeekClient(
            self.api_key, self.model, self.timeout_seconds
        )
        try:
            response = self.generate_content(client, build_report_user_prompt(request))
            return validate_response(response, request, self.model)
        except DeepSeekAPIError as first_error:
            if first_error.raw_content is None:
                raise SynthesisError(
                    "DeepSeek request failed before receiving a valid response."
                ) from first_error
            first_error_text = short_error(first_error)
            invalid_output = first_error.raw_content
        except SynthesisError as first_error:
            first_error_text = short_error(first_error)
            invalid_output = response_text(response)
        except Exception as exc:
            raise SynthesisError("DeepSeek request failed before receiving a valid response.") from exc
        try:
            repair_response = self.generate_content(
                client, build_repair_prompt(invalid_output, first_error_text)
            )
        except Exception as exc:
            raise SynthesisError(
                f"DeepSeek repair request failed after invalid response: {first_error_text}"
            ) from exc
        try:
            return validate_response(repair_response, request, self.model)
        except SynthesisError as repair_error:
            raise SynthesisError(
                "DeepSeek response did not match report schema after repair: "
                f"{short_error(repair_error)}"
            ) from repair_error

    def generate_content(self, client: Any, contents: str) -> dict[str, Any]:
        return client.generate_json(
            SYSTEM_PROMPT,
            contents,
            DeepSeekReportSchema,
            max_tokens=self.max_output_tokens,
            temperature=0.2,
        )


def validate_response(
    response: dict[str, Any], request: SynthesisRequest, model: str
) -> StructuredReportSchema:
    raw_report = normalize_report_payload(dict(response), request, model)
    try:
        internal_report = DeepSeekReportSchema.model_validate(raw_report)
        public_payload = validate_and_strip_supporting_quotes(
            internal_report.model_dump(mode="json")
        )
        return StructuredReportSchema.model_validate(public_payload)
    except ValidationError as exc:
        raise SynthesisError("DeepSeek response did not match report schema.") from exc


def response_text(response: Any) -> str:
    if isinstance(response, dict):
        return json.dumps(response, ensure_ascii=False)
    return "<empty DeepSeek response>"


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
        raise SynthesisError("DeepSeek response content was not valid JSON.")
    return cleaned[start : end + 1]


def normalize_report_payload(
    raw_report: dict[str, Any], request: SynthesisRequest, model: str
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
    raw_report["evidence"] = build_compact_report_payload(request)["evidence"]
    metadata = raw_report.get("metadata") if isinstance(raw_report.get("metadata"), dict) else {}
    metadata.setdefault("search_provider", "unknown")
    metadata.update(
        {
            "llm_provider": "deepseek",
            "llm_model": model,
            "source_count": len(raw_report["sources"]),
            "evidence_count": len(raw_report["evidence"]),
            "used_cv": request.include_cv,
            "used_cv_tailoring": request.include_cv_tailoring,
        }
    )
    metadata.setdefault("generation_duration_ms", None)
    raw_report["metadata"] = metadata
    return raw_report


def normalize_sources(
    raw_sources: Any, request: SynthesisRequest, accessed_at: str
) -> list[dict[str, Any]]:
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
        fallback = sources_by_id.get(item.get("id"), {})
        merged = {**fallback, **item}
        merged.setdefault("accessed_at", accessed_at)
        merged.setdefault("published_at", None)
        merged.setdefault("language", "es")
        normalized.append(merged)
    return normalized or list(sources_by_id.values())
