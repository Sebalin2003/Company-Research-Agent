from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from backend.app.core.config import Settings
from backend.app.domain.reports import StructuredReportSchema
from backend.app.llm.deepseek import DeepSeekClient, extract_json_object
from backend.app.llm.synthesizer import SynthesisError


CHAT_SYSTEM_PROMPT = (
    "Sos un asistente de investigacion laboral para usuarios en Argentina. "
    "Responde siempre en espanol rioplatense, usando solamente el contexto del informe provisto. "
    "Si el informe no contiene evidencia suficiente, decilo claramente. "
    "No inventes datos, fuentes, salarios, direcciones, etapas de entrevista ni vacantes. "
    "Devuelve JSON valido con answer y source_ids."
)


class ChatAnswerPayload(BaseModel):
    answer: str = Field(min_length=1)
    source_ids: list[str] = Field(default_factory=list)


class ReportChatService:
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self.client = client

    def answer(self, report: StructuredReportSchema, message: str) -> ChatAnswerPayload:
        if not self.settings.deepseek_api_key:
            raise SynthesisError("DEEPSEEK_API_KEY is required.")

        client = self.client or DeepSeekClient(
            api_key=self.settings.deepseek_api_key,
            model=self.settings.deepseek_model,
            timeout_seconds=self.settings.deepseek_timeout_seconds,
        )
        try:
            response = client.generate_json(
                CHAT_SYSTEM_PROMPT,
                build_chat_prompt(report, message),
                ChatAnswerPayload,
                temperature=0.2,
                max_tokens=1600,
            )
        except Exception as exc:
            raise SynthesisError("DeepSeek no pudo responder la pregunta del informe.") from exc
        return normalize_chat_response(response, report)


def build_chat_prompt(report: StructuredReportSchema, message: str) -> str:
    return json.dumps(
        {
            "instruction": (
                "Contesta la pregunta usando solo este informe guardado. "
                "Cita source_ids relevantes. Si falta evidencia, dilo explicitamente."
            ),
            "question": message,
            "report_context": compact_report_context(report),
        },
        ensure_ascii=False,
    )


def compact_report_context(report: StructuredReportSchema) -> dict[str, Any]:
    source_ids_with_evidence = {item.source_id for item in report.evidence}
    return {
        "company": report.company.model_dump(mode="json"),
        "sections": [
            {
                "type": section.type.value,
                "title": section.title,
                "summary": section.summary,
                "claim_texts": [claim.text for claim in section.claims[:5]],
            }
            for section in report.sections
        ],
        "sources": [
            {
                "id": source.id,
                "title": source.title,
                "url": str(source.url),
                "domain": source.domain,
                "source_type": source.source_type.value,
                "reliability_score": source.reliability_score,
            }
            for source in report.sources
            if source.id in source_ids_with_evidence
        ],
        "evidence": [
            {
                "id": item.id,
                "source_id": item.source_id,
                "topic": item.topic.value,
                "claim": item.claim,
                "excerpt": item.raw_text_excerpt,
                "confidence": item.confidence.value,
            }
            for item in report.evidence[:80]
        ],
        "warnings": [warning.message for warning in report.warnings],
    }


def normalize_chat_response(response: Any, report: StructuredReportSchema) -> ChatAnswerPayload:
    if isinstance(response, ChatAnswerPayload):
        payload = response
    elif isinstance(response, dict):
        payload = validate_chat_payload(response)
    else:
        text = getattr(response, "content", None)
        if not text:
            raise SynthesisError("DeepSeek no devolvio texto para el chat.")
        try:
            payload = validate_chat_payload(json.loads(text))
        except json.JSONDecodeError:
            payload = validate_chat_payload(json.loads(extract_json_object(text)))

    known_source_ids = {source.id for source in report.sources}
    return ChatAnswerPayload(
        answer=payload.answer,
        source_ids=[source_id for source_id in payload.source_ids if source_id in known_source_ids],
    )


def validate_chat_payload(raw_payload: dict[str, Any]) -> ChatAnswerPayload:
    try:
        return ChatAnswerPayload.model_validate(raw_payload)
    except ValidationError as exc:
        raise SynthesisError("DeepSeek no devolvio una respuesta de chat valida.") from exc
