from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from backend.app.core.config import Settings
from backend.app.domain.reports import EvidenceTopic, SectionType, StructuredReportSchema
from backend.app.llm.deepseek import DeepSeekClient, extract_json_object
from backend.app.llm.synthesizer import SynthesisError


INDEXED_EVIDENCE_TOPICS = {
    EvidenceTopic.argentina_presence,
    EvidenceTopic.employees,
    EvidenceTopic.salary,
    EvidenceTopic.benefits,
    EvidenceTopic.culture,
    EvidenceTopic.interview_process,
    EvidenceTopic.interview_questions,
    EvidenceTopic.open_roles,
}

MAX_EVIDENCE_CHUNKS_PER_TOPIC = 2
MAX_WARNING_CHUNKS = 4
EMBEDDING_REQUEST_INTERVAL_SECONDS = 0.7

BROAD_QUERY_MARKERS = (
    "compar",
    "todos",
    "todas",
    "cuales",
    "cuáles",
    "mejor",
    "ranking",
    "empresas",
    "guardados",
    "reportes",
    "informes",
    "entre ",
    "versus",
    " vs ",
)

RAG_CHAT_SYSTEM_PROMPT = (
    "Sos un asistente de investigacion laboral para usuarios en Argentina. "
    "Responde siempre en espanol rioplatense. Usa solamente los fragmentos RAG provistos. "
    "Si los fragmentos no alcanzan para responder, decilo claramente. "
    "No inventes salarios, direcciones, beneficios, etapas de entrevista, vacantes ni fuentes. "
    "Devuelve JSON valido con answer y citation_chunk_ids."
)


@dataclass(frozen=True)
class RAGChunk:
    id: str
    report_id: str
    company_id: str
    company_name: str
    chunk_key: str
    chunk_type: str
    chunk_title: str
    chunk_text: str
    source_ids: list[str]
    evidence_ids: list[str]
    embedding: list[float]
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    score: float = 0.0


@dataclass(frozen=True)
class RAGChatCitation:
    report_id: str
    company_name: str
    chunk_id: str
    chunk_title: str
    source_id: str | None
    title: str
    url: str | None


@dataclass(frozen=True)
class RAGChatResult:
    answer: str
    scope_used: str
    citations: list[RAGChatCitation]


class RAGChatPayload(BaseModel):
    answer: str = Field(min_length=1)
    citation_chunk_ids: list[str] = Field(default_factory=list)


class GoogleEmbeddingService:
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self.client = client

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.settings.gemini_api_key:
            raise SynthesisError("GEMINI_API_KEY is required for embeddings.")
        if not texts:
            return []

        client = self.client or genai.Client(
            api_key=self.settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=self.settings.gemini_timeout_ms),
        )
        vectors = []
        for index, text in enumerate(texts):
            if index:
                time.sleep(EMBEDDING_REQUEST_INTERVAL_SECONDS)
            vectors.append(self._embed_one(client, text))
        return vectors

    def _embed_one(self, client: Any, text: str) -> list[float]:
        response = self._embed_one_with_retry(client, text)

        embeddings = getattr(response, "embeddings", None)
        if not embeddings:
            raise SynthesisError("Gemini embeddings response was empty.")
        values = getattr(embeddings[0], "values", embeddings[0])
        return [float(value) for value in values]

    def _embed_one_with_retry(self, client: Any, text: str):
        for attempt in range(2):
            try:
                return client.models.embed_content(
                    model=self.settings.gemini_embedding_model,
                    contents=text,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.settings.gemini_embedding_dimensions
                    ),
                )
            except Exception as exc:
                if attempt == 0 and "429" in str(exc):
                    time.sleep(12)
                    continue
                raise SynthesisError("Gemini embeddings request failed.") from exc
        raise SynthesisError("Gemini embeddings request failed.")


class ReportEmbeddingIndexer:
    def __init__(self, embedding_service: GoogleEmbeddingService) -> None:
        self.embedding_service = embedding_service

    def build_chunks(self, report: StructuredReportSchema) -> list[RAGChunk]:
        chunks_without_embeddings = build_report_chunks(report)
        vectors = self.embedding_service.embed([chunk.chunk_text for chunk in chunks_without_embeddings])
        return [
            RAGChunk(
                **{
                    **chunk.__dict__,
                    "embedding": vector,
                    "embedding_provider": "google",
                    "embedding_model": self.embedding_service.settings.gemini_embedding_model,
                    "embedding_dimensions": len(vector),
                }
            )
            for chunk, vector in zip(chunks_without_embeddings, vectors, strict=True)
        ]


class RAGChatService:
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self.client = client
        self.embedding_service = GoogleEmbeddingService(settings)

    def answer(
        self,
        message: str,
        chunks: list[RAGChunk],
        source_lookup: dict[str, dict[str, str]],
        active_report_id: str | None = None,
    ) -> RAGChatResult:
        if not self.settings.gemini_api_key:
            raise SynthesisError("GEMINI_API_KEY is required for embeddings.")
        if not self.settings.deepseek_api_key:
            raise SynthesisError("DEEPSEEK_API_KEY is required for generation.")

        scope_used = choose_scope(message, active_report_id)
        if not chunks:
            return RAGChatResult(
                answer="No encontre informes indexados para responder con fuentes. Genera o reindexa informes primero.",
                scope_used=scope_used,
                citations=[],
            )
        query_vector = self.embedding_service.embed([message])[0]
        ranked_chunks = rank_chunks(
            query_vector=query_vector,
            chunks=chunks,
            top_k=self.settings.rag_top_k,
            active_report_id=active_report_id,
            scope_used=scope_used,
        )
        if not ranked_chunks:
            return RAGChatResult(
                answer="No encontre evidencia suficiente en los informes guardados para responder con fuentes.",
                scope_used=scope_used,
                citations=[],
            )

        client = self.client or DeepSeekClient(
            api_key=self.settings.deepseek_api_key,
            model=self.settings.deepseek_model,
            timeout_seconds=self.settings.deepseek_timeout_seconds,
        )
        try:
            response = client.generate_json(
                RAG_CHAT_SYSTEM_PROMPT,
                build_rag_chat_prompt(message, ranked_chunks),
                RAGChatPayload,
                temperature=0.2,
                max_tokens=1800,
            )
        except Exception as exc:
            raise SynthesisError("DeepSeek no pudo responder el chat RAG.") from exc

        payload = normalize_rag_chat_response(response)
        citations = build_rag_citations(payload.citation_chunk_ids, ranked_chunks, source_lookup)
        return RAGChatResult(
            answer=payload.answer,
            scope_used=scope_used,
            citations=citations,
        )


def build_report_chunks(report: StructuredReportSchema) -> list[RAGChunk]:
    chunks: list[RAGChunk] = []
    sources_by_id = {source.id: source for source in report.sources}
    evidence_by_id = {item.id: item for item in report.evidence}

    for section in report.sections:
        evidence_ids = [
            evidence_id
            for claim in section.claims
            for evidence_id in claim.evidence_ids
            if evidence_id in evidence_by_id
        ]
        source_ids = sorted(
            {
                evidence_by_id[evidence_id].source_id
                for evidence_id in evidence_ids
                if evidence_by_id[evidence_id].source_id in sources_by_id
            }
        )
        text = clean_chunk_text(
            "\n".join(
                [
                    f"Empresa: {report.company.name}",
                    f"Seccion: {section.title}",
                    section.summary,
                    *[claim.text for claim in section.claims[:5]],
                ]
            )
        )
        if text:
            chunks.append(
                empty_chunk(
                    report=report,
                    chunk_key=f"section:{section.type.value}",
                    chunk_type=f"section:{section.type.value}",
                    chunk_title=section.title,
                    chunk_text=text,
                    source_ids=source_ids,
                    evidence_ids=evidence_ids[:12],
                )
            )

    evidence_counts_by_topic: dict[EvidenceTopic, int] = {}
    for item in report.evidence:
        if item.topic not in INDEXED_EVIDENCE_TOPICS:
            continue
        if evidence_counts_by_topic.get(item.topic, 0) >= MAX_EVIDENCE_CHUNKS_PER_TOPIC:
            continue
        evidence_counts_by_topic[item.topic] = evidence_counts_by_topic.get(item.topic, 0) + 1
        source = sources_by_id.get(item.source_id)
        text = clean_chunk_text(
            "\n".join(
                [
                    f"Empresa: {report.company.name}",
                    f"Tipo: evidencia {item.topic.value}",
                    f"Fuente: {source.title if source else item.source_id}",
                    item.claim,
                    item.raw_text_excerpt or "",
                ]
            )
        )
        if text:
            chunks.append(
                empty_chunk(
                    report=report,
                    chunk_key=f"evidence:{item.id}",
                    chunk_type=f"evidence:{item.topic.value}",
                    chunk_title=f"Evidencia: {item.topic.value}",
                    chunk_text=text,
                    source_ids=[item.source_id],
                    evidence_ids=[item.id],
                )
            )

    for warning in report.warnings[:MAX_WARNING_CHUNKS]:
        text = clean_chunk_text(
            f"Empresa: {report.company.name}\nAdvertencia: {warning.message}"
        )
        if text:
            chunks.append(
                empty_chunk(
                    report=report,
                    chunk_key=f"warning:{warning.id}",
                    chunk_type="warning",
                    chunk_title="Advertencia",
                    chunk_text=text,
                    source_ids=[],
                    evidence_ids=[],
                )
            )

    return chunks


def empty_chunk(
    report: StructuredReportSchema,
    chunk_key: str,
    chunk_type: str,
    chunk_title: str,
    chunk_text: str,
    source_ids: list[str],
    evidence_ids: list[str],
) -> RAGChunk:
    return RAGChunk(
        id=str(uuid4()),
        report_id=report.report_id,
        company_id=report.company.id,
        company_name=report.company.name,
        chunk_key=chunk_key,
        chunk_type=chunk_type,
        chunk_title=chunk_title,
        chunk_text=chunk_text[:2200],
        source_ids=source_ids,
        evidence_ids=evidence_ids,
        embedding=[],
        embedding_provider="",
        embedding_model="",
        embedding_dimensions=0,
    )


def choose_scope(message: str, active_report_id: str | None) -> str:
    normalized = f" {message.lower()} "
    if any(marker in normalized for marker in BROAD_QUERY_MARKERS):
        return "all_reports"
    return "active_report" if active_report_id else "all_reports"


def rank_chunks(
    query_vector: list[float],
    chunks: list[RAGChunk],
    top_k: int,
    active_report_id: str | None,
    scope_used: str,
) -> list[RAGChunk]:
    scoped_chunks = chunks
    if scope_used == "active_report" and active_report_id:
        active_chunks = [chunk for chunk in chunks if chunk.report_id == active_report_id]
        scoped_chunks = active_chunks or chunks

    ranked = []
    for chunk in scoped_chunks:
        score = cosine_similarity(query_vector, chunk.embedding)
        if active_report_id and chunk.report_id == active_report_id:
            score += 0.08
        ranked.append(
            RAGChunk(
                **{
                    **chunk.__dict__,
                    "score": score,
                }
            )
        )
    return sorted(ranked, key=lambda item: item.score, reverse=True)[:top_k]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def build_rag_chat_prompt(message: str, chunks: list[RAGChunk]) -> str:
    return json.dumps(
        {
            "instruction": (
                "Responde usando solo estos fragmentos. Cita citation_chunk_ids relevantes. "
                "Si faltan datos exactos, indica que no estan disponibles en los informes guardados."
            ),
            "question": message,
            "chunks": [
                {
                    "chunk_id": chunk.id,
                    "report_id": chunk.report_id,
                    "company": chunk.company_name,
                    "type": chunk.chunk_type,
                    "title": chunk.chunk_title,
                    "text": chunk.chunk_text,
                    "source_ids": chunk.source_ids,
                    "evidence_ids": chunk.evidence_ids,
                }
                for chunk in chunks
            ],
        },
        ensure_ascii=False,
    )


def normalize_rag_chat_response(response: Any) -> RAGChatPayload:
    if isinstance(response, RAGChatPayload):
        return response
    if isinstance(response, dict):
        return validate_rag_chat_payload(response)

    text = getattr(response, "content", None)
    if not text:
        raise SynthesisError("DeepSeek no devolvio texto para el chat RAG.")
    try:
        return validate_rag_chat_payload(json.loads(text))
    except json.JSONDecodeError:
        return validate_rag_chat_payload(json.loads(extract_json_object(text)))


def validate_rag_chat_payload(raw_payload: dict[str, Any]) -> RAGChatPayload:
    try:
        return RAGChatPayload.model_validate(raw_payload)
    except ValidationError as exc:
        raise SynthesisError("DeepSeek no devolvio una respuesta RAG valida.") from exc


def build_rag_citations(
    chunk_ids: list[str],
    chunks: list[RAGChunk],
    source_lookup: dict[str, dict[str, str]],
) -> list[RAGChatCitation]:
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    citations: list[RAGChatCitation] = []
    seen: set[tuple[str, str | None]] = set()
    for chunk_id in chunk_ids:
        chunk = chunks_by_id.get(chunk_id)
        if not chunk:
            continue
        source_id = chunk.source_ids[0] if chunk.source_ids else None
        key = (chunk.id, source_id)
        if key in seen:
            continue
        seen.add(key)
        source = source_lookup.get(source_id or "", {})
        citations.append(
            RAGChatCitation(
                report_id=chunk.report_id,
                company_name=chunk.company_name,
                chunk_id=chunk.id,
                chunk_title=chunk.chunk_title,
                source_id=source_id,
                title=source.get("title") or chunk.chunk_title,
                url=source.get("url"),
            )
        )
    return citations


def clean_chunk_text(value: str) -> str:
    return " ".join(value.split())


def chunk_source_lookup(report: StructuredReportSchema) -> dict[str, dict[str, str]]:
    return {
        source.id: {
            "title": source.title,
            "url": str(source.url),
        }
        for source in report.sources
    }
