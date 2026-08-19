from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.config import Settings, get_settings
from backend.app.core.time import utc_now
from backend.app.db import models
from backend.app.db.conversation_repository import ConversationRepository
from backend.app.db.repositories import CompanyRepository, ReportRepository, safe_json_dict
from backend.app.domain.reports import ConfidenceLevel, EvidenceTopic, ReportStatus, SourceType
from backend.app.llm.deepseek import DeepSeekAPIError, DeepSeekClient, DeepSeekReportSynthesizer
from backend.app.llm.synthesizer import SynthesisError, SynthesisRequest
from backend.app.research.content_extractor import HttpContentExtractor
from backend.app.research.evidence_classifier import classify_evidence
from backend.app.research.scoring import score_search_result, stable_source_id
from backend.app.research.search_provider import FakeSearchProvider, TavilySearchProvider
from backend.app.research.types import ClassifiedEvidence, ExtractedContent, ScoredSource, SearchResult
from backend.app.services.rag_indexing import schedule_report_embedding_task
from backend.app.services.rag import GoogleEmbeddingService, rank_chunks


SYSTEM_PROMPT = """Sos Radar Laboral, un asistente laboral conversacional en español.
Elegí una sola función por turno. Usá respond para terminar y request_clarification si falta un dato esencial.
Un mensaje aislado o ambiguo como "google" requiere aclaración. Una consulta meteorológica sin ubicación
también requiere aclaración. Para datos actuales, priorizá fuentes oficiales y páginas de empleo oficiales.
Después de recibir una aclaración, avanzá con la mejor interpretación disponible sin volver a preguntar.
Si una vacante o pasantía actual no tiene evidencia de una fuente oficial o career page, indicá claramente
que no pudiste confirmarla oficialmente y agregá una advertencia.
La orientación general puede ser guidance sin citas. Datos sobre empresas, mercado, salarios, vacantes,
beneficios, entrevistas o informes deben ser grounded y citar IDs de evidencia disponibles.
No uses conocimiento previo del modelo como evidencia. No inventes fuentes ni datos. Las herramientas de CV
no están disponibles. search_web busca, inspect_page inspecciona solo resultados obtenidos, review_evidence
revisa cobertura, retrieve_reports recupera informes, compare_reports crea una comparación y finish_research
crea un informe o cierra la investigación. No describas razonamiento privado."""

GENERAL_GUIDANCE_RE = re.compile(
    r"\b(consejos?|recomendaciones? generales?|c[oó]mo mejorar|c[oó]mo preparar|qu[eé] es|"
    r"explicame|expl[ií]came|curr[ií]culum|\bcv\b|entrevista en general|"
    r"organizar.*b[uú]squeda laboral|ayudame.*b[uú]squeda laboral)\b",
    re.IGNORECASE,
)

SMALL_TALK_RE = re.compile(r"^(hola|buenas|gracias|ok|okay|sí|si|no|chau)[!. ]*$", re.IGNORECASE)
WEATHER_RE = re.compile(r"\b(clima|tiempo|temperatura|pron[oó]stico)\b", re.IGNORECASE)
CURRENT_OPPORTUNITY_RE = re.compile(
    r"\b(pasant[ií]as?|internships?|vacantes?|puesto abierto|empleo vigente|oportunidad laboral)\b",
    re.IGNORECASE,
)


class AgentSource(BaseModel):
    id: str
    title: str
    url: str
    snippet: str = ""
    domain: str = ""
    source_type: str = SourceType.unknown.value
    reliability_score: int = 1
    rank: int = 1
    topic: str = EvidenceTopic.general.value
    accessed_at: str
    inspected: bool = False


class AgentEvidence(BaseModel):
    id: str
    source_id: str
    topic: str
    claim: str
    excerpt: str = ""
    confidence: str = ConfidenceLevel.unknown.value


class AgentState(BaseModel):
    goal: str
    requires_grounding: bool = True
    sources: list[AgentSource] = Field(default_factory=list)
    evidence: list[AgentEvidence] = Field(default_factory=list)
    searched_queries: list[str] = Field(default_factory=list)
    inspected_source_ids: list[str] = Field(default_factory=list)
    retrieved_report_ids: list[str] = Field(default_factory=list)
    unresolved_topics: list[str] = Field(default_factory=list)
    generated_report_id: str | None = None
    generated_comparison_id: str | None = None
    clarification_response: dict[str, Any] | None = None
    force_finalize: bool = False
    response_message_id: str | None = None


class GroundedClaim(BaseModel):
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class AgentAnswer(BaseModel):
    answer: str = Field(min_length=1, max_length=12_000)
    answer_type: Literal["guidance", "grounded"]
    claims: list[GroundedClaim] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ComparisonPayload(BaseModel):
    companies: list[dict[str, str]]
    rows: list[dict[str, Any]]


class AgentPaused(Exception):
    pass


class AgentCompleted(Exception):
    pass


class AgentFailure(RuntimeError):
    pass


@dataclass
class ToolResult:
    payload: dict[str, Any]
    artifact_refs: list[dict[str, str]]


class AgentOrchestrator:
    def __init__(
        self,
        db: Session,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
        embedding_service: Any | None = None,
        search_provider: Any | None = None,
        extractor: Any | None = None,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.repo = ConversationRepository(db)
        self.report_repo = ReportRepository(db)
        self.client = client
        self.embedding_service = embedding_service
        self.search_provider = search_provider
        self.extractor = extractor
        self.run_elapsed_base = 0

    def run(self, task_id: str) -> None:
        task = self.repo.get_task(task_id)
        if not task or task.status != "pending":
            return
        try:
            self.repo.update_task(task, "running")
            self.repo.add_event(task, "task.started", {"task_run_id": task.id, "status": "running"})
            self.repo.add_event(
                task,
                "task.progress",
                {
                    "task_run_id": task.id,
                    "label": "DeepSeek está evaluando la solicitud",
                    "detail": "Decidiendo si responder o usar herramientas",
                    "status": "running",
                },
            )
            self.db.commit()
            state = self.load_state(task)
            usage = safe_json_dict(task.usage_json)
            usage.setdefault("model_turns", 0)
            usage.setdefault("searches", 0)
            usage.setdefault("inspections", 0)
            usage.setdefault("elapsed_ms", 0)
            self.run_elapsed_base = int(usage["elapsed_ms"])
            run_started = time.monotonic()
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self.build_context(task, state)},
            ]
            protocol_repair_used = False
            repair_tool_name: str | None = None

            while True:
                self.refresh_task(task)
                if task.status == "cancelled":
                    return
                self.enforce_budget(task, state, usage, run_started)
                if state.force_finalize:
                    self.repo.add_event(
                        task,
                        "tool.started",
                        {
                            "task_run_id": task.id,
                            "label": self.tool_label("respond"),
                            "status": "running",
                        },
                        tool_name="respond",
                        arguments={"answer_type": "grounded" if state.requires_grounding else "guidance"},
                    )
                    self.db.commit()
                    self.tool_respond(
                        task,
                        state,
                        usage,
                        {"answer_type": "grounded" if state.requires_grounding else "guidance"},
                    )
                client = self.get_client()
                clarification_only = self.requires_clarification(state)
                tools = self.function_tools(
                    final_only=state.force_finalize,
                    clarification_only=clarification_only,
                    allow_clarification=not bool(state.clarification_response),
                )
                allowed_names = [item["function"]["name"] for item in tools]
                tool_choice: str | dict[str, Any] = "required"
                if repair_tool_name:
                    tools = [
                        item for item in tools
                        if item["function"]["name"] == repair_tool_name
                    ]
                    allowed_names = [repair_tool_name]
                    tool_choice = {
                        "type": "function",
                        "function": {"name": repair_tool_name},
                    }
                response = client.select_tool(messages, tools, tool_choice=tool_choice)
                usage["model_turns"] += 1
                self.add_provider_usage(usage, response.usage)
                self.persist_usage(task, usage, run_started)
                calls = response.tool_calls
                assistant_message = response.assistant_message
                if len(calls) > 1:
                    calls = calls[:1]
                    assistant_message = dict(assistant_message)
                    assistant_message["tool_calls"] = [
                        item
                        for item in assistant_message.get("tool_calls", [])
                        if item.get("id") == calls[0].id
                    ]
                if not calls:
                    if protocol_repair_used:
                        raise AgentFailure(
                            "DeepSeek no pudo elegir una acción válida "
                            f"({len(calls)} acciones devueltas). Reintentá la solicitud."
                        )
                    protocol_repair_used = True
                    repair_tool_name = next(
                        (call.name for call in calls if call.name in allowed_names),
                        None,
                    )
                    if not repair_tool_name:
                        if allowed_names == ["request_clarification"]:
                            repair_tool_name = "request_clarification"
                        elif state.requires_grounding and "search_web" in allowed_names:
                            repair_tool_name = "search_web"
                        elif "respond" in allowed_names:
                            repair_tool_name = "respond"
                    messages.append(
                        {
                            "role": "user",
                            "content": "Seleccioná ahora exactamente una de las funciones permitidas; no respondas con texto libre.",
                        }
                    )
                    continue
                call = calls[0]
                name = call.name
                if name not in allowed_names:
                    if protocol_repair_used:
                        raise AgentFailure("DeepSeek intentó usar una acción no permitida.")
                    protocol_repair_used = True
                    messages.append(
                        {
                            "role": "user",
                            "content": "La acción anterior no está permitida. Seleccioná exactamente una función disponible.",
                        }
                    )
                    continue
                arguments = call.arguments
                self.repo.add_event(
                    task,
                    "tool.started",
                    {"task_run_id": task.id, "label": self.tool_label(name), "status": "running"},
                    tool_name=name,
                    arguments=self.safe_arguments(name, arguments),
                )
                self.db.commit()
                result = self.execute_tool(task, state, usage, name, arguments)
                self.save_state(task, state)
                self.persist_usage(task, usage, run_started)
                self.repo.add_event(
                    task,
                    "tool.completed",
                    {
                        "task_run_id": task.id,
                        "label": self.tool_label(name),
                        "status": "completed",
                        "summary": result.payload.get("summary") or result.payload.get("message"),
                    },
                    tool_name=name,
                    artifact_refs=result.artifact_refs,
                )
                self.db.commit()
                messages.append(assistant_message)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result.payload, ensure_ascii=False),
                    }
                )
                if state.force_finalize and name != "respond":
                    raise AgentFailure("DeepSeek no pudo finalizar la respuesta dentro del presupuesto.")
        except (AgentPaused, AgentCompleted):
            return
        except Exception as exc:
            self.fail_task(task_id, exc)

    def get_client(self):
        if self.client is not None:
            return self.client
        if not self.settings.deepseek_api_key:
            raise AgentFailure(
                "DeepSeek no está configurado. Agregá DEEPSEEK_API_KEY y reintentá."
            )
        self.client = DeepSeekClient(
            api_key=self.settings.deepseek_api_key,
            model=self.settings.deepseek_model,
            timeout_seconds=self.settings.deepseek_timeout_seconds,
        )
        return self.client

    def get_embedding_service(self):
        if self.embedding_service is None:
            self.embedding_service = GoogleEmbeddingService(self.settings)
        return self.embedding_service

    def get_search_provider(self):
        if self.search_provider is not None:
            return self.search_provider
        if self.settings.search_provider != "tavily":
            self.search_provider = FakeSearchProvider()
        else:
            self.search_provider = TavilySearchProvider(
                api_key=self.settings.search_provider_api_key or "",
                timeout_seconds=self.settings.research_search_timeout_seconds,
            )
        return self.search_provider

    def get_extractor(self):
        if self.extractor is None:
            self.extractor = HttpContentExtractor(
                timeout_seconds=self.settings.research_extract_timeout_seconds
            )
        return self.extractor

    def load_state(self, task: models.TaskRun) -> AgentState:
        raw = safe_json_dict(task.working_state_json)
        if raw.get("goal"):
            return AgentState.model_validate(raw)
        trigger = self.db.get(models.ConversationMessage, task.trigger_message_id)
        goal = trigger.content if trigger else "Responder al usuario"
        has_report = any(
            item.artifact_type == "report"
            for item in self.repo.list_artifacts(task.conversation_id)
            if item.message_id == task.trigger_message_id
        )
        return AgentState(
            goal=goal,
            requires_grounding=has_report or not bool(GENERAL_GUIDANCE_RE.search(goal)),
            sources=self.conversation_sources(task.conversation_id),
            evidence=self.conversation_evidence(task.conversation_id),
            clarification_response=raw.get("clarification_response"),
        )

    def conversation_sources(self, conversation_id: str) -> list[AgentSource]:
        citations = self.recent_citations(conversation_id)
        sources = {}
        for citation in citations:
            source_id = str(citation.get("source_id") or "")
            if not source_id or source_id in sources:
                continue
            sources[source_id] = AgentSource(
                id=source_id,
                title=str(citation.get("title") or citation.get("domain") or "Fuente previa"),
                url=str(citation.get("url") or ""),
                domain=str(citation.get("domain") or ""),
                source_type=str(citation.get("source_type") or SourceType.unknown.value),
                accessed_at=str(citation.get("accessed_at") or utc_now().isoformat()),
                inspected=True,
            )
        return list(sources.values())

    def conversation_evidence(self, conversation_id: str) -> list[AgentEvidence]:
        evidence = {}
        for citation in self.recent_citations(conversation_id):
            evidence_id = str(citation.get("evidence_id") or "")
            source_id = str(citation.get("source_id") or "")
            claim = str(citation.get("claim") or "")
            if evidence_id and source_id and claim and evidence_id not in evidence:
                evidence[evidence_id] = AgentEvidence(
                    id=evidence_id,
                    source_id=source_id,
                    topic=EvidenceTopic.general.value,
                    claim=claim,
                    excerpt=claim,
                    confidence=ConfidenceLevel.medium.value,
                )
        return list(evidence.values())

    def recent_citations(self, conversation_id: str) -> list[dict]:
        citations = []
        for message in self.repo.list_messages(conversation_id)[-12:]:
            citations.extend(json.loads(message.citations_json or "[]"))
        return citations[-20:]

    @staticmethod
    def requires_clarification(state: AgentState) -> bool:
        if state.clarification_response or state.force_finalize:
            return False
        goal = " ".join(state.goal.strip().split())
        if SMALL_TALK_RE.fullmatch(goal):
            return False
        if WEATHER_RE.search(goal) and not re.search(r"\b(en|para)\s+\w+", goal, re.IGNORECASE):
            return True
        return len(re.findall(r"\w+", goal, re.UNICODE)) <= 2

    def save_state(self, task: models.TaskRun, state: AgentState) -> None:
        task.working_state_json = state.model_dump_json()
        task.updated_at = utc_now()
        self.db.flush()

    def persist_usage(self, task: models.TaskRun, usage: dict, started: float) -> None:
        usage["elapsed_ms"] = self.run_elapsed_base + int((time.monotonic() - started) * 1000)
        task.usage_json = json.dumps(usage, ensure_ascii=False)
        self.db.flush()

    @staticmethod
    def add_provider_usage(usage: dict, provider_usage: dict[str, int]) -> None:
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cache_hit_tokens",
            "cache_miss_tokens",
        ):
            usage[key] = int(usage.get(key) or 0) + int(provider_usage.get(key) or 0)

    def refresh_task(self, task: models.TaskRun) -> None:
        self.db.refresh(task)

    def enforce_budget(
        self, task: models.TaskRun, state: AgentState, usage: dict, started: float
    ) -> None:
        if state.force_finalize:
            return
        budget = safe_json_dict(task.budget_json)
        elapsed = self.run_elapsed_base + int((time.monotonic() - started) * 1000)
        exhausted = (
            usage["model_turns"] >= int(budget.get("model_turns", 4))
            or elapsed >= int(budget.get("elapsed_seconds", 180)) * 1000
        )
        if not exhausted:
            return
        if state.evidence or not state.requires_grounding:
            state.force_finalize = True
            self.save_state(task, state)
            self.db.commit()
            return
        if budget.get("extension_used"):
            state.force_finalize = True
            self.save_state(task, state)
            self.db.commit()
            return
        self.pause(
            task,
            "awaiting_approval",
            "approval.required",
            {
                "type": "budget_extension",
                "prompt": "Se alcanzó el presupuesto estándar. ¿Querés ampliar la investigación una vez?",
                "options": [
                    {"id": "approved", "label": "Continuar"},
                    {"id": "rejected", "label": "Finalizar con lo disponible"},
                ],
            },
        )

    def enforce_tool_budget(
        self, task: models.TaskRun, state: AgentState, usage: dict, counter: str
    ) -> None:
        budget = safe_json_dict(task.budget_json)
        if int(usage.get(counter, 0)) < int(budget.get(counter, 0)):
            return
        if budget.get("extension_used"):
            state.force_finalize = True
            raise ValueError("Se agotó el presupuesto de esta herramienta; finalizá con lo disponible.")
        self.pause(
            task,
            "awaiting_approval",
            "approval.required",
            {
                "type": "budget_extension",
                "prompt": "Se alcanzó el presupuesto estándar. ¿Querés ampliar la investigación una vez?",
                "options": [
                    {"id": "approved", "label": "Continuar"},
                    {"id": "rejected", "label": "Finalizar con lo disponible"},
                ],
            },
        )

    def build_context(self, task: models.TaskRun, state: AgentState) -> str:
        conversation = self.repo.get(task.conversation_id)
        messages = self.repo.list_messages(task.conversation_id)[-12:]
        artifacts = self.repo.list_artifacts(task.conversation_id)
        context = {
            "conversation_summary": conversation.summary if conversation else None,
            "messages": [{"role": item.role, "content": item.content[:3000]} for item in messages],
            "active_context": safe_json_dict(conversation.active_context_json) if conversation else {},
            "artifacts": [
                {"type": item.artifact_type, "id": item.artifact_id} for item in artifacts[-12:]
            ],
            "task": state.model_dump(mode="json"),
            "budget": safe_json_dict(task.budget_json),
            "usage": safe_json_dict(task.usage_json),
        }
        return "Contexto de la tarea:\n" + json.dumps(context, ensure_ascii=False)[:24_000]

    def execute_tool(
        self,
        task: models.TaskRun,
        state: AgentState,
        usage: dict,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        handlers = {
            "respond": self.tool_respond,
            "request_clarification": self.tool_request_clarification,
            "search_web": self.tool_search_web,
            "inspect_page": self.tool_inspect_page,
            "review_evidence": self.tool_review_evidence,
            "retrieve_reports": self.tool_retrieve_reports,
            "compare_reports": self.tool_compare_reports,
            "finish_research": self.tool_finish_research,
        }
        handler = handlers.get(name)
        if not handler:
            return ToolResult({"ok": False, "error": "Herramienta no permitida."}, [])
        try:
            return handler(task, state, usage, arguments)
        except (AgentPaused, AgentCompleted):
            raise
        except (ValidationError, ValueError) as exc:
            return ToolResult({"ok": False, "error": str(exc)[:500]}, [])

    def tool_respond(self, task, state, usage, arguments) -> ToolResult:
        if state.force_finalize and state.requires_grounding and not state.evidence:
            arguments = AgentAnswer(
                answer=(
                    "No pude verificar la información solicitada porque la investigación "
                    "terminó antes de inspeccionar y validar las fuentes encontradas."
                ),
                answer_type="grounded",
                warnings=[
                    "Hay fuentes potenciales, pero no se validó evidencia suficiente para responder."
                ],
            ).model_dump(mode="json")
        elif not str(arguments.get("answer") or "").strip():
            arguments = self.generate_final_answer(
                task,
                state,
                usage,
                str(arguments.get("answer_type") or ("grounded" if state.requires_grounding else "guidance")),
            ).model_dump(mode="json")
        answer = AgentAnswer.model_validate(arguments)
        evidence_by_id = {item.id: item for item in state.evidence}
        cited_ids = [evidence_id for claim in answer.claims for evidence_id in claim.evidence_ids]
        invalid_ids = sorted(set(cited_ids) - set(evidence_by_id))
        if invalid_ids:
            raise ValueError(f"IDs de evidencia no disponibles: {', '.join(invalid_ids)}")
        if state.requires_grounding and answer.answer_type != "grounded":
            raise ValueError("Esta solicitud requiere una respuesta grounded con evidencia.")
        if answer.answer_type == "grounded" and (not answer.claims or not cited_ids):
            limited = bool(answer.warnings) and bool(
                re.search(r"\b(no hay evidencia|no pude verificar|sin evidencia)\b", answer.answer, re.I)
            )
            if not limited:
                raise ValueError("La respuesta grounded requiere afirmaciones con evidencia.")
        if CURRENT_OPPORTUNITY_RE.search(state.goal):
            cited_source_ids = {evidence_by_id[evidence_id].source_id for evidence_id in cited_ids}
            has_official_source = any(
                source.id in cited_source_ids
                and source.source_type in {SourceType.official.value, SourceType.career_page.value}
                for source in state.sources
            )
            caveat = re.search(
                r"\b(no pude confirmar|sin confirmaci[oó]n oficial|no hay confirmaci[oó]n oficial)\b",
                answer.answer,
                re.IGNORECASE,
            )
            if not has_official_source:
                if not caveat:
                    answer.answer = (
                        "No pude confirmar esta información en una fuente oficial de empleos. "
                        + answer.answer
                    )
                if not answer.warnings:
                    answer.warnings.append("La oportunidad solo cuenta con evidencia secundaria.")
        citations = self.build_citations(state, cited_ids)
        conversation = self.repo.get(task.conversation_id)
        if not conversation:
            raise AgentFailure("La conversación ya no existe.")
        message = self.repo.add_message(
            conversation,
            role="assistant",
            content="",
            status="streaming",
            citations=citations,
        )
        state.response_message_id = message.id
        self.save_state(task, state)
        self.repo.add_event(task, "message.started", {"message": self.message_payload(message)})
        self.db.commit()
        accumulated = ""
        for chunk in text_chunks(answer.answer, 240):
            accumulated += chunk
            self.repo.update_message_content(message, accumulated)
            self.repo.add_event(task, "message.delta", {"message_id": message.id, "delta": chunk})
            self.db.commit()
        self.repo.complete_message(message, answer.answer)
        self.repo.add_event(task, "message.completed", {"message": self.message_payload(message)})
        self.repo.add_event(
            task,
            "tool.completed",
            {"task_run_id": task.id, "label": self.tool_label("respond"), "status": "completed"},
            tool_name="respond",
        )
        self.repo.update_task(task, "completed")
        self.repo.add_event(task, "task.completed", {"task_run_id": task.id, "status": "completed"})
        self.db.commit()
        self.summarize_if_needed(conversation)
        raise AgentCompleted

    def generate_final_answer(
        self, task: models.TaskRun, state: AgentState, usage: dict, answer_type: str
    ) -> AgentAnswer:
        prompt = (
            f"Redactá ahora la respuesta final en español. El tipo requerido es {answer_type}. "
            "Para una respuesta grounded, cada afirmación factual debe aparecer en claims y citar "
            "solo IDs presentes en el contexto. No inventes IDs. La respuesta visible debe ser útil "
            "y coherente con esos claims. Para pasantías o vacantes actuales sin una fuente oficial, "
            "decí explícitamente que no pudiste confirmarlas oficialmente y agregá una advertencia.\n\n"
            + self.build_context(task, state)
        )
        client = self.get_client()
        payload = client.generate_json(
            SYSTEM_PROMPT,
            prompt[:24_000],
            AgentAnswer,
            max_tokens=4000,
            temperature=0.1,
        )
        self.add_provider_usage(usage, getattr(client, "last_usage", {}))
        answer = AgentAnswer.model_validate(payload)
        if answer.answer_type != answer_type:
            raise ValueError("DeepSeek devolvió un tipo de respuesta distinto del solicitado.")
        return answer

    def tool_request_clarification(self, task, state, usage, arguments) -> ToolResult:
        question = str(arguments.get("question") or "").strip()
        if not question:
            raise ValueError("La aclaración requiere una pregunta.")
        options = arguments.get("options") or []
        options = [
            {"id": str(item.get("id", ""))[:80], "label": str(item.get("label", ""))[:160]}
            for item in options[:4]
            if isinstance(item, dict) and item.get("id") and item.get("label")
        ]
        conversation = self.repo.get(task.conversation_id)
        if conversation:
            self.repo.add_message(conversation, role="assistant", content=question, status="completed")
        self.save_state(task, state)
        self.repo.add_event(
            task,
            "tool.completed",
            {"task_run_id": task.id, "label": self.tool_label("request_clarification"), "status": "completed"},
            tool_name="request_clarification",
        )
        self.pause(
            task,
            "needs_clarification",
            "clarification.required",
            {"type": "clarification", "prompt": question, "options": options},
        )

    def tool_search_web(self, task, state, usage, arguments) -> ToolResult:
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ValueError("La búsqueda requiere una consulta.")
        topic = EvidenceTopic(str(arguments.get("topic") or EvidenceTopic.general.value))
        limit = min(5, max(1, int(arguments.get("limit") or 3)))
        self.enforce_tool_budget(task, state, usage, "searches")
        usage["searches"] += 1
        normalized = " ".join(query.lower().split())
        if normalized in state.searched_queries:
            return ToolResult({"ok": True, "summary": "La consulta ya había sido ejecutada.", "sources": []}, [])
        state.searched_queries.append(normalized)
        results = self.get_search_provider().search(query, limit)
        existing = {item.id for item in state.sources}
        added = []
        for result in results:
            if urlparse(result.url).scheme not in {"http", "https"}:
                continue
            normalized_result = SearchResult(
                title=result.title,
                url=result.url,
                snippet=result.snippet,
                rank=result.rank,
                query_topic=topic,
            )
            scored = score_search_result("", normalized_result)
            if scored.source_id in existing:
                continue
            source = AgentSource(
                id=scored.source_id,
                title=scored.title,
                url=scored.url,
                snippet=scored.snippet,
                domain=scored.domain,
                source_type=scored.source_type.value,
                reliability_score=scored.reliability_score,
                rank=result.rank,
                topic=topic.value,
                accessed_at=utc_now().isoformat(),
            )
            state.sources.append(source)
            existing.add(source.id)
            added.append(source)
        return ToolResult(
            {
                "ok": True,
                "summary": f"Se encontraron {len(added)} fuentes nuevas.",
                "sources": [item.model_dump(mode="json") for item in added],
            },
            [],
        )

    def tool_inspect_page(self, task, state, usage, arguments) -> ToolResult:
        source_id = str(arguments.get("source_id") or "")
        source = next((item for item in state.sources if item.id == source_id), None)
        if not source:
            raise ValueError("Solo se pueden inspeccionar fuentes devueltas por search_web.")
        self.enforce_tool_budget(task, state, usage, "inspections")
        usage["inspections"] += 1
        if source_id in state.inspected_source_ids:
            return ToolResult({"ok": True, "summary": "La fuente ya había sido inspeccionada."}, [])
        content = self.get_extractor().extract(source.url)
        if content is None:
            source.inspected = True
            state.inspected_source_ids.append(source.id)
            return ToolResult(
                {
                    "ok": False,
                    "summary": "No se pudo extraer contenido util de la fuente.",
                    "source_id": source.id,
                    "source_type": source.source_type,
                },
                [],
            )
        scored = ScoredSource(
            source_id=source.id,
            title=source.title,
            url=source.url,
            domain=source.domain,
            source_type=SourceType(source.source_type),
            reliability_score=source.reliability_score,
            snippet=source.snippet,
            is_current=True,
        )
        classified = classify_evidence(scored, content)
        known = {item.id for item in state.evidence}
        added = []
        for item in classified:
            evidence_id = evidence_identifier(source.id, item.topic.value, item.raw_text_excerpt)
            if evidence_id in known:
                continue
            evidence = AgentEvidence(
                id=evidence_id,
                source_id=source.id,
                topic=item.topic.value,
                claim=item.claim,
                excerpt=item.raw_text_excerpt[:1200],
                confidence=item.confidence.value,
            )
            state.evidence.append(evidence)
            known.add(evidence.id)
            added.append(evidence)
        source.inspected = True
        state.inspected_source_ids.append(source.id)
        return ToolResult(
            {
                "ok": True,
                "summary": f"La fuente aportó {len(added)} evidencias.",
                "evidence": [item.model_dump(mode="json") for item in added],
            },
            [],
        )

    def tool_review_evidence(self, task, state, usage, arguments) -> ToolResult:
        coverage = {topic.value: 0 for topic in EvidenceTopic}
        for item in state.evidence:
            coverage[item.topic] = coverage.get(item.topic, 0) + 1
        missing = [topic for topic, count in coverage.items() if count == 0]
        state.unresolved_topics = missing
        return ToolResult(
            {
                "ok": True,
                "summary": f"Hay evidencia en {sum(1 for count in coverage.values() if count)} temas.",
                "coverage": coverage,
                "missing_topics": missing,
            },
            [],
        )

    def tool_retrieve_reports(self, task, state, usage, arguments) -> ToolResult:
        company_name = str(arguments.get("company_name") or "").strip() or None
        query = str(arguments.get("query") or "").strip().lower()
        limit = min(5, max(1, int(arguments.get("limit") or 3)))
        requested_ids = [str(item) for item in arguments.get("report_ids") or []]
        reports = self.report_repo.list_reports(limit=50, company_name=company_name)
        report_by_id = {report.id: report for report in reports if report.status == ReportStatus.completed.value}
        if requested_ids:
            for report_id in requested_ids:
                report = self.report_repo.get_by_id(report_id)
                if report and report.status == ReportStatus.completed.value:
                    report_by_id[report.id] = report
            report_by_id = {
                report_id: report_by_id[report_id]
                for report_id in requested_ids
                if report_id in report_by_id
            }
        matched_chunks = []
        chunks = [
            chunk
            for chunk in self.report_repo.list_embedding_chunks()
            if chunk.report_id in report_by_id
        ]
        if query and chunks:
            vector = self.get_embedding_service().embed([query])[0]
            matched_chunks = rank_chunks(vector, chunks, limit * 2, None, "all_reports")
        ordered_ids = list(dict.fromkeys(chunk.report_id for chunk in matched_chunks))
        lexical = []
        for report in report_by_id.values():
            if report.status != ReportStatus.completed.value:
                continue
            haystack = f"{report.company.name} {report.summary or ''}".lower()
            score = sum(1 for term in query.split() if len(term) > 2 and term in haystack)
            lexical.append((score, report))
        lexical.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
        ordered_ids.extend(
            report.id for _, report in lexical if report.id not in ordered_ids
        )
        selected = [report_by_id[report_id] for report_id in ordered_ids[:limit]]
        summaries = []
        for report in selected:
            self.import_report_evidence(state, report)
            if report.id not in state.retrieved_report_ids:
                state.retrieved_report_ids.append(report.id)
            summaries.append(
                {
                    "report_id": report.id,
                    "company": report.company.name,
                    "summary": report.summary,
                    "generated_at": report.generated_at.isoformat() if report.generated_at else None,
                    "fresh": bool(
                        report.valid_until
                        and report.valid_until
                        >= utc_now().replace(tzinfo=report.valid_until.tzinfo)
                    ),
                }
            )
        return ToolResult(
            {
                "ok": True,
                "summary": f"Se recuperaron {len(selected)} informes.",
                "reports": summaries,
                "relevant_chunks": [
                    {
                        "report_id": chunk.report_id,
                        "title": chunk.chunk_title,
                        "evidence_ids": chunk.evidence_ids,
                    }
                    for chunk in matched_chunks[: limit * 2]
                ],
            },
            [],
        )

    def tool_compare_reports(self, task, state, usage, arguments) -> ToolResult:
        report_ids = list(dict.fromkeys(str(item) for item in arguments.get("report_ids") or []))
        if not 2 <= len(report_ids) <= 4:
            raise ValueError("La comparación requiere entre 2 y 4 informes.")
        dimensions = [str(item).strip() for item in arguments.get("dimensions") or [] if str(item).strip()]
        if not dimensions:
            dimensions = ["business", "argentina_presence", "salary_benefits", "culture", "open_roles"]
        reports = [self.report_repo.get_by_id(report_id) for report_id in report_ids]
        if any(report is None or report.status != ReportStatus.completed.value for report in reports):
            raise ValueError("Todos los informes deben existir y estar completados.")
        structured = [self.report_repo.to_structured_report(report) for report in reports if report]
        rows = []
        citations = []
        for dimension in dimensions:
            values = []
            for report in structured:
                section = next(
                    (
                        item
                        for item in report.sections
                        if item.type.value == dimension or dimension.lower() in item.title.lower()
                    ),
                    None,
                )
                values.append(
                    {
                        "company": report.company.name,
                        "value": section.summary if section else "Sin evidencia disponible.",
                        "confidence": section.confidence.value if section else "unknown",
                        "missing_evidence": not section or section.missing_evidence,
                    }
                )
            rows.append({"dimension": dimension, "values": values})
        for report in structured:
            citations.extend(
                {
                    "report_id": report.report_id,
                    "source_id": source.id,
                    "title": source.title,
                    "url": str(source.url),
                    "domain": source.domain,
                }
                for source in report.sources[:8]
            )
        payload = ComparisonPayload(
            companies=[{"report_id": report.report_id, "name": report.company.name} for report in structured],
            rows=rows,
        )
        conversation = self.repo.get(task.conversation_id)
        if not conversation:
            raise AgentFailure("La conversación ya no existe.")
        comparison = self.repo.create_comparison(
            conversation,
            title="Comparación: " + " vs. ".join(report.company.name for report in structured),
            report_ids=report_ids,
            dimensions=dimensions,
            payload=payload.model_dump(mode="json"),
            citations=citations,
            warnings=["Las dimensiones sin evidencia se muestran explícitamente."],
        )
        self.repo.add_artifact_link(
            conversation,
            artifact_type="comparison",
            artifact_id=comparison.id,
            relationship_type="generated",
        )
        state.generated_comparison_id = comparison.id
        self.update_active_context(conversation, report_ids=report_ids)
        refs = [{"type": "comparison", "artifact_id": comparison.id}]
        self.repo.add_event(
            task,
            "artifact.created",
            {"task_run_id": task.id, "type": "comparison", "artifact_id": comparison.id},
            artifact_refs=refs,
        )
        return ToolResult(
            {"ok": True, "summary": "La comparación quedó guardada.", "comparison_id": comparison.id},
            refs,
        )

    def tool_finish_research(self, task, state, usage, arguments) -> ToolResult:
        output_type = str(arguments.get("output_type") or "answer")
        state.unresolved_topics = [str(item) for item in arguments.get("unresolved_topics") or []]
        if output_type != "report":
            return ToolResult(
                {
                    "ok": True,
                    "summary": "La investigación puede finalizar con una respuesta grounded.",
                    "evidence_count": len(state.evidence),
                },
                [],
            )
        company_name = str(arguments.get("company_name") or "").strip()
        if not company_name or not state.sources or not state.evidence:
            raise ValueError("Para crear un informe se requiere empresa, fuentes y evidencia.")
        conversation = self.repo.get(task.conversation_id)
        if not conversation:
            raise AgentFailure("La conversación ya no existe.")
        company = CompanyRepository(self.db).get_or_create(company_name)
        report = self.report_repo.create_report(company.id)
        self.report_repo.update_status(report, ReportStatus.running)
        self.db.commit()
        try:
            synthesizer = DeepSeekReportSynthesizer(
                api_key=self.settings.deepseek_api_key or "",
                model=self.settings.deepseek_model,
                client=self.get_client(),
                timeout_seconds=self.settings.deepseek_timeout_seconds,
            )
            structured = synthesizer.synthesize(
                SynthesisRequest(
                    report_id=report.id,
                    company_id=company.id,
                    company_name=company.name,
                    normalized_company_name=company.normalized_name,
                    sources=[self.to_scored_source(item) for item in state.sources],
                    evidence=[self.to_classified_evidence(item) for item in state.evidence],
                    include_cv=False,
                    include_cv_tailoring=False,
                    include_adapted_cv_draft=False,
                    cv_text=None,
                )
            )
            self.report_repo.save_structured_report(report, structured)
        except Exception as exc:
            self.db.rollback()
            report = self.report_repo.get_by_id(report.id)
            if report:
                self.report_repo.update_status(report, ReportStatus.failed, str(exc)[:500])
                self.db.commit()
            raise AgentFailure("DeepSeek no pudo generar un informe grounded válido.") from exc
        self.repo.add_artifact_link(
            conversation,
            artifact_type="report",
            artifact_id=report.id,
            relationship_type="generated",
        )
        state.generated_report_id = report.id
        self.update_active_context(conversation, report_ids=[report.id], company_ids=[company.id])
        refs = [{"type": "report", "artifact_id": report.id}]
        self.repo.add_event(
            task,
            "artifact.created",
            {"task_run_id": task.id, "type": "report", "artifact_id": report.id},
            artifact_refs=refs,
        )
        self.db.commit()
        try:
            schedule_report_embedding_task(report.id)
        except Exception:
            pass
        return ToolResult(
            {"ok": True, "summary": "El informe grounded quedó guardado.", "report_id": report.id},
            refs,
        )

    def import_report_evidence(self, state: AgentState, report: models.Report) -> None:
        structured = self.report_repo.to_structured_report(report)
        known_sources = {item.id for item in state.sources}
        known_evidence = {item.id for item in state.evidence}
        for source in structured.sources:
            source_id = f"{report.id}:{source.id}"
            if source_id not in known_sources:
                state.sources.append(
                    AgentSource(
                        id=source_id,
                        title=source.title,
                        url=str(source.url),
                        snippet=source.snippet or "",
                        domain=source.domain,
                        source_type=source.source_type.value,
                        reliability_score=source.reliability_score,
                        rank=1,
                        topic=EvidenceTopic.general.value,
                        accessed_at=source.accessed_at,
                        inspected=True,
                    )
                )
        for evidence in structured.evidence:
            evidence_id = f"{report.id}:{evidence.id}"
            if evidence_id not in known_evidence:
                state.evidence.append(
                    AgentEvidence(
                        id=evidence_id,
                        source_id=f"{report.id}:{evidence.source_id}",
                        topic=evidence.topic.value,
                        claim=evidence.claim,
                        excerpt=evidence.raw_text_excerpt or "",
                        confidence=evidence.confidence.value,
                    )
                )

    def build_citations(self, state: AgentState, evidence_ids: list[str]) -> list[dict]:
        sources = {item.id: item for item in state.sources}
        evidence = {item.id: item for item in state.evidence}
        citations = []
        cited_source_ids = set()
        for evidence_id in dict.fromkeys(evidence_ids):
            item = evidence[evidence_id]
            source = sources.get(item.source_id)
            if not source or source.id in cited_source_ids:
                continue
            cited_source_ids.add(source.id)
            citations.append(
                {
                    "evidence_id": item.id,
                    "source_id": source.id,
                    "title": source.title,
                    "url": source.url,
                    "domain": source.domain,
                    "source_type": source.source_type,
                    "claim": item.claim,
                    "accessed_at": source.accessed_at,
                }
            )
        return citations

    def pause(self, task, status, event_type, payload) -> None:
        task.pause_reason_json = json.dumps(payload, ensure_ascii=False)
        self.repo.update_task(task, status)
        self.repo.add_event(task, event_type, {"task_run_id": task.id, **payload})
        self.db.commit()
        raise AgentPaused

    def fail_task(self, task_id: str, error: Exception) -> None:
        self.db.rollback()
        task = self.repo.get_task(task_id)
        if not task or task.status in {"completed", "cancelled", "failed"}:
            return
        pause, stopping_reason = self.public_error(error)
        message = pause["message"]
        task.pause_reason_json = json.dumps(pause, ensure_ascii=False)
        self.repo.update_task(task, "failed", stopping_reason)
        self.repo.add_event(
            task,
            "task.failed",
            {"task_run_id": task.id, "status": "failed", "message": message[:500]},
        )
        self.db.commit()

    @staticmethod
    def public_error(error: Exception) -> tuple[dict[str, Any], str]:
        raw = str(error).strip()
        status_code = getattr(error, "status_code", None)
        if status_code == 401:
            return (
                {
                    "type": "provider_authentication",
                    "message": "La clave de DeepSeek no es válida. Revisá DEEPSEEK_API_KEY.",
                },
                "provider_authentication_failed",
            )
        if status_code == 402:
            return (
                {
                    "type": "insufficient_balance",
                    "message": "La cuenta de DeepSeek no tiene saldo suficiente. Revisá Billing en DeepSeek.",
                },
                "provider_insufficient_balance",
            )
        if status_code == 429:
            retry_after = str(getattr(error, "retry_after", "") or "")
            seconds = max(1, int(float(retry_after))) if re.fullmatch(r"[0-9.]+", retry_after) else 60
            retry_at = (utc_now() + timedelta(seconds=seconds)).isoformat()
            return (
                {
                    "type": "rate_limit",
                    "message": "DeepSeek alcanzó temporalmente su límite de solicitudes.",
                    "retry_at": retry_at,
                },
                "provider_rate_limited",
            )
        if isinstance(error, DeepSeekAPIError) or status_code in {500, 503}:
            retry_at = (utc_now() + timedelta(seconds=15)).isoformat()
            return (
                {
                    "type": "provider_unavailable",
                    "message": "DeepSeek está temporalmente no disponible. Reintentá en unos momentos.",
                    "retry_at": retry_at,
                },
                "provider_unavailable",
            )
        if isinstance(error, AgentFailure):
            return ({"type": "error", "message": raw[:500]}, "agent_failed")
        return (
            {
                "type": "error",
                "message": "No se pudo completar la tarea del agente. Reintentá la solicitud.",
            },
            "agent_failed",
        )

    def summarize_if_needed(self, conversation: models.Conversation) -> None:
        messages = self.repo.list_messages(conversation.id)
        if len(messages) <= 20:
            return
        older = messages[:-12]
        prompt = "Resumí en español estos mensajes en menos de 4000 caracteres, sin razonamiento privado:\n" + json.dumps(
            [{"role": item.role, "content": item.content} for item in older], ensure_ascii=False
        )
        try:
            response = self.get_client().generate_text(
                SYSTEM_PROMPT,
                prompt[:24_000],
                max_tokens=1200,
                temperature=0.1,
            )
            summary = response.content.strip()
            if summary:
                self.repo.update_summary(conversation, summary)
                self.db.commit()
        except Exception:
            self.db.rollback()

    def update_active_context(
        self,
        conversation: models.Conversation,
        *,
        report_ids: list[str] | None = None,
        company_ids: list[str] | None = None,
    ) -> None:
        context = safe_json_dict(conversation.active_context_json)
        if report_ids:
            context["active_report_ids"] = list(dict.fromkeys(report_ids))
        if company_ids:
            context["active_company_ids"] = list(dict.fromkeys(company_ids))
        self.repo.update_active_context(conversation, context)

    def to_scored_source(self, item: AgentSource) -> ScoredSource:
        return ScoredSource(
            source_id=item.id,
            title=item.title,
            url=item.url,
            domain=item.domain,
            source_type=SourceType(item.source_type),
            reliability_score=item.reliability_score,
            snippet=item.snippet,
            is_current=True,
        )

    def to_classified_evidence(self, item: AgentEvidence) -> ClassifiedEvidence:
        return ClassifiedEvidence(
            source_id=item.source_id,
            topic=EvidenceTopic(item.topic),
            claim=item.claim,
            raw_text_excerpt=item.excerpt,
            confidence=ConfidenceLevel(item.confidence),
        )

    def message_payload(self, message: models.ConversationMessage) -> dict:
        return {
            "message_id": message.id,
            "role": message.role,
            "content": message.content,
            "status": message.status,
            "citations": json.loads(message.citations_json or "[]"),
            "created_at": message.created_at.isoformat(),
            "completed_at": message.completed_at.isoformat() if message.completed_at else None,
        }

    @staticmethod
    def tool_label(name: str) -> str:
        return {
            "respond": "Preparando respuesta",
            "request_clarification": "Solicitando aclaración",
            "search_web": "Buscando fuentes públicas",
            "inspect_page": "Inspeccionando una fuente",
            "review_evidence": "Revisando cobertura de evidencia",
            "retrieve_reports": "Recuperando informes guardados",
            "compare_reports": "Comparando informes",
            "finish_research": "Finalizando investigación",
        }.get(name, "Ejecutando acción segura")

    @staticmethod
    def safe_arguments(name: str, arguments: dict) -> dict:
        allowed = {
            "search_web": {"query", "topic", "limit"},
            "inspect_page": {"source_id"},
            "retrieve_reports": {"company_name", "query", "limit", "report_ids"},
            "compare_reports": {"report_ids", "dimensions"},
            "finish_research": {"company_name", "output_type", "stopping_reason", "unresolved_topics"},
            "request_clarification": {"question", "options"},
            "respond": {"answer_type", "warnings"},
        }.get(name, set())
        return {key: value for key, value in arguments.items() if key in allowed}

    @staticmethod
    def function_tools(
        final_only: bool = False,
        clarification_only: bool = False,
        allow_clarification: bool = True,
    ) -> list[dict[str, Any]]:
        names = (
            ["respond"]
            if final_only
            else ["request_clarification"]
            if clarification_only
            else list(FUNCTION_SCHEMAS)
        )
        if not allow_clarification:
            names = [name for name in names if name != "request_clarification"]
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": FUNCTION_SCHEMAS[name]["description"],
                    "parameters": FUNCTION_SCHEMAS[name]["parameters"],
                },
            }
            for name in names
        ]


FUNCTION_SCHEMAS = {
    "respond": {
        "description": "Selecciona la finalización. El backend generará y validará la respuesta completa antes de mostrarla.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer_type": {"type": "string", "enum": ["guidance", "grounded"]},
            },
            "required": ["answer_type"],
        },
    },
    "request_clarification": {
        "description": "Pausa cuando falta un dato esencial.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "options": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}, "label": {"type": "string"}},
                        "required": ["id", "label"],
                    },
                },
            },
            "required": ["question"],
        },
    },
    "search_web": {
        "description": "Busca fuentes públicas con Tavily.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "topic": {"type": "string", "enum": [item.value for item in EvidenceTopic]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["query", "topic"],
        },
    },
    "inspect_page": {
        "description": "Extrae evidencia de una fuente ya obtenida.",
        "parameters": {
            "type": "object",
            "properties": {"source_id": {"type": "string"}},
            "required": ["source_id"],
        },
    },
    "review_evidence": {
        "description": "Revisa cobertura y temas faltantes.",
        "parameters": {"type": "object", "properties": {}},
    },
    "retrieve_reports": {
        "description": "Recupera informes históricos y su evidencia.",
        "parameters": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string"},
                "query": {"type": "string"},
                "report_ids": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer", "minimum": 1, "maximum": 5},
            },
        },
    },
    "compare_reports": {
        "description": "Crea un artefacto comparativo entre informes completados.",
        "parameters": {
            "type": "object",
            "properties": {
                "report_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 4},
                "dimensions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["report_ids"],
        },
    },
    "finish_research": {
        "description": "Finaliza la investigación y opcionalmente crea un informe.",
        "parameters": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string"},
                "output_type": {"type": "string", "enum": ["answer", "report"]},
                "stopping_reason": {"type": "string"},
                "unresolved_topics": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["output_type", "stopping_reason"],
        },
    },
}


def evidence_identifier(source_id: str, topic: str, excerpt: str) -> str:
    digest = hashlib.sha256(f"{source_id}|{topic}|{excerpt}".encode("utf-8")).hexdigest()[:16]
    return f"evidence_{digest}"


def text_chunks(value: str, size: int) -> list[str]:
    return [value[index : index + size] for index in range(0, len(value), size)] or [""]


def run_agent_task(task_id: str, bind: Engine) -> None:
    task_session = sessionmaker(bind=bind, autoflush=False, autocommit=False)
    with task_session() as db:
        AgentOrchestrator(db).run(task_id)
