from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal
from urllib.parse import urlparse
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.config import Settings, get_settings
from backend.app.core.time import utc_now
from backend.app.db import models
from backend.app.db.conversation_repository import ConversationRepository
from backend.app.db.repositories import CompanyRepository, ReportRepository, safe_json_dict
from backend.app.domain.companies import normalize_company_name
from backend.app.domain.cv import extract_candidate_signals, select_cv_evidence_lines
from backend.app.domain.jobs import extract_job_signals, job_signal_payload, select_job_evidence_lines
from backend.app.domain.reports import (
    ConfidenceLevel,
    EvidenceTopic,
    ReportStatus,
    SectionSchema,
    SectionType,
    SourceType,
    WarningSchema,
    WarningSeverity,
)
from backend.app.llm.cv_tailoring import DeepSeekCVTailoringService
from backend.app.llm.deepseek import DeepSeekAPIError, DeepSeekClient, DeepSeekReportSynthesizer
from backend.app.llm.synthesizer import SynthesisError, SynthesisRequest
from backend.app.research.content_extractor import HttpContentExtractor
from backend.app.research.evidence_classifier import classify_evidence
from backend.app.research.scoring import company_domain_tokens, score_search_result, stable_source_id
from backend.app.research.search_provider import FakeSearchProvider, TavilySearchProvider
from backend.app.research.types import ClassifiedEvidence, ExtractedContent, ScoredSource, SearchResult
from backend.app.services.rag_indexing import schedule_report_embedding_task
from backend.app.services.rag import GoogleEmbeddingService, rank_chunks
from backend.app.services.cv_library import CVLibraryService
from backend.app.services.report_quality import report_completion_issues


SYSTEM_PROMPT = """Sos Radar Laboral, un asistente laboral conversacional en español.
Elegí una sola función por turno. Usá respond para terminar y request_clarification si falta un dato esencial.
Un mensaje aislado o ambiguo como "google" requiere aclaración. Una consulta meteorológica sin ubicación
también requiere aclaración. Para datos actuales, combiná fuentes oficiales con fuentes laborales relevantes.
Después de recibir una aclaración, avanzá con la mejor interpretación disponible sin volver a preguntar.
Para presencia, cantidad de empleados y puestos, considerá LinkedIn como fuente laboral principal. Para
sueldos, cultura y entrevistas, considerá Glassdoor. Para vacantes en Argentina, considerá LinkedIn Jobs,
Computrabajo, Bumeran, ZonaJobs, Indeed, Get on Board y Portal Empleo, además de career pages oficiales.
Si una vacante o pasantía actual no tiene evidencia de una career page o de un portal laboral reconocido,
indicá claramente que no pudo confirmarse en una fuente laboral confiable y agregá una advertencia.
La orientación general sobre entrevistas puede ser guidance sin citas. Datos factuales sobre empresas,
mercado, salarios, vacantes, beneficios, entrevistas en una empresa concreta o informes deben ser grounded
y citar IDs de evidencia disponibles.
No uses conocimiento previo del modelo como evidencia. No inventes fuentes ni datos. search_web busca,
inspect_page inspecciona solo resultados obtenidos, review_evidence
revisa cobertura, retrieve_reports recupera informes, compare_reports crea una comparación y finish_research
crea un informe durable. Si la evidencia disponible ya alcanza para el pedido, finalizá sin seguir buscando.
Si no podés verificar información esencial, usá respond para explicarlo con claridad, sin inventar datos.
respond es una acción terminal: nunca digas que vas a buscar, investigar o continuar después de responder.
Para un informe explícito, cuando haya evidencia validada usá finish_research, no respond.
Para completar un informe, necesitás evidencia verificable del resumen, negocio y presencia en Argentina.
Para vacantes, buscá una publicación concreta con empresa, rol y ubicación argentina. Si el servidor indica
searched_not_verified después de intentar las cinco fuentes laborales, creá el informe igualmente y dejá
la sección de vacantes sin evidencia, con la advertencia indicada; nunca afirmes que no existen puestos.
Para investigar un informe, priorizá una fuente corporativa, una fuente laboral actual y después
plataformas de reseñas solo para sueldos, cultura o entrevistas. No presentes datos de reseñas como
hechos oficiales de la empresa. Para conteos dinámicos de vacantes, indicá la plataforma y que se
consultaron al momento de la investigación.
En búsquedas sobre una empresa, enviá siempre company_name a search_web. Para vacantes, seguí las
recommended_queries del servidor: careers primero, luego LinkedIn Jobs y después Computrabajo, Bumeran
y ZonaJobs, una plataforma por consulta. Evitá combinar varias plataformas
en una misma consulta y no inspecciones resultados dirigidos claramente a otro país.
No describas razonamiento privado. Las herramientas de CV solo pueden usarse cuando el usuario pide
usar o adaptar su CV, o cuando adjunta un CV o una descripción de puesto. En ese caso están disponibles
get_cv_profile, analyze_job_description y prepare_cv_recommendations.
Escribí respuestas limpias en párrafos breves. No uses Markdown: evitá encabezados con #, negritas,
comillas de bloque, separadores, bloques de código, viñetas decorativas y tablas con barras verticales."""

GROUNDING_REQUIRED_RE = re.compile(
    r"\b(investig(?:á|a|ar|ue)|compar(?:á|a|ar|e)|busc(?:á|a|ar)\s+fuentes?|"
    r"fuentes?|citas?|informes?|report(?:es)?|mercado laboral|salarios?|sueldos?|vacantes?|"
    r"pasant[ií]as?|internships?|beneficios laborales|career page|empleo vigente|"
    r"puesto abierto|oportunidad laboral|trabajar en|como empleador|como empresa|"
    r"cantidad de empleados|facturaci[oó]n|(?:a\s+)?qu[eé]\s+se\s+dedica)\b|"
    r"\bentrevista\b.{0,80}\ben\s+(?!general\b)",
    re.IGNORECASE,
)

SMALL_TALK_RE = re.compile(r"^(hola|buenas|gracias|ok|okay|sí|si|no|chau)[!. ]*$", re.IGNORECASE)
WEATHER_RE = re.compile(r"\b(clima|tiempo|temperatura|pron[oó]stico)\b", re.IGNORECASE)
CURRENT_OPPORTUNITY_RE = re.compile(
    r"\b(pasant[ií]as?|internships?|vacantes?|puesto abierto|empleo vigente|oportunidad laboral)\b",
    re.IGNORECASE,
)
FUTURE_RESEARCH_RE = re.compile(
    r"\b(?:voy|vamos)\s+a\s+(?:buscar|investigar|consultar|revisar)|"
    r"\b(?:buscar[eé]|investigar[eé]|consultar[eé]|continuar[eé])\b",
    re.IGNORECASE,
)
FULL_REPORT_CORE_TOPICS = (
    EvidenceTopic.business.value,
    EvidenceTopic.argentina_presence.value,
    EvidenceTopic.open_roles.value,
)
CORPORATE_SOURCE_TYPES = {SourceType.official.value, SourceType.career_page.value}
EMPLOYMENT_SOURCE_TYPES = {
    SourceType.career_page.value,
    SourceType.linkedin.value,
    SourceType.job_board.value,
}
ARGENTINA_JOB_BOARD_DOMAINS = (
    "ar.computrabajo.com",
    "computrabajo.com.ar",
    "bumeran.com.ar",
    "zonajobs.com.ar",
    "ar.indeed.com",
    "getonbrd.com",
    "portalempleo.gob.ar",
    "buscojobs.com",
    "talent.com",
    "jooble.org",
)
EMPLOYMENT_SEARCH_SEQUENCE = (
    "careers",
    "linkedin",
    "computrabajo",
    "bumeran",
    "zonajobs",
)
EMPLOYMENT_SEARCH_QUERIES = {
    "careers": "{company} careers jobs Argentina",
    "linkedin": 'site:linkedin.com/jobs/view "{company}" Argentina',
    "computrabajo": 'site:ar.computrabajo.com "{company}" empleo Argentina',
    "bumeran": 'site:bumeran.com.ar "{company}" empleo Argentina',
    "zonajobs": 'site:zonajobs.com.ar "{company}" empleo Argentina',
}
ARGENTINA_LOCATION_RE = re.compile(
    r"\b(argentina|buenos aires|caba|c[oó]rdoba|rosario|mendoza)\b",
    re.IGNORECASE,
)
JOB_DETAIL_URL_RE = re.compile(
    r"/(?:jobs?|empleos?|vacantes?|ofertas?(?:-de-trabajo)?)/(?:view/)?[^/?#]+",
    re.IGNORECASE,
)
GENERIC_JOB_PAGE_RE = re.compile(
    r"^(?:our\s+)?careers?$|^jobs?$|^.+:\s*jobs$|^join\s+our\s+team$|"
    r"^trabajar\s+en\s+.+|^empleos(?:\s+en\s+.+)?$",
    re.IGNORECASE,
)
OPEN_ROLES_LIMITATION = (
    "No se encontraron vacantes actuales verificables en Argentina en las fuentes consultadas; "
    "esto no significa que no existan."
)
LOCATION_SENSITIVE_TOPICS = {
    EvidenceTopic.salary.value,
    EvidenceTopic.benefits.value,
    EvidenceTopic.culture.value,
    EvidenceTopic.interview_process.value,
    EvidenceTopic.interview_questions.value,
    EvidenceTopic.open_roles.value,
}
FOREIGN_LOCATION_RE = re.compile(
    r"\b(bogot[aá]|colombia|m[eé]xico|chile|per[uú]|brasil|espa[nñ]a)\b",
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
    request_intent: Literal["standard", "full_report"] = "standard"
    company_name: str | None = None
    requires_grounding: bool = True
    sources: list[AgentSource] = Field(default_factory=list)
    evidence: list[AgentEvidence] = Field(default_factory=list)
    searched_queries: list[str] = Field(default_factory=list)
    employment_search_attempts: list[str] = Field(default_factory=list)
    inspected_source_ids: list[str] = Field(default_factory=list)
    retrieved_report_ids: list[str] = Field(default_factory=list)
    unresolved_topics: list[str] = Field(default_factory=list)
    generated_report_id: str | None = None
    generated_comparison_id: str | None = None
    selected_cv_id: str | None = None
    selected_cv_version_id: str | None = None
    selected_job_description_id: str | None = None
    recommendation_artifact_id: str | None = None
    clarification_response: dict[str, Any] | None = None
    pending_clarification: dict[str, Any] | None = None
    review_response: dict[str, Any] | None = None
    force_finalize: bool = False
    force_report_finalize: bool = False
    coverage_reminder_sent: bool = False
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
        usage: dict[str, Any] | None = None
        run_started: float | None = None
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
            usage = safe_json_dict(task.usage_json)
            for key in (
                "model_turns",
                "searches",
                "inspections",
                "elapsed_ms",
                "deepseek_ms",
                "search_ms",
                "inspection_ms",
                "evidence_ms",
                "artifact_ms",
                "tool_ms",
                "provider_requests",
                "tool_calls",
            ):
                usage.setdefault(key, 0)
            usage.setdefault(
                "first_progress_ms",
                elapsed_datetime_ms(task.created_at),
            )
            task.usage_json = json.dumps(usage, ensure_ascii=False)
            self.db.commit()
            state = self.load_state(task)
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
                if (
                    self.is_full_report_task(task)
                    and usage["model_turns"] >= 12
                    and not state.coverage_reminder_sent
                    and not self.can_create_report(state)
                ):
                    coverage = self.full_report_coverage(state)
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Control de cobertura del servidor. Quedan pocas decisiones. "
                                "Priorizá solamente los faltantes centrales y consultas separadas. "
                                + json.dumps(
                                    {
                                        "missing": coverage["missing"],
                                        "recommended_queries": coverage["recommended_queries"],
                                        "recommended_inspections": coverage["recommended_inspections"],
                                    },
                                    ensure_ascii=False,
                                )
                            ),
                        }
                    )
                    state.coverage_reminder_sent = True
                    self.save_state(task, state)
                    self.db.commit()
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
                clarification_retry = self.should_retry_clarification(state)
                clarification_only = self.requires_clarification(state)
                tools = self.function_tools(
                    final_only=state.force_finalize,
                    report_final_only=state.force_report_finalize,
                    clarification_only=clarification_only,
                    allow_clarification=(
                        not bool(state.clarification_response) or clarification_retry
                    ),
                    allow_research=state.requires_grounding,
                    report_needs_artifact=(
                        self.is_full_report_task(task)
                        and self.can_create_report(state)
                        and not state.generated_report_id
                    ),
                )
                required_report_action = (
                    self.required_full_report_action(state)
                    if self.is_full_report_task(task) and not repair_tool_name
                    else None
                )
                if required_report_action:
                    required_name = required_report_action["name"]
                    tools = [
                        item for item in tools
                        if item["function"]["name"] == required_name
                    ]
                    messages.append(
                        {
                            "role": "user",
                            "content": required_report_action["instruction"],
                        }
                    )
                allowed_names = [item["function"]["name"] for item in tools]
                tool_choice: str | dict[str, Any] = "required"
                if required_report_action:
                    tool_choice = {
                        "type": "function",
                        "function": {"name": required_report_action["name"]},
                    }
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
                        elif len(allowed_names) == 1:
                            repair_tool_name = allowed_names[0]
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
                arguments = (
                    required_report_action["arguments"]
                    if required_report_action
                    else call.arguments
                )
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
                if name == "finish_research" and result.payload.get("ok"):
                    self.complete_task(task, usage)
                messages.append(assistant_message)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result.payload, ensure_ascii=False),
                    }
                )
                if state.force_report_finalize and name != "finish_research":
                    raise AgentFailure("DeepSeek no pudo finalizar el informe dentro del presupuesto.")
                if state.force_finalize and name != "respond":
                    raise AgentFailure("DeepSeek no pudo finalizar la respuesta dentro del presupuesto.")
        except (AgentPaused, AgentCompleted):
            if usage is not None and run_started is not None:
                self.persist_usage(task, usage, run_started)
                self.db.commit()
            return
        except Exception as exc:
            if usage is not None and run_started is not None:
                if self.client is not None and getattr(self.client, "last_usage", {}).get(
                    "failed_request"
                ):
                    self.add_provider_usage(usage, self.client.last_usage)
                self.persist_usage(task, usage, run_started)
                self.db.commit()
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
        conversation = self.repo.get(task.conversation_id)
        context = safe_json_dict(conversation.active_context_json) if conversation else {}
        active_company = str(context.get("active_company_name") or "").strip()
        focused_goal = self.focused_company_goal(goal, active_company)
        inferred_company = (
            self.infer_company_from_report_goal(goal) or active_company
            if self.is_full_report_task(task)
            else active_company if focused_goal else None
        )
        prior_state = (
            self.latest_company_state(task, active_company)
            if focused_goal and active_company
            else None
        )
        if focused_goal:
            goal = focused_goal
        has_report = any(
            item.artifact_type == "report"
            for item in self.repo.list_artifacts(task.conversation_id)
            if item.message_id == task.trigger_message_id
        )
        return AgentState(
            goal=goal,
            request_intent=(
                "full_report"
                if self.is_full_report_task(task) or raw.get("request_intent") == "full_report"
                else "standard"
            ),
            requires_grounding=(
                bool(focused_goal)
                or has_report
                or self.is_full_report_task(task)
                or bool(GROUNDING_REQUIRED_RE.search(goal))
            ),
            company_name=inferred_company or None,
            sources=prior_state.sources if prior_state else self.conversation_sources(task.conversation_id),
            evidence=prior_state.evidence if prior_state else self.conversation_evidence(task.conversation_id),
            clarification_response=raw.get("clarification_response"),
            pending_clarification=raw.get("pending_clarification"),
        )

    @staticmethod
    def infer_company_from_report_goal(goal: str) -> str | None:
        match = re.search(
            r"\b(?:sobre|de|para)\s+(.+?)(?=\s+(?:como|en|del|de la|para)\b|[.!?,;:]?$)",
            goal.strip(),
            re.IGNORECASE,
        )
        if not match:
            return None
        company = re.sub(r"\s+", " ", match.group(1)).strip(" .,!?:;")
        return company or None

    @staticmethod
    def focused_company_goal(goal: str, company_name: str) -> str | None:
        if not company_name:
            return None
        focus = re.sub(r"[^a-záéíóúüñ]+", " ", goal.casefold()).strip()
        templates = {
            "vacante": "Vacantes actuales de {company} en Argentina.",
            "vacantes": "Vacantes actuales de {company} en Argentina.",
            "puesto": "Vacantes actuales de {company} en Argentina.",
            "puestos": "Vacantes actuales de {company} en Argentina.",
            "general": "Perfil general de {company} como empleador en Argentina.",
            "perfil": "Perfil general de {company} como empleador en Argentina.",
            "sueldo": "Sueldos de {company} en Argentina.",
            "sueldos": "Sueldos de {company} en Argentina.",
            "salario": "Salarios de {company} en Argentina.",
            "salarios": "Salarios de {company} en Argentina.",
            "cultura": "Cultura laboral de {company} en Argentina.",
            "entrevista": "Proceso de entrevistas de {company} en Argentina.",
            "entrevistas": "Proceso de entrevistas de {company} en Argentina.",
        }
        template = templates.get(focus)
        return template.format(company=company_name) if template else None

    def latest_company_state(
        self,
        task: models.TaskRun,
        company_name: str,
    ) -> AgentState | None:
        previous_tasks = (
            self.db.query(models.TaskRun)
            .filter(
                models.TaskRun.conversation_id == task.conversation_id,
                models.TaskRun.id != task.id,
            )
            .order_by(models.TaskRun.created_at.desc())
            .limit(20)
        )
        for previous in previous_tasks:
            raw = safe_json_dict(previous.working_state_json)
            if not raw.get("goal"):
                continue
            candidate = AgentState.model_validate(raw)
            if candidate.company_name and self.same_company(candidate.company_name, company_name):
                return candidate
        return None

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
        if state.force_finalize:
            return False
        if AgentOrchestrator.should_retry_clarification(state):
            return True
        if state.clarification_response:
            return False
        goal = " ".join(state.goal.strip().split())
        if SMALL_TALK_RE.fullmatch(goal):
            return False
        if WEATHER_RE.search(goal) and not re.search(r"\b(en|para)\s+\w+", goal, re.IGNORECASE):
            return True
        return len(re.findall(r"\w+", goal, re.UNICODE)) <= 2

    @staticmethod
    def should_retry_clarification(state: AgentState) -> bool:
        """Keep an unresolved name clarification from falling through to respond."""
        pending = state.pending_clarification or {}
        response = state.clarification_response or {}
        question = str(pending.get("question") or "").casefold()
        content = " ".join(str(response.get("content") or "").casefold().split())
        goal = " ".join(str(state.goal or "").casefold().split())
        if not content or not goal or content != goal:
            return False
        return "qué significa" in question or "que significa" in question

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
            "deepseek_ms",
            "provider_requests",
        ):
            usage[key] = int(usage.get(key) or 0) + int(provider_usage.get(key) or 0)

    def refresh_task(self, task: models.TaskRun) -> None:
        self.db.refresh(task)

    def enforce_budget(
        self, task: models.TaskRun, state: AgentState, usage: dict, started: float
    ) -> None:
        if state.force_finalize or state.force_report_finalize:
            return
        budget = safe_json_dict(task.budget_json)
        elapsed = self.run_elapsed_base + int((time.monotonic() - started) * 1000)
        if (
            self.is_full_report_task(task)
            and self.can_create_report(state)
            and usage["model_turns"] >= max(0, int(budget.get("model_turns", 4)) - 1)
        ):
            self.force_completion(task, state)
            return
        exhausted = (
            usage["model_turns"] >= int(budget.get("model_turns", 4))
            or elapsed >= int(budget.get("elapsed_seconds", 180)) * 1000
        )
        if not exhausted:
            return
        self.force_completion(task, state)

    def enforce_tool_budget(
        self, task: models.TaskRun, state: AgentState, usage: dict, counter: str
    ) -> bool:
        budget = safe_json_dict(task.budget_json)
        if int(usage.get(counter, 0)) < int(budget.get(counter, 0)):
            return True
        self.force_completion(task, state)
        return False

    @staticmethod
    def is_full_report_task(task: models.TaskRun) -> bool:
        return safe_json_dict(task.budget_json).get("profile") == "full_report"

    @staticmethod
    def full_report_coverage(state: AgentState) -> dict[str, Any]:
        sources = {source.id: source for source in state.sources}
        corporate_source = any(source.source_type in CORPORATE_SOURCE_TYPES for source in state.sources)
        career_source = any(source.source_type == SourceType.career_page.value for source in state.sources)
        employment_source = any(source.source_type in EMPLOYMENT_SOURCE_TYPES for source in state.sources)
        linkedin_source = any("linkedin.com" in source.domain.lower() for source in state.sources)
        review_source = any(
            source.source_type == SourceType.salary_review_platform.value for source in state.sources
        )
        argentina_job_board_source = any(
            source.source_type == SourceType.job_board.value
            and any(domain in source.domain.lower() for domain in ARGENTINA_JOB_BOARD_DOMAINS)
            for source in state.sources
        )
        covered_topics = {
            topic
            for topic in FULL_REPORT_CORE_TOPICS
            if topic != EvidenceTopic.open_roles.value
            and any(
                evidence.topic == topic
                and evidence.confidence != ConfidenceLevel.low.value
                and (source := sources.get(evidence.source_id)) is not None
                and source.source_type in CORPORATE_SOURCE_TYPES
                for evidence in state.evidence
            )
        }
        open_roles_verified = any(
            evidence.topic == EvidenceTopic.open_roles.value
            and evidence.confidence != ConfidenceLevel.low.value
            and (source := sources.get(evidence.source_id)) is not None
            and AgentOrchestrator.is_specific_argentina_job(
                source,
                evidence.excerpt,
                state.company_name or "",
            )
            for evidence in state.evidence
        )
        if open_roles_verified:
            covered_topics.add(EvidenceTopic.open_roles.value)
        attempts = list(
            dict.fromkeys(
                [
                    *state.employment_search_attempts,
                    *(
                        family
                        for query in state.searched_queries
                        if (family := AgentOrchestrator.employment_search_family(query))
                    ),
                ]
            )
        )
        employment_search_exhausted = all(
            family in attempts for family in EMPLOYMENT_SEARCH_SEQUENCE
        )
        open_roles_status = (
            "verified"
            if open_roles_verified
            else "searched_not_verified"
            if employment_search_exhausted
            else "search_pending"
        )
        missing = []
        if not corporate_source:
            missing.append("una fuente oficial o de carreras de la empresa")
        if not employment_source and not employment_search_exhausted:
            missing.append("una fuente laboral actual para las vacantes")
        if EvidenceTopic.business.value not in covered_topics:
            missing.append("el negocio principal")
        if EvidenceTopic.argentina_presence.value not in covered_topics:
            missing.append("la presencia de la empresa en Argentina")
        if open_roles_status == "search_pending":
            missing.append("vacantes actuales en Argentina")
        company_name = state.company_name or "la empresa"
        recommended_queries: list[dict[str, str]] = []
        recommended_inspections: list[dict[str, str]] = []
        if not corporate_source:
            recommended_queries.extend(
                [
                    {
                        "topic": EvidenceTopic.business.value,
                        "query": f"{company_name} sitio oficial Argentina",
                        "reason": "Falta una fuente corporativa.",
                    },
                    {
                        "topic": EvidenceTopic.open_roles.value,
                        "query": f"{company_name} careers jobs Argentina",
                        "reason": "Falta una página de carreras de la empresa.",
                    },
                ]
            )
        elif EvidenceTopic.business.value not in covered_topics:
            recommended_queries.append(
                {
                    "topic": EvidenceTopic.business.value,
                    "query": f"{company_name} sitio oficial servicios productos what we do",
                    "reason": "Falta evidencia corporativa sobre el negocio.",
                }
            )
        if EvidenceTopic.argentina_presence.value not in covered_topics:
            recommended_queries.append(
                {
                    "topic": EvidenceTopic.argentina_presence.value,
                    "query": f"{company_name} sitio oficial Argentina oficinas presencia",
                    "reason": "Falta evidencia corporativa sobre la presencia en Argentina.",
                }
            )
        if open_roles_status == "search_pending":
            next_family = next(
                family for family in EMPLOYMENT_SEARCH_SEQUENCE if family not in attempts
            )
            recommended_queries.append(
                {
                    "topic": EvidenceTopic.open_roles.value,
                    "source_family": next_family,
                    "query": EMPLOYMENT_SEARCH_QUERIES[next_family].format(company=company_name),
                    "reason": "Falta una vacante concreta y localizada en Argentina.",
                }
            )
        elif not review_source:
            recommended_queries.append(
                {
                    "topic": EvidenceTopic.culture.value,
                    "query": f"site:glassdoor.com {company_name} Argentina reviews entrevistas",
                    "reason": "Fuente opcional para cultura y entrevistas.",
                }
            )
        missing_topics = set(FULL_REPORT_CORE_TOPICS) - covered_topics
        for source in state.sources:
            if source.id in state.inspected_source_ids:
                continue
            useful_for = []
            if source.source_type in CORPORATE_SOURCE_TYPES:
                useful_for.extend(
                    topic
                    for topic in (
                        EvidenceTopic.business.value,
                        EvidenceTopic.argentina_presence.value,
                    )
                    if topic in missing_topics
                )
            if (
                open_roles_status == "search_pending"
                and source.source_type in EMPLOYMENT_SOURCE_TYPES
                and AgentOrchestrator.is_specific_argentina_job(
                    source,
                    "",
                    state.company_name or "",
                )
            ):
                useful_for.append(EvidenceTopic.open_roles.value)
            if useful_for:
                recommended_inspections.append(
                    {
                        "source_id": source.id,
                        "title": source.title,
                        "topics": ", ".join(dict.fromkeys(useful_for)),
                    }
                )
        return {
            "corporate_source": corporate_source,
            "career_source": career_source,
            "employment_source": employment_source,
            "linkedin_source": linkedin_source,
            "review_source": review_source,
            "argentina_job_board_source": argentina_job_board_source,
            "core_topics": {topic: topic in covered_topics for topic in FULL_REPORT_CORE_TOPICS},
            "open_roles_status": open_roles_status,
            "employment_search_attempts": attempts,
            "employment_search_exhausted": employment_search_exhausted,
            "missing": missing,
            "recommended_queries": recommended_queries,
            "recommended_inspections": recommended_inspections,
            "ready": not missing,
        }

    @staticmethod
    def employment_search_family(query: str) -> str | None:
        normalized = query.casefold()
        if "linkedin.com/jobs" in normalized or (
            "linkedin" in normalized
            and any(term in normalized for term in ("job", "empleo", "vacante"))
        ):
            return "linkedin"
        if "computrabajo" in normalized:
            return "computrabajo"
        if "bumeran" in normalized:
            return "bumeran"
        if "zonajobs" in normalized:
            return "zonajobs"
        if "career" in normalized:
            return "careers"
        return None

    @staticmethod
    def required_full_report_action(state: AgentState) -> dict[str, Any] | None:
        coverage = AgentOrchestrator.full_report_coverage(state)
        if not state.company_name or coverage["ready"]:
            return None
        if coverage["recommended_inspections"]:
            source_id = coverage["recommended_inspections"][0]["source_id"]
            return {
                "name": "inspect_page",
                "arguments": {"source_id": source_id},
                "instruction": (
                    "La cobertura corporativa central ya está completa. Inspeccioná ahora "
                    f"la fuente laboral recomendada con source_id {source_id}."
                ),
            }
        query_priority = {
            EvidenceTopic.business.value: 0,
            EvidenceTopic.argentina_presence.value: 1,
            EvidenceTopic.open_roles.value: 2,
        }
        recommendation = min(
            coverage["recommended_queries"],
            key=lambda item: query_priority.get(item.get("topic", ""), 99),
        )
        return {
            "name": "search_web",
            "arguments": {
                "query": recommendation["query"],
                "company_name": state.company_name,
                "topic": EvidenceTopic.open_roles.value,
                "limit": 5,
            },
            "instruction": (
                "La cobertura corporativa central ya está completa. Ejecutá ahora exactamente "
                f"la búsqueda laboral pendiente: {recommendation['query']}"
            ),
        }

    @staticmethod
    def is_specific_argentina_job(
        source: AgentSource,
        evidence_text: str,
        company_name: str,
    ) -> bool:
        if source.source_type not in EMPLOYMENT_SOURCE_TYPES or not company_name:
            return False
        context = f"{source.title} {source.snippet} {evidence_text} {source.url}".casefold()
        collapsed = re.sub(r"[^a-z0-9áéíóúüñ]+", "", context)
        company_matches = any(
            token in collapsed for token in company_domain_tokens(company_name)
        )
        domain = source.domain.casefold()
        path = urlparse(source.url).path.rstrip("/").casefold()
        if "linkedin.com" in domain:
            is_job_detail = bool(re.search(r"/jobs/view/[^/]+$", path))
        else:
            is_job_detail = bool(
                path not in {"/careers", "/jobs", "/empleos", "/vacantes"}
                and JOB_DETAIL_URL_RE.search(path)
            )
        return bool(
            company_matches
            and ARGENTINA_LOCATION_RE.search(context)
            and is_job_detail
            and not GENERIC_JOB_PAGE_RE.fullmatch(source.title.strip())
        )

    def set_company_context(self, task: models.TaskRun, state: AgentState, company_name: str) -> None:
        company_name = company_name.strip()
        if not company_name:
            if self.is_full_report_task(task):
                raise ValueError("search_web requiere company_name para un informe completo.")
            return
        if state.company_name and not self.same_company(state.company_name, company_name):
            raise ValueError("La empresa no puede cambiar durante la misma investigación.")
        is_new_company = not state.company_name
        state.company_name = state.company_name or company_name
        conversation = self.repo.get(task.conversation_id)
        if conversation:
            context = safe_json_dict(conversation.active_context_json)
            context["active_company_name"] = state.company_name
            self.repo.update_active_context(conversation, context)
        if not is_new_company:
            return
        for source in state.sources:
            scored = score_search_result(
                company_name,
                SearchResult(
                    title=source.title,
                    url=source.url,
                    snippet=source.snippet,
                    rank=source.rank,
                    query_topic=EvidenceTopic(source.topic),
                ),
            )
            source.source_type = scored.source_type.value
            source.reliability_score = scored.reliability_score

    @staticmethod
    def same_company(first: str, second: str) -> bool:
        first_tokens = set(company_domain_tokens(first))
        second_tokens = set(company_domain_tokens(second))
        return (
            bool(first_tokens & second_tokens)
            or normalize_company_name(first) == normalize_company_name(second)
        )

    @classmethod
    def can_create_report(cls, state: AgentState) -> bool:
        return bool(cls.full_report_coverage(state)["ready"])

    def force_completion(self, task: models.TaskRun, state: AgentState) -> None:
        if self.is_full_report_task(task) and self.can_create_report(state):
            state.force_report_finalize = True
        else:
            state.force_finalize = True
        self.save_state(task, state)
        self.db.commit()

    def build_context(self, task: models.TaskRun, state: AgentState) -> str:
        conversation = self.repo.get(task.conversation_id)
        messages = self.repo.list_messages(task.conversation_id)[-12:]
        artifacts = self.repo.list_artifacts(task.conversation_id)
        task_context = state.model_dump(mode="json")
        for source in task_context["sources"]:
            source["snippet"] = source["snippet"][:400]
        for evidence in task_context["evidence"]:
            evidence["excerpt"] = evidence["excerpt"][:500]
        context = {
            "conversation_summary": conversation.summary if conversation else None,
            "task": task_context,
            "messages": [{"role": item.role, "content": item.content[:1000]} for item in messages],
            "active_context": safe_json_dict(conversation.active_context_json) if conversation else {},
            "artifacts": [
                {"type": item.artifact_type, "id": item.artifact_id} for item in artifacts[-12:]
            ],
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
            "get_cv_profile": self.tool_get_cv_profile,
            "analyze_job_description": self.tool_analyze_job_description,
            "prepare_cv_recommendations": self.tool_prepare_cv_recommendations,
        }
        handler = handlers.get(name)
        if not handler:
            return ToolResult({"ok": False, "error": "Herramienta no permitida."}, [])
        started = time.monotonic()
        try:
            return handler(task, state, usage, arguments)
        except (AgentPaused, AgentCompleted):
            raise
        except (ValidationError, ValueError) as exc:
            return ToolResult({"ok": False, "error": str(exc)[:500]}, [])
        finally:
            elapsed = int((time.monotonic() - started) * 1000)
            usage["tool_calls"] = int(usage.get("tool_calls") or 0) + 1
            usage["tool_ms"] = int(usage.get("tool_ms") or 0) + elapsed
            timing_key = {
                "search_web": "search_ms",
                "inspect_page": "inspection_ms",
                "review_evidence": "evidence_ms",
                "finish_research": "artifact_ms",
                "compare_reports": "artifact_ms",
                "prepare_cv_recommendations": "artifact_ms",
            }.get(name)
            if timing_key:
                usage[timing_key] = int(usage.get(timing_key) or 0) + elapsed

    def tool_respond(self, task, state, usage, arguments) -> ToolResult:
        if state.force_finalize and state.requires_grounding and (
            not state.evidence or self.is_full_report_task(task)
        ):
            missing = self.full_report_coverage(state)["missing"] if self.is_full_report_task(task) else []
            arguments = AgentAnswer(
                answer=(
                    "No pude verificar evidencia suficiente para generar un informe completo. "
                    + (
                        "Faltó cubrir: " + ", ".join(missing) + ". "
                        if missing
                        else ""
                    )
                    + "Podés indicar una fuente, el país, el rol o un foco más específico para intentarlo de nuevo."
                ),
                answer_type="grounded",
                claims=(
                    [
                        GroundedClaim(
                            text="La evidencia disponible no cubre todos los requisitos del informe.",
                            evidence_ids=[item.id for item in state.evidence[:3]],
                        )
                    ]
                    if state.evidence
                    else []
                ),
                warnings=[
                    "Hay fuentes potenciales, pero no se validó evidencia suficiente para completar el informe."
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
        if FUTURE_RESEARCH_RE.search(answer.answer):
            raise ValueError(
                "respond es terminal: ejecutá una herramienta de investigación o entregá la respuesta final."
            )
        usage["response_kind"] = (
            "artifact"
            if state.generated_report_id
            or state.generated_comparison_id
            or state.recommendation_artifact_id
            else "grounded" if answer.answer_type == "grounded" else "direct"
        )
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
            has_employment_source = any(
                source.id in cited_source_ids
                and source.source_type
                in {SourceType.official.value, SourceType.career_page.value, SourceType.job_board.value}
                for source in state.sources
            )
            caveat = re.search(
                r"\b(no pude confirmar|sin confirmaci[oó]n oficial|no hay confirmaci[oó]n oficial)\b",
                answer.answer,
                re.IGNORECASE,
            )
            if not has_employment_source:
                if not caveat:
                    answer.answer = (
                        "No pude confirmar esta información en una fuente laboral confiable. "
                        + answer.answer
                    )
                if not answer.warnings:
                    answer.warnings.append("La oportunidad solo cuenta con evidencia secundaria.")
        answer.answer = clean_generated_answer(answer.answer)
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
        self.summarize_if_needed(conversation, usage)
        raise AgentCompleted

    def complete_task(self, task: models.TaskRun, usage: dict) -> None:
        self.repo.update_task(task, "completed")
        self.repo.add_event(task, "task.completed", {"task_run_id": task.id, "status": "completed"})
        self.db.commit()
        conversation = self.repo.get(task.conversation_id)
        if conversation:
            self.summarize_if_needed(conversation, usage)
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
        state.pending_clarification = {"question": question, "options": options}
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

    def tool_get_cv_profile(self, task, state, usage, arguments) -> ToolResult:
        library = CVLibraryService(self.db, self.settings)
        conversation = self.repo.get(task.conversation_id)
        if not conversation:
            raise AgentFailure("La conversación ya no existe.")
        artifacts = self.repo.list_artifacts(task.conversation_id)
        attached_ids = [
            item.artifact_id
            for item in artifacts
            if item.message_id == task.trigger_message_id and item.artifact_type == "cv"
        ]
        context = safe_json_dict(conversation.active_context_json)
        selected_ids = (state.clarification_response or {}).get("selected_option_ids") or []
        requested = str(arguments.get("cv_id") or "")
        candidate_id = requested or (attached_ids[-1] if attached_ids else "")
        if not candidate_id:
            candidate_id = next(
                (item for item in selected_ids if library.get_cv(str(item))),
                context.get("active_cv_id") or "",
            )
        cv = library.get_cv(str(candidate_id)) if candidate_id else library.default_or_only()
        if not cv:
            cvs = library.list_cvs()
            if not cvs:
                raise ValueError("No hay ningún CV guardado. Subí uno desde Mi CV o desde el compositor.")
            self.pause(
                task,
                "needs_clarification",
                "clarification.required",
                {
                    "type": "cv_selection",
                    "prompt": "Hay varios CV guardados. ¿Cuál querés usar?",
                    "options": [{"id": item.id, "label": item.display_name} for item in cvs],
                },
            )
        version = library.current_version(cv)
        signals = extract_candidate_signals(version.extracted_text)
        lines = select_cv_evidence_lines(version.extracted_text, signals)
        state.selected_cv_id = cv.id
        state.selected_cv_version_id = version.id
        context["active_cv_id"] = cv.id
        context["active_cv_version_id"] = version.id
        self.repo.update_active_context(conversation, context)
        return ToolResult(
            {
                "ok": True,
                "summary": f"Se recuperó {cv.display_name}.",
                "cv_id": cv.id,
                "cv_version_id": version.id,
                "profile": json.loads(version.structured_profile_json or "{}"),
                "evidence_lines": [
                    {"id": f"cv_line_{index}", "text": line}
                    for index, line in enumerate(lines, start=1)
                ],
            },
            [],
        )

    def tool_analyze_job_description(self, task, state, usage, arguments) -> ToolResult:
        conversation = self.repo.get(task.conversation_id)
        if not conversation:
            raise AgentFailure("La conversación ya no existe.")
        artifacts = self.repo.list_artifacts(task.conversation_id)
        attached_ids = [
            item.artifact_id
            for item in artifacts
            if item.message_id == task.trigger_message_id and item.artifact_type == "job_description"
        ]
        context = safe_json_dict(conversation.active_context_json)
        job_id = str(arguments.get("job_description_id") or "")
        job_id = job_id or (attached_ids[-1] if attached_ids else "")
        job_id = job_id or str(context.get("active_job_description_id") or "")
        job = self.db.get(models.JobDescription, job_id) if job_id else None
        if not job and (state.clarification_response or {}).get("content"):
            messages = self.repo.list_messages(task.conversation_id)
            user_message = next((item for item in reversed(messages) if item.role == "user"), None)
            if user_message:
                job = CVLibraryService(self.db, self.settings).create_job_description(
                    conversation,
                    user_message,
                    "Descripción del puesto",
                    str(state.clarification_response["content"]),
                )
        if not job or job.conversation_id != task.conversation_id:
            self.pause(
                task,
                "needs_clarification",
                "clarification.required",
                {
                    "type": "job_description",
                    "prompt": "Pegá la descripción del puesto que querés analizar.",
                    "options": [],
                },
            )
        signals = extract_job_signals(job.raw_text)
        lines = select_job_evidence_lines(job.raw_text, signals)
        state.selected_job_description_id = job.id
        context["active_job_description_id"] = job.id
        self.repo.update_active_context(conversation, context)
        return ToolResult(
            {
                "ok": True,
                "summary": "Se analizó la descripción del puesto.",
                "job_description_id": job.id,
                "signals": job_signal_payload(signals),
                "evidence_lines": [
                    {"id": f"job_line_{index}", "text": line}
                    for index, line in enumerate(lines, start=1)
                ],
            },
            [],
        )

    def tool_prepare_cv_recommendations(self, task, state, usage, arguments) -> ToolResult:
        if not state.selected_cv_id:
            self.tool_get_cv_profile(task, state, usage, arguments)
        library = CVLibraryService(self.db, self.settings)
        cv = library.get_cv(state.selected_cv_id or "")
        version = library.get_version(cv.id, state.selected_cv_version_id or "") if cv else None
        if not cv or not version:
            raise ValueError("El CV seleccionado ya no está disponible.")
        cv_signals = extract_candidate_signals(version.extracted_text)
        cv_lines = select_cv_evidence_lines(version.extracted_text, cv_signals)

        job = (
            self.db.get(models.JobDescription, state.selected_job_description_id)
            if state.selected_job_description_id
            else None
        )
        if not job and arguments.get("require_job_description", True):
            self.tool_analyze_job_description(task, state, usage, arguments)
            job = self.db.get(models.JobDescription, state.selected_job_description_id)
        job_signals = extract_job_signals(job.raw_text) if job else None
        job_lines = select_job_evidence_lines(job.raw_text, job_signals) if job and job_signals else []
        report = None
        report_id = str(arguments.get("report_id") or "")
        if report_id:
            stored_report = self.report_repo.get_by_id(report_id)
            if stored_report and stored_report.status == ReportStatus.completed.value:
                report = self.report_repo.to_structured_report(stored_report)
        tailoring = DeepSeekCVTailoringService(
            api_key=self.settings.deepseek_api_key or "",
            model=self.settings.deepseek_model,
            client=self.get_client(),
            timeout_seconds=self.settings.deepseek_timeout_seconds,
        ).build(
            company_name=str(arguments.get("company_name") or "la oportunidad"),
            signals=cv_signals,
            cv_evidence_lines=cv_lines,
            job_signals=job_signals,
            job_evidence_lines=job_lines,
            report=report,
            include_adapted_cv_draft=False,
        )
        self.add_provider_usage(usage, getattr(self.get_client(), "last_usage", {}))
        allowed_ids = {
            *(f"cv_line_{index}" for index in range(1, len(cv_lines) + 1)),
            *(f"job_line_{index}" for index in range(1, len(job_lines) + 1)),
            *(item.id for item in state.evidence),
        }
        for suggestion in tailoring.change_suggestions:
            if not suggestion.evidence_ids or not set(suggestion.evidence_ids).issubset(allowed_ids):
                raise ValueError("DeepSeek generó una recomendación sin evidencia válida.")

        now = utc_now()
        artifact = models.CVRecommendationArtifact(
            id=str(uuid4()),
            conversation_id=task.conversation_id,
            task_run_id=task.id,
            cv_id=cv.id,
            cv_version_id=version.id,
            job_description_id=job.id if job else None,
            status="awaiting_review",
            payload_json=tailoring.model_dump_json(),
            review_json="{}",
            draft_text=version.extracted_text,
            created_at=now,
            updated_at=now,
        )
        self.db.add(artifact)
        conversation = self.repo.get(task.conversation_id)
        if not conversation:
            raise AgentFailure("La conversación ya no existe.")
        self.repo.add_artifact_link(
            conversation,
            artifact_type="cv_recommendation",
            artifact_id=artifact.id,
            relationship_type="generated",
            message_id=task.trigger_message_id,
        )
        state.recommendation_artifact_id = artifact.id
        self.save_state(task, state)
        refs = [{"type": "cv_recommendation", "artifact_id": artifact.id}]
        self.repo.add_event(
            task,
            "artifact.created",
            {"task_run_id": task.id, "type": "cv_recommendation", "artifact_id": artifact.id},
            artifact_refs=refs,
        )
        self.pause(
            task,
            "awaiting_review",
            "review.required",
            {
                "type": "cv_recommendation",
                "artifact_id": artifact.id,
                "prompt": "Revisá las recomendaciones antes de guardar cambios en tu CV.",
                "suggestion_count": len(tailoring.change_suggestions),
            },
        )

    def tool_search_web(self, task, state, usage, arguments) -> ToolResult:
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ValueError("La búsqueda requiere una consulta.")
        self.set_company_context(task, state, str(arguments.get("company_name") or ""))
        topic = EvidenceTopic(str(arguments.get("topic") or EvidenceTopic.general.value))
        limit = min(5, max(1, int(arguments.get("limit") or 3)))
        if not self.enforce_tool_budget(task, state, usage, "searches"):
            return ToolResult(
                {"ok": False, "summary": "Se alcanzó el límite de búsquedas; finalizando con lo disponible."},
                [],
            )
        usage["searches"] += 1
        normalized = " ".join(query.lower().split())
        if normalized in state.searched_queries:
            return ToolResult({"ok": True, "summary": "La consulta ya había sido ejecutada.", "sources": []}, [])
        state.searched_queries.append(normalized)
        results = self.get_search_provider().search(query, limit)
        if topic == EvidenceTopic.open_roles:
            family = self.employment_search_family(query)
            if family and family not in state.employment_search_attempts:
                state.employment_search_attempts.append(family)
        existing = {item.id for item in state.sources}
        added = []
        filtered_by_location = 0
        targets_argentina = "argentina" in f"{state.goal} {state.company_name or ''}".lower()
        for result in results:
            if urlparse(result.url).scheme not in {"http", "https"}:
                continue
            result_context = f"{result.title} {result.snippet}"
            if (
                targets_argentina
                and topic.value in LOCATION_SENSITIVE_TOPICS
                and FOREIGN_LOCATION_RE.search(result_context)
                and "argentina" not in result_context.lower()
            ):
                filtered_by_location += 1
                continue
            normalized_result = SearchResult(
                title=result.title,
                url=result.url,
                snippet=result.snippet,
                rank=result.rank,
                query_topic=topic,
            )
            scored = score_search_result(state.company_name or "", normalized_result)
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
        payload = {
            "ok": True,
            "summary": f"Se encontraron {len(added)} fuentes nuevas.",
            "sources": [item.model_dump(mode="json") for item in added],
            "filtered_by_location": filtered_by_location,
        }
        if self.is_full_report_task(task):
            payload["report_coverage"] = self.full_report_coverage(state)
        return ToolResult(payload, [])

    def tool_inspect_page(self, task, state, usage, arguments) -> ToolResult:
        source_id = str(arguments.get("source_id") or "")
        source = next((item for item in state.sources if item.id == source_id), None)
        if not source:
            raise ValueError("Solo se pueden inspeccionar fuentes devueltas por search_web.")
        if not self.enforce_tool_budget(task, state, usage, "inspections"):
            return ToolResult(
                {"ok": False, "summary": "Se alcanzó el límite de inspecciones; finalizando con lo disponible."},
                [],
            )
        usage["inspections"] += 1
        if source_id in state.inspected_source_ids:
            return ToolResult({"ok": True, "summary": "La fuente ya había sido inspeccionada."}, [])
        content = self.get_extractor().extract(source.url)
        if content is None:
            if not source.snippet.strip():
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
            content = ExtractedContent(url=source.url, title=source.title, text=source.snippet)
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
            confidence = item.confidence
            if (
                item.topic == EvidenceTopic.open_roles
                and confidence == ConfidenceLevel.low
                and self.is_specific_argentina_job(
                    source,
                    item.raw_text_excerpt,
                    state.company_name or "",
                )
            ):
                confidence = ConfidenceLevel.medium
            evidence = AgentEvidence(
                id=evidence_id,
                source_id=source.id,
                topic=item.topic.value,
                claim=item.claim,
                excerpt=item.raw_text_excerpt[:1200],
                confidence=confidence.value,
            )
            state.evidence.append(evidence)
            known.add(evidence.id)
            added.append(evidence)
        source.inspected = True
        state.inspected_source_ids.append(source.id)
        payload = {
            "ok": True,
            "summary": f"La fuente aportó {len(added)} evidencias.",
            "evidence": [item.model_dump(mode="json") for item in added],
        }
        if self.is_full_report_task(task):
            payload["report_coverage"] = self.full_report_coverage(state)
        return ToolResult(payload, [])

    def tool_review_evidence(self, task, state, usage, arguments) -> ToolResult:
        coverage = {topic.value: 0 for topic in EvidenceTopic}
        for item in state.evidence:
            coverage[item.topic] = coverage.get(item.topic, 0) + 1
        missing = [topic for topic, count in coverage.items() if count == 0]
        state.unresolved_topics = missing
        report_coverage = self.full_report_coverage(state) if self.is_full_report_task(task) else None
        return ToolResult(
            {
                "ok": True,
                "summary": f"Hay evidencia en {sum(1 for count in coverage.values() if count)} temas.",
                "coverage": coverage,
                "missing_topics": missing,
                "report_coverage": report_coverage,
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
            message_id=task.trigger_message_id,
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
        self.set_company_context(task, state, company_name)
        coverage = self.full_report_coverage(state) if self.is_full_report_task(task) else None
        if coverage and not coverage["ready"]:
            raise ValueError(
                "Todavía falta evidencia central o completar las búsquedas laborales requeridas."
            )
        allow_unverified_open_roles = bool(
            coverage and coverage["open_roles_status"] == "searched_not_verified"
        )
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
            self.add_provider_usage(usage, getattr(self.get_client(), "last_usage", {}))
            if allow_unverified_open_roles:
                self.apply_open_roles_limitation(structured)
            completion_issues = report_completion_issues(
                structured,
                allow_unverified_open_roles=allow_unverified_open_roles,
            )
            if completion_issues:
                self.report_repo.update_status(
                    report,
                    ReportStatus.failed,
                    " ".join(completion_issues)[:500],
                )
                self.db.commit()
                self.tool_respond(
                    task,
                    state,
                    usage,
                    {
                        "answer": (
                            f"No pude verificar evidencia suficiente para generar un informe completo sobre "
                            f"{company.name}. "
                            + " ".join(completion_issues)
                            + " Puedo investigar una parte puntual si indicás qué información priorizar."
                        ),
                        "answer_type": "grounded",
                        "claims": [],
                        "warnings": completion_issues,
                    },
            )
            self.report_repo.save_structured_report(report, structured)
        except AgentCompleted:
            raise
        except SynthesisError as exc:
            if "did not match report schema after repair" not in str(exc):
                raise
            self.db.rollback()
            report = self.report_repo.get_by_id(report.id)
            if report:
                self.report_repo.update_status(report, ReportStatus.failed, str(exc)[:500])
                self.db.commit()
            self.tool_respond(
                task,
                state,
                usage,
                {
                    "answer": (
                        "No pude verificar citas exactas suficientes para generar el informe completo. "
                        "Las afirmaciones centrales siguieron sin respaldo después del intento de reparación. "
                        "La evidencia no se publicó como informe para evitar afirmaciones sin respaldo."
                    ),
                    "answer_type": "grounded",
                    "claims": [],
                    "warnings": [
                        "La síntesis no superó la validación de citas centrales."
                    ],
                },
            )
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
            message_id=task.trigger_message_id,
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

    @staticmethod
    def apply_open_roles_limitation(report) -> None:
        section = next(
            (item for item in report.sections if item.type == SectionType.open_roles),
            None,
        )
        if section is None:
            section = SectionSchema(
                type=SectionType.open_roles,
                title="Vacantes actuales en Argentina",
                summary=OPEN_ROLES_LIMITATION,
                claims=[],
                confidence=ConfidenceLevel.low,
                missing_evidence=True,
            )
            report.sections.append(section)
        else:
            section.title = "Vacantes actuales en Argentina"
            section.summary = OPEN_ROLES_LIMITATION
            section.claims = []
            section.confidence = ConfidenceLevel.low
            section.missing_evidence = True
        report.warnings = [
            warning
            for warning in report.warnings
            if warning.related_section != SectionType.open_roles
        ]
        report.warnings.append(
            WarningSchema(
                id="open_roles_not_verified",
                type="open_roles_not_verified",
                message=OPEN_ROLES_LIMITATION,
                severity=WarningSeverity.medium,
                related_section=SectionType.open_roles,
            )
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

    def summarize_if_needed(self, conversation: models.Conversation, usage: dict) -> None:
        messages = self.repo.list_messages(conversation.id)
        if len(messages) <= 20:
            return
        older = messages[:-12]
        prompt = "Resumí en español estos mensajes en menos de 4000 caracteres, sin razonamiento privado:\n" + json.dumps(
            [{"role": item.role, "content": item.content} for item in older], ensure_ascii=False
        )
        try:
            client = self.get_client()
            response = client.generate_text(
                SYSTEM_PROMPT,
                prompt[:24_000],
                max_tokens=1200,
                temperature=0.1,
            )
            self.add_provider_usage(usage, getattr(client, "last_usage", {}))
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
            "get_cv_profile": "Recuperando el CV seleccionado",
            "analyze_job_description": "Analizando la descripción del puesto",
            "prepare_cv_recommendations": "Preparando recomendaciones para el CV",
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
            "search_web": {"query", "company_name", "topic", "limit"},
            "inspect_page": {"source_id"},
            "retrieve_reports": {"company_name", "query", "limit", "report_ids"},
            "compare_reports": {"report_ids", "dimensions"},
            "finish_research": {"company_name", "output_type", "stopping_reason", "unresolved_topics"},
            "request_clarification": {"question", "options"},
            "respond": {"answer_type", "warnings"},
            "get_cv_profile": {"cv_id"},
            "analyze_job_description": {"job_description_id"},
            "prepare_cv_recommendations": {
                "cv_id",
                "job_description_id",
                "report_id",
                "company_name",
                "require_job_description",
            },
        }.get(name, set())
        return {key: value for key, value in arguments.items() if key in allowed}

    @staticmethod
    def function_tools(
        final_only: bool = False,
        report_final_only: bool = False,
        clarification_only: bool = False,
        allow_clarification: bool = True,
        allow_research: bool = True,
        report_needs_artifact: bool = False,
    ) -> list[dict[str, Any]]:
        names = (
            ["finish_research"]
            if report_final_only
            else ["respond"]
            if final_only
            else ["request_clarification"]
            if clarification_only
            else list(FUNCTION_SCHEMAS)
        )
        if report_needs_artifact:
            names = [name for name in names if name != "respond"]
        if not allow_clarification:
            names = [name for name in names if name != "request_clarification"]
        if not allow_research:
            names = [
                name
                for name in names
                if name
                not in {
                    "search_web",
                    "inspect_page",
                    "review_evidence",
                    "retrieve_reports",
                    "compare_reports",
                    "finish_research",
                }
            ]
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
        "description": "Finaliza con una respuesta completa que el backend validará antes de mostrarla.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "answer_type": {"type": "string", "enum": ["guidance", "grounded"]},
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "evidence_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["text", "evidence_ids"],
                    },
                },
                "warnings": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "answer_type", "claims", "warnings"],
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
        "description": "Busca fuentes públicas con Tavily. Para empleo en Argentina, incluí LinkedIn, Glassdoor, Computrabajo, Bumeran, ZonaJobs, Indeed, Get on Board o Portal Empleo cuando correspondan al tema.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "company_name": {
                    "type": "string",
                    "description": "Empresa estable investigada; obligatoria para informes completos.",
                },
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
    "get_cv_profile": {
        "description": "Recupera el CV adjunto, activo o predeterminado usando solo evidencia seleccionada.",
        "parameters": {
            "type": "object",
            "properties": {"cv_id": {"type": "string"}},
        },
    },
    "analyze_job_description": {
        "description": "Analiza la descripción de puesto adjunta o activa.",
        "parameters": {
            "type": "object",
            "properties": {"job_description_id": {"type": "string"}},
        },
    },
    "prepare_cv_recommendations": {
        "description": "Crea recomendaciones de CV basadas en evidencia y pausa para revisión humana.",
        "parameters": {
            "type": "object",
            "properties": {
                "cv_id": {"type": "string"},
                "job_description_id": {"type": "string"},
                "report_id": {"type": "string"},
                "company_name": {"type": "string"},
                "require_job_description": {"type": "boolean"},
            },
        },
    },
    "finish_research": {
        "description": "Finaliza la investigación creando un informe durable. Para una respuesta normal, usa respond.",
        "parameters": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string"},
                "output_type": {"type": "string", "enum": ["report"]},
                "stopping_reason": {"type": "string"},
                "unresolved_topics": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["company_name", "output_type", "stopping_reason"],
        },
    },
}


def elapsed_datetime_ms(started) -> int:
    now = utc_now()
    if started.tzinfo is None:
        now = now.replace(tzinfo=None)
    return max(0, int((now - started).total_seconds() * 1000))


def evidence_identifier(source_id: str, topic: str, excerpt: str) -> str:
    digest = hashlib.sha256(f"{source_id}|{topic}|{excerpt}".encode("utf-8")).hexdigest()[:16]
    return f"evidence_{digest}"


def clean_generated_answer(value: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if re.fullmatch(r"(?:[-*_]\s*){3,}", line):
            continue
        if re.fullmatch(r"\|?(?:\s*:?-{3,}:?\s*\|)+\s*", line):
            continue
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^>\s?", "", line)
        line = re.sub(r"^[-*+]\s+", "", line)
        line = line.replace("**", "").replace("__", "").replace("`", "")
        if line.startswith("|") and line.endswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|") if cell.strip()]
            line = f"{cells[0]}: {', '.join(cells[1:])}" if len(cells) > 1 else "".join(cells)
        cleaned_lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned_lines)).strip()


def text_chunks(value: str, size: int) -> list[str]:
    return [value[index : index + size] for index in range(0, len(value), size)] or [""]


def run_agent_task(task_id: str, bind: Engine) -> None:
    task_session = sessionmaker(bind=bind, autoflush=False, autocommit=False)
    with task_session() as db:
        AgentOrchestrator(db).run(task_id)
