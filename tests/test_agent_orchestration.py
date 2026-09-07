from __future__ import annotations

from collections.abc import Generator
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.core.config import Settings
from backend.app.db import models
from backend.app.db.conversation_repository import ConversationRepository
from backend.app.db.models import Base
from backend.app.db.repositories import CompanyRepository, ReportRepository
from backend.app.domain.reports import (
    ClaimSchema,
    ClaimType,
    CompanySchema,
    ConfidenceLevel,
    EvidenceSchema,
    EvidenceTopic,
    ReportMetadataSchema,
    ReportStatus,
    SectionSchema,
    SectionType,
    SourceSchema,
    SourceType,
    StructuredReportSchema,
)
from backend.app.llm.deepseek import DeepSeekAPIError, DeepSeekResponse, DeepSeekToolCall
from backend.app.llm.synthesizer import SynthesisError
from backend.app.research.content_extractor import ExtractedContent
from backend.app.research.evidence_classifier import classify_evidence
from backend.app.research.scoring import score_search_result
from backend.app.research.types import SearchResult
from backend.app.services.agent_orchestrator import (
    AgentAnswer,
    AgentCompleted,
    AgentEvidence,
    AgentOrchestrator,
    AgentSource,
    AgentState,
    GroundedClaim,
    clean_generated_answer,
    evidence_identifier,
)
from backend.app.services.report_builder import MockReportBuilder


class SequenceClient:
    def __init__(self, calls: list[tuple[str, dict]]) -> None:
        self.calls = list(calls)
        self.requests = []
        self.last_usage = {}

    def select_tool(self, messages, tools, **kwargs):
        self.requests.append({"messages": messages, "tools": tools, **kwargs})
        name, arguments = self.calls.pop(0)
        call = DeepSeekToolCall(id=f"call_{len(self.requests)}", name=name, arguments=arguments)
        return DeepSeekResponse(
            tool_calls=[call],
            assistant_message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": name, "arguments": "{}"},
                    }
                ],
            },
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "cache_hit_tokens": 5,
                "cache_miss_tokens": 5,
            },
        )


class FinalizingClient:
    def __init__(self) -> None:
        self.requests = []
        self.last_usage = {}

    def select_tool(self, messages, tools, **kwargs):
        self.requests.append({"messages": messages, "tools": tools, **kwargs})
        call = DeepSeekToolCall(id="call_1", name="respond", arguments={"answer_type": "guidance"})
        return DeepSeekResponse(
            tool_calls=[call],
            assistant_message={"role": "assistant", "content": None, "tool_calls": []},
        )

    def generate_json(self, *_args, **_kwargs):
        self.requests.append({"json": True})
        return AgentAnswer(
            answer="Practicá explicar decisiones técnicas con ejemplos concretos.",
            answer_type="guidance",
        ).model_dump(mode="json")


class GroundedFinalizingClient:
    def __init__(self) -> None:
        self.requests = []
        self.last_usage = {}

    def generate_json(self, *_args, **_kwargs):
        self.requests.append({"json": True})
        return AgentAnswer(
            answer="Acme desarrolla software.",
            answer_type="grounded",
            claims=[GroundedClaim(text="Actividad de Acme", evidence_ids=["evidence-1"])],
        ).model_dump(mode="json")


class MissingCallOnceClient:
    def __init__(self) -> None:
        self.requests = []

    def select_tool(self, messages, tools, **kwargs):
        self.requests.append({"messages": messages, "tools": tools, **kwargs})
        if len(self.requests) == 1:
            return DeepSeekResponse(content="respuesta libre")
        call = DeepSeekToolCall(
            id="call_2",
            name="respond",
            arguments={
                "answer": "Respuesta recuperada mediante una acción válida.",
                "answer_type": "guidance",
                "claims": [],
                "warnings": [],
            },
        )
        return DeepSeekResponse(
            tool_calls=[call],
            assistant_message={"role": "assistant", "content": None, "tool_calls": []},
        )


class RaisingClient:
    def select_tool(self, *_args, **_kwargs):
        raise DeepSeekAPIError(
            "DeepSeek API request failed.", status_code=429, retry_after="22.4"
        )


class MultipleCallsOnceClient:
    def __init__(self) -> None:
        self.requests = []

    def select_tool(self, messages, tools, **kwargs):
        self.requests.append({"messages": messages, "tools": tools, **kwargs})
        respond = DeepSeekToolCall(
            id="call_respond",
            name="respond",
            arguments={
                "answer": "Respuesta recuperada tras rechazar acciones paralelas.",
                "answer_type": "guidance",
                "claims": [],
                "warnings": [],
            },
        )
        if len(self.requests) == 1:
            search = DeepSeekToolCall(
                id="call_search",
                name="search_web",
                arguments={"query": "consulta innecesaria"},
            )
            return DeepSeekResponse(tool_calls=[respond, search])
        return DeepSeekResponse(tool_calls=[respond])


class FakeSearch:
    def __init__(self, result: SearchResult) -> None:
        self.result = result
        self.queries = []

    def search(self, query: str, limit: int):
        self.queries.append((query, limit))
        return [self.result]


class FakeExtractor:
    def __init__(self, content: ExtractedContent) -> None:
        self.content = content
        self.urls = []

    def extract(self, url: str):
        self.urls.append(url)
        return self.content


@pytest.fixture()
def db_session() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


def create_task(
    db: Session,
    content: str,
    *,
    budget: dict | None = None,
) -> tuple[models.Conversation, models.TaskRun]:
    repo = ConversationRepository(db)
    conversation = repo.create()
    message = repo.add_message(conversation, role="user", content=content, status="completed")
    task = repo.create_task(
        conversation,
        message,
        budget=budget
        or {
            "profile": "standard",
            "model_turns": 12,
            "searches": 6,
            "inspections": 10,
            "elapsed_seconds": 180,
        },
    )
    db.commit()
    return conversation, task


def settings(**changes) -> Settings:
    return Settings(
        deepseek_api_key="test-key",
        gemini_api_key="embedding-key",
        search_provider="mock",
        **changes,
    )


def completion_ready_report(report: models.Report) -> StructuredReportSchema:
    sources = [
        SourceSchema(
            id="source_official",
            title="Acme Argentina",
            url="https://acme.example/argentina",
            domain="acme.example",
            source_type=SourceType.official,
            reliability_score=5,
            accessed_at="2026-08-28T00:00:00+00:00",
        ),
        SourceSchema(
            id="source_jobs",
            title="Acme jobs in Argentina",
            url="https://linkedin.com/jobs/acme-argentina",
            domain="linkedin.com",
            source_type=SourceType.linkedin,
            reliability_score=4,
            accessed_at="2026-08-28T00:00:00+00:00",
        ),
    ]
    evidence = [
        EvidenceSchema(
            id="evidence_business",
            source_id="source_official",
            topic=EvidenceTopic.business,
            claim="Acme desarrolla software.",
            raw_text_excerpt="Acme desarrolla software en Argentina.",
            confidence=ConfidenceLevel.high,
        ),
        EvidenceSchema(
            id="evidence_presence",
            source_id="source_official",
            topic=EvidenceTopic.argentina_presence,
            claim="Acme opera en Buenos Aires.",
            raw_text_excerpt="Acme opera en Buenos Aires, Argentina.",
            confidence=ConfidenceLevel.high,
        ),
        EvidenceSchema(
            id="evidence_roles",
            source_id="source_jobs",
            topic=EvidenceTopic.open_roles,
            claim="Hay 2 vacantes activas.",
            raw_text_excerpt="2 vacantes activas de Acme en Argentina.",
            confidence=ConfidenceLevel.high,
        ),
    ]

    def section(section_type, title, text, evidence_id):
        return SectionSchema(
            type=section_type,
            title=title,
            summary=text,
            claims=[
                ClaimSchema(
                    id=f"claim_{section_type.value}",
                    type=ClaimType.fact,
                    text=text,
                    evidence_ids=[evidence_id],
                    confidence=ConfidenceLevel.high,
                )
            ],
            confidence=ConfidenceLevel.high,
        )

    return StructuredReportSchema(
        report_id=report.id,
        company=CompanySchema(
            id=report.company.id,
            name=report.company.name,
            normalized_name=report.company.normalized_name,
        ),
        status=ReportStatus.completed,
        sections=[
            section(SectionType.executive_summary, "Resumen", "Acme desarrolla software.", "evidence_business"),
            section(SectionType.business, "Negocio", "Acme desarrolla software.", "evidence_business"),
            section(SectionType.argentina_presence, "Argentina", "Acme opera en Buenos Aires.", "evidence_presence"),
            section(SectionType.open_roles, "Vacantes", "Hay 2 vacantes activas.", "evidence_roles"),
        ],
        sources=sources,
        evidence=evidence,
        metadata=ReportMetadataSchema(search_provider="tavily", llm_model="deepseek-test"),
    )


def test_direct_guidance_completes_with_persisted_tool_and_message_events(db_session: Session) -> None:
    conversation, task = create_task(db_session, "Dame un consejo general para una entrevista")
    client = SequenceClient(
        [
            (
                "respond",
                {
                    "answer": "Prepará ejemplos concretos y practicá respuestas breves.",
                    "answer_type": "guidance",
                    "claims": [],
                    "warnings": [],
                },
            )
        ]
    )

    AgentOrchestrator(db_session, settings(), client=client).run(task.id)

    db_session.refresh(task)
    assert task.status == "completed"
    messages = ConversationRepository(db_session).list_messages(conversation.id)
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[-1].content.startswith("Prepará ejemplos")
    usage = __import__("json").loads(task.usage_json)
    assert usage["total_tokens"] == 12
    assert usage["cache_hit_tokens"] == 5
    event_types = [event.event_type for event in ConversationRepository(db_session).list_events(conversation.id)]
    assert event_types == [
        "task.started",
        "task.progress",
        "tool.started",
        "message.started",
        "message.delta",
        "message.completed",
        "tool.completed",
        "task.completed",
    ]


def test_general_interview_preparation_cannot_use_research_tools(db_session: Session) -> None:
    _, task = create_task(
        db_session,
        "¿Cómo puedo prepararme para mi primera entrevista técnica?",
    )
    client = SequenceClient(
        [
            (
                "respond",
                {
                    "answer": "Practicá explicar tus decisiones y prepará ejemplos concretos.",
                    "answer_type": "guidance",
                    "claims": [],
                    "warnings": [],
                },
            )
        ]
    )

    AgentOrchestrator(db_session, settings(), client=client).run(task.id)

    db_session.refresh(task)
    assert task.status == "completed"
    tool_names = {item["function"]["name"] for item in client.requests[0]["tools"]}
    assert not tool_names & {
        "search_web",
        "inspect_page",
        "review_evidence",
        "retrieve_reports",
        "compare_reports",
        "finish_research",
    }
    assert __import__("json").loads(task.usage_json)["searches"] == 0


def test_general_junior_ai_profile_question_cannot_use_research_tools(
    db_session: Session,
) -> None:
    conversation, task = create_task(
        db_session,
        "¿Qué debería incluir un perfil profesional para un puesto junior de inteligencia artificial?",
    )
    client = SequenceClient(
        [
            (
                "respond",
                {
                    "answer": "## Perfil\n\n**Incluí** proyectos concretos.\n\n---\n\n- Explicá tu aporte.",
                    "answer_type": "guidance",
                    "claims": [],
                    "warnings": [],
                },
            )
        ]
    )

    AgentOrchestrator(db_session, settings(), client=client).run(task.id)

    db_session.refresh(task)
    assert task.status == "completed"
    tool_names = {item["function"]["name"] for item in client.requests[0]["tools"]}
    assert not tool_names & {
        "search_web",
        "inspect_page",
        "review_evidence",
        "retrieve_reports",
        "compare_reports",
        "finish_research",
    }
    assert __import__("json").loads(task.usage_json)["searches"] == 0
    assistant = ConversationRepository(db_session).list_messages(conversation.id)[-1]
    assert assistant.content == "Perfil\n\nIncluí proyectos concretos.\n\nExplicá tu aporte."


def test_company_specific_interview_preparation_still_requires_grounding(
    db_session: Session,
) -> None:
    _, task = create_task(
        db_session,
        "¿Cómo puedo prepararme para una entrevista técnica en Google?",
    )

    state = AgentOrchestrator(db_session, settings()).load_state(task)

    assert state.requires_grounding is True


@pytest.mark.parametrize(
    ("focus", "expected_goal"),
    [
        ("vacantes", "Vacantes actuales de Globant en Argentina."),
        ("general", "Perfil general de Globant como empleador en Argentina."),
        ("sueldos", "Sueldos de Globant en Argentina."),
        ("cultura", "Cultura laboral de Globant en Argentina."),
        ("entrevistas", "Proceso de entrevistas de Globant en Argentina."),
    ],
)
def test_short_focus_uses_active_company_as_a_standard_grounded_request(
    db_session: Session,
    focus: str,
    expected_goal: str,
) -> None:
    conversation, previous = create_task(
        db_session,
        "Generá un informe completo sobre Globant Argentina",
        budget={
            "profile": "full_report",
            "model_turns": 16,
            "searches": 12,
            "inspections": 20,
            "elapsed_seconds": 300,
        },
    )
    prior_state = AgentState(
        goal="Generá un informe completo sobre Globant Argentina",
        company_name="Globant",
        sources=[
            AgentSource(
                id="globant-official",
                title="Globant",
                url="https://www.globant.com/es",
                domain="globant.com",
                source_type="official",
                accessed_at="2026-09-04T00:00:00+00:00",
            )
        ],
        evidence=[
            AgentEvidence(
                id="globant-business",
                source_id="globant-official",
                topic="business",
                claim="Globant ofrece soluciones digitales.",
                excerpt="Globant ofrece soluciones digitales.",
                confidence="medium",
            )
        ],
    )
    previous.working_state_json = prior_state.model_dump_json()
    conversation.active_context_json = '{"active_company_name":"Globant"}'
    repo = ConversationRepository(db_session)
    message = repo.add_message(conversation, role="user", content=focus, status="completed")
    task = repo.create_task(
        conversation,
        message,
        budget={
            "profile": "standard",
            "model_turns": 12,
            "searches": 6,
            "inspections": 10,
            "elapsed_seconds": 180,
        },
    )
    db_session.commit()

    state = AgentOrchestrator(db_session, settings()).load_state(task)

    assert state.goal == expected_goal
    assert state.company_name == "Globant"
    assert state.requires_grounding is True
    assert state.sources[0].id == "globant-official"
    assert __import__("json").loads(task.budget_json)["profile"] == "standard"


def test_bare_continue_still_requires_clarification_with_an_active_company(
    db_session: Session,
) -> None:
    conversation, _ = create_task(db_session, "Consulta anterior")
    conversation.active_context_json = '{"active_company_name":"Globant"}'
    repo = ConversationRepository(db_session)
    message = repo.add_message(conversation, role="user", content="continue", status="completed")
    task = repo.create_task(conversation, message, budget={"profile": "standard"})
    db_session.commit()

    state = AgentOrchestrator(db_session, settings()).load_state(task)

    assert state.goal == "continue"
    assert AgentOrchestrator.requires_clarification(state) is True


def test_company_context_is_saved_for_later_focused_requests(db_session: Session) -> None:
    conversation, task = create_task(db_session, "Investigá Globant")
    state = AgentState(goal="Investigá Globant")

    AgentOrchestrator(db_session, settings()).set_company_context(task, state, "Globant")

    assert __import__("json").loads(conversation.active_context_json)["active_company_name"] == "Globant"


def test_terminal_response_cannot_only_promise_future_research(db_session: Session) -> None:
    _, task = create_task(db_session, "Buscá vacantes de Globant")
    state = AgentState(goal="Buscá vacantes de Globant", requires_grounding=True)

    with pytest.raises(ValueError, match="respond es terminal"):
        AgentOrchestrator(db_session, settings()).tool_respond(
            task,
            state,
            {},
            {
                "answer": "Voy a buscar vacantes actuales y después te respondo.",
                "answer_type": "grounded",
                "claims": [],
                "warnings": ["Búsqueda pendiente."],
            },
        )


def test_clean_generated_answer_removes_markdown_but_keeps_normal_punctuation() -> None:
    raw = (
        "## Perfil profesional\n\n"
        "**Resumen:** Python, SQL y proyectos.\n\n"
        "---\n"
        "> Explicá tu aporte.\n"
        "- Mostrá resultados.\n\n"
        "| Área | Contenido |\n"
        "| --- | --- |\n"
        "| Stack | `Python` y SQL |"
    )

    cleaned = clean_generated_answer(raw)

    assert cleaned == (
        "Perfil profesional\n\n"
        "Resumen: Python, SQL y proyectos.\n\n"
        "Explicá tu aporte.\n"
        "Mostrá resultados.\n\n"
        "Área: Contenido\n"
        "Stack: Python y SQL"
    )


def test_respond_action_uses_buffered_structured_finalization(db_session: Session) -> None:
    conversation, task = create_task(db_session, "Dame un consejo general para una entrevista")
    client = FinalizingClient()

    AgentOrchestrator(db_session, settings(), client=client).run(task.id)

    db_session.refresh(task)
    assert task.status == "completed"
    assert len(client.requests) == 2
    assistant = ConversationRepository(db_session).list_messages(conversation.id)[-1]
    assert assistant.content.startswith("Practicá explicar")


def test_decision_turn_forces_function_mode_and_repairs_one_missing_call(db_session: Session) -> None:
    _, task = create_task(db_session, "Dame un consejo general para una entrevista")
    client = MissingCallOnceClient()

    AgentOrchestrator(db_session, settings(), client=client).run(task.id)

    db_session.refresh(task)
    assert task.status == "completed"
    assert len(client.requests) == 2
    assert client.requests[0]["tools"]
    assert client.requests[1]["tool_choice"] == {
        "type": "function",
        "function": {"name": "respond"},
    }


def test_respond_tool_requests_the_complete_validated_answer() -> None:
    respond = next(
        item for item in AgentOrchestrator.function_tools()
        if item["function"]["name"] == "respond"
    )

    assert respond["function"]["parameters"]["required"] == [
        "answer",
        "answer_type",
        "claims",
        "warnings",
    ]
    finish = next(
        item for item in AgentOrchestrator.function_tools()
        if item["function"]["name"] == "finish_research"
    )
    assert finish["function"]["parameters"]["properties"]["output_type"]["enum"] == [
        "report"
    ]


def test_compact_context_keeps_evidence_after_long_conversation(db_session: Session) -> None:
    conversation, task = create_task(db_session, "Investigá Acme")
    repo = ConversationRepository(db_session)
    for _ in range(12):
        repo.add_message(conversation, role="user", content="x" * 3000, status="completed")
    state = AgentState(
        goal="Investigá Acme",
        evidence=[
            AgentEvidence(
                id="evidence-must-remain",
                source_id="source-1",
                topic="general",
                claim="Acme publicó una vacante.",
                excerpt="e" * 1200,
            )
        ],
    )

    context = AgentOrchestrator(db_session, settings()).build_context(task, state)

    assert "evidence-must-remain" in context
    assert len(context) < 24_000


def test_parallel_tool_calls_are_serialized_to_one_controlled_action(
    db_session: Session,
) -> None:
    _, task = create_task(db_session, "Dame un consejo general para una entrevista")
    client = MultipleCallsOnceClient()

    AgentOrchestrator(db_session, settings(), client=client).run(task.id)

    db_session.refresh(task)
    assert task.status == "completed"
    assert len(client.requests) == 1
    events = ConversationRepository(db_session).list_events(task.conversation_id)
    assert all(event.tool_name != "search_web" for event in events)


def test_ambiguous_single_word_only_allows_clarification(db_session: Session) -> None:
    _, task = create_task(db_session, "google")
    client = SequenceClient(
        [
            (
                "request_clarification",
                {
                    "question": "¿Querés investigar Google, buscar vacantes o hacer otra consulta?",
                    "options": [],
                },
            )
        ]
    )

    AgentOrchestrator(db_session, settings(), client=client).run(task.id)

    db_session.refresh(task)
    assert task.status == "needs_clarification"
    pending = __import__("json").loads(task.working_state_json)["pending_clarification"]
    assert pending["question"] == "¿Querés investigar Google, buscar vacantes o hacer otra consulta?"
    declarations = client.requests[0]["tools"]
    assert [item["function"]["name"] for item in declarations] == ["request_clarification"]


def test_repeated_name_answer_repeats_unresolved_clarification(db_session: Session) -> None:
    _, task = create_task(db_session, "stanley")
    task.working_state_json = __import__("json").dumps(
        {
            "goal": "stanley",
            "requires_grounding": False,
            "clarification_response": {"content": "stanley", "selected_option_ids": []},
            "pending_clarification": {
                "question": 'Mencionaste "Stanley". ¿Qué significa Stanley en este contexto?',
                "options": [],
            },
        }
    )
    db_session.commit()
    client = SequenceClient(
        [
            (
                "request_clarification",
                {"question": "¿Querés investigar Stanley como empresa?", "options": []},
            )
        ]
    )

    AgentOrchestrator(db_session, settings(), client=client).run(task.id)

    db_session.refresh(task)
    assert task.status == "needs_clarification"
    assert [item["function"]["name"] for item in client.requests[0]["tools"]] == [
        "request_clarification"
    ]


def test_explicit_report_infers_company_and_requires_initial_search(db_session: Session) -> None:
    conversation, task = create_task(
        db_session,
        "Generá un informe sobre Accenture como empleador en Argentina.",
        budget={
            "profile": "full_report",
            "model_turns": 16,
            "searches": 12,
            "inspections": 20,
            "elapsed_seconds": 300,
        },
    )
    conversation.active_context_json = '{"active_company_name":"Globant"}'
    db_session.commit()

    orchestrator = AgentOrchestrator(db_session, settings())
    state = orchestrator.load_state(task)

    assert state.company_name == "Accenture"
    action = orchestrator.required_full_report_action(state)
    assert action is not None
    assert action["name"] == "search_web"
    assert action["arguments"]["company_name"] == "Accenture"


def test_weather_without_location_requires_clarification(db_session: Session) -> None:
    _, task = create_task(db_session, "clima de hoy")
    client = SequenceClient(
        [("request_clarification", {"question": "¿De qué ciudad querés el clima?", "options": []})]
    )

    AgentOrchestrator(db_session, settings(), client=client).run(task.id)

    db_session.refresh(task)
    assert task.status == "needs_clarification"


def test_web_search_and_inspection_ground_factual_answer(db_session: Session) -> None:
    url = "https://example.com/company"
    search_result = SearchResult(
        title="Empresa Example",
        url=url,
        snippet="Servicios de software y productos digitales.",
        rank=1,
        query_topic=EvidenceTopic.business,
    )
    content = ExtractedContent(
        url=url,
        title="Empresa Example",
        text="La empresa ofrece servicios de software y productos digitales para clientes regionales.",
        language="es",
    )
    scored = score_search_result("", search_result)
    classified = classify_evidence(scored, content)[0]
    evidence_id = evidence_identifier(
        scored.source_id,
        classified.topic.value,
        classified.raw_text_excerpt,
    )
    client = SequenceClient(
        [
            ("search_web", {"query": "Empresa Example negocio", "topic": "business", "limit": 3}),
            ("inspect_page", {"source_id": scored.source_id}),
            (
                "respond",
                {
                    "answer": "Empresa Example ofrece servicios de software y productos digitales.",
                    "answer_type": "grounded",
                    "claims": [{"text": "Oferta tecnológica", "evidence_ids": [evidence_id]}],
                    "warnings": [],
                },
            ),
        ]
    )
    conversation, task = create_task(db_session, "¿A qué se dedica Empresa Example?")

    AgentOrchestrator(
        db_session,
        settings(),
        client=client,
        search_provider=FakeSearch(search_result),
        extractor=FakeExtractor(content),
    ).run(task.id)

    db_session.refresh(task)
    assert task.status == "completed"
    assistant = ConversationRepository(db_session).list_messages(conversation.id)[-1]
    citations = __import__("json").loads(assistant.citations_json)
    assert citations[0]["evidence_id"] == evidence_id
    assert citations[0]["url"] == url


def test_clarification_pauses_and_can_continue_from_persisted_state(db_session: Session) -> None:
    conversation, task = create_task(db_session, "Dame un consejo general adaptado a mi situación")
    first_client = SequenceClient(
        [
            (
                "request_clarification",
                {
                    "question": "¿Para qué tipo de entrevista te estás preparando?",
                    "options": [{"id": "tecnica", "label": "Técnica"}],
                },
            )
        ]
    )
    AgentOrchestrator(db_session, settings(), client=first_client).run(task.id)
    db_session.refresh(task)
    assert task.status == "needs_clarification"
    assert "entrevista" in (task.pause_reason_json or "")

    repo = ConversationRepository(db_session)
    repo.add_message(conversation, role="user", content="Técnica", status="completed")
    working_state = __import__("json").loads(task.working_state_json)
    working_state["clarification_response"] = {"content": "Técnica", "selected_option_ids": []}
    task.working_state_json = __import__("json").dumps(working_state)
    task.pause_reason_json = None
    repo.update_task(task, "pending")
    db_session.commit()
    second_client = SequenceClient(
        [
            (
                "respond",
                {
                    "answer": "Practicá explicar decisiones técnicas y sus tradeoffs.",
                    "answer_type": "guidance",
                    "claims": [],
                    "warnings": [],
                },
            )
        ]
    )
    AgentOrchestrator(db_session, settings(), client=second_client).run(task.id)
    db_session.refresh(task)
    assert task.status == "completed"
    resumed_tools = second_client.requests[0]["tools"]
    assert "request_clarification" not in [item["function"]["name"] for item in resumed_tools]


def test_budget_exhaustion_completes_with_a_transparent_limited_answer(db_session: Session) -> None:
    conversation, task = create_task(
        db_session,
        "Investigá una empresa",
        budget={
            "profile": "standard",
            "model_turns": 0,
            "searches": 0,
            "inspections": 0,
            "elapsed_seconds": 0,
        },
    )
    AgentOrchestrator(db_session, settings(), client=SequenceClient([])).run(task.id)
    db_session.refresh(task)
    assert task.status == "completed"
    events = ConversationRepository(db_session).list_events(task.conversation_id)
    assert events[-1].event_type == "task.completed"
    assistant = ConversationRepository(db_session).list_messages(conversation.id)[-1]
    assert "No pude verificar" in assistant.content


def test_budget_exhaustion_forces_guidance_finalization_without_extra_decision(
    db_session: Session,
) -> None:
    conversation, task = create_task(
        db_session,
        "Dame un consejo general para entrevistas",
        budget={
            "profile": "standard",
            "model_turns": 0,
            "searches": 0,
            "inspections": 0,
            "elapsed_seconds": 180,
        },
    )
    client = FinalizingClient()

    AgentOrchestrator(db_session, settings(), client=client).run(task.id)

    db_session.refresh(task)
    assert task.status == "completed"
    assert len(client.requests) == 1
    assert ConversationRepository(db_session).list_messages(conversation.id)[-1].role == "assistant"


def test_budget_exhaustion_with_evidence_completes_a_grounded_answer(
    db_session: Session,
) -> None:
    conversation, task = create_task(
        db_session,
        "¿A qué se dedica Acme?",
        budget={
            "profile": "standard",
            "model_turns": 0,
            "searches": 6,
            "inspections": 10,
            "elapsed_seconds": 180,
        },
    )
    task.working_state_json = AgentState(
        goal="¿A qué se dedica Acme?",
        sources=[
            AgentSource(
                id="source-1",
                title="Acme oficial",
                url="https://example.com/acme",
                domain="example.com",
                source_type="official",
                reliability_score=90,
                rank=1,
                topic="business",
                accessed_at="2026-08-17T00:00:00+00:00",
                inspected=True,
            ),
        ],
        evidence=[
            AgentEvidence(
                id="evidence-1",
                source_id="source-1",
                topic="business",
                claim="Acme desarrolla software.",
                excerpt="Acme desarrolla software.",
                confidence="high",
            ),
        ],
    ).model_dump_json()
    db_session.commit()
    client = GroundedFinalizingClient()

    AgentOrchestrator(db_session, settings(), client=client).run(task.id)

    db_session.refresh(task)
    assert task.status == "completed"
    assert client.requests == [{"json": True}]
    assistant = ConversationRepository(db_session).list_messages(conversation.id)[-1]
    assert __import__("json").loads(assistant.citations_json)[0]["evidence_id"] == "evidence-1"


def test_full_report_budget_reserves_finish_research_for_a_saved_artifact(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    conversation, task = create_task(
        db_session,
        "Generá un informe de Acme",
        budget={
            "profile": "full_report",
            "model_turns": 1,
            "searches": 12,
            "inspections": 20,
            "elapsed_seconds": 300,
        },
    )
    task.working_state_json = AgentState(
        goal="Generá un informe de Acme",
        company_name="Acme",
        sources=[
            AgentSource(
                id="source-1",
                title="Acme oficial",
                url="https://example.com/acme",
                snippet="Acme desarrolla software.",
                domain="example.com",
                source_type="official",
                reliability_score=90,
                rank=1,
                topic="business",
                accessed_at="2026-08-17T00:00:00+00:00",
                inspected=True,
            ),
            AgentSource(
                id="source-2",
                title="Software Engineer en Acme",
                url="https://www.linkedin.com/jobs/view/software-engineer-acme-123",
                snippet="Acme busca Software Engineer en Buenos Aires, Argentina.",
                domain="linkedin.com",
                source_type="job_board",
                reliability_score=4,
                rank=1,
                topic="open_roles",
                accessed_at="2026-08-17T00:00:00+00:00",
                inspected=True,
            ),
        ],
        evidence=[
            AgentEvidence(
                id="evidence-1",
                source_id="source-1",
                topic="business",
                claim="Acme desarrolla software.",
                excerpt="Acme desarrolla software.",
                confidence="high",
            ),
            AgentEvidence(
                id="evidence-2",
                source_id="source-1",
                topic="argentina_presence",
                claim="Acme opera en Buenos Aires.",
                excerpt="Acme opera en Buenos Aires, Argentina.",
                confidence="high",
            ),
            AgentEvidence(
                id="evidence-3",
                source_id="source-2",
                topic="open_roles",
                claim="Acme busca Software Engineer.",
                excerpt="Acme busca Software Engineer en Buenos Aires, Argentina.",
                confidence="high",
            ),
        ],
    ).model_dump_json()
    db_session.commit()

    def fake_synthesize(_self, request):
        report = db_session.get(models.Report, request.report_id)
        return completion_ready_report(report)

    monkeypatch.setattr(
        "backend.app.services.agent_orchestrator.DeepSeekReportSynthesizer.synthesize",
        fake_synthesize,
    )
    monkeypatch.setattr(
        "backend.app.services.agent_orchestrator.schedule_report_embedding_task",
        lambda _report_id: None,
    )
    client = SequenceClient(
        [
            (
                "finish_research",
                {
                    "company_name": "Acme",
                    "output_type": "report",
                    "stopping_reason": "Evidencia suficiente.",
                    "unresolved_topics": [],
                },
            )
        ]
    )

    AgentOrchestrator(db_session, settings(), client=client).run(task.id)

    db_session.refresh(task)
    assert task.status == "completed"
    assert [item["function"]["name"] for item in client.requests[0]["tools"]] == [
        "finish_research"
    ]
    assert any(
        item.artifact_type == "report"
        for item in ConversationRepository(db_session).list_artifacts(conversation.id)
    )
    assert "approval.required" not in [
        item.event_type for item in ConversationRepository(db_session).list_events(conversation.id)
    ]


def test_full_report_budget_returns_limited_answer_without_core_coverage(
    db_session: Session,
) -> None:
    conversation, task = create_task(
        db_session,
        "GenerÃ¡ un informe de Acme",
        budget={
            "profile": "full_report",
            "model_turns": 0,
            "searches": 12,
            "inspections": 20,
            "elapsed_seconds": 300,
        },
    )
    task.working_state_json = AgentState(
        goal="GenerÃ¡ un informe de Acme",
        sources=[
            AgentSource(
                id="source-1",
                title="Acme oficial",
                url="https://example.com/acme",
                domain="example.com",
                source_type="official",
                reliability_score=5,
                accessed_at="2026-08-28T00:00:00+00:00",
                inspected=True,
            )
        ],
        evidence=[
            AgentEvidence(
                id="evidence-1",
                source_id="source-1",
                topic="business",
                claim="Acme desarrolla software.",
                excerpt="Acme desarrolla software.",
                confidence="high",
            )
        ],
    ).model_dump_json()
    db_session.commit()

    AgentOrchestrator(db_session, settings(), client=SequenceClient([])).run(task.id)

    db_session.refresh(task)
    message = ConversationRepository(db_session).list_messages(conversation.id)[-1]
    assert task.status == "completed"
    assert __import__("json").loads(task.usage_json)["model_turns"] == 0
    assert "No pude verificar evidencia suficiente" in message.content
    assert not ConversationRepository(db_session).list_artifacts(conversation.id)


def test_review_evidence_returns_full_report_coverage(db_session: Session) -> None:
    _, task = create_task(
        db_session,
        "GenerÃ¡ un informe de Acme",
        budget={"profile": "full_report", "model_turns": 16, "searches": 12, "inspections": 20, "elapsed_seconds": 300},
    )
    state = AgentState(
        goal="GenerÃ¡ un informe de Acme",
        sources=[
            AgentSource(
                id="official",
                title="Acme oficial",
                url="https://acme.example",
                domain="acme.example",
                source_type="official",
                reliability_score=5,
                accessed_at="2026-08-28T00:00:00+00:00",
            )
        ],
        evidence=[
            AgentEvidence(
                id="business",
                source_id="official",
                topic="business",
                claim="Acme desarrolla software.",
                excerpt="Acme desarrolla software.",
                confidence="high",
            )
        ],
    )

    result = AgentOrchestrator(db_session, settings()).tool_review_evidence(task, state, {}, {})

    coverage = result.payload["report_coverage"]
    assert coverage["corporate_source"] is True
    assert coverage["employment_source"] is False
    assert coverage["core_topics"]["business"] is True
    assert coverage["core_topics"]["open_roles"] is False
    assert coverage["ready"] is False


def test_full_report_search_requires_company_and_classifies_owned_careers(
    db_session: Session,
) -> None:
    _, task = create_task(
        db_session,
        "Generá un informe completo sobre Globant Argentina",
        budget={
            "profile": "full_report",
            "model_turns": 16,
            "searches": 12,
            "inspections": 20,
            "elapsed_seconds": 300,
        },
    )
    result = SearchResult(
        title="Careers at Globant",
        url="https://www.globant.com/careers",
        snippet="Explore jobs in Argentina.",
        rank=1,
        query_topic=EvidenceTopic.open_roles,
    )
    state = AgentState(goal="Generá un informe completo sobre Globant Argentina")
    orchestrator = AgentOrchestrator(
        db_session,
        settings(),
        search_provider=FakeSearch(result),
    )

    with pytest.raises(ValueError, match="requiere company_name"):
        orchestrator.tool_search_web(
            task,
            state,
            {"searches": 0},
            {"query": "Globant careers", "topic": "open_roles"},
        )

    tool_result = orchestrator.tool_search_web(
        task,
        state,
        {"searches": 0},
        {
            "query": "Globant careers Argentina",
            "company_name": "Globant Argentina",
            "topic": "open_roles",
        },
    )

    assert state.company_name == "Globant Argentina"
    assert state.sources[0].source_type == SourceType.career_page.value
    assert tool_result.payload["report_coverage"]["career_source"] is True


def test_company_context_reclassifies_sources_and_cannot_change_company(
    db_session: Session,
) -> None:
    _, task = create_task(
        db_session,
        "Generá un informe completo sobre Globant Argentina",
        budget={
            "profile": "full_report",
            "model_turns": 16,
            "searches": 12,
            "inspections": 20,
            "elapsed_seconds": 300,
        },
    )
    state = AgentState(
        goal="Generá un informe completo sobre Globant Argentina",
        sources=[
            AgentSource(
                id="globant",
                title="Globant",
                url="https://www.globant.com/about",
                domain="globant.com",
                source_type=SourceType.secondary.value,
                accessed_at="2026-08-28T00:00:00+00:00",
            )
        ],
    )
    orchestrator = AgentOrchestrator(db_session, settings())

    orchestrator.set_company_context(task, state, "Globant Argentina")

    assert state.sources[0].source_type == SourceType.official.value
    with pytest.raises(ValueError, match="no puede cambiar"):
        orchestrator.set_company_context(task, state, "IBM Argentina")


def test_full_report_coverage_recommends_sources_individually_and_stops_when_ready() -> None:
    weak_state = AgentState(
        goal="Generá un informe completo sobre Globant Argentina",
        company_name="Globant Argentina",
        sources=[
            AgentSource(
                id="linkedin",
                title="Globant jobs",
                url="https://www.linkedin.com/jobs/globant",
                domain="linkedin.com",
                source_type=SourceType.job_board.value,
                accessed_at="2026-08-28T00:00:00+00:00",
            ),
            AgentSource(
                id="official",
                title="About Globant",
                url="https://www.globant.com/about",
                domain="globant.com",
                source_type=SourceType.official.value,
                accessed_at="2026-08-28T00:00:00+00:00",
            ),
        ],
    )

    weak_coverage = AgentOrchestrator.full_report_coverage(weak_state)
    weak_queries = [item["query"] for item in weak_coverage["recommended_queries"]]

    assert any("careers" in query for query in weak_queries)
    assert not any("computrabajo" in query for query in weak_queries)
    assert any(
        item["source_id"] == "official"
        and EvidenceTopic.business.value in item["topics"]
        for item in weak_coverage["recommended_inspections"]
    )

    weak_state.employment_search_attempts = ["careers", "linkedin"]
    next_coverage = AgentOrchestrator.full_report_coverage(weak_state)
    assert next_coverage["recommended_queries"][-1]["source_family"] == "computrabajo"

    weak_state.sources.append(
        AgentSource(
            id="careers",
            title="Junior Software Engineer en Globant",
            url="https://www.globant.com/careers/job/junior-software-engineer-123",
            snippet="Globant busca Junior Software Engineer en Buenos Aires, Argentina.",
            domain="globant.com",
            source_type=SourceType.career_page.value,
            accessed_at="2026-08-28T00:00:00+00:00",
        )
    )
    weak_state.evidence.append(
        AgentEvidence(
            id="roles",
            source_id="careers",
            topic=EvidenceTopic.open_roles.value,
            claim="Globant busca Junior Software Engineer.",
            excerpt="Globant busca Junior Software Engineer en Buenos Aires, Argentina.",
            confidence=ConfidenceLevel.medium.value,
        )
    )
    ready_coverage = AgentOrchestrator.full_report_coverage(weak_state)
    ready_queries = [item["query"] for item in ready_coverage["recommended_queries"]]

    assert not any(
        domain in query
        for query in ready_queries
        for domain in ("ar.computrabajo.com", "bumeran.com.ar", "zonajobs.com.ar")
    )


def test_full_report_forces_each_pending_employment_source_after_core_coverage() -> None:
    state = AgentState(
        goal="Generá un informe completo sobre Globant Argentina",
        company_name="Globant",
        employment_search_attempts=["careers"],
        sources=[
            AgentSource(
                id="official",
                title="Globant official",
                url="https://www.globant.com/about",
                domain="globant.com",
                source_type=SourceType.official.value,
                accessed_at="2026-09-04T00:00:00+00:00",
            )
        ],
        evidence=[
            AgentEvidence(
                id="business",
                source_id="official",
                topic=EvidenceTopic.business.value,
                claim="Globant ofrece servicios de transformación digital.",
                excerpt="Globant ofrece servicios de transformación digital.",
                confidence=ConfidenceLevel.medium.value,
            ),
            AgentEvidence(
                id="presence",
                source_id="official",
                topic=EvidenceTopic.argentina_presence.value,
                claim="Globant tiene oficinas en Argentina.",
                excerpt="Globant tiene oficinas en Buenos Aires, Argentina.",
                confidence=ConfidenceLevel.high.value,
            ),
        ],
    )

    action = AgentOrchestrator.required_full_report_action(state)

    assert action["name"] == "search_web"
    assert action["arguments"]["query"] == 'site:linkedin.com/jobs/view "Globant" Argentina'
    assert action["arguments"]["topic"] == EvidenceTopic.open_roles.value
    assert AgentOrchestrator.employment_search_family(
        "Globant careers jobs Argentina LinkedIn"
    ) == "linkedin"

    state.employment_search_attempts.extend(
        ["linkedin", "computrabajo", "bumeran", "zonajobs"]
    )
    assert AgentOrchestrator.required_full_report_action(state) is None


def test_full_report_prioritizes_uninspected_corporate_source_for_missing_business() -> None:
    state = AgentState(
        goal="Generá un informe completo sobre Globant Argentina",
        company_name="Globant",
        sources=[
            AgentSource(
                id="official",
                title="Globant solutions and AI",
                url="https://www.globant.com/about",
                snippet="Globant ofrece soluciones de ingeniería, IA y transformación digital.",
                domain="globant.com",
                source_type=SourceType.official.value,
                accessed_at="2026-09-04T00:00:00+00:00",
            )
        ],
    )

    action = AgentOrchestrator.required_full_report_action(state)

    assert action["name"] == "inspect_page"
    assert action["arguments"] == {"source_id": "official"}


def test_only_a_specific_argentina_job_covers_open_roles() -> None:
    generic_linkedin = AgentSource(
        id="linkedin-company-jobs",
        title="Globant: Jobs",
        url="https://www.linkedin.com/company/globant/jobs",
        snippet="78 empleos en todo el mundo.",
        domain="linkedin.com",
        source_type=SourceType.linkedin.value,
        accessed_at="2026-09-04T00:00:00+00:00",
    )
    careers_landing = AgentSource(
        id="careers-landing",
        title="Careers",
        url="https://www.globant.com/careers",
        snippet="Explorá oportunidades en Argentina.",
        domain="globant.com",
        source_type=SourceType.career_page.value,
        accessed_at="2026-09-04T00:00:00+00:00",
    )
    concrete_job = AgentSource(
        id="linkedin-job",
        title="Junior Java Developer en Globant",
        url="https://www.linkedin.com/jobs/view/junior-java-developer-at-globant-123",
        snippet="Globant busca Junior Java Developer en Buenos Aires, Argentina.",
        domain="linkedin.com",
        source_type=SourceType.linkedin.value,
        accessed_at="2026-09-04T00:00:00+00:00",
    )

    assert not AgentOrchestrator.is_specific_argentina_job(
        generic_linkedin, "Globant tiene puestos abiertos.", "Globant Argentina"
    )
    assert not AgentOrchestrator.is_specific_argentina_job(
        careers_landing, "Globant tiene puestos abiertos.", "Globant Argentina"
    )
    assert AgentOrchestrator.is_specific_argentina_job(
        concrete_job,
        "Globant busca Junior Java Developer en Buenos Aires, Argentina.",
        "Globant Argentina",
    )


def test_full_report_search_filters_foreign_location_results(db_session: Session) -> None:
    _, task = create_task(
        db_session,
        "Generá un informe completo sobre Globant Argentina",
        budget={
            "profile": "full_report",
            "model_turns": 16,
            "searches": 12,
            "inspections": 20,
            "elapsed_seconds": 300,
        },
    )
    result = SearchResult(
        title="Sueldos de Globant en Bogotá",
        url="https://example.com/globant-bogota",
        snippet="Salarios para Colombia.",
        rank=1,
        query_topic=EvidenceTopic.salary,
    )
    state = AgentState(goal="Generá un informe completo sobre Globant Argentina")

    tool_result = AgentOrchestrator(
        db_session,
        settings(),
        search_provider=FakeSearch(result),
    ).tool_search_web(
        task,
        state,
        {"searches": 0},
        {
            "query": "Globant salarios Argentina",
            "company_name": "Globant Argentina",
            "topic": "salary",
        },
    )

    assert state.sources == []
    assert tool_result.payload["filtered_by_location"] == 1


def test_forced_grounded_finalization_without_evidence_returns_limited_answer(
    db_session: Session,
) -> None:
    conversation, task = create_task(db_session, "Investigá las pasantías de Acme")
    task.working_state_json = AgentState(
        goal="Investigá las pasantías de Acme",
        requires_grounding=True,
        force_finalize=True,
    ).model_dump_json()
    db_session.commit()
    client = SequenceClient([])

    AgentOrchestrator(db_session, settings(), client=client).run(task.id)

    db_session.refresh(task)
    assert task.status == "completed"
    assert client.requests == []
    assistant = ConversationRepository(db_session).list_messages(conversation.id)[-1]
    assert "No pude verificar" in assistant.content


def test_new_task_reuses_recent_conversation_citations(db_session: Session) -> None:
    repo = ConversationRepository(db_session)
    conversation = repo.create()
    repo.add_message(conversation, role="user", content="Investigá Acme", status="completed")
    repo.add_message(
        conversation,
        role="assistant",
        content="Acme desarrolla software.",
        status="completed",
        citations=[
            {
                "evidence_id": "prior-evidence",
                "source_id": "prior-source",
                "title": "Acme Careers",
                "url": "https://example.com/careers",
                "domain": "example.com",
                "source_type": "career_page",
                "claim": "Acme desarrolla software.",
                "accessed_at": "2026-08-18T00:00:00+00:00",
            }
        ],
    )
    trigger = repo.add_message(
        conversation,
        role="user",
        content="¿Qué oportunidades ofrece Acme?",
        status="completed",
    )
    task = repo.create_task(conversation, trigger)
    db_session.commit()

    state = AgentOrchestrator(db_session, settings(), client=SequenceClient([])).load_state(task)

    assert [source.id for source in state.sources] == ["prior-source"]
    assert [evidence.id for evidence in state.evidence] == ["prior-evidence"]


def test_comparison_is_owned_by_conversation_but_reports_survive(db_session: Session) -> None:
    report_repo = ReportRepository(db_session)
    report_ids = []
    for company_name in ("Acme", "Globex"):
        company = CompanyRepository(db_session).get_or_create(company_name)
        report = report_repo.create_report(company.id)
        structured = MockReportBuilder(settings()).build(
            report,
            include_cv=False,
            include_cv_tailoring=False,
            include_adapted_cv_draft=False,
        )
        report_repo.save_structured_report(report, structured)
        report_ids.append(report.id)
    conversation, task = create_task(db_session, "Compará los informes")
    orchestrator = AgentOrchestrator(db_session, settings(), client=SequenceClient([]))
    state = AgentState(goal="Comparar", requires_grounding=True)

    result = orchestrator.tool_compare_reports(
        task,
        state,
        {},
        {"report_ids": report_ids, "dimensions": ["business", "culture"]},
    )
    db_session.commit()

    comparison_id = result.payload["comparison_id"]
    assert db_session.get(models.ComparisonArtifact, comparison_id) is not None
    ConversationRepository(db_session).delete(conversation)
    db_session.commit()
    assert db_session.get(models.ComparisonArtifact, comparison_id) is None
    assert all(report_repo.get_by_id(report_id) is not None for report_id in report_ids)


def test_finish_research_creates_and_links_grounded_report(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    conversation, task = create_task(db_session, "Generá un informe de Acme")
    state = AgentState(
        goal="Generar informe",
        requires_grounding=True,
        sources=[
            AgentSource(
                id="source-1",
                title="Acme oficial",
                url="https://example.com/acme",
                snippet="Acme desarrolla software.",
                domain="example.com",
                source_type="official",
                reliability_score=90,
                rank=1,
                topic="business",
                accessed_at="2026-08-17T00:00:00+00:00",
                inspected=True,
            )
        ],
        evidence=[
            AgentEvidence(
                id="evidence-1",
                source_id="source-1",
                topic="business",
                claim="Acme desarrolla software.",
                excerpt="Acme desarrolla software.",
                confidence="high",
            )
        ],
    )

    def fake_synthesize(_self, request):
        report = db_session.get(models.Report, request.report_id)
        return completion_ready_report(report)

    monkeypatch.setattr(
        "backend.app.services.agent_orchestrator.DeepSeekReportSynthesizer.synthesize",
        fake_synthesize,
    )
    monkeypatch.setattr(
        "backend.app.services.agent_orchestrator.schedule_report_embedding_task",
        lambda _report_id: None,
    )
    result = AgentOrchestrator(db_session, settings(), client=SequenceClient([])).tool_finish_research(
        task,
        state,
        {},
        {"company_name": "Acme", "output_type": "report", "unresolved_topics": []},
    )

    report_id = result.payload["report_id"]
    report = ReportRepository(db_session).get_by_id(report_id)
    assert report is not None and report.status == "completed"
    links = ConversationRepository(db_session).list_artifacts(conversation.id)
    assert any(link.artifact_type == "report" and link.artifact_id == report_id for link in links)


def test_finish_research_saves_report_with_open_roles_warning_only_after_all_sources(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    conversation, task = create_task(
        db_session,
        "GenerÃ¡ un informe de Acme",
        budget={
            "profile": "full_report",
            "model_turns": 16,
            "searches": 12,
            "inspections": 20,
            "elapsed_seconds": 300,
        },
    )
    task.working_state_json = AgentState(
        goal="Generá un informe de Acme",
        company_name="Acme",
        employment_search_attempts=[
            "careers",
            "linkedin",
            "computrabajo",
            "bumeran",
            "zonajobs",
        ],
        sources=[
            AgentSource(
                id="source-1",
                title="Acme oficial",
                url="https://acme.example/about",
                domain="acme.example",
                source_type="official",
                reliability_score=5,
                accessed_at="2026-08-28T00:00:00+00:00",
                inspected=True,
            ),
            AgentSource(
                id="source-2",
                title="Acme en Argentina",
                url="https://news.example/acme-argentina",
                domain="news.example",
                source_type="secondary",
                reliability_score=3,
                accessed_at="2026-08-28T00:00:00+00:00",
                inspected=True,
            ),
        ],
        evidence=[
            AgentEvidence(
                id="evidence-1",
                source_id="source-1",
                topic="business",
                claim="Acme desarrolla software.",
                excerpt="Acme desarrolla software.",
                confidence="high",
            ),
            AgentEvidence(
                id="evidence-2",
                source_id="source-1",
                topic="argentina_presence",
                claim="Acme opera en Buenos Aires.",
                excerpt="Acme opera en Buenos Aires, Argentina.",
                confidence="high",
            ),
        ],
    ).model_dump_json()
    db_session.commit()

    def fake_synthesize(_self, request):
        report = db_session.get(models.Report, request.report_id)
        structured = completion_ready_report(report)
        structured.sources[1] = structured.sources[1].model_copy(
            update={
                "title": "Contexto laboral de Acme",
                "url": "https://news.example/acme-argentina",
                "domain": "news.example",
                "source_type": SourceType.secondary,
            }
        )
        return structured

    monkeypatch.setattr(
        "backend.app.services.agent_orchestrator.DeepSeekReportSynthesizer.synthesize",
        fake_synthesize,
    )
    monkeypatch.setattr(
        "backend.app.services.agent_orchestrator.schedule_report_embedding_task",
        lambda _report_id: None,
    )
    orchestrator = AgentOrchestrator(db_session, settings(), client=SequenceClient([]))
    state = orchestrator.load_state(task)
    result = orchestrator.tool_finish_research(
        task,
        state,
        {},
        {"company_name": "Acme", "output_type": "report", "unresolved_topics": []},
    )

    assert result.payload["report_id"]
    report = ReportRepository(db_session).to_structured_report(
        ReportRepository(db_session).get_by_id(result.payload["report_id"])
    )
    open_roles = next(item for item in report.sections if item.type == SectionType.open_roles)
    assert open_roles.missing_evidence is True
    assert open_roles.claims == []
    assert "esto no significa que no existan" in open_roles.summary
    assert any(warning.type == "open_roles_not_verified" for warning in report.warnings)


def test_finish_research_rejects_open_roles_warning_before_all_sources(
    db_session: Session,
) -> None:
    _, task = create_task(
        db_session,
        "Generá un informe de Acme",
        budget={
            "profile": "full_report",
            "model_turns": 16,
            "searches": 12,
            "inspections": 20,
            "elapsed_seconds": 300,
        },
    )
    state = AgentState(
        goal="Generá un informe de Acme",
        company_name="Acme",
        employment_search_attempts=["careers", "linkedin", "computrabajo", "bumeran"],
        sources=[
            AgentSource(
                id="official",
                title="Acme oficial",
                url="https://acme.example/about",
                domain="acme.example",
                source_type="official",
                accessed_at="2026-09-04T00:00:00+00:00",
            )
        ],
        evidence=[
            AgentEvidence(
                id="business",
                source_id="official",
                topic="business",
                claim="Acme desarrolla software.",
                excerpt="Acme desarrolla software.",
                confidence="high",
            ),
            AgentEvidence(
                id="presence",
                source_id="official",
                topic="argentina_presence",
                claim="Acme opera en Argentina.",
                excerpt="Acme opera en Buenos Aires, Argentina.",
                confidence="high",
            ),
        ],
    )

    with pytest.raises(ValueError, match="búsquedas laborales"):
        AgentOrchestrator(db_session, settings()).tool_finish_research(
            task,
            state,
            {},
            {"company_name": "Acme", "output_type": "report", "unresolved_topics": []},
        )


def test_failed_quote_repair_finishes_with_a_transparent_response(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conversation, task = create_task(
        db_session,
        "Generá un informe completo sobre Acme Argentina",
        budget={
            "profile": "full_report",
            "model_turns": 16,
            "searches": 12,
            "inspections": 20,
            "elapsed_seconds": 300,
        },
    )
    state = AgentState(
        goal="Generá un informe completo sobre Acme Argentina",
        company_name="Acme",
        sources=[
            AgentSource(
                id="official",
                title="Acme",
                url="https://acme.example/about",
                domain="acme.example",
                source_type="official",
                accessed_at="2026-09-04T00:00:00+00:00",
            ),
            AgentSource(
                id="job",
                title="Software Engineer en Acme",
                url="https://www.linkedin.com/jobs/view/software-engineer-acme-123",
                snippet="Acme busca Software Engineer en Buenos Aires, Argentina.",
                domain="linkedin.com",
                source_type="job_board",
                accessed_at="2026-09-04T00:00:00+00:00",
            ),
        ],
        evidence=[
            AgentEvidence(
                id="business",
                source_id="official",
                topic="business",
                claim="Acme desarrolla software.",
                excerpt="Acme desarrolla software.",
                confidence="high",
            ),
            AgentEvidence(
                id="presence",
                source_id="official",
                topic="argentina_presence",
                claim="Acme opera en Argentina.",
                excerpt="Acme opera en Buenos Aires, Argentina.",
                confidence="high",
            ),
            AgentEvidence(
                id="role",
                source_id="job",
                topic="open_roles",
                claim="Acme busca Software Engineer.",
                excerpt="Acme busca Software Engineer en Buenos Aires, Argentina.",
                confidence="medium",
            ),
        ],
    )

    def failed_synthesis(_self, _request):
        raise SynthesisError(
            "DeepSeek response did not match report schema after repair: supporting quote"
        )

    monkeypatch.setattr(
        "backend.app.services.agent_orchestrator.DeepSeekReportSynthesizer.synthesize",
        failed_synthesis,
    )

    with pytest.raises(AgentCompleted):
        AgentOrchestrator(db_session, settings()).tool_finish_research(
            task,
            state,
            {},
            {"company_name": "Acme", "output_type": "report", "unresolved_topics": []},
        )

    assert not ConversationRepository(db_session).list_artifacts(conversation.id)
    assistant = ConversationRepository(db_session).list_messages(conversation.id)[-1]
    assert "después del intento de reparación" in assistant.content
    failed_reports = db_session.query(models.Report).filter_by(status="failed").all()
    assert failed_reports


def test_missing_deepseek_configuration_fails_transparently(db_session: Session) -> None:
    _, task = create_task(db_session, "Dame un consejo general")
    AgentOrchestrator(db_session, Settings(deepseek_api_key=None, search_provider="mock")).run(task.id)
    db_session.refresh(task)
    assert task.status == "failed"
    assert "DEEPSEEK_API_KEY" in (task.pause_reason_json or "")


def test_provider_quota_error_is_sanitized_and_has_retry_time(db_session: Session) -> None:
    _, task = create_task(db_session, "¿Google tiene pasantías actualmente?")

    AgentOrchestrator(db_session, settings(), client=RaisingClient()).run(task.id)

    db_session.refresh(task)
    pause = __import__("json").loads(task.pause_reason_json or "{}")
    assert task.status == "failed"
    assert task.stopping_reason == "provider_rate_limited"
    assert pause["type"] == "rate_limit"
    assert pause["retry_at"]
    assert "DeepSeek API request failed" not in pause["message"]


@pytest.mark.parametrize(
    ("status_code", "expected_type"),
    [(401, "provider_authentication"), (402, "insufficient_balance"), (503, "provider_unavailable")],
)
def test_deepseek_provider_errors_are_sanitized(status_code: int, expected_type: str) -> None:
    pause, _ = AgentOrchestrator.public_error(
        DeepSeekAPIError("private provider response", status_code=status_code)
    )

    assert pause["type"] == expected_type
    assert "private provider response" not in pause["message"]


def test_invalid_tool_call_is_rejected_without_stopping_recovery(db_session: Session) -> None:
    _, task = create_task(db_session, "Dame un consejo general")
    client = SequenceClient(
        [
            ("delete_file", {"path": "secret.txt"}),
            (
                "respond",
                {
                    "answer": "No necesito acceder a archivos para darte orientación general.",
                    "answer_type": "guidance",
                    "claims": [],
                    "warnings": [],
                },
            ),
        ]
    )
    AgentOrchestrator(db_session, settings(), client=client).run(task.id)
    db_session.refresh(task)
    assert task.status == "completed"
    events = ConversationRepository(db_session).list_events(task.conversation_id)
    assert all(event.tool_name != "delete_file" for event in events)


def test_unsupported_evidence_id_cannot_complete_factual_answer(db_session: Session) -> None:
    _, task = create_task(db_session, "Contame datos actuales de Acme")
    client = SequenceClient(
        [
            (
                "respond",
                {
                    "answer": "Acme tiene datos actuales.",
                    "answer_type": "grounded",
                    "claims": [{"text": "Dato", "evidence_ids": ["invented"]}],
                    "warnings": [],
                },
            ),
            (
                "request_clarification",
                {"question": "¿Qué aspecto de Acme querés investigar?", "options": []},
            ),
        ]
    )
    AgentOrchestrator(db_session, settings(), client=client).run(task.id)
    db_session.refresh(task)
    assert task.status == "needs_clarification"
    messages = ConversationRepository(db_session).list_messages(task.conversation_id)
    assert all(message.content != "Acme tiene datos actuales." for message in messages)


def test_current_internship_claim_without_official_source_adds_caveat(
    db_session: Session,
) -> None:
    _, task = create_task(db_session, "¿Google tiene pasantías vigentes?")
    state = AgentState(
        goal="¿Google tiene pasantías vigentes?",
        requires_grounding=True,
        sources=[
            AgentSource(
                id="secondary-source",
                title="Artículo sobre pasantías",
                url="https://example.com/pasantias",
                domain="example.com",
                source_type="news_media",
                reliability_score=50,
                rank=1,
                topic="open_roles",
                accessed_at="2026-08-18T00:00:00+00:00",
                inspected=True,
            )
        ],
        evidence=[
            AgentEvidence(
                id="secondary-evidence",
                source_id="secondary-source",
                topic="open_roles",
                claim="Un artículo anuncia pasantías.",
                excerpt="El artículo anuncia un programa de pasantías.",
                confidence="medium",
            )
        ],
    )

    with pytest.raises(AgentCompleted):
        AgentOrchestrator(db_session, settings(), client=SequenceClient([])).tool_respond(
            task,
            state,
            {},
            {
                "answer": "Google tiene pasantías vigentes.",
                "answer_type": "grounded",
                "claims": [
                    {"text": "Hay pasantías", "evidence_ids": ["secondary-evidence"]}
                ],
                "warnings": [],
            },
        )
    assistant = ConversationRepository(db_session).list_messages(task.conversation_id)[-1]
    assert assistant.content.startswith("No pude confirmar esta información en una fuente laboral confiable")


def test_current_internship_claim_on_recognized_job_board_needs_no_official_caveat(
    db_session: Session,
) -> None:
    _, task = create_task(db_session, "¿Google tiene pasantías vigentes?")
    state = AgentState(
        goal="¿Google tiene pasantías vigentes?",
        requires_grounding=True,
        sources=[
            AgentSource(
                id="job-board-source",
                title="Pasantía en Computrabajo",
                url="https://www.computrabajo.com.ar/google-pasantia",
                domain="computrabajo.com.ar",
                source_type="job_board",
                reliability_score=4,
                rank=1,
                topic="open_roles",
                accessed_at="2026-08-18T00:00:00+00:00",
                inspected=True,
            )
        ],
        evidence=[
            AgentEvidence(
                id="job-board-evidence",
                source_id="job-board-source",
                topic="open_roles",
                claim="El portal publica una pasantía.",
                excerpt="Pasantía de Google en Argentina.",
                confidence="medium",
            )
        ],
    )

    with pytest.raises(AgentCompleted):
        AgentOrchestrator(db_session, settings(), client=SequenceClient([])).tool_respond(
            task,
            state,
            {},
            {
                "answer": "Google tiene pasantías vigentes.",
                "answer_type": "grounded",
                "claims": [
                    {"text": "Hay pasantías", "evidence_ids": ["job-board-evidence"]}
                ],
                "warnings": [],
            },
        )
    assistant = ConversationRepository(db_session).list_messages(task.conversation_id)[-1]
    assert not assistant.content.startswith("No pude confirmar")


def test_inspection_uses_search_snippet_when_a_job_board_blocks_extraction(
    db_session: Session,
) -> None:
    _, task = create_task(db_session, "Buscá pasantías de Acme")
    state = AgentState(
        goal="Buscá pasantías de Acme",
        company_name="Acme",
        requires_grounding=True,
        sources=[
            AgentSource(
                id="linkedin-job",
                title="Acme pasantía",
                url="https://www.linkedin.com/jobs/view/123",
                snippet="Oferta laboral: pasantía de datos en Argentina.",
                domain="linkedin.com",
                source_type="job_board",
                reliability_score=4,
                rank=1,
                topic="open_roles",
                accessed_at="2026-08-18T00:00:00+00:00",
            )
        ],
    )

    result = AgentOrchestrator(
        db_session,
        settings(),
        extractor=FakeExtractor(None),
    ).tool_inspect_page(task, state, {"inspections": 0}, {"source_id": "linkedin-job"})

    assert result.payload["ok"] is True
    evidence = next(item for item in state.evidence if item.topic == "open_roles")
    assert evidence.confidence == ConfidenceLevel.medium.value
