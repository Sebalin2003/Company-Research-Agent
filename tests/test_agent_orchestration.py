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
from backend.app.domain.reports import EvidenceTopic
from backend.app.llm.deepseek import DeepSeekAPIError, DeepSeekResponse, DeepSeekToolCall
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
            "model_turns": 12,
            "searches": 6,
            "inspections": 10,
            "elapsed_seconds": 180,
            "extension_used": False,
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
    declarations = client.requests[0]["tools"]
    assert [item["function"]["name"] for item in declarations] == ["request_clarification"]


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


def test_budget_exhaustion_pauses_for_one_extension(db_session: Session) -> None:
    _, task = create_task(
        db_session,
        "Investigá una empresa",
        budget={
            "model_turns": 0,
            "searches": 0,
            "inspections": 0,
            "elapsed_seconds": 0,
            "extension_used": False,
        },
    )
    AgentOrchestrator(db_session, settings(), client=SequenceClient([])).run(task.id)
    db_session.refresh(task)
    assert task.status == "awaiting_approval"
    events = ConversationRepository(db_session).list_events(task.conversation_id)
    assert events[-1].event_type == "approval.required"


def test_budget_exhaustion_forces_guidance_finalization_without_extra_decision(
    db_session: Session,
) -> None:
    conversation, task = create_task(
        db_session,
        "Dame un consejo general para entrevistas",
        budget={
            "model_turns": 0,
            "searches": 0,
            "inspections": 0,
            "elapsed_seconds": 180,
            "extension_used": False,
        },
    )
    client = FinalizingClient()

    AgentOrchestrator(db_session, settings(), client=client).run(task.id)

    db_session.refresh(task)
    assert task.status == "completed"
    assert len(client.requests) == 1
    assert ConversationRepository(db_session).list_messages(conversation.id)[-1].role == "assistant"


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
        return MockReportBuilder(settings()).build(
            report,
            include_cv=False,
            include_cv_tailoring=False,
            include_adapted_cv_draft=False,
        )

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
    assert assistant.content.startswith("No pude confirmar esta información en una fuente oficial")
