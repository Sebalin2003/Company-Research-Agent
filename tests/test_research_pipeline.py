from __future__ import annotations

from backend.app.domain.companies import normalize_company_name, validate_company_name
from backend.app.domain.reports import ConfidenceLevel, EvidenceTopic, SourceType
from backend.app.research.content_extractor import FakeContentExtractor
from backend.app.research.evidence_classifier import classify_evidence
from backend.app.research.pipeline import ResearchPipeline
from backend.app.research.query_builder import (
    build_company_research_queries,
    build_fast_company_research_queries,
)
from backend.app.research.scoring import classify_source_type, score_search_result
from backend.app.research.search_provider import FakeSearchProvider
from backend.app.research.types import ExtractedContent, ScoredSource, SearchResult


def test_normalize_and_validate_company_name() -> None:
    assert normalize_company_name("  Mercado   Libre  ") == "mercado libre"
    assert validate_company_name("  Acme  ") == "acme"


def test_build_company_research_queries_covers_required_topics() -> None:
    queries = build_company_research_queries("Mercado Libre")
    topics = {query.topic for query in queries}
    query_text = "\n".join(query.text for query in queries).lower()

    assert EvidenceTopic.business in topics
    assert EvidenceTopic.argentina_presence in topics
    assert EvidenceTopic.salary in topics
    assert EvidenceTopic.culture in topics
    assert EvidenceTopic.interview_process in topics
    assert EvidenceTopic.open_roles in topics
    assert any("Mercado Libre" in query.text for query in queries)
    assert "glassdoor" in query_text
    assert "indeed" in query_text
    assert "openqube" in query_text
    assert "computrabajo" in query_text
    assert "bumeran" in query_text
    assert "zonajobs" in query_text
    assert "portal empleo" in query_text
    assert "remuneracion empleo argentina" in query_text
    assert "direccion oficinas argentina" in query_text
    assert "cantidad de empleados" in query_text
    assert "total employees company profile" in query_text
    assert "rango salarial argentina" in query_text
    assert "sueldo it trainee argentina" in query_text
    assert "sueldo pasante it argentina" in query_text
    assert "sueldo internship it argentina" in query_text
    assert "sueldo desarrollador junior argentina" in query_text
    assert "salario junior it argentina" in query_text
    assert "trainee program salario argentina" in query_text
    assert "sueldos trainee junior glassdoor argentina" in query_text
    assert "salario junior it indeed argentina" in query_text
    assert "sueldos junior trainee openqube" in query_text
    assert "etapas entrevista seleccion" in query_text


def test_fast_company_research_queries_cover_required_topics() -> None:
    queries = build_fast_company_research_queries("Mercado Libre")
    topics = {query.topic for query in queries}
    query_text = "\n".join(query.text for query in queries).lower()

    assert EvidenceTopic.business in topics
    assert EvidenceTopic.argentina_presence in topics
    assert EvidenceTopic.employees in topics
    assert EvidenceTopic.salary in topics
    assert EvidenceTopic.culture in topics
    assert EvidenceTopic.interview_process in topics
    assert EvidenceTopic.interview_questions in topics
    assert EvidenceTopic.open_roles in topics
    assert "sueldo it trainee argentina" in query_text
    assert "glassdoor" in query_text
    assert "openqube" in query_text


def test_source_type_classification_prefers_company_owned_domains() -> None:
    assert (
        classify_source_type(
            "Acme",
            "https://careers.acme.com/jobs",
            EvidenceTopic.open_roles,
        )
        == SourceType.career_page
    )
    assert (
        classify_source_type(
            "Acme",
            "https://www.acme.com/about",
            EvidenceTopic.business,
        )
        == SourceType.official
    )
    assert (
        classify_source_type(
            "IBM Argentina",
            "https://latam.newsroom.ibm.com/2023-10-ibm-argentina",
            EvidenceTopic.argentina_presence,
        )
        == SourceType.official
    )
    assert (
        classify_source_type(
            "Globant Argentina",
            "https://www.globant.com/about",
            EvidenceTopic.business,
        )
        == SourceType.official
    )
    assert (
        classify_source_type(
            "Globant Argentina",
            "https://www.globant.com/careers",
            EvidenceTopic.open_roles,
        )
        == SourceType.career_page
    )


def test_official_digital_services_page_provides_business_evidence() -> None:
    source = ScoredSource(
        source_id="globant-official",
        title="Globant AI Powerhouse",
        url="https://www.globant.com/es",
        domain="globant.com",
        source_type=SourceType.official,
        reliability_score=5,
        snippet="Soluciones enterprise impulsadas por inteligencia artificial.",
        is_current=True,
    )
    content = ExtractedContent(
        url=source.url,
        title=source.title,
        text=(
            "Transformamos organizaciones mediante inteligencia artificial. "
            "Nuestras soluciones combinan ingeniería, innovación y diseño."
        ),
    )

    business = next(
        item for item in classify_evidence(source, content)
        if item.topic == EvidenceTopic.business
    )

    assert business.confidence == ConfidenceLevel.medium


def test_source_type_classifies_relevant_employment_platforms() -> None:
    sources = {
        "https://www.linkedin.com/company/acme": SourceType.linkedin,
        "https://www.linkedin.com/jobs/view/123": SourceType.job_board,
        "https://www.glassdoor.com.ar/Acme": SourceType.salary_review_platform,
        "https://www.computrabajo.com.ar/acme": SourceType.job_board,
        "https://www.bumeran.com.ar/acme": SourceType.job_board,
        "https://www.zonajobs.com.ar/acme": SourceType.job_board,
        "https://portalempleo.gob.ar/OfertasLaborales": SourceType.job_board,
    }

    for url, expected in sources.items():
        assert classify_source_type("Acme", url, EvidenceTopic.open_roles) == expected


def test_score_search_result_uses_source_type_https_and_rank() -> None:
    result = SearchResult(
        title="Acme Careers",
        url="https://careers.acme.com/jobs",
        snippet="Empleos abiertos",
        rank=1,
        query_topic=EvidenceTopic.open_roles,
    )

    source = score_search_result("Acme", result)

    assert source.source_type == SourceType.career_page
    assert source.reliability_score == 5
    assert source.domain == "careers.acme.com"


def test_evidence_classifier_detects_multiple_topics() -> None:
    source = ScoredSource(
        source_id="source_1",
        title="Acme Jobs",
        url="https://careers.acme.com/jobs",
        domain="careers.acme.com",
        source_type=SourceType.career_page,
        reliability_score=5,
        snippet="Empleos y beneficios en Argentina",
        is_current=True,
    )
    content = ExtractedContent(
        url=source.url,
        title="Trabaja en Acme Argentina",
        text="Tenemos empleo en Buenos Aires, beneficios y proceso de entrevista.",
    )

    evidence = classify_evidence(source, content)
    topics = {item.topic for item in evidence}

    assert EvidenceTopic.argentina_presence in topics
    assert EvidenceTopic.benefits in topics
    assert EvidenceTopic.interview_process in topics
    assert all(
        item.confidence in {ConfidenceLevel.low, ConfidenceLevel.medium, ConfidenceLevel.high}
        for item in evidence
    )


def test_evidence_classifier_marks_concrete_details_with_higher_confidence() -> None:
    source = ScoredSource(
        source_id="source_1",
        title="Acme Company Profile",
        url="https://example.com/acme",
        domain="example.com",
        source_type=SourceType.company_database,
        reliability_score=3,
        snippet="Acme tiene 12.500 empleados.",
        is_current=True,
    )
    content = ExtractedContent(
        url=source.url,
        title="Acme Buenos Aires",
        text=(
            "Direccion: Av. Corrientes 1234, Buenos Aires. "
            "Employees: 12,500. Sueldo mensual ARS 1.200.000. "
            "Proceso de entrevista: recruiter, tecnica y manager."
        ),
    )

    evidence = classify_evidence(source, content)
    confidence_by_topic = {item.topic: item.confidence for item in evidence}

    assert confidence_by_topic[EvidenceTopic.argentina_presence] == ConfidenceLevel.high
    assert confidence_by_topic[EvidenceTopic.employees] == ConfidenceLevel.high
    assert confidence_by_topic[EvidenceTopic.salary] == ConfidenceLevel.high
    assert confidence_by_topic[EvidenceTopic.interview_process] == ConfidenceLevel.high


def test_evidence_classifier_detects_it_entry_level_salary_evidence() -> None:
    source = ScoredSource(
        source_id="source_1",
        title="Acme sueldos IT junior",
        url="https://www.glassdoor.com.ar/Sueldos/Acme-sueldos.htm",
        domain="glassdoor.com.ar",
        source_type=SourceType.salary_review_platform,
        reliability_score=3,
        snippet="Sueldo trainee IT Argentina.",
        is_current=True,
    )
    content = ExtractedContent(
        url=source.url,
        title="Sueldos IT trainee y junior",
        text="Desarrollador junior: ARS 900.000 mensual. Pasante IT: $450.000.",
    )

    evidence = classify_evidence(source, content)
    salary_evidence = [item for item in evidence if item.topic == EvidenceTopic.salary]

    assert salary_evidence
    assert salary_evidence[0].confidence == ConfidenceLevel.high


def test_research_pipeline_deduplicates_sources_and_classifies_evidence() -> None:
    queries = build_company_research_queries("Acme")
    shared_result = SearchResult(
        title="Acme Careers",
        url="https://careers.acme.com/jobs",
        snippet="Empleos en Argentina",
        rank=1,
        query_topic=EvidenceTopic.open_roles,
    )
    search_provider = FakeSearchProvider(
        {
            queries[0].text: [shared_result],
            queries[-1].text: [shared_result],
        }
    )
    extractor = FakeContentExtractor(
        {
            shared_result.url: ExtractedContent(
                url=shared_result.url,
                title="Acme Careers Argentina",
                text="Busquedas de empleo en Argentina y beneficios para empleados.",
            )
        }
    )

    result = ResearchPipeline(search_provider, extractor).run("Acme")

    assert len(result.sources) == 1
    assert result.sources[0].source_type == SourceType.career_page
    assert {item.topic for item in result.evidence} >= {
        EvidenceTopic.open_roles,
        EvidenceTopic.argentina_presence,
    }


def test_research_pipeline_preserves_salary_review_sources_with_official_sources() -> None:
    queries = build_company_research_queries("Acme")
    official = SearchResult(
        title="Acme Oficial",
        url="https://www.acme.com/about",
        snippet="Acme es una empresa industrial.",
        rank=1,
        query_topic=EvidenceTopic.business,
    )
    salary = SearchResult(
        title="Sueldos Acme Argentina",
        url="https://www.glassdoor.com.ar/Sueldos/Acme-sueldos.htm",
        snippet="Sueldos y salarios de Acme en Argentina publicados por empleados.",
        rank=2,
        query_topic=EvidenceTopic.salary,
    )
    search_provider = FakeSearchProvider(
        {
            queries[0].text: [official],
            next(query.text for query in queries if "Glassdoor" in query.text): [salary],
        }
    )
    extractor = FakeContentExtractor(
        {
            official.url: ExtractedContent(
                url=official.url,
                title=official.title,
                text="Acme es una empresa industrial con productos para clientes corporativos.",
            ),
            salary.url: ExtractedContent(
                url=salary.url,
                title=salary.title,
                text="Sueldos y salarios de empleados en Argentina. Rangos publicados varian por rol.",
            ),
        }
    )

    result = ResearchPipeline(search_provider, extractor).run("Acme")

    assert {source.source_type for source in result.sources} >= {
        SourceType.official,
        SourceType.salary_review_platform,
    }
    assert EvidenceTopic.salary in {item.topic for item in result.evidence}


def test_research_pipeline_preserves_interview_and_open_role_sources() -> None:
    queries = build_company_research_queries("Acme")
    official = SearchResult(
        title="Acme Oficial",
        url="https://www.acme.com/about",
        snippet="Acme es una empresa.",
        rank=1,
        query_topic=EvidenceTopic.business,
    )
    interview = SearchResult(
        title="Acme Interview Questions",
        url="https://www.glassdoor.com.ar/Entrevista/Acme-preguntas-entrevista.htm",
        snippet="Preguntas de entrevista, proceso y experiencia de candidatos.",
        rank=2,
        query_topic=EvidenceTopic.interview_questions,
    )
    jobs = SearchResult(
        title="Acme Jobs",
        url="https://jobs.acme.com/argentina",
        snippet="Vacantes y busquedas abiertas en Argentina.",
        rank=2,
        query_topic=EvidenceTopic.open_roles,
    )
    search_provider = FakeSearchProvider(
        {
            queries[0].text: [official],
            next(query.text for query in queries if "interview questions" in query.text): [
                interview
            ],
            queries[-1].text: [jobs],
        }
    )
    extractor = FakeContentExtractor(
        {
            official.url: ExtractedContent(
                url=official.url,
                title=official.title,
                text="Acme es una empresa.",
            ),
            interview.url: ExtractedContent(
                url=interview.url,
                title=interview.title,
                text="Proceso de entrevista con preguntas conductuales y tecnicas.",
            ),
            jobs.url: ExtractedContent(
                url=jobs.url,
                title=jobs.title,
                text="Vacantes, trabajos y busquedas abiertas en Argentina.",
            ),
        }
    )

    result = ResearchPipeline(search_provider, extractor).run("Acme")
    topics = {item.topic for item in result.evidence}

    assert any(source.url == interview.url for source in result.sources)
    assert any(source.url == jobs.url for source in result.sources)
    assert EvidenceTopic.interview_process in topics
    assert EvidenceTopic.interview_questions in topics
    assert EvidenceTopic.open_roles in topics


def test_research_pipeline_deepens_only_missing_topics() -> None:
    fast_queries = build_fast_company_research_queries("Acme")
    fast_business = fast_queries[0]
    fast_salary = next(query for query in fast_queries if query.topic == EvidenceTopic.salary)
    all_queries = build_company_research_queries("Acme")
    deep_salary = next(query for query in all_queries if "salario junior IT Indeed" in query.text)
    deep_employees = next(query for query in all_queries if "cantidad de empleados" in query.text)
    search_provider = FakeSearchProvider(
        {
            fast_business.text: [
                SearchResult(
                    title="Acme Oficial",
                    url="https://www.acme.com/about",
                    snippet="Acme es una empresa de tecnologia.",
                    rank=1,
                    query_topic=EvidenceTopic.business,
                )
            ],
            fast_salary.text: [
                SearchResult(
                    title="Acme salarios junior",
                    url="https://glassdoor.com/acme-salarios",
                    snippet="Desarrollador junior IT: ARS 900.000 mensual.",
                    rank=1,
                    query_topic=EvidenceTopic.salary,
                )
            ],
            deep_employees.text: [
                SearchResult(
                    title="Acme empleados",
                    url="https://example.com/acme-empleados",
                    snippet="Acme tiene 5.000 empleados.",
                    rank=1,
                    query_topic=EvidenceTopic.employees,
                )
            ],
            deep_salary.text: [
                SearchResult(
                    title="No deberia buscar salario",
                    url="https://example.com/no",
                    snippet="No usado",
                    rank=1,
                    query_topic=EvidenceTopic.salary,
                )
            ],
        }
    )
    extractor = FakeContentExtractor(
        {
            "https://www.acme.com/about": ExtractedContent(
                url="https://www.acme.com/about",
                title="Acme Oficial",
                text="Acme es una empresa de tecnologia.",
            ),
            "https://glassdoor.com/acme-salarios": ExtractedContent(
                url="https://glassdoor.com/acme-salarios",
                title="Acme salarios junior",
                text="Desarrollador junior IT: ARS 900.000 mensual.",
            ),
            "https://example.com/acme-empleados": ExtractedContent(
                url="https://example.com/acme-empleados",
                title="Acme empleados",
                text="Acme tiene 5.000 empleados.",
            ),
        }
    )

    result = ResearchPipeline(search_provider, extractor, results_per_query=1).run("Acme")
    urls = {source.url for source in result.sources}

    assert "https://example.com/acme-empleados" in urls
    assert "https://example.com/no" not in urls
    assert result.search_count < len(all_queries)
    assert result.extracted_url_count == 2
    assert result.snippet_only_count == 1


def test_research_pipeline_isolates_search_and_extraction_failures() -> None:
    class PartiallyFailingSearchProvider:
        def search(self, query: str, limit: int):
            if "sitio oficial" in query:
                raise RuntimeError("search boom")
            if "carreras empleos" in query:
                return [
                    SearchResult(
                        title="Acme Jobs",
                        url="https://jobs.acme.com",
                        snippet="Vacantes en Argentina.",
                        rank=1,
                        query_topic=EvidenceTopic.open_roles,
                    )
                ]
            return []

    class FailingExtractor:
        def extract(self, url: str):
            raise RuntimeError("extract boom")

    result = ResearchPipeline(
        PartiallyFailingSearchProvider(),
        FailingExtractor(),
        results_per_query=1,
        enable_deepening=False,
    ).run("Acme")

    assert result.sources[0].url == "https://jobs.acme.com"
    assert EvidenceTopic.open_roles in {item.topic for item in result.evidence}
    assert result.search_count == len(build_fast_company_research_queries("Acme"))
    assert result.extracted_url_count == 1
    assert result.search_duration_ms >= 0
    assert result.extraction_duration_ms >= 0


def test_research_pipeline_caps_extraction_and_uses_snippets_for_protected_domains() -> None:
    queries = build_fast_company_research_queries("Acme")
    official_query = queries[0]
    salary_query = next(query for query in queries if query.topic == EvidenceTopic.salary)
    results = [
        SearchResult(
            title=f"Acme official {index}",
            url=f"https://www.acme.com/page-{index}",
            snippet="Acme es una empresa con operaciones en Argentina.",
            rank=index,
            query_topic=EvidenceTopic.business,
        )
        for index in range(1, 8)
    ]
    glassdoor = SearchResult(
        title="Acme sueldos Glassdoor",
        url="https://www.glassdoor.com.ar/Sueldos/Acme-sueldos.htm",
        snippet="Desarrollador junior IT: ARS 900.000 mensual.",
        rank=1,
        query_topic=EvidenceTopic.salary,
    )
    search_provider = FakeSearchProvider(
        {
            official_query.text: results,
            salary_query.text: [glassdoor],
        }
    )
    extractor = RecordingExtractor()

    result = ResearchPipeline(
        search_provider,
        extractor,
        results_per_query=7,
        enable_deepening=False,
        max_extract_urls=3,
        max_extract_urls_per_topic=2,
    ).run("Acme")

    assert len(extractor.extracted_urls) == 2
    assert glassdoor.url not in extractor.extracted_urls
    assert result.extracted_url_count == 2
    assert result.extraction_skipped_count == 6
    assert result.snippet_only_count == 6
    assert any(source.url == glassdoor.url for source in result.sources)
    assert EvidenceTopic.salary in {item.topic for item in result.evidence}


def test_research_pipeline_extracts_protected_domain_when_snippet_is_empty() -> None:
    salary_query = next(
        query for query in build_fast_company_research_queries("Acme")
        if query.topic == EvidenceTopic.salary
    )
    glassdoor = SearchResult(
        title="Acme sueldos Glassdoor",
        url="https://www.glassdoor.com.ar/Sueldos/Acme-sueldos.htm",
        snippet="",
        rank=1,
        query_topic=EvidenceTopic.salary,
    )
    extractor = RecordingExtractor(
        {
            glassdoor.url: ExtractedContent(
                url=glassdoor.url,
                title=glassdoor.title,
                text="Desarrollador junior IT: ARS 900.000 mensual.",
            )
        }
    )

    result = ResearchPipeline(
        FakeSearchProvider({salary_query.text: [glassdoor]}),
        extractor,
        results_per_query=1,
        enable_deepening=False,
    ).run("Acme")

    assert extractor.extracted_urls == [glassdoor.url]
    assert result.extracted_url_count == 1
    assert result.snippet_only_count == 0


class RecordingExtractor:
    def __init__(self, content_by_url: dict[str, ExtractedContent] | None = None) -> None:
        self.content_by_url = content_by_url or {}
        self.extracted_urls: list[str] = []

    def extract(self, url: str) -> ExtractedContent | None:
        self.extracted_urls.append(url)
        return self.content_by_url.get(
            url,
            ExtractedContent(url=url, title=url, text="Acme opera en Argentina."),
        )
