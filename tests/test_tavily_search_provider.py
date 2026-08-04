from __future__ import annotations

import json

import httpx
import pytest

from backend.app.domain.reports import EvidenceTopic, SourceType
from backend.app.research.content_extractor import FakeContentExtractor
from backend.app.research.pipeline import ResearchPipeline
from backend.app.research.search_provider import SearchProviderError, TavilySearchProvider
from backend.app.research.types import ExtractedContent


def test_tavily_search_provider_sends_expected_request_and_maps_results() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "query": "Acme empleos Argentina",
                "results": [
                    {
                        "title": "Acme Careers",
                        "url": "https://careers.acme.com/jobs",
                        "content": "Empleos abiertos en Argentina.",
                        "score": 0.91,
                    }
                ],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = TavilySearchProvider(api_key="tvly-test", http_client=client)

    results = provider.search("Acme empleos Argentina", limit=3)

    assert seen["url"] == TavilySearchProvider.endpoint
    assert seen["authorization"] == "Bearer tvly-test"
    assert seen["body"]["query"] == "Acme empleos Argentina"
    assert seen["body"]["max_results"] == 3
    assert seen["body"]["country"] == "argentina"
    assert results[0].title == "Acme Careers"
    assert results[0].snippet == "Empleos abiertos en Argentina."
    assert results[0].rank == 1
    assert results[0].query_topic == EvidenceTopic.general


def test_tavily_search_provider_raises_for_http_error() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(401, json={"error": "bad key"}))
    )
    provider = TavilySearchProvider(api_key="bad-key", http_client=client)

    with pytest.raises(SearchProviderError):
        provider.search("Acme", limit=1)


def test_research_pipeline_overrides_tavily_general_topic_with_query_topic() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Acme Jobs",
                            "url": "https://careers.acme.com/jobs",
                            "content": "Empleos en Argentina y beneficios.",
                        }
                    ]
                },
            )
        )
    )
    provider = TavilySearchProvider(api_key="tvly-test", http_client=client)
    extractor = FakeContentExtractor(
        {
            "https://careers.acme.com/jobs": ExtractedContent(
                url="https://careers.acme.com/jobs",
                title="Acme Jobs",
                text="Empleos en Argentina y beneficios.",
            )
        }
    )

    result = ResearchPipeline(provider, extractor, results_per_query=1).run("Acme")

    assert result.sources[0].source_type == SourceType.career_page
    assert {item.topic for item in result.evidence} >= {
        EvidenceTopic.open_roles,
        EvidenceTopic.argentina_presence,
    }
