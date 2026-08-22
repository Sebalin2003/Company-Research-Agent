from __future__ import annotations

from backend.app.services.performance import estimate_cost_usd, performance_warnings, sanitize_usage


def test_usage_is_sanitized_and_private_fields_are_omitted() -> None:
    safe = sanitize_usage(
        {
            "elapsed_ms": 1200.8,
            "provider_requests": 2,
            "response_kind": "grounded",
            "prompt": "PRIVATE CV TEXT",
            "working_state": {"secret": True},
            "negative": -5,
        }
    )

    assert safe == {
        "elapsed_ms": 1200,
        "provider_requests": 2,
        "response_kind": "grounded",
    }


def test_performance_budgets_warn_without_failing() -> None:
    warnings = performance_warnings(
        {
            "acceptance_ms": 501,
            "first_progress_ms": 1001,
            "elapsed_ms": 75_001,
            "response_kind": "grounded",
        }
    )

    assert [item["metric"] for item in warnings] == [
        "acceptance_ms",
        "first_progress_ms",
        "grounded",
    ]


def test_optional_cost_uses_cache_rate_without_hardcoded_prices() -> None:
    usage = {"prompt_tokens": 1000, "cache_hit_tokens": 400, "completion_tokens": 500}

    assert estimate_cost_usd(usage) is None
    assert estimate_cost_usd(
        usage,
        input_per_million=1.0,
        output_per_million=2.0,
        cache_hit_per_million=0.1,
    ) == 0.00164


def test_historical_research_usage_uses_grounded_budget() -> None:
    assert performance_warnings({"elapsed_ms": 30_000, "searches": 1}) == []
