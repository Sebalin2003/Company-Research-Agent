from __future__ import annotations

from typing import Any


WARNING_BUDGETS_MS = {
    "acceptance_ms": 500,
    "first_progress_ms": 1_000,
    "direct": 20_000,
    "grounded": 75_000,
    "artifact": 120_000,
}

PUBLIC_USAGE_KEYS = {
    "acceptance_ms",
    "first_progress_ms",
    "elapsed_ms",
    "deepseek_ms",
    "search_ms",
    "inspection_ms",
    "evidence_ms",
    "artifact_ms",
    "tool_ms",
    "model_turns",
    "searches",
    "inspections",
    "provider_requests",
    "tool_calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cache_hit_tokens",
    "cache_miss_tokens",
    "response_kind",
}


def sanitize_usage(raw: dict[str, Any]) -> dict[str, int | str]:
    result: dict[str, int | str] = {}
    for key in PUBLIC_USAGE_KEYS:
        value = raw.get(key)
        if key == "response_kind":
            if value in {"direct", "grounded", "artifact"}:
                result[key] = str(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            result[key] = max(0, int(value))
    return result


def performance_warnings(usage: dict[str, Any]) -> list[dict[str, int | str]]:
    safe = sanitize_usage(usage)
    checks = ["acceptance_ms", "first_progress_ms"]
    kind = str(safe.get("response_kind") or ("grounded" if safe.get("searches") else "direct"))
    checks.append(kind)
    warnings = []
    for name in checks:
        actual_key = "elapsed_ms" if name in {"direct", "grounded", "artifact"} else name
        actual = int(safe.get(actual_key) or 0)
        budget = WARNING_BUDGETS_MS[name]
        if actual > budget:
            warnings.append({"metric": name, "actual_ms": actual, "budget_ms": budget})
    return warnings


def estimate_cost_usd(
    usage: dict[str, Any],
    *,
    input_per_million: float | None = None,
    output_per_million: float | None = None,
    cache_hit_per_million: float | None = None,
) -> float | None:
    if input_per_million is None or output_per_million is None:
        return None
    safe = sanitize_usage(usage)
    prompt = int(safe.get("prompt_tokens") or 0)
    hits = min(prompt, int(safe.get("cache_hit_tokens") or 0))
    misses = max(0, prompt - hits)
    cache_rate = input_per_million if cache_hit_per_million is None else cache_hit_per_million
    completion = int(safe.get("completion_tokens") or 0)
    return round(
        (misses * input_per_million + hits * cache_rate + completion * output_per_million)
        / 1_000_000,
        8,
    )
