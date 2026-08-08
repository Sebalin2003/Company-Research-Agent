from __future__ import annotations

from dataclasses import dataclass, field

from backend.app.domain.cv import (
    contains_term,
    extract_candidate_signals,
    normalize_text,
    useful_lines,
)

REQUIREMENT_MARKERS = (
    "requisito",
    "requirements",
    "excluyente",
    "experiencia",
    "conocimiento",
    "must have",
    "se requiere",
)
RESPONSIBILITY_MARKERS = (
    "responsabilidad",
    "responsibilities",
    "tareas",
    "funciones",
    "vas a",
    "seras responsable",
    "trabajaras",
)


@dataclass(frozen=True)
class JobSignals:
    role: str | None = None
    seniority: str | None = None
    hard_skills: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)
    responsibilities: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    confidence: str = "unknown"


def extract_job_signals(job_description: str) -> JobSignals:
    candidate_like = extract_candidate_signals(job_description)
    lines = useful_lines(job_description)
    responsibilities = marked_lines(lines, RESPONSIBILITY_MARKERS, 5)
    requirements = marked_lines(lines, REQUIREMENT_MARKERS, 6)
    signal_count = sum(
        len(items)
        for items in (
            candidate_like.roles,
            candidate_like.hard_skills,
            candidate_like.tools,
            candidate_like.languages,
            candidate_like.education,
            responsibilities,
            requirements,
        )
    )
    confidence = "medium" if signal_count >= 4 else "low" if signal_count else "unknown"
    return JobSignals(
        role=candidate_like.roles[0] if candidate_like.roles else None,
        seniority=candidate_like.seniority,
        hard_skills=candidate_like.hard_skills,
        tools=candidate_like.tools,
        languages=candidate_like.languages,
        education=candidate_like.education,
        responsibilities=responsibilities,
        requirements=requirements,
        confidence=confidence,
    )


def select_job_evidence_lines(
    job_description: str,
    signals: JobSignals,
    max_lines: int = 10,
) -> list[str]:
    terms = {
        *(signals.hard_skills),
        *(signals.tools),
        *(signals.languages),
    }
    if signals.role:
        terms.add(signals.role)
    if signals.seniority:
        terms.add(signals.seniority)

    selected = []
    marked = {*signals.requirements, *signals.responsibilities, *signals.education}
    for line in useful_lines(job_description):
        normalized = normalize_text(line)
        if line in marked or any(contains_term(normalized, term) for term in terms):
            selected.append(line)
        if len(selected) >= max_lines:
            break
    return list(dict.fromkeys(selected))


def job_signal_payload(signals: JobSignals) -> dict:
    return {
        "role": signals.role,
        "seniority": signals.seniority,
        "hard_skills": signals.hard_skills,
        "tools": signals.tools,
        "languages": signals.languages,
        "education": signals.education,
        "responsibilities": signals.responsibilities,
        "requirements": signals.requirements,
        "confidence": signals.confidence,
    }


def marked_lines(lines: list[str], markers: tuple[str, ...], limit: int) -> list[str]:
    return [
        line
        for line in lines
        if any(marker in normalize_text(line) for marker in markers)
    ][:limit]
