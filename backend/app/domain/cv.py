from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.app.domain.reports import CVProfileSchema, ConfidenceLevel


@dataclass(frozen=True)
class CandidateSignals:
    roles: list[str] = field(default_factory=list)
    seniority: str | None = None
    industries: list[str] = field(default_factory=list)
    hard_skills: list[str] = field(default_factory=list)
    soft_skills: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    achievements: list[str] = field(default_factory=list)
    confidence: str = "unknown"


def validate_cv_text(cv_text: str | None, max_characters: int) -> str | None:
    if cv_text is None:
        return None
    cleaned = cv_text.strip()
    if not cleaned:
        raise ValueError("El CV no puede estar vacio.")
    if len(cleaned) > max_characters:
        raise ValueError("El CV supera el limite permitido para esta version.")
    return cleaned


ROLE_KEYWORDS = {
    "analista de datos": ("analista de datos", "data analyst"),
    "project manager": ("project manager", "jefe de proyecto", "lider de proyecto"),
    "desarrollador": ("desarrollador", "developer", "software engineer", "programador"),
    "marketing": ("marketing", "growth", "performance"),
    "ventas": ("ventas", "sales", "comercial"),
    "recursos humanos": ("recursos humanos", "human resources", "rrhh"),
    "administracion": ("administrativo", "administrativa"),
}

HARD_SKILLS = (
    "python",
    "sql",
    "excel",
    "power bi",
    "tableau",
    "looker",
    "javascript",
    "typescript",
    "react",
    "node",
    "java",
    "sap",
    "salesforce",
)

TOOLS = (
    "bigquery",
    "postgresql",
    "mysql",
    "jira",
    "trello",
    "notion",
    "figma",
    "google analytics",
    "hubspot",
)

INDUSTRIES = (
    "fintech",
    "banca",
    "retail",
    "ecommerce",
    "e-commerce",
    "salud",
    "educacion",
    "logistica",
    "seguros",
    "industria",
)

SOFT_SKILLS = {
    "comunicacion": ("comunicacion", "comunicar", "stakeholders"),
    "liderazgo": ("liderazgo", "lider", "coordine", "coordinar"),
    "trabajo en equipo": ("equipo", "colaboracion", "colaborativo"),
    "resolucion de problemas": ("problemas", "mejora", "optimizacion"),
}


def extract_candidate_signals(cv_text: str) -> CandidateSignals:
    text = normalize_text(cv_text)
    roles = find_keyword_group(text, ROLE_KEYWORDS)
    hard_skills = find_terms(text, HARD_SKILLS)
    tools = find_terms(text, TOOLS)
    industries = normalize_industries(find_terms(text, INDUSTRIES))
    soft_skills = find_keyword_group(text, SOFT_SKILLS)
    languages = extract_languages(text)
    education = extract_education(cv_text)
    achievements = extract_achievements(cv_text)
    seniority = extract_seniority(text)

    signal_count = sum(
        len(items)
        for items in [roles, hard_skills, tools, industries, soft_skills, languages, education, achievements]
    )
    confidence = "medium" if signal_count >= 4 else "low" if signal_count else "unknown"
    return CandidateSignals(
        roles=roles,
        seniority=seniority,
        industries=industries,
        hard_skills=hard_skills,
        soft_skills=soft_skills,
        tools=tools,
        education=education,
        languages=languages,
        achievements=achievements,
        confidence=confidence,
    )


def candidate_signals_to_schema(signals: CandidateSignals) -> CVProfileSchema:
    return CVProfileSchema(
        roles=signals.roles,
        seniority=signals.seniority,
        industries=signals.industries,
        hard_skills=signals.hard_skills,
        soft_skills=signals.soft_skills,
        tools=signals.tools,
        education=signals.education,
        languages=signals.languages,
        achievements=signals.achievements,
        confidence=ConfidenceLevel(signals.confidence),
    )


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def find_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if contains_term(text, term)]


def find_keyword_group(text: str, groups: dict[str, tuple[str, ...]]) -> list[str]:
    return [label for label, keywords in groups.items() if any(contains_term(text, keyword) for keyword in keywords)]


def contains_term(text: str, term: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) is not None


def normalize_industries(industries: list[str]) -> list[str]:
    normalized = ["ecommerce" if item == "e-commerce" else item for item in industries]
    return list(dict.fromkeys(normalized))


def extract_seniority(text: str) -> str | None:
    if "semi senior" in text or "ssr" in text:
        return "semi senior"
    if "senior" in text or "sr" in text:
        return "senior"
    if "junior" in text or "jr" in text:
        return "junior"
    return None


def extract_languages(text: str) -> list[str]:
    languages = []
    for language in ("espanol", "ingles", "portugues", "frances", "aleman"):
        if language in text:
            match = re.search(rf"{language}\s+(basico|intermedio|avanzado|nativo)", text)
            languages.append(f"{language} {match.group(1)}" if match else language)
    return languages


def extract_education(cv_text: str) -> list[str]:
    education = []
    for line in useful_lines(cv_text):
        lowered = line.lower()
        if any(term in lowered for term in ("licenciatura", "ingenieria", "tecnicatura", "universidad", "mba")):
            education.append(line)
    return education[:5]


def extract_achievements(cv_text: str) -> list[str]:
    achievements = []
    for line in useful_lines(cv_text):
        lowered = line.lower()
        if any(
            term in lowered
            for term in ("automat", "reduj", "aument", "mejor", "lider", "implemente", "optimice", "optimiz")
        ):
            achievements.append(line)
    return achievements[:6]


def useful_lines(value: str) -> list[str]:
    return [
        line.strip(" -\t")
        for line in value.splitlines()
        if 8 <= len(line.strip(" -\t")) <= 240
    ]
