from __future__ import annotations

import re

from backend.app.domain.reports import ConfidenceLevel, EvidenceTopic
from backend.app.research.types import ClassifiedEvidence, ExtractedContent, ScoredSource


TOPIC_KEYWORDS: dict[EvidenceTopic, tuple[str, ...]] = {
    EvidenceTopic.business: (
        "empresa",
        "plataforma",
        "plataformas",
        "servicio",
        "servicios",
        "producto",
        "productos",
        "negocio",
        "solucion",
        "solución",
        "soluciones",
        "ingenieria",
        "ingeniería",
        "consultoria",
        "consultoría",
        "transformacion digital",
        "transformación digital",
        "inteligencia artificial",
    ),
    EvidenceTopic.argentina_presence: (
        "argentina",
        "buenos aires",
        "cordoba",
        "rosario",
        "oficina",
        "oficinas",
        "direccion",
        "dirección",
        "domicilio",
        "sede",
        "contacto",
        "remote",
        "remoto",
        "hibrido",
        "híbrido",
    ),
    EvidenceTopic.employees: (
        "empleados",
        "employees",
        "personas",
        "equipo",
        "headcount",
        "plantilla",
        "trabajadores",
    ),
    EvidenceTopic.salary: (
        "sueldo",
        "sueldos",
        "salario",
        "salarios",
        "salary",
        "remuneracion",
        "remuneración",
        "rango salarial",
        "mensual",
        "ars",
        "it",
        "tecnologia",
        "tecnologÃ­a",
        "sistemas",
        "desarrollador",
        "developer",
        "programador",
        "software",
        "trainee",
        "junior",
        "jr",
        "pasantia",
        "pasantÃ­a",
        "pasante",
        "internship",
        "intern",
    ),
    EvidenceTopic.benefits: (
        "beneficio",
        "beneficios",
        "benefits",
        "prepaga",
        "obra social",
        "vacaciones",
        "bono",
        "bonus",
        "capacitacion",
        "capacitación",
    ),
    EvidenceTopic.culture: ("cultura", "ambiente", "opiniones", "reviews"),
    EvidenceTopic.interview_process: (
        "entrevista",
        "interview",
        "proceso",
        "etapa",
        "etapas",
        "seleccion",
        "selección",
        "recruiter",
        "rrhh",
        "dificultad",
    ),
    EvidenceTopic.interview_questions: ("preguntas", "questions", "technical interview"),
    EvidenceTopic.open_roles: (
        "empleo",
        "trabajo",
        "vacante",
        "oferta",
        "pasantia",
        "pasantía",
        "jobs",
        "careers",
        "busquedas",
    ),
}

SALARY_VALUE_RE = re.compile(
    r"(\$|ars|usd|pesos?)\s*\d|"
    r"\d[\d\.\,]*\s*(ars|usd|pesos?)|"
    r"\d[\d\.\,]*\s*(k|mil|millones?)",
    re.IGNORECASE,
)
EMPLOYEE_COUNT_RE = re.compile(
    r"\d[\d\.\,]*\s*(empleados|employees|personas|trabajadores)|"
    r"(empleados|employees|personas|trabajadores)\s*[:\-]?\s*\d",
    re.IGNORECASE,
)
ADDRESS_RE = re.compile(
    r"\b(av\.?|avenida|calle|ruta|bouchard|alem|corrientes|cordoba|córdoba|"
    r"buenos aires|caba|piso|oficina|sede|domicilio|direccion|dirección)\b",
    re.IGNORECASE,
)
INTERVIEW_STAGE_RE = re.compile(
    r"\b(etapa|etapas|ronda|rondas|recruiter|rrhh|tecnica|técnica|manager|"
    r"assessment|psicotecnico|psicotécnico|entrevista final)\b",
    re.IGNORECASE,
)


def classify_evidence(source: ScoredSource, content: ExtractedContent) -> list[ClassifiedEvidence]:
    text = normalize_text(f"{content.title}. {source.snippet}. {content.text}")
    evidence: list[ClassifiedEvidence] = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            evidence.append(
                ClassifiedEvidence(
                    source_id=source.source_id,
                    topic=topic,
                    claim=build_claim(topic, content.title or source.title),
                    raw_text_excerpt=excerpt(content.text or source.snippet),
                    confidence=confidence_for_match(topic, text),
                )
            )
    if not evidence:
        evidence.append(
            ClassifiedEvidence(
                source_id=source.source_id,
                topic=EvidenceTopic.general,
                claim="La fuente contiene informacion general que requiere revision.",
                raw_text_excerpt=excerpt(content.text or source.snippet),
                confidence=ConfidenceLevel.low,
            )
        )
    return evidence


def build_claim(topic: EvidenceTopic, title: str) -> str:
    topic_labels = {
        EvidenceTopic.business: "negocio principal",
        EvidenceTopic.argentina_presence: "presencia en Argentina",
        EvidenceTopic.employees: "cantidad de empleados",
        EvidenceTopic.salary: "sueldos",
        EvidenceTopic.benefits: "beneficios",
        EvidenceTopic.culture: "cultura laboral",
        EvidenceTopic.interview_process: "proceso de entrevista",
        EvidenceTopic.open_roles: "busquedas abiertas",
        EvidenceTopic.general: "informacion general",
        EvidenceTopic.interview_questions: "preguntas de entrevista",
    }
    return f"La fuente aporta senales sobre {topic_labels[topic]}: {title}."


def confidence_for_match(topic: EvidenceTopic, text: str) -> ConfidenceLevel:
    hits = sum(1 for keyword in TOPIC_KEYWORDS.get(topic, ()) if keyword in text)
    if topic == EvidenceTopic.salary and SALARY_VALUE_RE.search(text):
        return ConfidenceLevel.high if hits >= 2 else ConfidenceLevel.medium
    if topic == EvidenceTopic.employees and EMPLOYEE_COUNT_RE.search(text):
        return ConfidenceLevel.high if hits >= 2 else ConfidenceLevel.medium
    if topic == EvidenceTopic.argentina_presence and ADDRESS_RE.search(text):
        return ConfidenceLevel.high if hits >= 2 else ConfidenceLevel.medium
    if topic == EvidenceTopic.interview_process and INTERVIEW_STAGE_RE.search(text):
        return ConfidenceLevel.high if hits >= 2 else ConfidenceLevel.medium
    if hits >= 2:
        return ConfidenceLevel.medium
    return ConfidenceLevel.low


def excerpt(text: str, max_length: int = 320) -> str:
    cleaned = " ".join(text.split())
    return cleaned[:max_length]


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())
