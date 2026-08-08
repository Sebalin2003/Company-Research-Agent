from __future__ import annotations

import json

from backend.app.domain.reports import ConfidenceLevel, EvidenceTopic, SourceType
from backend.app.domain.reports import StructuredReportSchema
from backend.app.llm.synthesizer import SynthesisRequest
from backend.app.research.types import ClassifiedEvidence, ScoredSource


MAX_EXCERPT_CHARACTERS = 450
MAX_TOTAL_EVIDENCE = 18
TOPIC_EVIDENCE_LIMITS = {
    EvidenceTopic.business: 2,
    EvidenceTopic.argentina_presence: 2,
    EvidenceTopic.employees: 2,
    EvidenceTopic.salary: 3,
    EvidenceTopic.benefits: 2,
    EvidenceTopic.culture: 2,
    EvidenceTopic.interview_process: 2,
    EvidenceTopic.interview_questions: 2,
    EvidenceTopic.open_roles: 2,
    EvidenceTopic.general: 1,
}


SYSTEM_PROMPT = """\
Sos un asistente de investigacion laboral para personas que buscan trabajo en Argentina.
Tu respuesta debe ser JSON valido y todo texto visible para el usuario debe estar en espanol.
Usa solo la evidencia entregada. No inventes fuentes, empleados, sueldos, beneficios, puestos ni datos del CV.
Cuando falte evidencia, declaralo como missing_evidence.
Las afirmaciones factuales deben citar evidence_ids y copiar un supporting_quote textual del raw_text_excerpt citado.
supporting_quote es obligatorio para claims fact e inference, debe tener como maximo 450 caracteres y no debe ser una parafrasis.
Las claims recommendation y missing_evidence deben usar supporting_quote=null.
El informe debe respetar esta estructura de secciones y este orden:
1. executive_summary: resumen breve y lo primero que debe saber una persona candidata.
2. business: que hace la empresa, productos, servicios y modelo de negocio si hay evidencia.
3. argentina_presence: oficinas, contratacion, mercado local y senales remoto/hibrido si hay evidencia.
4. employees: cantidad aproximada de empleados, fuente y nivel de confianza.
5. salary_benefits: sueldos y beneficios, priorizando rangos IT para trainee, pasantia/internship y junior.
6. culture: cultura laboral, temas positivos/negativos y cuidado con reviews escasas.
7. interview_process: etapas, evaluaciones y dificultad solo si hay evidencia.
8. interview_questions: preguntas posibles separadas por generales, empresa, conductuales y tecnicas cuando sea util.
9. open_roles: busquedas abiertas y links a carreras o portales laborales.
No omitas secciones: si falta evidencia para una seccion, inclui una seccion con missing_evidence=true.
Para direccion, cantidad de empleados, rango salarial, beneficios, etapas de entrevista y vacantes:
- informa valores exactos solo si aparecen explicitamente en raw_text_excerpt o snippet;
- si la fuente habla del tema pero no da un valor concreto, escribe "No disponible en las fuentes consultadas" o "Evidencia insuficiente";
- no infieras rangos salariales, direcciones, headcount ni etapas de entrevista a partir de conocimiento general.
En salary_benefits:
- busca primero sueldos IT de trainee, pasantia/internship y junior en Argentina;
- separa esos niveles si hay evidencia numerica por rol;
- si solo hay salarios generales o de otros niveles, aclara que no son especificos para IT trainee/pasantia/junior;
- no conviertas ni actualices montos por inflacion si la fuente no lo dice.
Usa resumenes concisos. Evita copiar textos largos de las fuentes.
Inclui como maximo una claim factual por seccion; si la seccion es incierta, usa missing_evidence.
"""


def build_report_user_prompt(request: SynthesisRequest) -> str:
    payload = build_compact_report_payload(request)
    return (
        "Genera un informe compatible con el schema del backend. "
        "Inclui todas las secciones requeridas del PRD, fuentes, evidencia, advertencias y metadata. "
        "Si hay CV, genera preparacion personalizada. "
        "Si se pidio tailoring, genera sugerencias seguras sin modificar el CV original. "
        "Cita solo evidence_ids presentes en el contexto. "
        "Para cada claim fact o inference, inclui supporting_quote copiando textualmente hasta 450 caracteres del raw_text_excerpt citado. "
        "No muestres ni expliques supporting_quote fuera del campo JSON interno. "
        "Escribi resumenes breves y como maximo una claim por seccion. "
        "Para direccion, empleados, salarios, beneficios, entrevistas y vacantes, usa valores exactos solo si aparecen en el contexto; "
        "si no aparecen, escribe \"No disponible en las fuentes consultadas\" o \"Evidencia insuficiente\". "
        "En sueldos, prioriza perfiles IT trainee, pasantia/internship y junior en Argentina; "
        "si no hay montos para esos niveles, indicalo explicitamente y no uses rangos generales como si fueran entry-level. "
        "Contexto estructurado compactado:\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def build_compact_report_payload(request: SynthesisRequest) -> dict:
    selected_evidence = compact_evidence(request.evidence, request.sources)
    selected_source_ids = {item.source_id for item in selected_evidence}
    payload = {
        "report_id": request.report_id,
        "company": {
            "id": request.company_id,
            "name": request.company_name,
            "normalized_name": request.normalized_company_name,
        },
        "sources": [
            {
                "id": source.source_id,
                "title": source.title,
                "url": source.url,
                "domain": source.domain,
                "source_type": source.source_type.value,
                "reliability_score": source.reliability_score,
                "is_current": source.is_current,
            }
            for source in request.sources
            if source.source_id in selected_source_ids
        ],
        "evidence": [
            {
                "id": f"evidence_{index + 1}",
                "source_id": item.source_id,
                "topic": item.topic.value,
                "claim": item.claim,
                "raw_text_excerpt": truncate_text(item.raw_text_excerpt, MAX_EXCERPT_CHARACTERS),
                "confidence": item.confidence.value,
            }
            for index, item in enumerate(selected_evidence)
        ],
        "cv": {
            "provided": request.include_cv,
            "tailoring_requested": request.include_cv_tailoring,
            "adapted_draft_requested": request.include_adapted_cv_draft,
            "text": request.cv_text if request.include_cv else None,
        },
    }
    return (
        payload
    )


def build_repair_prompt(invalid_output: str, validation_error: str) -> str:
    return (
        "La respuesta anterior no cumplio el schema JSON requerido. "
        "Devuelve solo JSON valido compatible con el schema del backend. "
        "No agregues Markdown ni explicaciones. "
        "Conserva los hechos y evidencia_ids disponibles; no inventes fuentes ni datos. "
        "Error de validacion resumido:\n"
        f"{truncate_text(validation_error, 2000)}\n"
        "Respuesta invalida a reparar:\n"
        f"{truncate_text(invalid_output, 12000)}"
    )


def compact_evidence(
    evidence: list[ClassifiedEvidence],
    sources: list[ScoredSource],
) -> list[ClassifiedEvidence]:
    source_by_id = {source.source_id: source for source in sources}
    selected = []
    for topic in EvidenceTopic:
        topic_items = [item for item in evidence if item.topic == topic]
        topic_items.sort(
            key=lambda item: evidence_rank(item, source_by_id.get(item.source_id)),
            reverse=True,
        )
        selected.extend(topic_items[: TOPIC_EVIDENCE_LIMITS.get(topic, 1)])
    selected.sort(
        key=lambda item: evidence_rank(item, source_by_id.get(item.source_id)),
        reverse=True,
    )
    return selected[:MAX_TOTAL_EVIDENCE]


def evidence_rank(item: ClassifiedEvidence, source: ScoredSource | None) -> tuple[int, int, int, int]:
    source_type = source.source_type if source else SourceType.unknown
    reliability = source.reliability_score if source else 0
    return (
        confidence_score(item.confidence),
        topic_source_score(item.topic, source_type),
        reliability,
        -len(item.raw_text_excerpt or ""),
    )


def confidence_score(confidence: ConfidenceLevel) -> int:
    return {
        ConfidenceLevel.high: 4,
        ConfidenceLevel.medium: 3,
        ConfidenceLevel.low: 2,
        ConfidenceLevel.unknown: 1,
    }[confidence]


def topic_source_score(topic: EvidenceTopic, source_type: SourceType) -> int:
    if topic in {EvidenceTopic.salary, EvidenceTopic.benefits, EvidenceTopic.culture}:
        if source_type == SourceType.salary_review_platform:
            return 4
        if source_type == SourceType.job_board:
            return 3
    if topic in {EvidenceTopic.interview_process, EvidenceTopic.interview_questions}:
        if source_type in {SourceType.salary_review_platform, SourceType.job_board}:
            return 3
    if topic == EvidenceTopic.open_roles:
        if source_type in {SourceType.career_page, SourceType.job_board, SourceType.linkedin}:
            return 3
    if source_type in {SourceType.official, SourceType.career_page}:
        return 2
    return 1


def truncate_text(value: str | None, max_characters: int) -> str | None:
    if value is None or len(value) <= max_characters:
        return value
    return value[: max_characters - 3].rstrip() + "..."


def report_response_json_schema() -> dict:
    return StructuredReportSchema.model_json_schema()

