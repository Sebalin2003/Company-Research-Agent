from __future__ import annotations

from backend.app.domain.cv import CandidateSignals, candidate_signals_to_schema
from backend.app.domain.jobs import JobSignals
from backend.app.domain.reports import (
    AdaptedCVDraftSchema,
    ConfidenceLevel,
    CVTailoringSchema,
    CVTailoringSuggestionSchema,
    CVTailoringSuggestionType,
    FitSummarySchema,
    PersonalizedPreparationItemSchema,
    PersonalizedPreparationSchema,
    WarningSchema,
    WarningSeverity,
)


def build_personalized_preparation(
    signals: CandidateSignals,
    evidence_ids: list[str],
    job_signals: JobSignals | None = None,
) -> PersonalizedPreparationSchema:
    cv_profile = candidate_signals_to_schema(signals)
    matching_skills = matching_job_terms(signals, job_signals)
    missing_skills = missing_job_terms(signals, job_signals)
    main_strength = first_available(
        matching_skills,
        signals.achievements,
        signals.hard_skills,
        signals.roles,
        default="tu experiencia verificable",
    )
    evidence_id_list = evidence_ids[:1]
    confidence = ConfidenceLevel(signals.confidence)
    warnings = []
    if signals.confidence in {"low", "unknown"}:
        warnings.append(
            WarningSchema(
                id="warning_cv_detail",
                type="insufficient_cv_detail",
                message="El CV tiene pocas senales estructuradas; la preparacion es conservadora.",
                severity=WarningSeverity.medium,
            )
        )

    fit_summary = (
        "La preparacion conecta las senales verificables del CV con la informacion disponible "
        "de la empresa."
    )
    if job_signals:
        target = job_signals.role or "el puesto"
        fit_summary = (
            f"Para {target}, se detectaron {len(matching_skills)} coincidencias verificables "
            f"y {len(missing_skills)} requisitos que no aparecen claramente en el CV."
        )

    gap_text = (
        f"Prepara evidencia sobre {missing_skills[0]}; el requisito no aparece claramente en tu CV."
        if missing_skills
        else "Prepara una explicacion breve sobre areas del puesto que no aparezcan claramente en tu CV."
    )
    question_skill = first_available(
        matching_skills,
        job_signals.hard_skills if job_signals else [],
        signals.hard_skills,
        default="una habilidad clave",
    )

    return PersonalizedPreparationSchema(
        cv_profile=cv_profile,
        fit_summary=FitSummarySchema(
            summary=fit_summary,
            confidence=confidence,
            evidence_ids=evidence_id_list,
        ),
        strengths_to_highlight=[
            PersonalizedPreparationItemSchema(
                text=f"Destaca {main_strength} con un ejemplo concreto y medible.",
                reason="La recomendacion usa contenido detectado en el CV, sin agregar experiencia nueva.",
                evidence_ids=evidence_id_list,
                confidence=confidence,
            )
        ],
        gaps_to_prepare=[
            PersonalizedPreparationItemSchema(
                text=gap_text,
                reason="La brecha se presenta como un tema a preparar, no como experiencia del candidato.",
                evidence_ids=evidence_id_list,
                confidence=ConfidenceLevel.low,
            )
        ],
        suggested_pitch=PersonalizedPreparationItemSchema(
            text=build_pitch(signals, job_signals.role if job_signals else None),
            reason="El pitch se arma solo con senales presentes en el CV.",
            evidence_ids=[],
            confidence=confidence,
        ),
        personalized_questions=[
            PersonalizedPreparationItemSchema(
                text=f"Contame sobre una experiencia donde aplicaste {question_skill} para resolver un problema.",
                reason="Pregunta generada desde la coincidencia entre el CV y el puesto.",
                evidence_ids=[],
                confidence=confidence,
            )
        ],
        questions_for_company=[
            PersonalizedPreparationItemSchema(
                text="Que desafios tendria este rol durante los primeros tres meses?",
                reason="Pregunta util para validar expectativas del puesto.",
                evidence_ids=[],
                confidence=ConfidenceLevel.medium,
            )
        ],
        warnings=warnings,
    )


def build_cv_tailoring(
    signals: CandidateSignals,
    company_name: str,
    evidence_ids: list[str],
    include_adapted_cv_draft: bool,
    job_signals: JobSignals | None = None,
) -> CVTailoringSchema:
    evidence_id_list = evidence_ids[:1]
    confidence = ConfidenceLevel(signals.confidence)
    matching_skills = matching_job_terms(signals, job_signals)
    missing_skills = missing_job_terms(signals, job_signals)
    suggestions = [
        CVTailoringSuggestionSchema(
            id="tailoring_emphasize_skills",
            type=CVTailoringSuggestionType.emphasize,
            original_text=None,
            suggested_text=(
                f"Enfatiza {', '.join((matching_skills or signals.hard_skills)[:4])} en el perfil y en los logros principales."
                if matching_skills or signals.hard_skills
                else "Enfatiza logros verificables que sean relevantes para la empresa."
            ),
            reason="La sugerencia reordena o enfatiza informacion ya presente en el CV.",
            evidence_ids=evidence_id_list,
            requires_user_confirmation=False,
            confidence=confidence,
        )
    ]
    if missing_skills:
        suggestions.append(
            CVTailoringSuggestionSchema(
                id="tailoring_job_gap_if_true",
                type=CVTailoringSuggestionType.add_only_if_true,
                original_text=None,
                suggested_text=(
                    f"Agrega experiencia con {missing_skills[0]} solo si es real y verificable."
                ),
                reason="El puesto menciona este requisito, pero no aparece en las senales del CV.",
                evidence_ids=evidence_id_list,
                requires_user_confirmation=True,
                confidence=ConfidenceLevel.low,
            )
        )
    if signals.achievements:
        suggestions.append(
            CVTailoringSuggestionSchema(
                id="tailoring_rewrite_achievement",
                type=CVTailoringSuggestionType.rewrite,
                original_text=signals.achievements[0],
                suggested_text=f"Reescribe este logro con accion, herramienta e impacto: {signals.achievements[0]}",
                reason="Mejora claridad sin cambiar el hecho original.",
                evidence_ids=[],
                requires_user_confirmation=False,
                confidence=confidence,
            )
        )
    else:
        suggestions.append(
            CVTailoringSuggestionSchema(
                id="tailoring_add_metric_if_true",
                type=CVTailoringSuggestionType.add_only_if_true,
                original_text=None,
                suggested_text="Agrega metricas de impacto solo si son reales y verificables.",
                reason="El CV no muestra resultados medibles suficientes.",
                evidence_ids=[],
                requires_user_confirmation=True,
                confidence=ConfidenceLevel.low,
            )
        )

    draft = None
    if include_adapted_cv_draft:
        draft = AdaptedCVDraftSchema(
            title=f"CV adaptado para {company_name}",
            content_markdown=build_adapted_cv_draft(company_name, signals),
            included_suggestion_ids=[
                suggestion.id
                for suggestion in suggestions
                if suggestion.type != CVTailoringSuggestionType.add_only_if_true
            ],
            excluded_suggestion_ids=[
                suggestion.id
                for suggestion in suggestions
                if suggestion.type == CVTailoringSuggestionType.add_only_if_true
            ],
        )

    warnings = []
    if signals.confidence in {"low", "unknown"}:
        warnings.append(
            WarningSchema(
                id="warning_cv_tailoring_detail",
                type="insufficient_cv_detail",
                message="La adaptacion del CV es limitada porque faltan logros, herramientas o roles claros.",
                severity=WarningSeverity.medium,
            )
        )

    return CVTailoringSchema(
        positioning_summary=(
            f"Para {company_name}"
            f"{f' y el rol {job_signals.role}' if job_signals and job_signals.role else ''}, "
            "conviene priorizar experiencia y logros verificables que ya aparecen en el CV."
        ),
        change_suggestions=suggestions,
        adapted_cv_draft=draft,
        warnings=warnings,
    )


def build_pitch(signals: CandidateSignals, target_role: str | None = None) -> str:
    role = first_available(signals.roles, default="profesional")
    skills = ", ".join(signals.hard_skills[:3]) if signals.hard_skills else "herramientas relevantes"
    industry = f" en {signals.industries[0]}" if signals.industries else ""
    target = f" Para el rol de {target_role}," if target_role else ""
    return (
        f"Soy {role}{industry}, con experiencia usando {skills} para resolver problemas de negocio."
        f"{target} puedo aportar esas capacidades sin agregar experiencia no verificada."
    )


def build_adapted_cv_draft(company_name: str, signals: CandidateSignals) -> str:
    lines = [
        f"# CV adaptado para {company_name}",
        "",
        "## Perfil",
        build_pitch(signals),
    ]
    if signals.hard_skills or signals.tools:
        lines.extend(["", "## Habilidades relevantes", ", ".join(signals.hard_skills + signals.tools)])
    if signals.achievements:
        lines.extend(["", "## Logros seleccionados"])
        lines.extend(f"- {achievement}" for achievement in signals.achievements[:4])
    return "\n".join(lines)


def first_available(*groups: list[str], default: str) -> str:
    for group in groups:
        if group:
            return group[0]
    return default


def matching_job_terms(
    signals: CandidateSignals,
    job_signals: JobSignals | None,
) -> list[str]:
    if not job_signals:
        return []
    candidate_terms = {
        *signals.hard_skills,
        *signals.tools,
        *signals.languages,
    }
    return [
        term
        for term in [*job_signals.hard_skills, *job_signals.tools, *job_signals.languages]
        if term in candidate_terms
    ]


def missing_job_terms(
    signals: CandidateSignals,
    job_signals: JobSignals | None,
) -> list[str]:
    if not job_signals:
        return []
    candidate_terms = {
        *signals.hard_skills,
        *signals.tools,
        *signals.languages,
    }
    return [
        term
        for term in [*job_signals.hard_skills, *job_signals.tools, *job_signals.languages]
        if term not in candidate_terms
    ]
