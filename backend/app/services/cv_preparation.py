from __future__ import annotations

from backend.app.domain.cv import CandidateSignals, candidate_signals_to_schema
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


def build_personalized_preparation(signals: CandidateSignals, evidence_ids: list[str]) -> PersonalizedPreparationSchema:
    cv_profile = candidate_signals_to_schema(signals)
    main_strength = first_available(
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

    return PersonalizedPreparationSchema(
        cv_profile=cv_profile,
        fit_summary=FitSummarySchema(
            summary=(
                "La preparacion conecta las senales verificables del CV con la informacion disponible "
                "de la empresa."
            ),
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
                text="Prepara una explicacion breve sobre areas de la empresa que no aparezcan claramente en tu CV.",
                reason="Evita sobreprometer y ayuda a responder brechas de contexto.",
                evidence_ids=evidence_id_list,
                confidence=ConfidenceLevel.low,
            )
        ],
        suggested_pitch=PersonalizedPreparationItemSchema(
            text=build_pitch(signals),
            reason="El pitch se arma solo con senales presentes en el CV.",
            evidence_ids=[],
            confidence=confidence,
        ),
        personalized_questions=[
            PersonalizedPreparationItemSchema(
                text=f"Contame sobre una experiencia donde aplicaste {first_available(signals.hard_skills, default='una habilidad clave')} para resolver un problema.",
                reason="Pregunta generada desde habilidades detectadas en el CV.",
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
) -> CVTailoringSchema:
    evidence_id_list = evidence_ids[:1]
    confidence = ConfidenceLevel(signals.confidence)
    suggestions = [
        CVTailoringSuggestionSchema(
            id="tailoring_emphasize_skills",
            type=CVTailoringSuggestionType.emphasize,
            original_text=None,
            suggested_text=(
                f"Enfatiza {', '.join(signals.hard_skills[:4])} en el perfil y en los logros principales."
                if signals.hard_skills
                else "Enfatiza logros verificables que sean relevantes para la empresa."
            ),
            reason="La sugerencia reordena o enfatiza informacion ya presente en el CV.",
            evidence_ids=evidence_id_list,
            requires_user_confirmation=False,
            confidence=confidence,
        )
    ]
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
            f"Para {company_name}, conviene priorizar experiencia y logros verificables que ya aparecen en el CV."
        ),
        change_suggestions=suggestions,
        adapted_cv_draft=draft,
        warnings=warnings,
    )


def build_pitch(signals: CandidateSignals) -> str:
    role = first_available(signals.roles, default="profesional")
    skills = ", ".join(signals.hard_skills[:3]) if signals.hard_skills else "herramientas relevantes"
    industry = f" en {signals.industries[0]}" if signals.industries else ""
    return f"Soy {role}{industry}, con experiencia usando {skills} para resolver problemas de negocio."


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
