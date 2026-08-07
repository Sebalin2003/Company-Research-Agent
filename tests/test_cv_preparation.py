from __future__ import annotations

from backend.app.domain.cv import extract_candidate_signals, select_cv_evidence_lines
from backend.app.services.cv_preparation import build_cv_tailoring, build_personalized_preparation


def test_extract_candidate_signals_detects_cv_facts() -> None:
    cv_text = """
    Analista de datos semi senior en fintech y retail.
    Experiencia con SQL, Python, Power BI y BigQuery.
    Ingles intermedio.
    Licenciatura en Administracion, Universidad de Buenos Aires.
    Optimice reportes comerciales y reduje tiempos de seguimiento.
    """

    signals = extract_candidate_signals(cv_text)

    assert signals.roles == ["analista de datos"]
    assert signals.seniority == "semi senior"
    assert signals.industries == ["fintech", "retail"]
    assert signals.hard_skills == ["python", "sql", "power bi"]
    assert signals.tools == ["bigquery"]
    assert signals.languages == ["ingles intermedio"]
    assert signals.education == ["Licenciatura en Administracion, Universidad de Buenos Aires."]
    assert signals.achievements == ["Optimice reportes comerciales y reduje tiempos de seguimiento."]
    assert signals.confidence == "medium"


def test_personalized_preparation_uses_detected_cv_profile() -> None:
    signals = extract_candidate_signals("Data analyst con SQL y Excel. Mejore tableros de gestion.")

    preparation = build_personalized_preparation(signals, ["evidence_1"])

    assert preparation.cv_profile.roles == ["analista de datos"]
    assert preparation.cv_profile.hard_skills == ["sql", "excel"]
    assert "Mejore tableros de gestion." in preparation.strengths_to_highlight[0].text
    assert preparation.fit_summary.evidence_ids == ["evidence_1"]
    assert preparation.suggested_pitch is not None
    assert "analista de datos" in preparation.suggested_pitch.text


def test_select_cv_evidence_lines_keeps_only_relevant_cv_lines() -> None:
    cv_text = """
    Data analyst con SQL y Excel.
    Esta linea privada no coincide con senales.
    Universidad de Buenos Aires.
    Optimice tableros comerciales.
    """
    signals = extract_candidate_signals(cv_text)

    lines = select_cv_evidence_lines(cv_text, signals)

    assert "Data analyst con SQL y Excel." in lines
    assert "Universidad de Buenos Aires." in lines
    assert "Optimice tableros comerciales." in lines
    assert "Esta linea privada no coincide con senales." not in lines


def test_cv_tailoring_keeps_unverified_metrics_out_of_draft() -> None:
    signals = extract_candidate_signals("Project manager con experiencia en banca y Jira.")

    tailoring = build_cv_tailoring(
        signals=signals,
        company_name="Banco Ejemplo",
        evidence_ids=["evidence_1"],
        include_adapted_cv_draft=True,
    )

    add_if_true = [
        suggestion
        for suggestion in tailoring.change_suggestions
        if suggestion.type == "add_only_if_true"
    ][0]
    assert add_if_true.requires_user_confirmation is True
    assert tailoring.adapted_cv_draft is not None
    assert add_if_true.id in tailoring.adapted_cv_draft.excluded_suggestion_ids
    assert add_if_true.id not in tailoring.adapted_cv_draft.included_suggestion_ids
    assert "Agrega metricas" not in tailoring.adapted_cv_draft.content_markdown
