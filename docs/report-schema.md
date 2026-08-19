# Report Schema

## Purpose

This document defines the structured report contract for the company research agent.

The schema must be used consistently by:

- backend API responses;
- database persistence;
- DeepSeek synthesis prompts;
- frontend rendering;
- tests.

The report is Spanish-first, evidence-backed, and candidate-oriented. It must support company-only research, optional CV-based personalized preparation, and optional CV tailoring.

## Design Principles

- Store structured data, not only rendered text.
- Every factual claim should reference evidence when evidence exists.
- Missing evidence must be explicit.
- Inferences must be labeled as inferences.
- CV-based preparation is optional and must not be required for report generation.
- CV tailoring is optional and must never overwrite the original CV.
- User-facing text must be in Spanish.
- Internal enum values should remain stable and English-based for implementation clarity.

## Top-Level Report Object

```json
{
  "schema_version": "1.0",
  "report_id": "uuid",
  "company": {},
  "status": "completed",
  "language": "es-AR",
  "generated_at": "2026-07-24T00:00:00Z",
  "valid_until": "2026-08-23T00:00:00Z",
  "sections": [],
  "personalized_preparation": null,
  "cv_tailoring": null,
  "sources": [],
  "evidence": [],
  "warnings": [],
  "metadata": {}
}
```

Required fields:

- `schema_version`
- `report_id`
- `company`
- `status`
- `language`
- `generated_at`
- `sections`
- `sources`
- `evidence`
- `warnings`
- `metadata`

Optional fields:

- `valid_until`
- `personalized_preparation`
- `cv_tailoring`

## Enums

### Report Status

Allowed values:

- `pending`
- `running`
- `completed`
- `failed`

### Language

MVP value:

- `es-AR`

### Section Type

Allowed values:

- `executive_summary`
- `business`
- `argentina_presence`
- `employees`
- `salary_benefits`
- `culture`
- `interview_process`
- `interview_questions`
- `open_roles`
- `personalized_preparation`
- `cv_tailoring`
- `sources`
- `warnings`

Required report content sections must appear in this order:

1. `executive_summary`
2. `business`
3. `argentina_presence`
4. `employees`
5. `salary_benefits`
6. `culture`
7. `interview_process`
8. `interview_questions`
9. `open_roles`

`sources` and `warnings` may be rendered as dedicated report blocks, but they remain required parts of the user-facing report. If evidence is missing for a content section, return the section with `missing_evidence = true` instead of omitting it.

### Source Type

Allowed values:

- `official`
- `career_page`
- `linkedin`
- `job_board`
- `salary_review_platform`
- `news_media`
- `company_database`
- `secondary`
- `unknown`

### Evidence Topic

Allowed values:

- `business`
- `argentina_presence`
- `employees`
- `salary`
- `benefits`
- `culture`
- `interview_process`
- `interview_questions`
- `open_roles`
- `general`

### Claim Type

Allowed values:

- `fact`
- `inference`
- `recommendation`
- `missing_evidence`

### Confidence Level

Allowed values:

- `high`
- `medium`
- `low`
- `unknown`

## Company Object

```json
{
  "id": "uuid",
  "name": "Mercado Libre",
  "normalized_name": "mercado libre",
  "possible_aliases": ["MercadoLibre", "MELI"],
  "ambiguity_warning": null
}
```

Fields:

- `id`: internal company id.
- `name`: display name.
- `normalized_name`: normalized search and persistence key.
- `possible_aliases`: optional known aliases found during research.
- `ambiguity_warning`: Spanish warning if the company name may refer to multiple entities.

## Section Object

```json
{
  "type": "business",
  "title": "Negocio principal",
  "summary": "Mercado Libre opera una plataforma regional de comercio electronico y servicios financieros.",
  "claims": [],
  "confidence": "medium",
  "missing_evidence": false
}
```

Fields:

- `type`: stable section enum.
- `title`: Spanish title for UI display.
- `summary`: Spanish section summary.
- `claims`: list of claim objects.
- `confidence`: overall confidence for the section.
- `missing_evidence`: true when there is not enough evidence for the section.

## Claim Object

```json
{
  "id": "claim_001",
  "type": "fact",
  "text": "La empresa opera una plataforma de comercio electronico y servicios financieros.",
  "evidence_ids": ["evidence_001", "evidence_002"],
  "confidence": "high"
}
```

Fields:

- `id`: stable id within the report.
- `type`: fact, inference, recommendation, or missing evidence.
- `text`: Spanish user-facing claim.
- `evidence_ids`: evidence supporting the claim. Required for facts when evidence exists.
- `confidence`: claim-level confidence.

Rules:

- A `fact` should have at least one `evidence_id`.
- An `inference` may have evidence but must be worded as interpretation, not certainty.
- A `recommendation` is allowed for interview prep and CV preparation.
- `missing_evidence` claims should explain what could not be verified.

## Source Object

```json
{
  "id": "source_001",
  "title": "Mercado Libre - Careers",
  "url": "https://www.mercadolibre.com.ar/careers",
  "domain": "mercadolibre.com.ar",
  "source_type": "career_page",
  "reliability_score": 5,
  "accessed_at": "2026-07-24T00:00:00Z",
  "published_at": null,
  "snippet": "Trabaja en Mercado Libre...",
  "language": "es",
  "is_current": true
}
```

Fields:

- `id`: stable id within the report.
- `title`: source title.
- `url`: canonical URL.
- `domain`: parsed domain.
- `source_type`: source classification.
- `reliability_score`: integer from 1 to 5.
- `accessed_at`: timestamp when source was fetched.
- `published_at`: source publication date if available.
- `snippet`: short extracted source text.
- `language`: detected language if available.
- `is_current`: whether source appears current enough for the claim type.

## Evidence Object

```json
{
  "id": "evidence_001",
  "source_id": "source_001",
  "topic": "open_roles",
  "claim": "La pagina de carreras muestra busquedas abiertas en Argentina.",
  "raw_text_excerpt": "Ver oportunidades abiertas en Argentina...",
  "confidence": "medium"
}
```

Fields:

- `id`: stable id within the report.
- `source_id`: source backing this evidence.
- `topic`: evidence topic enum.
- `claim`: normalized Spanish evidence claim.
- `raw_text_excerpt`: short excerpt used to support the claim.
- `confidence`: evidence confidence.

Rules:

- Evidence must link to exactly one source.
- Evidence should be short enough to display or audit.
- Evidence should not include full scraped pages.

## Warning Object

```json
{
  "id": "warning_001",
  "type": "missing_salary_data",
  "message": "No se encontro informacion salarial confiable y actualizada para Argentina.",
  "severity": "medium",
  "related_section": "salary_benefits"
}
```

Fields:

- `id`: stable id within the report.
- `type`: implementation-defined warning key.
- `message`: Spanish user-facing warning.
- `severity`: `low`, `medium`, or `high`.
- `related_section`: optional section type.

Common warning types:

- `missing_salary_data`
- `missing_employee_data`
- `missing_argentina_presence`
- `stale_sources`
- `conflicting_sources`
- `company_name_ambiguity`
- `weak_review_sample`
- `missing_open_roles`
- `insufficient_cv_detail`

## Personalized Preparation Object

This object is `null` when the user does not provide a CV.

```json
{
  "cv_profile": {},
  "fit_summary": {},
  "strengths_to_highlight": [],
  "gaps_to_prepare": [],
  "suggested_pitch": {},
  "personalized_questions": [],
  "star_answer_outlines": [],
  "questions_for_company": [],
  "warnings": []
}
```

## CV Tailoring Object

This object is `null` when the user does not request CV tailoring.

```json
{
  "positioning_summary": "Conviene enfatizar experiencia en analisis comercial, automatizacion de reportes y colaboracion con areas de negocio.",
  "change_suggestions": [],
  "adapted_cv_draft": null,
  "warnings": []
}
```

Rules:

- Must not overwrite the original CV.
- Must not invent facts.
- Must keep uncertain additions out of the adapted draft.
- Must label user-confirmation suggestions as `add_only_if_true`.

## CV Tailoring Suggestion Object

```json
{
  "id": "tailoring_001",
  "type": "rewrite",
  "original_text": "Hice reportes para ventas.",
  "suggested_text": "Desarrolle reportes comerciales para dar seguimiento a ventas y detectar oportunidades de mejora.",
  "reason": "La empresa muestra roles donde la capacidad de convertir datos en decisiones comerciales puede ser relevante.",
  "evidence_ids": ["evidence_010"],
  "requires_user_confirmation": false,
  "confidence": "medium"
}
```

Allowed `type` values:

- `rewrite`
- `reorder`
- `emphasize`
- `add_only_if_true`

Fields:

- `id`: stable id within the report.
- `type`: type of suggested change.
- `original_text`: original CV fragment when available.
- `suggested_text`: Spanish suggested replacement or addition.
- `reason`: Spanish explanation.
- `evidence_ids`: company or job evidence supporting the suggestion.
- `requires_user_confirmation`: true for any suggestion that may introduce a detail not explicit in the CV.
- `confidence`: confidence level.

Rules:

- `rewrite`, `reorder`, and `emphasize` must preserve facts already present in the CV.
- `add_only_if_true` must never be included in the adapted CV draft automatically.
- Suggestions based on company/job context should cite evidence.

## Adapted CV Draft Object

```json
{
  "title": "CV adaptado para Mercado Libre",
  "content_markdown": "## Perfil\nAnalista de datos con experiencia en...",
  "included_suggestion_ids": ["tailoring_001"],
  "excluded_suggestion_ids": ["tailoring_004"],
  "warnings": []
}
```

Fields:

- `title`: Spanish draft title.
- `content_markdown`: adapted CV draft in Markdown.
- `included_suggestion_ids`: tailoring suggestions included in the draft.
- `excluded_suggestion_ids`: suggestions intentionally excluded from the draft.
- `warnings`: Spanish warnings related to the draft.

Rules:

- The draft must be a separate artifact from the original CV.
- The draft must not include `add_only_if_true` suggestions unless the user confirms them in a future workflow.
- The draft must not add employers, dates, credentials, tools, metrics, or responsibilities not present in the CV.

## CV Profile Object

```json
{
  "roles": ["Analista de datos"],
  "seniority": "semi senior",
  "industries": ["retail", "fintech"],
  "hard_skills": ["SQL", "Python", "Power BI"],
  "soft_skills": ["comunicacion con stakeholders"],
  "tools": ["BigQuery", "Excel"],
  "education": ["Licenciatura en Administracion"],
  "languages": ["espanol", "ingles intermedio"],
  "achievements": [
    "Automatizo reportes comerciales y redujo tiempos operativos."
  ],
  "confidence": "medium"
}
```

Rules:

- Extract only what appears in the CV.
- Do not infer sensitive attributes.
- Do not make employability judgments.
- If CV text is vague, add an `insufficient_cv_detail` warning.

## Fit Summary Object

```json
{
  "summary": "Tu experiencia en analisis de datos puede ser relevante si la empresa busca perfiles orientados a producto, operaciones o BI.",
  "confidence": "medium",
  "evidence_ids": ["evidence_010"]
}
```

Rules:

- Must distinguish company evidence from CV evidence.
- Must avoid definitive fit/no-fit conclusions.

## Personalized Preparation Item

Use this shape for strengths, gaps, personalized questions, and questions for the company.

```json
{
  "text": "Destaca proyectos donde hayas convertido datos en decisiones comerciales.",
  "reason": "El CV muestra experiencia en reportes comerciales y la empresa publica roles vinculados a operaciones.",
  "evidence_ids": ["evidence_010"],
  "confidence": "medium"
}
```

## STAR Answer Outline

```json
{
  "question": "Contame sobre una vez que usaste datos para mejorar un proceso.",
  "situation": "Elegir un proyecto con problema operativo claro.",
  "task": "Explicar tu responsabilidad concreta.",
  "action": "Describir analisis, herramientas usadas y coordinacion con stakeholders.",
  "result": "Cerrar con impacto medible.",
  "cv_basis": "Automatizacion de reportes comerciales.",
  "confidence": "medium"
}
```

## Metadata Object

```json
{
  "search_provider": "tavily",
  "llm_provider": "deepseek",
  "llm_model": "model-id",
  "source_count": 8,
  "evidence_count": 14,
  "used_cv": true,
  "generation_duration_ms": 12000
}
```

Fields:

- `search_provider`: selected search provider.
- `llm_provider`: must be `deepseek` for newly generated reports.
- `llm_model`: configured DeepSeek model id.
- `source_count`: number of sources retained.
- `evidence_count`: number of evidence items retained.
- `used_cv`: whether CV input was used.
- `generation_duration_ms`: total generation time.

## Minimal Completed Report Example

```json
{
  "schema_version": "1.0",
  "report_id": "rep_123",
  "company": {
    "id": "comp_123",
    "name": "Mercado Libre",
    "normalized_name": "mercado libre",
    "possible_aliases": ["MercadoLibre", "MELI"],
    "ambiguity_warning": null
  },
  "status": "completed",
  "language": "es-AR",
  "generated_at": "2026-07-24T00:00:00Z",
  "valid_until": "2026-08-23T00:00:00Z",
  "sections": [
    {
      "type": "business",
      "title": "Negocio principal",
      "summary": "La empresa opera en comercio electronico y servicios financieros.",
      "claims": [
        {
          "id": "claim_001",
          "type": "fact",
          "text": "La empresa opera una plataforma de comercio electronico y servicios financieros.",
          "evidence_ids": ["evidence_001"],
          "confidence": "high"
        }
      ],
      "confidence": "high",
      "missing_evidence": false
    }
  ],
  "personalized_preparation": null,
  "sources": [
    {
      "id": "source_001",
      "title": "Mercado Libre",
      "url": "https://www.mercadolibre.com.ar/",
      "domain": "mercadolibre.com.ar",
      "source_type": "official",
      "reliability_score": 5,
      "accessed_at": "2026-07-24T00:00:00Z",
      "published_at": null,
      "snippet": "Sitio oficial de Mercado Libre.",
      "language": "es",
      "is_current": true
    }
  ],
  "evidence": [
    {
      "id": "evidence_001",
      "source_id": "source_001",
      "topic": "business",
      "claim": "La empresa opera una plataforma de comercio electronico y servicios financieros.",
      "raw_text_excerpt": "Comercio electronico y servicios financieros...",
      "confidence": "high"
    }
  ],
  "warnings": [],
  "metadata": {
    "search_provider": "tavily",
    "llm_provider": "deepseek",
    "llm_model": "model-id",
    "source_count": 1,
    "evidence_count": 1,
    "used_cv": false,
    "used_cv_tailoring": false,
    "generation_duration_ms": 12000
  }
}
```

## DeepSeek Output Contract

DeepSeek synthesis must return JSON compatible with this schema.

Rules for prompts:

- Provide only collected evidence and source metadata as factual context.
- Instruct the model to write all user-facing text in Spanish.
- Instruct the model to never cite unsupported knowledge.
- Require `evidence_ids` for factual claims.
- Require `missing_evidence` claims when a section lacks support.
- Require CV tailoring suggestions to preserve CV truth and label uncertain additions as `add_only_if_true`.
- Validate the returned JSON server-side before saving.

If the model returns invalid JSON or unsupported claims, the backend should reject the response and either retry with a stricter prompt or return a failed report with a clear error state.

