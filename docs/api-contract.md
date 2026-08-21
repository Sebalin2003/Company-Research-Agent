# API Contract

## Purpose

This document defines the HTTP API contract between the frontend and backend.

The API must support:

- starting company research;
- polling or retrieving saved reports;
- listing recent reports;
- retrieving reports for a company;
- optional CV-based preparation;
- PDF/DOCX CV text extraction;
- optional CV tailoring suggestions and adapted CV drafts;
- report-grounded company chat;
- privacy cleanup for CV-derived data.

Primary references:

- `docs/report-schema.md`
- `docs/database-model.md`

## General Rules

- Base path: `/api`
- Request and response body format: JSON.
- User-facing strings must be Spanish.
- Internal enum values use stable English keys.
- Timestamps use ISO 8601 UTC.
- IDs are strings. MVP may use UUID strings.
- API responses must not expose raw CV text by default.
- Reports can be generated without CV input.
- CV tailoring requires CV input.

## Conversational and CV Library Contract (Implemented)

Conversation messages accept a validated attachment union: `report` and `cv` use an existing `artifact_id`; `job_description` carries `content` and an optional `title`. Job descriptions are owned by their conversation and are persisted with the message.

Persistent CV endpoints:

- `POST /api/cvs`: multipart PDF/DOCX upload with optional `display_name`;
- `GET /api/cvs`: metadata summaries only;
- `GET /api/cvs/{cv_id}`: current editable text, structured signals, and versions;
- `PATCH /api/cvs/{cv_id}`: rename, set default, or create a text-edit version;
- `POST /api/cvs/{cv_id}/versions`: upload a replacement version;
- `GET /api/cvs/{cv_id}/versions/{version_id}`: inspect historical text and signals;
- `GET /api/cvs/{cv_id}/versions/{version_id}/file`: read an uploaded original;
- `DELETE /api/cvs/{cv_id}/versions/{version_id}` and `DELETE /api/cvs/{cv_id}`: confirmed deletion;
- `GET /api/cv-recommendations/{artifact_id}`: retrieve the durable draft and review state.

`POST /api/task-runs/{task_run_id}/resume` accepts `response_type: "review"` with the recommendation `artifact_id`, one decision per suggestion, explicit truth confirmation for accepted `add_only_if_true` items, optional `draft_text`, and `save_as_cv_version`. Saving creates a text-only version and never changes an uploaded binary.

Complete CV text is returned only by explicit CV detail/version endpoints. CV lists, conversation responses, task state, and SSE events contain metadata or artifact IDs only.

## Status Codes

Use these status codes consistently:

- `200 OK`: successful read or completed synchronous operation.
- `201 Created`: report request accepted and report record created.
- `202 Accepted`: report generation started but not completed.
- `400 Bad Request`: invalid request body or unsupported option.
- `404 Not Found`: report or company not found.
- `409 Conflict`: request conflicts with current report state.
- `422 Unprocessable Entity`: request is valid JSON but fails validation.
- `500 Internal Server Error`: unexpected backend failure.
- `502 Bad Gateway`: search, DeepSeek generation, or Google embedding failure.

## Error Response

All errors should use this shape:

```json
{
  "error": {
    "code": "invalid_company_name",
    "message": "El nombre de la empresa es obligatorio.",
    "details": {}
  }
}
```

Fields:

- `code`: stable machine-readable error code.
- `message`: Spanish user-facing message.
- `details`: optional object for validation fields or debug-safe metadata.

Common error codes:

- `invalid_company_name`
- `invalid_cv_text`
- `cv_tailoring_requires_cv`
- `report_not_found`
- `company_not_found`
- `report_generation_failed`
- `search_provider_failed`
- `llm_provider_failed`
- `invalid_report_schema`
- `cv_data_not_found`
- `invalid_cv_file`
- `report_not_ready`
- `chat_provider_failed`

## Report Status Flow

Reports use this lifecycle:

1. `pending`: report record exists but work has not started.
2. `running`: research pipeline is active.
3. `completed`: report is valid and ready for display.
4. `failed`: report generation failed.

The frontend must handle all statuses.

For MVP, `POST /api/research` should return `202 Accepted` after creating the report record. The frontend should poll `GET /api/reports/{report_id}` until completion.

## Endpoints

## `POST /api/cv/extract` (Compatibility)

Extracts CV text from a transient uploaded file.

Request type: `multipart/form-data`

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `file` | file | yes | `.pdf` or `.docx` only. |

Response:

```json
{
  "filename": "cv.pdf",
  "content_type": "application/pdf",
  "character_count": 12000,
  "cv_text": "Texto extraido del CV..."
}
```

Rules:

- The backend must not persist the uploaded file.
- The frontend should copy `cv_text` into the editable CV textarea.
- Extracted text uses the same max length as pasted `cv_text`.

## `POST /api/research`

Starts a new research report or returns a reusable fresh report.

### Request

```json
{
  "company_name": "Mercado Libre",
  "force_refresh": false,
  "cv_text": "Texto opcional del CV del usuario",
  "include_cv_tailoring": false,
  "include_adapted_cv_draft": false
}
```

Fields:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `company_name` | string | yes | Company name entered by user. |
| `force_refresh` | boolean | no | Defaults to false. If true, bypass fresh cached reports. |
| `cv_text` | string/null | no | Optional pasted CV text. |
| `include_cv_tailoring` | boolean | no | Defaults to false. Requires `cv_text`. |
| `include_adapted_cv_draft` | boolean | no | Defaults to false. Requires `include_cv_tailoring`. |

Validation:

- `company_name` must not be empty after trimming.
- `cv_text` may be omitted.
- If provided, `cv_text` must not be empty after trimming.
- `include_cv_tailoring = true` requires `cv_text`.
- `include_adapted_cv_draft = true` requires `include_cv_tailoring = true`.

Privacy:

- The API must not return raw `cv_text`.
- The backend should prefer discarding raw CV text after extracting structured signals.

### `202 Accepted` Response

```json
{
  "report_id": "rep_123",
  "status": "running",
  "company": {
    "id": "comp_123",
    "name": "Mercado Libre",
    "normalized_name": "mercado libre"
  },
  "status_url": "/api/reports/rep_123",
  "reused_existing_report": false
}
```

### `200 OK` Reused Completed Report Response

This response is allowed only when the backend reuses an existing fresh completed report and `force_refresh` is false.

```json
{
  "report_id": "rep_123",
  "status": "completed",
  "company": {
    "id": "comp_123",
    "name": "Mercado Libre",
    "normalized_name": "mercado libre"
  },
  "reused_existing_report": false,
  "report": {
    "schema_version": "1.0",
    "report_id": "rep_123",
    "company": {},
    "status": "completed",
    "language": "es-AR",
    "generated_at": "2026-07-24T00:00:00Z",
    "sections": [],
    "personalized_preparation": null,
    "cv_tailoring": null,
    "sources": [],
    "evidence": [],
    "warnings": [],
    "metadata": {}
  }
}
```

### Error Cases

`400 Bad Request`

```json
{
  "error": {
    "code": "cv_tailoring_requires_cv",
    "message": "Para adaptar el CV, primero tenes que pegar o subir un CV.",
    "details": {
      "field": "cv_text"
    }
  }
}
```

`502 Bad Gateway`

```json
{
  "error": {
    "code": "llm_provider_failed",
    "message": "No se pudo generar el informe con el proveedor de IA. Intentalo nuevamente.",
    "details": {
      "provider": "deepseek"
    }
  }
}
```

## `GET /api/reports/{report_id}`

Returns a report by id.

### Response: `200 OK`

For `completed` reports:

```json
{
  "report": {
    "schema_version": "1.0",
    "report_id": "rep_123",
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
}
```

For `pending` or `running` reports:

```json
{
  "report_id": "rep_123",
  "status": "running",
  "company": {
    "id": "comp_123",
    "name": "Mercado Libre",
    "normalized_name": "mercado libre"
  },
  "progress": {
    "stage": "collecting_sources",
    "message": "Buscando fuentes confiables sobre la empresa."
  }
}
```

For `failed` reports:

```json
{
  "report_id": "rep_123",
  "status": "failed",
  "company": {
    "id": "comp_123",
    "name": "Mercado Libre",
    "normalized_name": "mercado libre"
  },
  "error": {
    "code": "report_generation_failed",
    "message": "No se pudo generar el informe."
  }
}
```

## `POST /api/reports/{report_id}/chat`

Answers a follow-up question using only the completed saved report.

Request:

```json
{
  "message": "Que deberia preparar para la entrevista?"
}
```

Response:

```json
{
  "report_id": "rep_123",
  "answer": "Segun el informe, conviene preparar...",
  "citations": [
    {
      "source_id": "src_123",
      "title": "Careers",
      "url": "https://example.com/careers"
    }
  ],
  "created_at": "2026-07-24T00:00:00Z"
}
```

Rules:

- Only completed reports can be used for factual chat answers.
- If evidence is insufficient, answer with uncertainty instead of inventing details.
- Save user and assistant messages per report.
- This endpoint remains for compatibility; the frontend should use `POST /api/chat`.

### Error Cases

`404 Not Found`

```json
{
  "error": {
    "code": "report_not_found",
    "message": "No se encontro el informe solicitado.",
    "details": {
      "report_id": "rep_123"
    }
  }
}
```

## `POST /api/chat`

Answers from the unified chatbot using RAG over saved report embeddings.

Request:

```json
{
  "message": "Que empresas tienen mejores senales para roles junior IT?",
  "active_report_id": "rep_123"
}
```

Response:

```json
{
  "answer": "Segun los reportes guardados...",
  "scope_used": "all_reports",
  "citations": [
    {
      "source_id": "src_123",
      "title": "Careers",
      "url": "https://example.com/careers",
      "report_id": "rep_123",
      "company_name": "Acme",
      "chunk_title": "Sueldos y beneficios"
    }
  ],
  "created_at": "2026-08-01T00:00:00Z"
}
```

Rules:

- Uses Google embeddings and SQLite vector search over saved completed reports.
- Prioritizes the active report for direct questions and searches all reports for broad or comparative questions.
- Saves user and assistant messages in global chat history.
- Does not embed raw CV text or CV-derived content in v1.

## `POST /api/rag/reindex`

Backfills embeddings for completed reports that do not have RAG chunks yet.

Response:

```json
{
  "indexed_reports": 3,
  "indexed_chunks": 42
}
```

## `GET /api/reports`

Returns recent reports for the report history view.

### Query Parameters

| Parameter | Type | Required | Notes |
| --- | --- | --- | --- |
| `limit` | integer | no | Defaults to 20. Max 100. |
| `offset` | integer | no | Defaults to 0. |
| `status` | string | no | Optional status filter. |
| `company_name` | string | no | Optional search by company name. |

### Response: `200 OK`

```json
{
  "items": [
    {
      "report_id": "rep_123",
      "company": {
        "id": "comp_123",
        "name": "Mercado Libre",
        "normalized_name": "mercado libre"
      },
      "status": "completed",
      "summary": "Empresa regional de comercio electronico y servicios financieros.",
      "generated_at": "2026-07-24T00:00:00Z",
      "valid_until": "2026-08-23T00:00:00Z",
      "used_cv": true,
      "used_cv_tailoring": true
    }
  ],
  "pagination": {
    "limit": 20,
    "offset": 0,
    "total": 1
  }
}
```

Rules:

- This endpoint must not return full report bodies.
- This endpoint must not return raw CV text.

## `GET /api/companies/{company_id}/reports`

Returns report history for a specific company.

### Query Parameters

| Parameter | Type | Required | Notes |
| --- | --- | --- | --- |
| `limit` | integer | no | Defaults to 20. Max 100. |
| `offset` | integer | no | Defaults to 0. |

### Response: `200 OK`

```json
{
  "company": {
    "id": "comp_123",
    "name": "Mercado Libre",
    "normalized_name": "mercado libre"
  },
  "items": [
    {
      "report_id": "rep_123",
      "status": "completed",
      "summary": "Empresa regional de comercio electronico y servicios financieros.",
      "generated_at": "2026-07-24T00:00:00Z",
      "valid_until": "2026-08-23T00:00:00Z",
      "used_cv": false,
      "used_cv_tailoring": false
    }
  ],
  "pagination": {
    "limit": 20,
    "offset": 0,
    "total": 1
  }
}
```

## `DELETE /api/reports/{report_id}/cv-data`

Deletes CV-derived data for a report.

This endpoint exists for privacy cleanup. It should remove candidate profile data, personalized preparation, CV tailoring suggestions, and adapted CV drafts associated with the report.

### Response: `200 OK`

```json
{
  "report_id": "rep_123",
  "deleted": {
    "candidate_profile": true,
    "personalized_preparation": true,
    "cv_tailoring": true
  }
}
```

Rules:

- This must not delete the company report itself.
- The returned report should still be readable, but `personalized_preparation` and `cv_tailoring` should be `null`.
- If no CV data exists, return `200 OK` with all deleted flags as false.

## `DELETE /api/reports/{report_id}`

Deletes one saved report from history.

Response:

```json
{
  "report_id": "rep_123",
  "deleted_report": true,
  "deleted_company": true
}
```

Rules:

- Delete sections, claims, sources, evidence, CV-derived data, and chat messages for that report.
- If no other reports exist for the company, delete the company too.
- If other reports exist for the company, keep the company.

## Request Size Limits

Suggested MVP limits:

- `company_name`: 200 characters.
- `cv_text`: 50,000 characters.

If limits are exceeded, return `422 Unprocessable Entity`.

Example:

```json
{
  "error": {
    "code": "invalid_cv_text",
    "message": "El CV supera el limite permitido para esta version.",
    "details": {
      "max_characters": 50000
    }
  }
}
```

## Frontend Behavior Contract

The frontend should:

- submit company name and optional CV text through `POST /api/research`;
- allow PDF/DOCX upload through `POST /api/cv/extract`, then let the user review/edit extracted text;
- set `include_cv_tailoring` only when the user explicitly enables it;
- set `include_adapted_cv_draft` only when the user asks for a draft;
- poll `GET /api/reports/{report_id}` while status is `pending` or `running`;
- render `completed` reports using the report schema;
- render `failed` reports as recoverable errors;
- never assume `personalized_preparation` exists;
- never assume `cv_tailoring` exists;
- explain that CV tailoring does not modify the original CV.
- show a report-grounded chat after a completed report exists;
- show guide-only chat before a report exists;
- allow deleting individual history items.

## Backend Validation Contract

Before returning or saving a completed report, the backend must validate:

- report JSON matches `docs/report-schema.md`;
- all factual claims cite evidence;
- all cited evidence belongs to the same report;
- all evidence references valid sources;
- source reliability score is between 1 and 5;
- `metadata.llm_provider` is `deepseek` for newly generated reports;
- CV tailoring suggestions do not include unsupported claims;
- `add_only_if_true` suggestions are not included in adapted drafts by default.

## Open Questions

- Which search provider will be used first?
- Should adapted CV drafts be exportable through a separate endpoint later?

