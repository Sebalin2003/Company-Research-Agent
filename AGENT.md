# Agent Guide: Enterprise Research Agent

## Project Context

This project is a Spanish-first company research agent for job seekers in Argentina. The product is an **API + frontend** application that helps users investigate companies before applying or interviewing.

The agent must research reliable public sources, generate a structured Spanish report, cite evidence, show uncertainty, optionally personalize interview preparation using the user's CV, and optionally suggest safe CV tailoring for a target company.

Primary reference document:

- `PRD.md`

## Product Goals

- Help job seekers understand a company before applying or interviewing.
- Support companies from any industry, not only technology.
- Produce practical insights in Spanish for users in Argentina.
- Save generated reports for later access.
- Support optional CV-based personalized preparation.
- Support optional CV tailoring suggestions without modifying the original CV.
- Keep factual claims tied to sources wherever possible.

## MVP Scope

The MVP must include:

- API backend.
- Responsive frontend.
- Company search input.
- Optional pasted CV input.
- Saved company research reports.
- Reliable source collection.
- Evidence classification.
- Source reliability scoring.
- Spanish report synthesis.
- CV-to-company fit preparation.
- CV tailoring suggestions and optional adapted CV draft.
- Warnings for weak, missing, stale, or conflicting evidence.

Report sections should include:

- resumen ejecutivo;
- negocio principal;
- presencia en Argentina;
- cantidad de empleados;
- sueldos y beneficios;
- cultura laboral;
- proceso de entrevista;
- posibles preguntas de entrevista;
- preparacion personalizada basada en CV, when CV is provided;
- sugerencias de adaptacion del CV, when requested;
- busquedas abiertas;
- fuentes;
- advertencias.

## Non-MVP Scope

Do not implement these unless explicitly requested:

- user accounts;
- saved candidate profiles;
- automatic company comparison;
- email alerts;
- scheduled refreshes;
- scraping behind logins;
- browser automation for protected pages;
- paid-source integrations beyond the chosen search provider.

## Recommended Stack

Backend:

- Python.
- FastAPI.
- Pydantic.
- SQLAlchemy or SQLModel.
- Alembic.
- SQLite for MVP.
- PostgreSQL later if deployment requires multi-user persistence.
- pytest.

Frontend:

- React.
- TypeScript.
- Vite.
- Spanish-first UI.
- Mobile-first responsive design.

Research and extraction:

- Search provider API such as Tavily, SerpAPI, Bing Web Search API, or Brave Search API.
- `httpx` for HTTP requests.
- `BeautifulSoup` or `trafilatura` for page extraction.
- PDF/DOCX parsing is supported for CV file uploads; uploaded files must remain transient.

LLM:

- Use **Google Gemini** by default.
- Keep provider-specific code behind an internal abstraction.
- The LLM must synthesize from evidence, not act as a source of truth.

Suggested abstraction:

- `ReportSynthesizer` interface.
- `GeminiReportSynthesizer` implementation.

## Architecture Rules

- Keep source collection deterministic.
- Keep source scoring deterministic.
- Keep evidence classification testable.
- Use the LLM only for Spanish synthesis, candidate-oriented insights, interview preparation, and uncertainty wording.
- Store structured report data, not only rendered Markdown.
- Preserve source metadata and citation mapping.
- Make report generation work without a CV.
- Make CV personalization optional and additive.

## Evidence and Quality Rules

The system must not:

- invent exact employee counts;
- invent salary ranges;
- claim current job openings without current source evidence;
- use LLM training knowledge as a cited source;
- hide uncertainty;
- overgeneralize anonymous employee reviews;
- make sensitive inferences from CV content;
- expose raw CV text outside the report-generation context.
- modify the original CV without explicit user action;
- invent or exaggerate CV content.

The system should:

- cite sources for factual claims;
- separate facts from inferences;
- flag weak evidence;
- prefer official, recent, and direct sources;
- show source reliability;
- say when evidence is insufficient;
- keep user-facing output in Spanish.

## CV-Based Preparation Frame

Use a **CV-to-company fit mapping** framework.

Steps:

1. Extract candidate signals:
   - roles;
   - seniority;
   - industries;
   - hard skills;
   - soft skills;
   - tools;
   - achievements;
   - education;
   - languages.

2. Map signals to company context:
   - company business;
   - Argentina presence;
   - open roles;
   - interview process;
   - culture evidence;
   - likely role families.

3. Generate preparation:
   - fortalezas para destacar;
   - brechas o puntos a preparar;
   - pitch sugerido;
   - preguntas probables personalizadas;
   - respuestas recomendadas en formato STAR;
   - preguntas para hacerle a la empresa;
   - alertas de evidencia.

4. Add caveats:
   - distinguish evidence-backed guidance from broad inference;
   - warn when job-opening evidence is weak or missing;
   - warn when the CV lacks detail.

## CV Tailoring Frame

The agent may suggest CV changes for a target company, but it must not overwrite the original CV.

Use a **safe CV tailoring** framework.

Rules:

- Preserve truth from the original CV.
- Never invent experience, employers, education, certifications, tools, metrics, dates, or responsibilities.
- Separate suggestions from the original CV.
- Generate an adapted CV draft only when explicitly requested.
- Label suggestions as `rewrite`, `reorder`, `emphasize`, or `add_only_if_true`.
- Use `add_only_if_true` when the change requires confirmation from the user.
- Explain why each suggestion is relevant.
- Cite company or job evidence when the suggestion depends on researched context.

Tailoring output should include:

- resumen de enfoque;
- sugerencias de cambios;
- borrador adaptado opcional;
- advertencias.

## Frontend Guidelines

- UI text must be in Spanish.
- Design mobile-first and verify mobile, tablet, and desktop layouts.
- The company input and CV input must be usable on narrow screens.
- CV tailoring controls must make clear that the original CV is not modified.
- Report sections should stack vertically on mobile.
- Long source URLs must wrap without horizontal overflow.
- Use clear loading, empty, error, and saved-report states.
- Do not build a marketing landing page as the main experience.
- The first screen should let the user start researching a company.

## API Guidelines

Core endpoints expected by the PRD:

- `POST /api/research`
- `GET /api/reports/{report_id}`
- `GET /api/companies/{company_id}/reports`
- `GET /api/reports`

`POST /api/research` should accept:

- company name;
- force refresh flag;
- optional CV text.
- optional CV tailoring flag.

Responses should expose:

- report id;
- status;
- company name;
- generated timestamp;
- structured report;
- sources;
- warnings.

## Data Guidelines

Persist at least:

- companies;
- reports;
- sources;
- evidence;
- optional candidate profile data derived from CV;
- optional personalized preparation.
- optional CV tailoring suggestions and adapted CV draft.

Avoid storing unnecessary raw personal data. If raw CV text is stored during MVP development, keep that decision explicit and easy to revisit.

## Testing Guidelines

Use mocked external services in tests.

Backend tests should cover:

- company name normalization;
- query generation;
- source scoring;
- evidence classification;
- report persistence;
- report schema validation;
- CV text parsing;
- candidate signal extraction;
- CV-to-company fit mapping;
- no-source cases;
- conflicting-source cases.

Frontend tests should cover:

- report rendering;
- empty states;
- loading states;
- source display;
- CV input states;
- personalized preparation rendering;
- responsive behavior.

Before reporting a task complete, run the relevant tests and at least a syntax/type/style check when available.

## Implementation Principles

- Make surgical changes.
- Do not add speculative abstractions.
- Keep provider-specific integrations isolated.
- Prefer simple, testable modules over complex orchestration.
- Do not implement features outside the PRD unless the user asks.
- If a requirement is ambiguous, state assumptions before coding.
- Every changed line should trace to the current task.

