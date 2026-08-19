# Company Research Agent

Company Research Agent is a FastAPI web app for researching companies and preparing for job applications or interviews. It is built for Spanish-first career preparation, especially for job seekers who want a clear company report backed by public evidence instead of generic AI advice.

The app can research a company, generate a structured report, cite sources, show uncertainty when evidence is missing, optionally use CV text for personalized preparation, and answer follow-up questions from saved reports.

## What It Does

- Generates Spanish company research reports.
- Collects public evidence through a configurable search pipeline.
- Extracts and classifies evidence before asking the LLM to synthesize the report.
- Uses DeepSeek V4 Flash for agent decisions and text generation, with Google Gemini retained only for embeddings.
- Stores reports, sources, evidence, claims, chat history, and optional CV-derived structured data in SQLite.
- Supports optional CV upload or pasted CV text for interview preparation and CV tailoring suggestions.
- Avoids returning or embedding raw CV text by default.
- Includes a simple frontend served by FastAPI.

## Project Structure

```text
backend/
  app/
    api/          FastAPI routes and response schemas
    core/         configuration and shared utilities
    db/           SQLite models, session setup, repositories
    domain/       domain objects for reports, companies, and CVs
    llm/          DeepSeek client, prompts, and synthesis logic
    research/     search, extraction, scoring, and evidence pipeline
    services/     report generation, RAG chat, CV extraction, indexing

frontend/
  index.html      main web UI
  static/         JavaScript and CSS

docs/             API, database, architecture, and report schema notes
tests/            pytest coverage for backend, frontend static checks, and flows
```

## How The Flow Works

1. The user submits a company name from the frontend.
2. The frontend calls `POST /api/research`.
3. FastAPI creates a company and report record in SQLite.
4. A background task runs the research pipeline.
5. The pipeline searches, extracts useful page content, classifies evidence, and scores sources.
6. DeepSeek receives a limited, evidence-backed context and returns structured JSON.
7. Pydantic validates the generated report.
8. The report is saved and can be read with `GET /api/reports/{report_id}`.
9. Chat endpoints answer follow-up questions using saved report evidence.

The conversational agent lets DeepSeek choose among bounded tools. The backend still executes tools and enforces evidence, citation, budget, validation, and persistence rules.

## Requirements

- Python 3.12 recommended
- A virtual environment
- Optional Tavily API key for real web search
- DeepSeek API key for agent decisions and text generation
- Google Gemini API key for semantic embeddings

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Configuration

Create a local `.env` file from the example:

```powershell
Copy-Item .env.example .env
```

Important settings:

```env
DATABASE_URL=sqlite:///./enterprise_research_agent.db
SEARCH_PROVIDER=mock
TAVILY_API_KEY=
DEEPSEEK_API_KEY=
DEEPSEEK_MODEL=deepseek-v4-flash
GEMINI_API_KEY=
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
```

Use `SEARCH_PROVIDER=mock` for local development without external search. Use `SEARCH_PROVIDER=tavily` when you want real company research through Tavily.

The local `.env`, `.venv`, and SQLite database are ignored by Git.

## Run Locally

Start the app:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

The API docs are available at:

```text
http://127.0.0.1:8000/docs
```

## Main API Endpoints

- `POST /api/research` starts a company research report.
- `GET /api/reports/{report_id}` returns report status or the completed report.
- `GET /api/reports` lists recent reports.
- `POST /api/cv/extract` extracts text from a PDF or DOCX CV upload.
- `POST /api/reports/{report_id}/chat` answers questions from one completed report.
- `POST /api/chat` answers broader questions using saved report embeddings.
- `POST /api/rag/reindex` backfills embeddings for completed reports.
- `DELETE /api/reports/{report_id}/cv-data` deletes CV-derived data for one report.
- `DELETE /api/reports/{report_id}` deletes a saved report.
- `DELETE /api/reports` deletes all saved reports.

## Example Research Request

```json
{
  "company_name": "Mercado Libre",
  "force_refresh": false,
  "cv_text": null,
  "include_cv_tailoring": false,
  "include_adapted_cv_draft": false
}
```

The response includes a `report_id` and a `status_url`. Poll the status URL until the report is completed or failed.

## Testing

Run the test suite with the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The suite covers the report flow, research pipeline, DeepSeek synthesis validation, Gemini embeddings, RAG behavior, CV preparation, and frontend static checks.

## Evidence And Privacy Safeguards

- Reports include sources, evidence, confidence, warnings, and missing-evidence signals.
- Factual claims are tied to evidence IDs.
- The backend validates generated reports before saving completed output.
- CV use is optional.
- Raw CV text is not exposed by default.
- CV-derived report data can be deleted separately from the company report.
- The adapted CV draft must not invent experience, dates, credentials, employers, metrics, or tools.

## Current Status

This is an MVP-style local web app with a FastAPI backend, SQLite persistence, a static frontend, tests, and documentation. It is ready for local development and experimentation with mock or real providers depending on environment configuration.
