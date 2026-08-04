from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import create_app


def test_frontend_index_is_served() -> None:
    client = TestClient(create_app())

    response = client.get("/")

    assert response.status_code == 200
    assert "Radar Laboral" in response.text
    assert "/static/app.js?v=14" in response.text
    assert "/static/styles.css?v=14" in response.text
    assert 'id="cvFile"' in response.text
    assert 'class="upload-control"' in response.text
    assert 'id="chatForm"' in response.text
    assert 'class="chat-composer"' in response.text
    assert 'id="chatToggle"' in response.text
    assert 'class="chat-panel chat-collapsed"' in response.text
    assert 'id="chatInput" type="text"' in response.text
    assert 'id="clearHistory"' in response.text
    assert "Borrar historial" in response.text
    assert "Informe en espa&ntilde;ol con fuentes" not in response.text
    assert ">Pregunta<" not in response.text
    assert response.text.index('class="query-panel"') < response.text.index('class="report-panel"')
    assert response.text.index('id="reportView"') < response.text.index('id="chatPanel"')
    assert response.text.index('id="chatPanel"') < response.text.index('class="history-panel"')


def test_frontend_assets_are_served() -> None:
    client = TestClient(create_app())

    css_response = client.get("/static/styles.css")
    js_response = client.get("/static/app.js")

    assert css_response.status_code == 200
    assert "grid-template-columns" in css_response.text
    assert "minmax(440px, 1.16fr)" in css_response.text
    assert ".report-panel" in css_response.text
    assert "grid-template-rows: minmax(0, 1fr) auto" in css_response.text
    assert "height: calc(100vh - 116px)" in css_response.text
    assert ".report-view" in css_response.text
    assert "overflow-y: auto" in css_response.text
    assert ".history-list" in css_response.text
    assert "grid-template-rows: auto minmax(0, 1fr)" in css_response.text
    assert "scrollbar-width: thin" in css_response.text
    assert "--scroll-thumb:" in css_response.text
    assert ".chat-composer input" in css_response.text
    assert ".chat-panel.chat-collapsed .chat-messages" in css_response.text
    assert ".chat-panel {\n  align-self: start;\n  background: transparent;" in css_response.text
    assert "border-top: 1px solid var(--line)" in css_response.text
    assert "max-height: 180px" in css_response.text
    assert "min-height: 72px" in css_response.text
    assert js_response.status_code == 200
    assert "fetchJson" in js_response.text
    assert "buildCitationIndex" in js_response.text
    assert "renderSectionCitations" in js_response.text
    assert "/api/cv/extract" in js_response.text
    assert "data-delete-report-id" in js_response.text
    assert "aria-label=\"Eliminar informe" in js_response.text
    assert r"\u00bfEliminar todo el historial? Esta acci\u00f3n no se puede deshacer." in js_response.text
    assert 'fetchJson("/api/reports", { method: "DELETE" })' in js_response.text
    assert 'fetchJson("/api/chat"' in js_response.text
    assert "active_report_id" in js_response.text
    assert "scopeLabelText" in js_response.text
    assert "updateChatState" in js_response.text
    assert "renderRagStatusChip" in js_response.text
    assert "active_report_pending_index" in js_response.text
    assert "Indexando chat" in js_response.text
    assert "Chat listo" in js_response.text
    assert ".meta-chip.rag-indexing" in css_response.text
    assert "renderClaimMeta" not in js_response.text
    assert "Evidencia:" not in js_response.text
