from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import create_app


def test_conversational_frontend_index_is_served() -> None:
    client = TestClient(create_app())

    response = client.get("/")

    assert response.status_code == 200
    assert "Radar Laboral" in response.text
    assert "/static/app.js?v=48" in response.text
    assert "/static/styles.css?v=28" in response.text
    assert "viewport-fit=cover" in response.text

    # Two-region conversational shell and accessible landmarks.
    assert 'id="sidebar"' in response.text
    assert 'id="newChatButton"' in response.text
    assert 'id="conversationSearch"' in response.text
    assert 'id="conversationNav"' in response.text
    assert 'id="cvLibraryButton"' in response.text
    assert 'id="mainWorkspace"' in response.text
    assert 'id="conversationView"' in response.text
    assert 'id="conversationTranscript"' in response.text
    assert 'id="taskAnnouncements"' in response.text
    assert 'aria-busy="false"' in response.text
    assert 'aria-label="Conversaci&oacute;n"' in response.text
    assert "Agente DeepSeek" in response.text

    # Sticky composer and attachment controls.
    assert 'id="composerForm"' in response.text
    assert 'id="composerText"' in response.text
    assert "textarea" in response.text
    assert 'id="attachmentButton"' in response.text
    assert 'id="attachmentMenu"' in response.text
    assert 'id="attachmentChips"' in response.text
    assert 'data-attachment-action="stored-cv"' in response.text
    assert 'data-attachment-action="upload-cv"' in response.text
    assert 'data-attachment-action="job"' in response.text
    assert 'data-attachment-action="report"' in response.text
    assert 'title="Disponible en la pr&oacute;xima etapa"' not in response.text
    assert 'id="cvFileInput"' in response.text
    assert 'accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"' in response.text

    # Persistent CV management and native dialogs are present.
    assert 'id="cvWorkspace"' in response.text
    assert 'aria-labelledby="cvWorkspaceTitle"' in response.text
    assert 'id="jobDialog"' in response.text
    assert 'id="cvSelectDialog"' in response.text
    assert 'id="renameDialog"' in response.text
    assert 'id="confirmDialog"' in response.text

    # The former form/report/history layout is gone.
    for obsolete_hook in (
        'id="researchForm"',
        'id="companyName"',
        'id="includeTailoring"',
        'id="chatToggle"',
        'id="historyList"',
        'class="query-panel"',
        'class="history-panel"',
        'chat-collapsed',
    ):
        assert obsolete_hook not in response.text

    assert response.text.index('id="sidebar"') < response.text.index('id="mainWorkspace"')


def test_conversational_frontend_assets_are_served() -> None:
    client = TestClient(create_app())

    css_response = client.get("/static/styles.css")
    js_response = client.get("/static/app.js")

    assert css_response.status_code == 200
    assert js_response.status_code == 200

    css = css_response.text
    js = js_response.text

    # Full-height two-region layout, responsive drawer, and mobile comparison fallback.
    assert "--sidebar-width: 284px" in css
    assert "height: 100dvh" in css
    assert "grid-template-columns: var(--sidebar-width) minmax(0, 1fr)" in css
    assert ".conversation-transcript" in css
    assert "overflow-y: auto" in css
    assert ".composer-dock" in css
    assert "env(safe-area-inset-bottom)" in css
    assert "@media (max-width: 899px)" in css
    assert ".sidebar-open .sidebar" in css
    assert "@media (max-width: 639px)" in css
    assert ".comparison-table" in css
    assert ".comparison-mobile" in css
    assert ".cv-workspace" in css

    # Safety, accessibility, and long-content resilience.
    assert "overflow-wrap: anywhere" in css
    assert ":focus-visible" in css
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert "@media (pointer: coarse)" in css
    assert "linear-gradient" not in css
    assert ".conversation-nav::-webkit-scrollbar" in css
    assert ".conversation-transcript::-webkit-scrollbar" in css
    assert "scrollbar-width: thin" in css
    assert ".chat-composer textarea:focus-visible" in css

    # Sidebar rows are uncluttered and CV versions keep only useful actions.
    assert "conversation-origin" not in js
    assert '>Ver</button>' not in js

    # Conversations use REST/SSE; fixture data and browser persistence are absent.
    assert "seedConversations" not in js
    assert "seedCVs" not in js
    assert "dataSource" in js
    assert "listConversations" in js
    assert '"/api/conversations"' in js
    assert "/messages`" in js
    assert "/api/task-runs/" in js
    assert "new EventSource" in js
    assert "pollActiveTaskState" in js
    assert "events?after_event_id=${cursor}" in js
    assert "eventCursors" in js
    assert "eventStreamUrl" in js
    assert "aria-busy" in js
    assert "dialogReturnFocus" in js
    assert "restoreFocus: true" in js
    assert 'document.querySelector("dialog[open]")' in js
    assert "resumeTask" in js
    assert "renderPauseRequest" in js
    assert "renderComparisonArtifact" in js
    assert "retryFailedTask" in js
    assert "retryTask(taskId)" in js
    assert "data-retry-task" in js
    assert "syncRetryCountdown" in js
    assert "data-retry-at" in js
    assert "clarification.required" in js
    assert "approval.required" in js
    assert "artifact.created" in js
    assert "await refreshActiveConversation();" in js
    assert "Last-Event-ID" not in js
    assert "listLegacyReports" in js
    assert '"/api/reports?limit=20"' in js
    assert "getLegacyReport" in js
    assert "/api/research" not in js
    assert "/api/chat" not in js
    assert "/api/cv/extract" not in js
    assert "localStorage" not in js
    assert "URL.createObjectURL" not in js
    assert "URL.revokeObjectURL" not in js

    # Persistent chat, progress, legacy reports, and CV management have render paths.
    for renderer in (
        "renderLiveProgress",
        "renderTaskState",
        "renderLegacyReportArtifact",
        "renderCVWorkspaceV3",
        "renderCVRecommendationArtifact",
    ):
        assert renderer in js

    assert "requestSubmit" in js
    assert "showConfirmation" in js
    assert "showRenameDialog" in js
    assert "safeUrl" in js
    assert "escapeHtml" in js
    assert "Respuesta local" not in js
    assert "Generación con DeepSeek" in js
    assert "Disponible en la próxima etapa" not in js
    assert '"/api/cvs"' in js
    assert '"review.required"' in js
    assert "save_as_cv_version" in js
    assert "Guardado" in js
