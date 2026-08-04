const state = {
  currentReportId: null,
  currentReportReady: false,
  chatExpanded: false,
  pollTimer: null,
  indexPollTimer: null,
};

const els = {
  form: document.querySelector("#researchForm"),
  companyName: document.querySelector("#companyName"),
  cvFile: document.querySelector("#cvFile"),
  cvFileName: document.querySelector("#cvFileName"),
  cvFileStatus: document.querySelector("#cvFileStatus"),
  cvText: document.querySelector("#cvText"),
  chatForm: document.querySelector("#chatForm"),
  chatPanel: document.querySelector("#chatPanel"),
  chatToggle: document.querySelector("#chatToggle"),
  chatInput: document.querySelector("#chatInput"),
  chatMessages: document.querySelector("#chatMessages"),
  chatHint: document.querySelector("#chatHint"),
  chatSubmit: document.querySelector("#chatSubmit"),
  includeTailoring: document.querySelector("#includeTailoring"),
  includeDraft: document.querySelector("#includeDraft"),
  clearButton: document.querySelector("#clearButton"),
  refreshHistory: document.querySelector("#refreshHistory"),
  clearHistory: document.querySelector("#clearHistory"),
  connectionStatus: document.querySelector("#connectionStatus"),
  emptyState: document.querySelector("#emptyState"),
  loadingState: document.querySelector("#loadingState"),
  loadingMessage: document.querySelector("#loadingMessage"),
  errorState: document.querySelector("#errorState"),
  errorMessage: document.querySelector("#errorMessage"),
  reportView: document.querySelector("#reportView"),
  historyList: document.querySelector("#historyList"),
};

els.form.addEventListener("submit", handleSubmit);
els.cvFile.addEventListener("change", handleCvUpload);
els.chatForm.addEventListener("submit", handleChatSubmit);
els.chatToggle.addEventListener("click", () => setChatExpanded(!state.chatExpanded));
els.clearButton.addEventListener("click", clearForm);
els.refreshHistory.addEventListener("click", loadHistory);
els.clearHistory.addEventListener("click", deleteAllHistory);
els.includeDraft.addEventListener("change", () => {
  if (els.includeDraft.checked) {
    els.includeTailoring.checked = true;
  }
});
els.includeTailoring.addEventListener("change", () => {
  if (!els.includeTailoring.checked) {
    els.includeDraft.checked = false;
  }
});

loadHistory();
updateChatState();

async function handleSubmit(event) {
  event.preventDefault();
  const cvText = els.cvText.value.trim();
  const payload = {
    company_name: els.companyName.value.trim(),
    cv_text: cvText || null,
    include_cv_tailoring: els.includeTailoring.checked,
    include_adapted_cv_draft: els.includeDraft.checked,
  };

  if (!payload.company_name) {
    showError("Ingres\u00e1 el nombre de la empresa.");
    return;
  }
  if (payload.include_cv_tailoring && !payload.cv_text) {
    showError("Peg\u00e1 tu CV para activar la adaptaci\u00f3n.");
    return;
  }

  setBusy(true);
  showLoading("La investigaci\u00f3n est\u00e1 en cola.");
  try {
    const response = await fetchJson("/api/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    els.chatMessages.innerHTML = "";
    state.currentReportId = response.report_id;
    state.currentReportReady = false;
    updateChatState();
    pollReport(response.status_url);
    loadHistory();
  } catch (error) {
    showError(error.message);
    setBusy(false);
  }
}

async function handleCvUpload() {
  const file = els.cvFile.files?.[0];
  if (!file) {
    return;
  }
  const formData = new FormData();
  formData.append("file", file);
  els.cvFileName.textContent = file.name;
  els.cvFileStatus.textContent = "Extrayendo texto...";
  try {
    const payload = await fetchJson("/api/cv/extract", {
      method: "POST",
      body: formData,
    });
    els.cvText.value = payload.cv_text || "";
    els.cvFileName.textContent = payload.filename;
    els.cvFileStatus.textContent = `Texto listo para revisar (${payload.character_count} caracteres).`;
  } catch (error) {
    els.cvFile.value = "";
    els.cvFileName.textContent = "No se pudo leer el archivo";
    els.cvFileStatus.textContent = "Prob\u00e1 con un PDF o DOCX legible.";
    showError(error.message);
  }
}

async function pollReport(statusUrl) {
  clearPoll();
  const tick = async () => {
    try {
      const payload = await fetchJson(statusUrl);
      if (payload.report) {
        clearPoll();
        setBusy(false);
        renderReport(payload.report);
        loadHistory();
        return;
      }
      if (payload.status === "failed") {
        clearPoll();
        setBusy(false);
        showError(payload.error?.message || "No se pudo generar el informe.");
        return;
      }
      showLoading(payload.progress?.message || "La investigaci\u00f3n sigue en curso.");
    } catch (error) {
      clearPoll();
      setBusy(false);
      showError(error.message);
    }
  };
  await tick();
  state.pollTimer = window.setInterval(tick, 1800);
}

async function loadHistory() {
  try {
    const payload = await fetchJson("/api/reports?limit=20");
    renderHistory(payload.items || []);
  } catch {
    els.clearHistory.disabled = true;
    els.historyList.innerHTML = `<p>No se pudo cargar el historial.</p>`;
  }
}

function renderReport(report) {
  if (state.currentReportId !== report.report_id) {
    els.chatMessages.innerHTML = "";
  }
  state.currentReportId = report.report_id;
  state.currentReportReady = report.status === "completed";
  updateChatState(report.company.name);
  hideAllStates();
  els.connectionStatus.textContent = "Informe listo";
  els.reportView.classList.remove("hidden");
  els.reportView.innerHTML = `
    <header class="report-header">
      <p class="eyebrow">Informe generado</p>
      <h2>${escapeHtml(report.company.name)}</h2>
      <div class="report-meta">
        ${chip(`Fuentes: ${report.metadata.source_count}`)}
        ${chip(`Hallazgos: ${report.metadata.evidence_count}`)}
        ${chip(report.metadata.used_cv ? "Con CV" : "Sin CV")}
        ${chip(report.metadata.search_provider)}
        ${renderRagStatusChip(report.metadata)}
      </div>
      ${
        report.metadata.used_cv
          ? `<button class="danger-button" type="button" id="deleteCvData">Eliminar datos del CV</button>`
          : ""
      }
    </header>

    ${renderSections(report)}
    ${renderPreparation(report.personalized_preparation)}
    ${renderTailoring(report.cv_tailoring)}
    ${renderSources(report.sources || [])}
    ${renderWarnings(report.warnings || [])}
  `;

  const deleteButton = document.querySelector("#deleteCvData");
  if (deleteButton) {
    deleteButton.addEventListener("click", () => deleteCvData(report.report_id));
  }
  watchRagIndexStatus(report);
}

function renderSections(report) {
  const sections = report.sections || [];
  if (!sections.length) {
    return "";
  }
  const citationIndex = buildCitationIndex(report);
  return `
    <section class="section-grid">
      ${sections.map((section) => renderSection(section, citationIndex)).join("")}
    </section>
  `;
}

function renderSection(section, citationIndex) {
  return `
    <div class="report-card">
      <h3>${escapeHtml(section.title)}</h3>
      <p>${escapeHtml(section.summary)}</p>
      ${renderSectionCitations(section, citationIndex)}
    </div>
  `;
}

function renderSectionCitations(section, citationIndex) {
  const citations = collectSectionCitations(section, citationIndex);
  if (!citations.length) {
    return "";
  }

  return `
    <div class="section-citations">
      <small>Fuentes relacionadas</small>
      <div class="citation-list" aria-label="Fuentes de la secci\u00f3n">
        ${citations.map((citation) => `
          <a href="${escapeAttribute(citation.url)}" target="_blank" rel="noreferrer">
            ${escapeHtml(citation.title)}
          </a>
        `).join("")}
      </div>
    </div>
  `;
}

function collectSectionCitations(section, citationIndex) {
  const citationsByKey = new Map();
  (section.claims || []).forEach((claim) => {
    (claim.evidence_ids || []).forEach((evidenceId) => {
      const citation = citationIndex.get(evidenceId);
      if (!citation) {
        return;
      }
      citationsByKey.set(`${citation.title}|${citation.url}`, citation);
    });
  });
  return Array.from(citationsByKey.values());
}

function renderPreparation(preparation) {
  if (!preparation) {
    return "";
  }
  return `
    <section class="report-card cv-block">
      <h3>Preparaci\u00f3n personalizada</h3>
      ${preparation.suggested_pitch ? `<p>${escapeHtml(preparation.suggested_pitch.text)}</p>` : ""}
      ${renderItems("Fortalezas", preparation.strengths_to_highlight)}
      ${renderItems("Brechas a preparar", preparation.gaps_to_prepare)}
      ${renderItems("Preguntas posibles", preparation.personalized_questions)}
      ${renderItems("Preguntas para la empresa", preparation.questions_for_company)}
    </section>
  `;
}

function renderTailoring(tailoring) {
  if (!tailoring) {
    return "";
  }
  const suggestions = tailoring.change_suggestions?.map((item) => ({
    text: item.suggested_text,
    reason: item.requires_user_confirmation ? "Requiere confirmaci\u00f3n" : item.reason,
  }));
  return `
    <section class="report-card cv-block">
      <h3>Adaptaci\u00f3n del CV</h3>
      <p>${escapeHtml(tailoring.positioning_summary)}</p>
      ${renderItems("Cambios sugeridos", suggestions)}
      ${
        tailoring.adapted_cv_draft
          ? `<pre class="draft">${escapeHtml(tailoring.adapted_cv_draft.content_markdown)}</pre>`
          : ""
      }
    </section>
  `;
}

function renderSources(sources) {
  if (!sources.length) {
    return "";
  }
  return `
    <section class="report-card">
      <h3>Fuentes</h3>
      <ul class="source-list">
        ${sources.map((source) => `
          <li>
            <a href="${escapeAttribute(source.url)}" target="_blank" rel="noreferrer">${escapeHtml(source.title)}</a>
            <span>${escapeHtml(source.domain)} &middot; ${sourceTypeLabel(source.source_type)} &middot; confianza ${source.reliability_score}/5</span>
            ${source.accessed_at ? `<small>Consultada/generada: ${formatDate(source.accessed_at)}</small>` : ""}
          </li>
        `).join("")}
      </ul>
    </section>
  `;
}

function renderWarnings(warnings) {
  if (!warnings.length) {
    return "";
  }
  return `
    <section class="report-card">
      <h3>Advertencias</h3>
      <ul class="claim-list">
        ${warnings.map((warning) => `<li class="warning">${escapeHtml(warning.message)}</li>`).join("")}
      </ul>
    </section>
  `;
}

function renderItems(title, items = []) {
  if (!items.length) {
    return "";
  }
  return `
    <h3>${escapeHtml(title)}</h3>
    <ul class="claim-list">
      ${items.map((item) => `
        <li>
          ${escapeHtml(item.text)}
          ${item.reason ? `<small>${escapeHtml(item.reason)}</small>` : ""}
        </li>
      `).join("")}
    </ul>
  `;
}

function renderHistory(items) {
  els.clearHistory.disabled = !items.length;
  if (!items.length) {
    els.historyList.innerHTML = `<p>Todav\u00eda no hay informes guardados.</p>`;
    return;
  }
  els.historyList.innerHTML = items.map((item) => `
    <article class="history-item">
      <button class="history-open" type="button" data-report-id="${escapeAttribute(item.report_id)}">
        <strong>${escapeHtml(item.company.name)}</strong>
        <span>${escapeHtml(item.status)}${item.used_cv ? " &middot; con CV" : ""}</span>
        ${item.summary ? `<span>${escapeHtml(item.summary)}</span>` : ""}
      </button>
      <button class="history-delete" type="button" data-delete-report-id="${escapeAttribute(item.report_id)}" aria-label="Eliminar informe de ${escapeAttribute(item.company.name)}" title="Eliminar informe">&times;</button>
    </article>
  `).join("");
  document.querySelectorAll("[data-report-id]").forEach((button) => {
    button.addEventListener("click", () => pollReport(`/api/reports/${button.dataset.reportId}`));
  });
  document.querySelectorAll("[data-delete-report-id]").forEach((button) => {
    button.addEventListener("click", () => deleteReport(button.dataset.deleteReportId));
  });
}

async function handleChatSubmit(event) {
  event.preventDefault();
  const message = els.chatInput.value.trim();
  if (!message) {
    return;
  }
  setChatExpanded(true);
  appendChatMessage("user", message);
  els.chatInput.value = "";

  try {
    const payload = await fetchJson("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        active_report_id: state.currentReportReady ? state.currentReportId : null,
      }),
    });
    appendChatMessage("assistant", payload.answer, payload.citations || [], payload.scope_used);
  } catch (error) {
    appendChatMessage("assistant", error.message);
  }
}

function appendChatMessage(role, text, citations = [], scope = null) {
  els.chatMessages.querySelector(".chat-empty")?.remove();
  const label = role === "user" ? "Vos" : "Radar Laboral";
  const scopeLabel = role === "assistant" && scope ? `<small>${scopeLabelText(scope)}</small>` : "";
  const citationLinks = citations.length
    ? `<div class="chat-citations">${citations.map((citation) => `
        <a href="${escapeAttribute(citation.url || "#")}" target="_blank" rel="noreferrer">
          ${escapeHtml(citation.company_name ? `${citation.company_name}: ${citation.title}` : citation.title)}
        </a>
      `).join("")}</div>`
    : "";
  els.chatMessages.insertAdjacentHTML(
    "beforeend",
    `<div class="chat-message ${role}">
      <strong>${label}</strong>
      <p>${escapeHtml(text)}</p>
      ${scopeLabel}
      ${citationLinks}
    </div>`
  );
  els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

async function deleteCvData(reportId) {
  try {
    await fetchJson(`/api/reports/${reportId}/cv-data`, { method: "DELETE" });
    const payload = await fetchJson(`/api/reports/${reportId}`);
    if (payload.report) {
      renderReport(payload.report);
    }
    loadHistory();
  } catch (error) {
    showError(error.message);
  }
}

async function deleteReport(reportId) {
  if (!window.confirm("Eliminar este informe del historial?")) {
    return;
  }
  try {
    await fetchJson(`/api/reports/${reportId}`, { method: "DELETE" });
    if (state.currentReportId === reportId) {
      clearCurrentReport();
    }
    loadHistory();
  } catch (error) {
    showError(error.message);
  }
}

async function deleteAllHistory() {
  if (!window.confirm("\u00bfEliminar todo el historial? Esta acci\u00f3n no se puede deshacer.")) {
    return;
  }
  try {
    await fetchJson("/api/reports", { method: "DELETE" });
    clearCurrentReport();
    loadHistory();
  } catch (error) {
    showError(error.message);
  }
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = payload.error?.message || payload.detail?.[0]?.msg || payload.detail || "Error de solicitud.";
    throw new Error(Array.isArray(message) ? message.join(", ") : message);
  }
  return payload;
}

function clearForm() {
  els.form.reset();
  els.cvFileName.textContent = "PDF o DOCX";
  els.cvFileStatus.textContent = "Ning\u00fan archivo seleccionado.";
  clearPoll();
  setBusy(false);
  clearCurrentReport();
}

function clearCurrentReport() {
  clearIndexPoll();
  state.currentReportId = null;
  state.currentReportReady = false;
  els.chatMessages.innerHTML = "";
  updateChatState();
  hideAllStates();
  els.emptyState.classList.remove("hidden");
  els.connectionStatus.textContent = "Listo";
}

function updateChatState(companyName = null) {
  const isReady = Boolean(state.currentReportId && state.currentReportReady);
  els.chatHint.textContent = isReady
    ? `Pregunt\u00e1 sobre ${companyName || "este informe"} o compar\u00e1 con reportes guardados.`
    : "Pregunt\u00e1 sobre los reportes guardados. Si no hay datos, te lo voy a decir.";
  renderChatEmptyState(isReady);
}

function renderChatEmptyState(isReady) {
  const hasConversation = Array.from(els.chatMessages.children).some(
    (child) => !child.classList.contains("chat-empty")
  );
  if (hasConversation) {
    return;
  }
  els.chatMessages.innerHTML = `
    <div class="chat-empty">
      ${isReady ? "Hac\u00e9 una pregunta sobre este informe o comparalo con otros." : "Abr\u00ed el chat para preguntar sobre reportes guardados."}
    </div>
  `;
}

function setChatExpanded(isExpanded) {
  state.chatExpanded = isExpanded;
  els.chatPanel.classList.toggle("chat-collapsed", !isExpanded);
  els.chatToggle.setAttribute("aria-expanded", String(isExpanded));
  els.chatToggle.textContent = isExpanded ? "Cerrar" : "Abrir";
  if (isExpanded) {
    els.chatInput.focus();
  }
}

function scopeLabelText(scope) {
  const labels = {
    active_report: "Usando informe actual",
    active_report_pending_index: "Usando informe actual mientras se indexa",
    all_reports: "Comparando reportes guardados",
  };
  return labels[scope] || "Usando reportes guardados";
}

function showLoading(message) {
  hideAllStates();
  els.loadingMessage.textContent = message;
  els.loadingState.classList.remove("hidden");
  els.connectionStatus.textContent = "Investigando";
}

function showError(message) {
  hideAllStates();
  els.errorMessage.textContent = message;
  els.errorState.classList.remove("hidden");
  els.connectionStatus.textContent = "Revisar";
}

function hideAllStates() {
  els.emptyState.classList.add("hidden");
  els.loadingState.classList.add("hidden");
  els.errorState.classList.add("hidden");
  els.reportView.classList.add("hidden");
}

function setBusy(isBusy) {
  els.form.querySelector(".primary-button").disabled = isBusy;
}

function clearPoll() {
  if (state.pollTimer) {
    window.clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

function clearIndexPoll() {
  if (state.indexPollTimer) {
    window.clearInterval(state.indexPollTimer);
    state.indexPollTimer = null;
  }
}

function watchRagIndexStatus(report) {
  clearIndexPoll();
  if (!["pending", "indexing"].includes(report.metadata?.rag_index_status || "pending")) {
    return;
  }
  state.indexPollTimer = window.setInterval(async () => {
    try {
      const payload = await fetchJson(`/api/reports/${report.report_id}`);
      if (payload.report && state.currentReportId === report.report_id) {
        renderReport(payload.report);
      }
    } catch {
      clearIndexPoll();
    }
  }, 5000);
}

function chip(text) {
  return `<span class="meta-chip">${escapeHtml(text)}</span>`;
}

function renderRagStatusChip(metadata = {}) {
  const status = metadata.rag_index_status || "pending";
  const count = Number(metadata.rag_indexed_chunk_count || 0);
  const labels = {
    ready: `Chat listo${count ? ` · ${count}` : ""}`,
    indexing: "Indexando chat",
    pending: "Chat parcial",
    failed: "Chat parcial",
  };
  const title = status === "failed" && metadata.rag_index_error
    ? metadata.rag_index_error
    : "Estado de indexacion para preguntas con RAG.";
  return `<span class="meta-chip rag-${escapeAttribute(status)}" title="${escapeAttribute(title)}">${escapeHtml(labels[status] || "Chat parcial")}</span>`;
}

function buildCitationIndex(report) {
  const sourcesById = new Map((report.sources || []).map((source) => [source.id, source]));
  const index = new Map();
  (report.evidence || []).forEach((evidence) => {
    const source = sourcesById.get(evidence.source_id);
    if (!source) {
      return;
    }
    index.set(evidence.id, {
      title: source.title || source.domain || source.url,
      url: source.url,
    });
  });
  return index;
}

function sourceTypeLabel(value) {
  const labels = {
    official: "oficial",
    career_page: "empleos",
    linkedin: "LinkedIn",
    job_board: "portal laboral",
    salary_review_platform: "sueldos/reviews",
    news_media: "medio",
    company_database: "base de empresas",
    secondary: "secundaria",
    unknown: "sin clasificar",
  };
  return labels[value] || "sin clasificar";
}

function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return escapeHtml(value);
  }
  return new Intl.DateTimeFormat("es-AR", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttribute(value) {
  return escapeHtml(value).replaceAll("`", "&#096;");
}
