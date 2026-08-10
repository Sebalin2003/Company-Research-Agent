const state = {
  currentReportId: null,
  currentReportReady: false,
  chatExpanded: false,
  currentDraft: "",
  lastResearchPayload: null,
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
  jobDescription: document.querySelector("#jobDescription"),
  jobDescriptionCount: document.querySelector("#jobDescriptionCount"),
  cvDisclosure: document.querySelector("#cvDisclosure"),
  cvTextDisclosure: document.querySelector("#cvTextDisclosure"),
  chatForm: document.querySelector("#chatForm"),
  chatPanel: document.querySelector("#chatPanel"),
  chatToggle: document.querySelector("#chatToggle"),
  chatInput: document.querySelector("#chatInput"),
  chatMessages: document.querySelector("#chatMessages"),
  chatHint: document.querySelector("#chatHint"),
  chatSubmit: document.querySelector("#chatSubmit"),
  includeTailoring: document.querySelector("#includeTailoring"),
  includeDraft: document.querySelector("#includeDraft"),
  formError: document.querySelector("#formError"),
  clearButton: document.querySelector("#clearButton"),
  retryButton: document.querySelector("#retryButton"),
  refreshHistory: document.querySelector("#refreshHistory"),
  clearHistory: document.querySelector("#clearHistory"),
  connectionStatus: document.querySelector("#connectionStatus"),
  emptyState: document.querySelector("#emptyState"),
  loadingState: document.querySelector("#loadingState"),
  loadingCompany: document.querySelector("#loadingCompany"),
  loadingMessage: document.querySelector("#loadingMessage"),
  errorState: document.querySelector("#errorState"),
  errorMessage: document.querySelector("#errorMessage"),
  reportView: document.querySelector("#reportView"),
  historyList: document.querySelector("#historyList"),
};

els.form.addEventListener("submit", handleSubmit);
els.cvFile.addEventListener("change", handleCvUpload);
els.cvText.addEventListener("input", updateCvControls);
els.jobDescription.addEventListener("input", updateJobDescriptionCount);
els.chatForm.addEventListener("submit", handleChatSubmit);
els.chatToggle.addEventListener("click", () => setChatExpanded(!state.chatExpanded));
els.clearButton.addEventListener("click", clearForm);
els.retryButton.addEventListener("click", retryResearch);
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
  updateCvControls();
});

loadHistory();
updateChatState();
updateCvControls();
updateJobDescriptionCount();

async function handleSubmit(event) {
  event.preventDefault();
  const cvText = els.cvText.value.trim();
  const jobDescription = els.jobDescription.value.trim();
  const payload = {
    company_name: els.companyName.value.trim(),
    cv_text: cvText || null,
    job_description: jobDescription || null,
    include_cv_tailoring: els.includeTailoring.checked,
    include_adapted_cv_draft: els.includeDraft.checked,
  };

  if (!payload.company_name) {
    showFormError("Ingres\u00e1 el nombre de la empresa.");
    els.companyName.focus();
    return;
  }
  if (payload.include_cv_tailoring && !payload.cv_text) {
    showFormError("Sub\u00ed o peg\u00e1 tu CV para activar las sugerencias.");
    els.cvDisclosure.open = true;
    return;
  }
  if (payload.job_description && !payload.cv_text) {
    showFormError("Agreg\u00e1 tu CV para comparar el puesto.");
    els.cvDisclosure.open = true;
    els.cvTextDisclosure.open = true;
    els.cvText.focus();
    return;
  }

  hideFormError();
  state.lastResearchPayload = payload;
  setBusy(true);
  showLoading("La investigaci\u00f3n est\u00e1 en cola.", payload.company_name);
  focusResultOnNarrowScreen();
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
    els.cvTextDisclosure.open = false;
    els.cvFileName.textContent = payload.filename;
    els.cvFileStatus.textContent = `Texto listo para revisar (${payload.character_count} caracteres).`;
    updateCvControls();
  } catch (error) {
    els.cvFile.value = "";
    els.cvFileName.textContent = "No se pudo leer el archivo";
    els.cvFileStatus.textContent = "Prob\u00e1 con un PDF o DOCX legible.";
    showFormError(error.message);
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
        loadHistory();
        return;
      }
      showLoading(
        payload.progress?.message || "La investigaci\u00f3n sigue en curso.",
        payload.company?.name || state.lastResearchPayload?.company_name
      );
    } catch (error) {
      clearPoll();
      setBusy(false);
      showError(error.message);
    }
  };
  await tick();
  state.pollTimer = window.setInterval(tick, 1800);
}

async function retryResearch() {
  if (!state.lastResearchPayload) {
    els.companyName.focus();
    return;
  }
  setBusy(true);
  showLoading("Volviendo a iniciar la investigaci\u00f3n.", state.lastResearchPayload.company_name);
  try {
    const response = await fetchJson("/api/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state.lastResearchPayload),
    });
    state.currentReportId = response.report_id;
    state.currentReportReady = false;
    pollReport(response.status_url);
  } catch (error) {
    setBusy(false);
    showError(error.message);
  }
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
  state.currentDraft = report.cv_tailoring?.adapted_cv_draft?.content_markdown || "";
  updateChatState(report.company.name);
  hideAllStates();
  els.connectionStatus.textContent = "Informe listo";
  els.reportView.classList.remove("hidden");
  const incompleteSections = (report.sections || []).filter((section) => section.missing_evidence).length;
  els.reportView.innerHTML = `
    <header class="report-header">
      <div class="report-title-row">
        <div>
          <span class="context-label">Informe generado</span>
          <h2 id="reportTitle">${escapeHtml(report.company.name)}</h2>
        </div>
        ${report.generated_at ? `<time datetime="${escapeAttribute(report.generated_at)}">${formatDate(report.generated_at)}</time>` : ""}
      </div>
      <div class="report-meta">
        ${chip(`Fuentes: ${report.metadata.source_count}`)}
        ${chip(`Hallazgos: ${report.metadata.evidence_count}`)}
        ${chip(report.metadata.used_cv ? "Con CV" : "Sin CV")}
        ${report.metadata.used_job_description ? chip("Con puesto", "cv-context") : ""}
        ${incompleteSections ? chip(`${incompleteSections} secciones con evidencia limitada`, "warning") : chip("Evidencia suficiente", "success")}
        ${renderRagStatusChip(report.metadata)}
      </div>
      ${
        report.metadata.used_cv
          ? `<button class="danger-button" type="button" id="deleteCvData">Eliminar datos del CV</button>`
          : ""
      }
      <details class="technical-details">
        <summary>Detalles t&eacute;cnicos</summary>
        <p>Proveedor de b&uacute;squeda: ${escapeHtml(report.metadata.search_provider)} &middot; Modelo: ${escapeHtml(report.metadata.llm_model)}</p>
      </details>
    </header>

    ${renderReportNavigation(report)}
    <div class="report-content">
      <section class="report-band" id="empresa" aria-labelledby="companySectionTitle">
        <div class="band-heading">
          <span class="context-label">Empresa</span>
          <h2 id="companySectionTitle">Qu&eacute; sabemos</h2>
        </div>
        ${renderSections(report)}
      </section>
      ${renderPreparation(report.personalized_preparation)}
      ${renderTailoring(report.cv_tailoring)}
      <section class="report-band" id="fuentes" aria-labelledby="sourcesTitle">
        <div class="band-heading">
          <span class="context-label">Trazabilidad</span>
          <h2 id="sourcesTitle">Fuentes y advertencias</h2>
        </div>
        ${renderSources(report.sources || [])}
        ${renderWarnings(report.warnings || [], { globalOnly: true })}
      </section>
    </div>
  `;

  const deleteButton = document.querySelector("#deleteCvData");
  if (deleteButton) {
    deleteButton.addEventListener("click", () => deleteCvData(report.report_id));
  }
  document.querySelector("#copyDraft")?.addEventListener("click", copyAdaptedDraft);
  els.reportView.focus({ preventScroll: window.innerWidth > 760 });
  if (window.innerWidth <= 760) {
    els.reportView.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" });
  }
  watchRagIndexStatus(report);
}

function renderReportNavigation(report) {
  const links = [
    ["empresa", "Empresa"],
    ...(report.personalized_preparation ? [["preparacion", "Preparaci\u00f3n"]] : []),
    ...(report.cv_tailoring ? [["cv", "CV"]] : []),
    ["fuentes", "Fuentes"],
  ];
  return `
    <nav class="report-nav" aria-label="Secciones del informe">
      ${links.map(([id, label]) => `<a href="#${id}">${label}</a>`).join("")}
    </nav>
  `;
}

function renderSections(report) {
  const sections = report.sections || [];
  if (!sections.length) {
    return "";
  }
  const citationIndex = buildCitationIndex(report);
  return `
    <div class="section-grid">
      ${sections.map((section) => renderSection(section, citationIndex, report.warnings || [])).join("")}
    </div>
  `;
}

function renderSection(section, citationIndex, warnings) {
  const sectionWarnings = warnings.filter((warning) => warning.related_section === section.type);
  return `
    <section class="evidence-section" id="report-${escapeAttribute(section.type)}">
      <div class="evidence-heading">
        <h3>${escapeHtml(section.title)}</h3>
        <div class="evidence-status">
          ${confidenceBadge(section.confidence)}
          ${section.missing_evidence ? `<span class="status-badge warning">Evidencia limitada</span>` : ""}
        </div>
      </div>
      <p>${escapeHtml(section.summary)}</p>
      ${renderSectionCitations(section, citationIndex)}
      ${renderWarnings(sectionWarnings)}
    </section>
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
    <section class="report-band preparation-band" id="preparacion" aria-labelledby="preparationTitle">
      <div class="band-heading">
        <span class="context-label">Preparaci&oacute;n</span>
        <h2 id="preparationTitle">Tu perfil frente a la empresa</h2>
      </div>
      ${
        preparation.fit_summary
          ? `<div class="fit-summary">
              <div>${confidenceBadge(preparation.fit_summary.confidence)}<strong>Lectura de encaje</strong></div>
              <p>${escapeHtml(preparation.fit_summary.summary)}</p>
            </div>`
          : ""
      }
      ${
        preparation.suggested_pitch
          ? `<div class="pitch-block">
              <h3>Presentaci&oacute;n sugerida</h3>
              <p>${escapeHtml(preparation.suggested_pitch.text)}</p>
            </div>`
          : ""
      }
      <div class="preparation-groups">
        ${renderItems("Fortalezas para destacar", preparation.strengths_to_highlight)}
        ${renderItems("Brechas para preparar", preparation.gaps_to_prepare)}
        ${renderItems("Preguntas que podr&iacute;an hacerte", preparation.personalized_questions)}
        ${renderItems("Preguntas para la empresa", preparation.questions_for_company)}
      </div>
      ${renderWarnings(preparation.warnings || [])}
    </section>
  `;
}

function renderTailoring(tailoring) {
  if (!tailoring) {
    return "";
  }
  return `
    <section class="report-band cv-band" id="cv" aria-labelledby="cvTitle">
      <div class="band-heading cv-heading">
        <div>
          <span class="context-label">Sugerencias asistidas por IA</span>
          <h2 id="cvTitle">Adaptaci&oacute;n del CV</h2>
        </div>
        <span class="review-label">Revisar antes de usar</span>
      </div>
      <p class="section-lead">${escapeHtml(tailoring.positioning_summary)}</p>
      ${renderTailoringSuggestions(tailoring.change_suggestions || [])}
      ${
        tailoring.adapted_cv_draft
          ? `<div class="draft-block">
              <div class="draft-heading">
                <div>
                  <h3>${escapeHtml(tailoring.adapted_cv_draft.title)}</h3>
                  <small>${tailoring.adapted_cv_draft.excluded_suggestion_ids?.length || 0} sugerencias no verificadas excluidas.</small>
                </div>
                <button class="secondary-button compact-button" id="copyDraft" type="button">Copiar borrador</button>
              </div>
              <p class="copy-feedback hidden" id="copyFeedback" role="status">Borrador copiado.</p>
              <pre class="draft">${escapeHtml(tailoring.adapted_cv_draft.content_markdown)}</pre>
              ${renderWarnings(tailoring.adapted_cv_draft.warnings || [])}
            </div>`
          : ""
      }
      ${renderWarnings(tailoring.warnings || [])}
    </section>
  `;
}

function renderTailoringSuggestions(suggestions) {
  if (!suggestions.length) {
    return `<p class="empty-inline">No hay cambios seguros para sugerir con la informaci&oacute;n disponible.</p>`;
  }
  return `
    <div class="suggestion-list">
      ${suggestions.map((item) => `
        <article class="suggestion-item ${item.type === "add_only_if_true" ? "needs-confirmation" : ""}">
          <div class="suggestion-meta">
            <span class="suggestion-type">${suggestionTypeLabel(item.type)}</span>
            ${confidenceBadge(item.confidence)}
            ${item.requires_user_confirmation ? `<span class="status-badge warning">Confirmar dato</span>` : `<span class="status-badge success">Basado en tu CV</span>`}
          </div>
          ${
            item.original_text
              ? `<div class="text-comparison">
                  <span>Texto actual</span>
                  <p>${escapeHtml(item.original_text)}</p>
                </div>`
              : ""
          }
          <div class="text-comparison proposed">
            <span>Texto sugerido</span>
            <p>${escapeHtml(item.suggested_text)}</p>
          </div>
          <small>${escapeHtml(item.reason)}</small>
        </article>
      `).join("")}
    </div>
  `;
}

function renderSources(sources) {
  if (!sources.length) {
    return "";
  }
  return `
    <div class="sources-block">
      <h3>Fuentes consultadas</h3>
      <ul class="source-list">
        ${sources.map((source) => `
          <li>
            <a href="${escapeAttribute(source.url)}" target="_blank" rel="noreferrer">${escapeHtml(source.title)}</a>
            <span>${escapeHtml(source.domain)} &middot; ${sourceTypeLabel(source.source_type)} &middot; confianza ${source.reliability_score}/5</span>
            ${source.accessed_at ? `<small>Consultada/generada: ${formatDate(source.accessed_at)}</small>` : ""}
          </li>
        `).join("")}
      </ul>
    </div>
  `;
}

function renderWarnings(warnings, options = {}) {
  const visibleWarnings = options.globalOnly
    ? warnings.filter((warning) => !warning.related_section)
    : warnings;
  if (!visibleWarnings.length) {
    return "";
  }
  return `
    <div class="warning-list" aria-label="Advertencias">
      ${visibleWarnings.map((warning) => `
        <div class="warning-item severity-${escapeAttribute(warning.severity || "medium")}">
          ${warningSeverityLabel(warning.severity) ? `<strong>${warningSeverityLabel(warning.severity)}</strong>` : ""}
          <p>${escapeHtml(warning.message)}</p>
        </div>
      `).join("")}
    </div>
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
          <div class="item-heading">
            ${item.confidence ? confidenceBadge(item.confidence) : ""}
            <p>${escapeHtml(item.text)}</p>
          </div>
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
    <article class="history-item ${state.currentReportId === item.report_id ? "selected" : ""}">
      <button class="history-open" type="button" data-report-id="${escapeAttribute(item.report_id)}">
        <strong>${escapeHtml(item.company.name)}</strong>
        <span>${reportStatusLabel(item.status)}${item.used_cv ? " &middot; con CV" : ""}${item.used_job_description ? " &middot; con puesto" : ""}${item.generated_at ? ` &middot; ${formatShortDate(item.generated_at)}` : ""}</span>
        ${item.summary ? `<span>${escapeHtml(item.summary)}</span>` : ""}
      </button>
      <button class="history-delete" type="button" data-delete-report-id="${escapeAttribute(item.report_id)}" aria-label="Eliminar informe de ${escapeAttribute(item.company.name)}" title="Eliminar informe">&#128465;</button>
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
  els.cvDisclosure.open = false;
  els.cvTextDisclosure.open = false;
  els.cvFileName.textContent = "PDF o DOCX";
  els.cvFileStatus.textContent = "Ning\u00fan archivo seleccionado.";
  hideFormError();
  updateCvControls();
  updateJobDescriptionCount();
  clearPoll();
  setBusy(false);
  clearCurrentReport();
}

function clearCurrentReport() {
  clearIndexPoll();
  state.currentReportId = null;
  state.currentReportReady = false;
  state.currentDraft = "";
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

function showLoading(message, companyName = "") {
  hideAllStates();
  els.loadingMessage.textContent = message;
  els.loadingCompany.textContent = companyName || "";
  els.loadingState.classList.remove("hidden");
  els.connectionStatus.textContent = "Investigando";
}

function showError(message) {
  hideAllStates();
  els.errorMessage.textContent = message;
  els.errorState.classList.remove("hidden");
  els.connectionStatus.textContent = "Revisar";
  els.errorState.focus();
}

function hideAllStates() {
  els.emptyState.classList.add("hidden");
  els.loadingState.classList.add("hidden");
  els.errorState.classList.add("hidden");
  els.reportView.classList.add("hidden");
}

function setBusy(isBusy) {
  const submitButton = els.form.querySelector(".primary-button");
  submitButton.disabled = isBusy;
  submitButton.textContent = isBusy ? "Investigando..." : "Generar informe";
  els.form.setAttribute("aria-busy", String(isBusy));
}

function updateCvControls() {
  const hasCv = Boolean(els.cvText.value.trim());
  els.includeTailoring.disabled = !hasCv;
  if (!hasCv) {
    els.includeTailoring.checked = false;
  }
  els.includeDraft.disabled = !hasCv || !els.includeTailoring.checked;
  if (els.includeDraft.disabled) {
    els.includeDraft.checked = false;
  }
  els.cvDisclosure.classList.toggle("has-cv", hasCv);
  const disclosureAction = els.cvDisclosure.querySelector(".disclosure-action");
  if (disclosureAction) {
    disclosureAction.textContent = hasCv ? "CV listo" : "Agregar";
  }
}

function updateJobDescriptionCount() {
  const count = els.jobDescription.value.length;
  els.jobDescriptionCount.textContent = `${new Intl.NumberFormat("es-AR").format(count)} / 20.000`;
}

function showFormError(message) {
  els.formError.textContent = message;
  els.formError.classList.remove("hidden");
}

function hideFormError() {
  els.formError.textContent = "";
  els.formError.classList.add("hidden");
}

function focusResultOnNarrowScreen() {
  if (window.innerWidth <= 760) {
    els.loadingState.focus();
    els.loadingState.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" });
  }
}

function prefersReducedMotion() {
  return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches || false;
}

async function copyAdaptedDraft() {
  const button = document.querySelector("#copyDraft");
  const feedback = document.querySelector("#copyFeedback");
  if (!button || !state.currentDraft) {
    return;
  }
  try {
    await navigator.clipboard.writeText(state.currentDraft);
    button.textContent = "Copiado";
    feedback?.classList.remove("hidden");
    window.setTimeout(() => {
      button.textContent = "Copiar borrador";
      feedback?.classList.add("hidden");
    }, 2200);
  } catch {
    button.textContent = "No se pudo copiar";
    feedback?.classList.remove("hidden");
  }
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

function chip(text, variant = "") {
  return `<span class="meta-chip ${escapeAttribute(variant)}">${escapeHtml(text)}</span>`;
}

function renderRagStatusChip(metadata = {}) {
  const status = metadata.rag_index_status || "pending";
  const count = Number(metadata.rag_indexed_chunk_count || 0);
  const labels = {
    ready: `Chat listo${count ? ` · ${count}` : ""}`,
    indexing: "Preparando chat",
    pending: "Preparando chat",
    failed: "Chat con informaci\u00f3n parcial",
  };
  const title = status === "failed"
    ? "No se pudo preparar toda la informaci\u00f3n para el chat."
    : "Estado de preparaci\u00f3n del chat.";
  return `<span class="meta-chip rag-${escapeAttribute(status)}" title="${escapeAttribute(title)}">${escapeHtml(labels[status] || "Preparando chat")}</span>`;
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

function formatShortDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  return new Intl.DateTimeFormat("es-AR", {
    day: "2-digit",
    month: "short",
  }).format(date);
}

function confidenceBadge(value) {
  const labels = {
    high: "Confianza alta",
    medium: "Confianza media",
    low: "Confianza baja",
    unknown: "Confianza no definida",
  };
  const safeValue = labels[value] ? value : "unknown";
  return `<span class="confidence-badge confidence-${safeValue}">${labels[safeValue]}</span>`;
}

function claimTypeLabel(value) {
  const labels = {
    fact: "Hecho",
    inference: "Inferencia",
    recommendation: "Recomendaci\u00f3n",
    missing_evidence: "Dato no disponible",
  };
  return labels[value] || "Hallazgo";
}

function suggestionTypeLabel(value) {
  const labels = {
    rewrite: "Reescribir",
    reorder: "Reordenar",
    emphasize: "Destacar",
    add_only_if_true: "Agregar solo si es cierto",
  };
  return labels[value] || "Sugerencia";
}

function warningSeverityLabel(value) {
  const labels = {
    high: "Revisi\u00f3n necesaria",
    medium: "",
    low: "Nota",
  };
  return labels[value] ?? labels.medium;
}

function reportStatusLabel(value) {
  const labels = {
    pending: "En cola",
    running: "Investigando",
    completed: "Informe listo",
    failed: "No completado",
  };
  return labels[value] || "Estado desconocido";
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
