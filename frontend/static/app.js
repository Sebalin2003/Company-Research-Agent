const ACTIVE_TASK_STATUSES = new Set([
  "pending",
  "running",
  "needs_clarification",
  "awaiting_approval",
  "awaiting_review",
]);

const state = {
  activeView: "conversation",
  activeConversation: null,
  activeLegacyReportId: null,
  previousConversationId: null,
  conversations: [],
  legacyReports: [],
  legacyStatus: "loading",
  search: "",
  drafts: {},
  attachments: {},
  scrollPositions: {},
  eventSource: null,
  eventPollTimer: null,
  seenEventIds: new Set(),
  progress: null,
  artifactPayloads: {},
  pendingRename: null,
  pendingConfirmation: null,
  searchTimer: null,
  retryTimer: null,
  cvs: [],
  selectedCVId: null,
  cvDetail: null,
  cvStatus: "loading",
  cvUploadTarget: null,
  cvViewedVersion: null,
  pendingCVRename: null,
};

const els = {
  appShell: document.querySelector("#appShell"),
  sidebarBackdrop: document.querySelector("#sidebarBackdrop"),
  openSidebar: document.querySelector("#openSidebar"),
  closeSidebar: document.querySelector("#closeSidebar"),
  brandButton: document.querySelector("#brandButton"),
  newChatButton: document.querySelector("#newChatButton"),
  conversationSearch: document.querySelector("#conversationSearch"),
  conversationNav: document.querySelector("#conversationNav"),
  cvLibraryButton: document.querySelector("#cvLibraryButton"),
  cvSidebarSummary: document.querySelector("#cvSidebarSummary"),
  settingsButton: document.querySelector("#settingsButton"),
  workspaceTitle: document.querySelector("#workspaceTitle"),
  workspaceStatus: document.querySelector("#workspaceStatus"),
  conversationMenuButton: document.querySelector("#conversationMenuButton"),
  conversationMenu: document.querySelector("#conversationMenu"),
  demoNotice: document.querySelector("#demoNotice"),
  conversationView: document.querySelector("#conversationView"),
  conversationTranscript: document.querySelector("#conversationTranscript"),
  cvWorkspace: document.querySelector("#cvWorkspace"),
  composerForm: document.querySelector("#composerForm"),
  composerText: document.querySelector("#composerText"),
  attachmentButton: document.querySelector("#attachmentButton"),
  attachmentMenu: document.querySelector("#attachmentMenu"),
  attachmentChips: document.querySelector("#attachmentChips"),
  sendButton: document.querySelector("#sendButton"),
  renameDialog: document.querySelector("#renameDialog"),
  renameDialogTitle: document.querySelector("#renameDialogTitle"),
  renameDialogHint: document.querySelector("#renameDialogHint"),
  renameInput: document.querySelector("#renameInput"),
  confirmRename: document.querySelector("#confirmRename"),
  confirmDialog: document.querySelector("#confirmDialog"),
  confirmDialogTitle: document.querySelector("#confirmDialogTitle"),
  confirmDialogMessage: document.querySelector("#confirmDialogMessage"),
  confirmAction: document.querySelector("#confirmAction"),
  toastRegion: document.querySelector("#toastRegion"),
  cvFileInput: document.querySelector("#cvFileInput"),
  jobDialog: document.querySelector("#jobDialog"),
  jobDialogForm: document.querySelector("#jobDialogForm"),
  jobDescriptionInput: document.querySelector("#jobDescriptionInput"),
  cvSelectDialog: document.querySelector("#cvSelectDialog"),
  cvSelectInput: document.querySelector("#cvSelectInput"),
  confirmCVSelection: document.querySelector("#confirmCVSelection"),
};

const dataSource = {
  async request(path, options = {}) {
    const response = await fetch(path, {
      ...options,
      headers: typeof options.body === "string" ? { "Content-Type": "application/json", ...(options.headers || {}) } : options.headers,
    });
    if (response.status === 204) return null;
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error?.message || "No se pudo completar la operación.");
    return payload;
  },
  listConversations(query = "") {
    const params = new URLSearchParams({ limit: "100" });
    if (query.trim()) params.set("q", query.trim());
    return this.request(`/api/conversations?${params}`);
  },
  createConversation() {
    return this.request("/api/conversations", { method: "POST", body: JSON.stringify({ title: null }) });
  },
  getConversation(conversationId) {
    return this.request(`/api/conversations/${encodeURIComponent(conversationId)}`);
  },
  renameConversation(conversationId, title) {
    return this.request(`/api/conversations/${encodeURIComponent(conversationId)}`, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    });
  },
  deleteConversation(conversationId) {
    return this.request(`/api/conversations/${encodeURIComponent(conversationId)}`, { method: "DELETE" });
  },
  sendMessage(conversationId, content, attachments) {
    return this.request(`/api/conversations/${encodeURIComponent(conversationId)}/messages`, {
      method: "POST",
      body: JSON.stringify({
        content,
        attachments: attachments.map((item) => ({
          type: item.type,
          artifact_id: item.artifactId || null,
          content: item.content || null,
          title: item.title || null,
        })),
      }),
    });
  },
  cancelTask(taskId) {
    return this.request(`/api/task-runs/${encodeURIComponent(taskId)}/cancel`, { method: "POST" });
  },
  retryTask(taskId) {
    return this.request(`/api/task-runs/${encodeURIComponent(taskId)}/retry`, { method: "POST" });
  },
  resumeTask(taskId, payload) {
    return this.request(`/api/task-runs/${encodeURIComponent(taskId)}/resume`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },
  async listLegacyReports() {
    const payload = await this.request("/api/reports?limit=20");
    return payload.items || [];
  },
  getLegacyReport(reportId) {
    return this.request(`/api/reports/${encodeURIComponent(reportId)}`);
  },
  getComparison(comparisonId) {
    return this.request(`/api/comparisons/${encodeURIComponent(comparisonId)}`);
  },
  listCVs() {
    return this.request("/api/cvs");
  },
  getCV(cvId) {
    return this.request(`/api/cvs/${encodeURIComponent(cvId)}`);
  },
  getCVVersion(cvId, versionId) {
    return this.request(`/api/cvs/${encodeURIComponent(cvId)}/versions/${encodeURIComponent(versionId)}`);
  },
  updateCV(cvId, payload) {
    return this.request(`/api/cvs/${encodeURIComponent(cvId)}`, { method: "PATCH", body: JSON.stringify(payload) });
  },
  uploadCV(file, cvId = null) {
    const form = new FormData();
    form.append("file", file);
    return this.request(cvId ? `/api/cvs/${encodeURIComponent(cvId)}/versions` : "/api/cvs", { method: "POST", body: form });
  },
  deleteCV(cvId) {
    return this.request(`/api/cvs/${encodeURIComponent(cvId)}`, { method: "DELETE" });
  },
  deleteCVVersion(cvId, versionId) {
    return this.request(`/api/cvs/${encodeURIComponent(cvId)}/versions/${encodeURIComponent(versionId)}`, { method: "DELETE" });
  },
  getCVRecommendation(artifactId) {
    return this.request(`/api/cv-recommendations/${encodeURIComponent(artifactId)}`);
  },
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function safeUrl(value) {
  try {
    const url = new URL(value);
    return ["http:", "https:"].includes(url.protocol) ? url.href : "#";
  } catch {
    return "#";
  }
}

function formatText(value) {
  return escapeHtml(value)
    .split(/\n{2,}/)
    .map((paragraph) => `<p>${paragraph.replaceAll("\n", "<br>")}</p>`)
    .join("");
}

function icon(name) {
  const paths = {
    chat: '<path d="M5 5h14v10H9l-4 4z" />',
    more: '<circle cx="5" cy="12" r="1" fill="currentColor" stroke="none" /><circle cx="12" cy="12" r="1" fill="currentColor" stroke="none" /><circle cx="19" cy="12" r="1" fill="currentColor" stroke="none" />',
    file: '<path d="M7 3h7l4 4v14H7z" /><path d="M14 3v5h5" />',
    report: '<path d="M6 3h12v18H6zM9 8h6M9 12h6M9 16h4" />',
  };
  return `<svg viewBox="0 0 24 24" aria-hidden="true">${paths[name] || paths.file}</svg>`;
}

function activeConversationId() {
  return state.activeConversation?.conversation_id || null;
}

function activeTask() {
  return state.activeConversation?.current_task || null;
}

function isTaskActive(task = activeTask()) {
  return Boolean(task && ACTIVE_TASK_STATUSES.has(task.status));
}

function groupForDate(value) {
  const date = new Date(value);
  const now = new Date();
  const startToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const startYesterday = new Date(startToday);
  startYesterday.setDate(startYesterday.getDate() - 1);
  if (date >= startToday) return "Hoy";
  if (date >= startYesterday) return "Ayer";
  return "Anteriores";
}

function statusLabel(status) {
  const labels = {
    ready: "Lista",
    pending: "Pendiente",
    running: "Respuesta en curso",
    needs_clarification: "Necesita aclaración",
    awaiting_approval: "Esperando aprobación",
    awaiting_review: "Esperando revisión",
    completed: "Completada",
    cancelled: "Cancelada",
    failed: "Interrumpida",
  };
  return labels[status] || "Lista";
}

function renderSidebar() {
  const groups = ["Hoy", "Ayer", "Anteriores"];
  let html = groups.map((group) => {
    const items = state.conversations.filter((item) => groupForDate(item.updated_at) === group);
    if (!items.length) return "";
    return `
      <section class="conversation-group" aria-labelledby="group-${group}">
        <h2 id="group-${group}">${group}</h2>
        ${items.map(renderConversationRow).join("")}
      </section>`;
  }).join("");

  const query = state.search.trim().toLocaleLowerCase("es");
  const legacy = state.legacyReports.filter((item) => item.company.name.toLocaleLowerCase("es").includes(query));
  if (state.legacyStatus === "loading") {
    html += '<section class="conversation-group"><h2>Informes guardados</h2><p class="sidebar-empty">Cargando informes…</p></section>';
  } else if (state.legacyStatus === "error") {
    html += '<section class="conversation-group"><h2>Informes guardados</h2><p class="sidebar-empty">No se pudieron cargar.</p></section>';
  } else if (legacy.length) {
    html += `
      <section class="conversation-group" aria-labelledby="legacyGroupTitle">
        <h2 id="legacyGroupTitle">Informes guardados</h2>
        ${legacy.map(renderLegacyRow).join("")}
      </section>`;
  }

  if (!html.trim()) {
    html = `<p class="sidebar-empty">${state.search ? "No hay resultados." : "Todavía no hay conversaciones."}</p>`;
  }
  els.conversationNav.innerHTML = html;
  els.cvSidebarSummary.textContent = "Disponible próximamente";
  els.cvLibraryButton.classList.toggle("active", state.activeView === "cv");
  els.cvSidebarSummary.textContent = state.cvs.length
    ? `${state.cvs.length} CV${state.cvs.length === 1 ? "" : "s"} guardado${state.cvs.length === 1 ? "" : "s"}`
    : "Sin CV guardados";
}

function renderConversationRow(item) {
  const selected = state.activeView === "conversation"
    && activeConversationId() === item.conversation_id
    && !state.activeLegacyReportId;
  return `
    <div class="conversation-row ${selected ? "selected" : ""}">
      <button class="conversation-open" type="button" data-conversation-id="${escapeHtml(item.conversation_id)}">
        ${icon("chat")}
        <span>${escapeHtml(item.title)}</span>
      </button>
      <div class="conversation-row-actions">
        <button class="mini-icon-button" type="button" data-conversation-menu="${escapeHtml(item.conversation_id)}" aria-label="Renombrar ${escapeHtml(item.title)}">${icon("more")}</button>
      </div>
    </div>`;
}

function renderLegacyRow(item) {
  const selected = state.activeLegacyReportId === item.report_id;
  return `
    <div class="conversation-row ${selected ? "selected" : ""}">
      <button class="conversation-open" type="button" data-legacy-report-id="${escapeHtml(item.report_id)}">
        ${icon("report")}
        <span>${escapeHtml(item.company.name)}</span>
      </button>
    </div>`;
}

function setHeader(title, status, statusClass = "") {
  els.workspaceTitle.textContent = title;
  els.workspaceStatus.className = `workspace-status ${statusClass}`.trim();
  els.workspaceStatus.innerHTML = `<span aria-hidden="true"></span>${escapeHtml(status)}`;
}

function setOriginNotice(kind, label, text) {
  els.demoNotice.innerHTML = `<span class="origin-badge ${kind}">${escapeHtml(label)}</span><span>${escapeHtml(text)}</span>`;
}

function renderActiveView({ scrollToBottom = false } = {}) {
  renderSidebar();
  if (state.activeView === "cv") {
    renderCVWorkspaceV3();
    return;
  }
  renderConversation({ scrollToBottom });
}

function renderConversation({ scrollToBottom = false } = {}) {
  els.cvWorkspace.classList.add("hidden");
  els.conversationView.classList.remove("hidden");
  els.cvLibraryButton.classList.remove("active");

  if (state.activeLegacyReportId) {
    renderLegacyConversation();
    return;
  }

  const conversation = state.activeConversation;
  const running = ["pending", "running"].includes(conversation?.current_task?.status);
  const taskActive = isTaskActive();
  setHeader(
    conversation?.title || "Nueva conversación",
    statusLabel(conversation?.current_task?.status || "ready"),
    running ? "running" : conversation?.current_task?.status === "failed" ? "paused" : "",
  );
  setOriginNotice("local", "DeepSeek", "Generación con DeepSeek · Conversaciones, herramientas y eventos guardados localmente en SQLite.");
  els.conversationMenuButton.classList.toggle("hidden", !conversation);
  els.composerText.disabled = taskActive;
  els.attachmentButton.disabled = taskActive;
  els.sendButton.disabled = false;
  els.composerText.placeholder = "Escribí un mensaje o pedí una tarea";

  const messages = conversation?.messages || [];
  let transcript = messages.length ? messages.map(renderMessage).join("") : renderEmptyConversation();
  transcript += renderGeneratedArtifacts(conversation);
  if (running) transcript += renderLiveProgress();
  if (["needs_clarification", "awaiting_approval"].includes(conversation?.current_task?.status)) {
    transcript += renderPauseRequest(conversation.current_task);
  }
  if (conversation?.current_task?.status === "cancelled") {
    transcript += renderTaskState("Tarea cancelada", "Cancelaste la respuesta del agente antes de que terminara.", "cancelled");
  }
  if (conversation?.current_task?.status === "failed") {
    transcript += renderTaskState(
      "Tarea interrumpida",
      conversation.current_task.pause?.message || "Volvé a enviar el mensaje para intentarlo nuevamente.",
      "failed",
      true,
      conversation.current_task.pause?.retry_at,
    );
  }
  els.conversationTranscript.innerHTML = transcript;
  syncRetryCountdown();
  syncComposer();

  if (scrollToBottom) {
    requestAnimationFrame(() => { els.conversationTranscript.scrollTop = els.conversationTranscript.scrollHeight; });
  } else if (conversation) {
    els.conversationTranscript.scrollTop = state.scrollPositions[conversation.conversation_id] || 0;
  }
}

function renderEmptyConversation() {
  return `
    <div class="empty-conversation">
      <div>
        <div class="empty-mark" aria-hidden="true">✦</div>
        <h2>¿En qué querés trabajar?</h2>
        <p>Conversá con DeepSeek o pedile que investigue, recupere informes y compare empresas.</p>
        <div class="starter-prompts" aria-label="Ejemplos de pedidos">
          <button class="starter-prompt" type="button" data-starter-prompt="Investigá Mercado Libre para una entrevista junior.">Investigá Mercado Libre para una entrevista junior.</button>
          <button class="starter-prompt" type="button" data-starter-prompt="Compará Globant y Accenture para roles de datos.">Compará Globant y Accenture para roles de datos.</button>
          <button class="starter-prompt" type="button" data-starter-prompt="Ayudame a organizar mi búsqueda laboral junior.">Ayudame a organizar mi búsqueda laboral junior.</button>
        </div>
      </div>
    </div>`;
}

function artifactsForMessage(messageId) {
  return (state.activeConversation?.artifacts || []).filter((item) => item.message_id === messageId);
}

function reportName(reportId) {
  return state.legacyReports.find((item) => item.report_id === reportId)?.company.name || "Informe guardado";
}

function attachmentLabel(item) {
  if (item.type === "cv") return state.cvs.find((cv) => cv.cv_id === item.artifact_id)?.display_name || "CV guardado";
  if (item.type === "job_description") return "Descripción de puesto";
  return reportName(item.artifact_id);
}

function renderMessage(message) {
  const attachments = artifactsForMessage(message.message_id);
  const attachmentHtml = attachments.length
    ? `<div class="message-attachments">${attachments.map((item) => `<span class="inline-attachment">${icon(item.type === "report" ? "report" : "file")}${escapeHtml(attachmentLabel(item))}</span>`).join("")}</div>`
    : "";
  if (message.role === "user") {
    return `<article class="message user" data-message-id="${escapeHtml(message.message_id)}"><div class="user-message-body"><div class="message-content">${formatText(message.content)}</div>${attachmentHtml}</div></article>`;
  }
  const content = message.content || "Preparando respuesta…";
  const citations = (message.citations || []).length
    ? `<div class="source-links message-citations">${renderSources(message.citations)}</div>`
    : "";
  return `
    <article class="message assistant" data-message-id="${escapeHtml(message.message_id)}">
      <div class="assistant-message">
        <span class="assistant-mark" aria-hidden="true"></span>
        <div>
          <div class="message-content">${formatText(content)}</div>${citations}
          <div class="message-meta"><span>Radar Laboral</span><span class="origin-badge local">DeepSeek</span></div>
        </div>
      </div>
    </article>`;
}

function renderLiveProgress() {
  const progress = state.progress || {
    label: "DeepSeek está evaluando la solicitud",
    detail: "Esperando eventos persistentes",
    status: "running",
  };
  return `
    <section class="research-progress" data-artifact-type="progress">
      <strong>Procesando mensaje</strong>
      <div class="progress-step ${escapeHtml(progress.status)}">
        <span class="progress-step-icon"></span>
        <span><strong>${escapeHtml(progress.label)}</strong><small>${escapeHtml(progress.detail)}</small></span>
      </div>
    </section>`;
}

function renderPauseRequest(task) {
  const pause = task.pause || {};
  if (task.status === "awaiting_approval") {
    return `<section class="decision-card warning" data-artifact-type="approval-request">
      <strong>Se necesita tu aprobación</strong><p>${escapeHtml(pause.prompt || "¿Querés ampliar la investigación?")}</p>
      <div class="decision-actions"><button class="primary-button" type="button" data-task-resume="approval" data-decision="approved">Continuar</button><button class="secondary-button" type="button" data-task-resume="approval" data-decision="rejected">Finalizar con lo disponible</button></div>
    </section>`;
  }
  const options = (pause.options || []).map((option) => `<button class="secondary-button" type="button" data-task-resume="clarification" data-option-id="${escapeHtml(option.id)}">${escapeHtml(option.label)}</button>`).join("");
  return `<section class="decision-card" data-artifact-type="clarification-request">
    <strong>Necesito una aclaración</strong><p>${escapeHtml(pause.prompt || "Agregá la información necesaria para continuar.")}</p>
    ${options ? `<div class="decision-actions">${options}</div>` : ""}
    <label class="decision-input"><span>Respuesta</span><textarea rows="2" data-clarification-input></textarea></label>
    <button class="primary-button" type="button" data-task-resume="clarification-text">Responder</button>
  </section>`;
}

function renderGeneratedArtifacts(conversation) {
  const generated = (conversation?.artifacts || []).filter((item) => item.relationship_type === "generated");
  return generated.map((item) => {
    const payload = state.artifactPayloads[`${item.type}:${item.artifact_id}`];
    if (!payload) return `<section class="artifact artifact-loading" data-artifact-type="${escapeHtml(item.type)}"><p>Cargando artefacto…</p></section>`;
    if (payload.unavailable) return `<section class="evidence-warning" data-artifact-type="artifact-missing"><strong>Artefacto no disponible</strong><p>El contenido fue eliminado o no se pudo recuperar.</p></section>`;
    if (item.type === "report") return renderLegacyReportArtifact(payload.report, item.artifact_id);
    if (item.type === "comparison") return renderComparisonArtifact(payload);
    if (item.type === "cv_recommendation") return renderCVRecommendationArtifact(payload);
    return "";
  }).join("");
}

function renderComparisonArtifact(comparison) {
  const companies = comparison.payload?.companies || [];
  const rows = comparison.payload?.rows || [];
  return `<section class="artifact" data-artifact-type="comparison" data-artifact-id="${escapeHtml(comparison.comparison_id)}">
    <header class="artifact-header"><div><h3>${escapeHtml(comparison.title)}</h3><p>${escapeHtml(formatDate(comparison.created_at))}</p></div><span class="origin-badge saved">Guardado</span></header>
    <div class="artifact-body">
      <table class="comparison-table"><thead><tr><th>Dimensión</th>${companies.map((company) => `<th>${escapeHtml(company.name)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr><th>${escapeHtml(row.dimension)}</th>${row.values.map((value) => `<td><p>${escapeHtml(value.value)}</p><span class="confidence-badge ${escapeHtml(value.confidence)}">${escapeHtml(value.confidence)}</span></td>`).join("")}</tr>`).join("")}</tbody></table>
      <div class="comparison-mobile">${rows.map((row) => `<section class="artifact-section"><h4>${escapeHtml(row.dimension)}</h4>${row.values.map((value) => `<p><strong>${escapeHtml(value.company)}:</strong> ${escapeHtml(value.value)}</p>`).join("")}</section>`).join("")}</div>
    </div>
    <footer class="artifact-footer"><div class="source-links">${renderSources((comparison.citations || []).slice(0, 8))}</div><span class="confidence-badge medium">${comparison.report_ids.length} informes</span></footer>
  </section>`;
}

function renderCVRecommendationArtifact(artifact) {
  const suggestions = artifact.payload?.change_suggestions || [];
  const review = artifact.review?.decisions || [];
  const decisions = new Map(review.map((item) => [item.suggestion_id, item]));
  const editable = artifact.status === "awaiting_review";
  const manual = artifact.review?.manual_suggestion_ids || [];
  return `<section class="artifact cv-review-artifact" data-artifact-type="cv-recommendation" data-artifact-id="${escapeHtml(artifact.artifact_id)}">
    <header class="artifact-header"><div><h3>Recomendaciones para tu CV</h3><p>${escapeHtml(artifact.payload?.positioning_summary || "Revisá cada cambio antes de guardarlo.")}</p></div><span class="origin-badge ${editable ? "warning" : "saved"}">${editable ? "Revisión" : "Revisado"}</span></header>
    <div class="artifact-body review-list">${suggestions.map((suggestion) => {
      const saved = decisions.get(suggestion.id);
      return `<article class="review-item ${escapeHtml(saved?.decision || "")}" data-review-suggestion="${escapeHtml(suggestion.id)}" data-review-type="${escapeHtml(suggestion.type)}">
        <h4>${escapeHtml(suggestion.type)} · ${escapeHtml(suggestion.confidence)}</h4>
        ${suggestion.original_text ? `<p><strong>Original:</strong> ${escapeHtml(suggestion.original_text)}</p>` : ""}
        <label class="review-edit"><span>Propuesta</span><textarea rows="3" data-review-text ${editable ? "" : "disabled"}>${escapeHtml(saved?.edited_text || suggestion.suggested_text)}</textarea></label>
        <p>${escapeHtml(suggestion.reason)}</p>
        ${suggestion.type === "add_only_if_true" ? `<label class="truth-check"><input type="checkbox" data-truth-confirmed ${saved?.truth_confirmed ? "checked" : ""} ${editable ? "" : "disabled"}> Confirmo que este dato es verdadero</label>` : ""}
        <div class="review-actions">
          <label><input type="radio" name="review-${escapeHtml(suggestion.id)}" value="accepted" ${saved?.decision === "accepted" ? "checked" : ""} ${editable ? "" : "disabled"}> Aceptar</label>
          <label><input type="radio" name="review-${escapeHtml(suggestion.id)}" value="edited" ${saved?.decision === "edited" ? "checked" : ""} ${editable ? "" : "disabled"}> Usar edición</label>
          <label><input type="radio" name="review-${escapeHtml(suggestion.id)}" value="rejected" ${saved?.decision === "rejected" ? "checked" : ""} ${editable ? "" : "disabled"}> Rechazar</label>
        </div>
      </article>`;
    }).join("")}</div>
    ${manual.length ? `<div class="evidence-warning"><strong>Cambios manuales pendientes</strong><p>${manual.length} sugerencia${manual.length === 1 ? " no pudo" : "s no pudieron"} aplicarse automáticamente porque el reemplazo era ambiguo o no era texto final.</p></div>` : ""}
    <div class="artifact-body cv-draft"><label><strong>Borrador editable</strong><textarea class="cv-text-editor" data-review-draft ${editable ? "" : "disabled"}>${escapeHtml(artifact.draft_text || "")}</textarea></label></div>
    ${editable ? `<footer class="artifact-footer"><div class="review-actions"><button class="secondary-button" type="button" data-submit-review="save">Guardar revisión</button><button class="primary-button" type="button" data-submit-review="version">Crear nueva versión de CV</button></div></footer>` : ""}
  </section>`;
}

function renderTaskState(title, content, status, retry = false, retryAt = null) {
  const retryAttribute = retryAt ? ` data-retry-at="${escapeHtml(retryAt)}"` : "";
  return `<section class="task-error" data-artifact-type="task-${escapeHtml(status)}"><strong>${escapeHtml(title)}</strong><p>${escapeHtml(content)}</p>${retry ? `<button class="secondary-button" type="button" data-retry-task${retryAttribute}>Reintentar</button>` : ""}</section>`;
}

function syncRetryCountdown() {
  window.clearTimeout(state.retryTimer);
  state.retryTimer = null;
  const button = els.conversationTranscript.querySelector("[data-retry-at]");
  if (!button) return;
  const remaining = Math.max(0, Math.ceil((Date.parse(button.dataset.retryAt) - Date.now()) / 1000));
  button.disabled = remaining > 0;
  button.textContent = remaining > 0 ? `Reintentar en ${remaining}s` : "Reintentar";
  if (remaining > 0) state.retryTimer = window.setTimeout(syncRetryCountdown, 1000);
}

function syncComposer() {
  const key = activeConversationId() || "new";
  els.composerText.value = state.drafts[key] || "";
  state.attachments[key] ||= [];
  renderAttachmentChips(state.attachments[key]);
  resizeComposer();
  const running = isTaskActive();
  els.sendButton.classList.toggle("running", running);
  els.sendButton.setAttribute("aria-label", running ? "Cancelar tarea" : "Enviar mensaje");
}

function renderAttachmentChips(items = []) {
  els.attachmentChips.innerHTML = items.map((item) => `
    <span class="attachment-chip">
      <span>${escapeHtml(item.label)}</span>
      <small class="origin-badge saved">Guardado</small>
      <button type="button" data-remove-attachment="${escapeHtml(item.id)}" aria-label="Quitar ${escapeHtml(item.label)}">×</button>
    </span>`).join("");
}

async function createNewConversation() {
  saveDraftAndScroll();
  closeEventStream();
  try {
    const created = await dataSource.createConversation();
    await loadConversations();
    await openConversation(created.conversation_id, { focusComposer: true });
  } catch (error) {
    showToast(error.message);
  }
}

async function openConversation(conversationId, { focusComposer = false } = {}) {
  saveDraftAndScroll();
  closeEventStream();
  state.activeView = "conversation";
  state.activeLegacyReportId = null;
  state.progress = null;
  try {
    state.activeConversation = await dataSource.getConversation(conversationId);
    await hydrateConversationArtifacts(state.activeConversation);
    state.previousConversationId = conversationId;
    renderActiveView();
    if (["pending", "running"].includes(activeTask()?.status)) {
      const cursor = state.activeConversation.last_event_id || 0;
      openEventStream(`/api/conversations/${encodeURIComponent(conversationId)}/events?after_event_id=${cursor}`);
    }
    if (focusComposer) requestAnimationFrame(() => els.composerText.focus());
  } catch (error) {
    showToast(error.message);
    await loadConversations();
  }
  closeSidebar();
}

function openLegacyReport(reportId) {
  saveDraftAndScroll();
  closeEventStream();
  state.activeView = "conversation";
  state.activeLegacyReportId = reportId;
  renderActiveView();
  closeSidebar();
}

async function openCVLibraryV3() {
  saveDraftAndScroll();
  state.activeView = "cv";
  state.activeLegacyReportId = null;
  await loadCVs();
  renderActiveView();
  closeSidebar();
}

function renderCVWorkspaceV3() {
  els.conversationView.classList.add("hidden");
  els.cvWorkspace.classList.remove("hidden");
  els.conversationMenuButton.classList.add("hidden");
  setHeader("Mi CV", state.cvStatus === "loading" ? "Cargando" : "Biblioteca local", "");
  setOriginNotice("local", "Local", "Tus CV y sus versiones se guardan en este equipo.");
  const header = `<div class="cv-workspace-header">
    <div><h2 id="cvWorkspaceTitle">Biblioteca de CV</h2><p>Administrá el texto editable y conservá cada archivo original.</p></div>
    <div class="cv-actions"><button class="secondary-button" type="button" data-cv-action="back">Volver al chat</button><button class="primary-button" type="button" data-cv-action="upload">Subir CV</button></div>
  </div>`;
  if (state.cvStatus === "loading") {
    els.cvWorkspace.innerHTML = `${header}<div class="cv-empty"><p>Cargando CV guardados…</p></div>`;
    return;
  }
  if (state.cvStatus === "error") {
    els.cvWorkspace.innerHTML = `${header}<div class="cv-empty"><p>No se pudo cargar la biblioteca de CV.</p><button class="secondary-button" type="button" data-cv-action="reload">Reintentar</button></div>`;
    return;
  }
  if (!state.cvs.length) {
    els.cvWorkspace.innerHTML = `${header}<div class="cv-empty"><p>No hay CV guardados. Subí un PDF o DOCX para comenzar.</p><button class="primary-button" type="button" data-cv-action="upload">Subir primer CV</button></div>`;
    return;
  }
  const detail = state.cvDetail;
  els.cvWorkspace.innerHTML = `${header}<div class="cv-layout">
    <nav class="cv-list" aria-label="CV guardados">${state.cvs.map((cv) => `<button class="cv-list-item ${cv.cv_id === state.selectedCVId ? "selected" : ""}" type="button" data-cv-action="select" data-cv-id="${escapeHtml(cv.cv_id)}"><strong>${escapeHtml(cv.display_name)}</strong><small>Versión ${cv.current_version.version_number}${cv.is_default ? " · Predeterminado" : ""}</small></button>`).join("")}</nav>
    <section class="cv-editor">${detail ? renderCVDetail(detail) : '<div class="cv-empty"><p>Cargando detalle…</p></div>'}</section>
  </div>`;
}

function renderCVDetail(cv) {
  const shown = state.cvViewedVersion || cv;
  const shownVersion = state.cvViewedVersion || cv.current_version;
  const signals = shown.structured_profile || {};
  return `<div class="cv-summary-bar"><div><h3>${escapeHtml(cv.display_name)} ${cv.is_default ? '<span class="origin-badge saved">Predeterminado</span>' : ""}</h3><p>Versión actual ${cv.current_version.version_number} · ${escapeHtml(cv.current_version.content_type || "Solo texto")} · Actualizado ${escapeHtml(formatDate(cv.updated_at))}</p></div>
    <div class="cv-actions"><button class="secondary-button" type="button" data-cv-action="rename">Renombrar</button>${cv.is_default ? "" : '<button class="secondary-button" type="button" data-cv-action="default">Usar por defecto</button>'}<button class="secondary-button" type="button" data-cv-action="replace">Subir reemplazo</button><button class="danger-button" type="button" data-cv-action="delete">Eliminar</button></div></div>
    ${state.cvViewedVersion ? `<section class="evidence-warning"><strong>Versión histórica ${shownVersion.version_number}</strong><p>Esta vista es de solo lectura.</p><button class="secondary-button" type="button" data-cv-action="current">Volver a la versión actual</button></section>` : ""}
    <section class="cv-section"><div class="cv-section-heading"><h3>Resumen</h3></div><div class="profile-signals">${renderProfileSignals(signals)}</div></section>
    <section class="cv-section"><div class="cv-section-heading"><h3>Texto extraído</h3>${state.cvViewedVersion ? "" : '<button class="primary-button" type="button" data-cv-action="save-text">Guardar como nueva versión</button>'}</div><textarea class="cv-text-editor" data-cv-text ${state.cvViewedVersion ? "disabled" : ""}>${escapeHtml(shown.extracted_text || "")}</textarea></section>
    <section class="cv-section"><div class="cv-section-heading"><h3>Versiones</h3></div><div class="version-list">${cv.versions.map((version) => `<div class="version-row"><div><strong>Versión ${version.version_number}</strong><small>${escapeHtml(version.created_from)} · ${escapeHtml(formatDate(version.created_at))}</small></div><div class="cv-actions">${version.has_file ? `<a class="secondary-button" href="/api/cvs/${encodeURIComponent(cv.cv_id)}/versions/${encodeURIComponent(version.version_id)}/file" target="_blank" rel="noopener">Archivo original</a>` : ""}${cv.versions.length > 1 ? `<button class="danger-button" type="button" data-cv-action="delete-version" data-version-id="${escapeHtml(version.version_id)}">Eliminar</button>` : ""}</div></div>`).join("")}</div></section>`;
}

function renderProfileSignals(signals) {
  const groups = [
    ["Habilidades", signals.hard_skills],
    ["Idiomas", signals.languages],
    ["Roles", signals.roles],
    ["Educación", signals.education],
  ].filter(([, values]) => Array.isArray(values) && values.length);
  return groups.length ? groups.map(([label, values]) => `<div><strong>${label}</strong><p>${values.map(escapeHtml).join(" · ")}</p></div>`).join("") : "<p>No se detectaron señales estructuradas.</p>";
}

async function renderLegacyConversation() {
  const reportId = state.activeLegacyReportId;
  const summary = state.legacyReports.find((item) => item.report_id === reportId);
  setHeader(summary?.company.name || "Informe guardado", "Cargando informe…", "running");
  setOriginNotice("saved", "Guardado", "Informe real del sistema actual · Modo de lectura.");
  els.conversationMenuButton.classList.add("hidden");
  els.composerText.disabled = true;
  els.attachmentButton.disabled = true;
  els.sendButton.disabled = true;
  els.composerText.placeholder = "Informe guardado en modo de lectura";
  els.attachmentChips.innerHTML = "";
  els.conversationTranscript.innerHTML = renderLegacyLoading();
  try {
    const payload = await dataSource.getLegacyReport(reportId);
    if (state.activeLegacyReportId !== reportId) return;
    if (!payload.report) throw new Error(payload.error?.message || "El informe todavía no está listo.");
    setHeader(payload.report.company?.name || summary?.company.name || "Informe guardado", "Informe guardado", "");
    els.conversationTranscript.innerHTML = `
      <article class="message assistant">
        <div class="assistant-message"><span class="assistant-mark" aria-hidden="true"></span><div><div class="message-content"><p>Este informe fue creado con la versión actual del sistema y se muestra como un artefacto de solo lectura.</p></div>${renderLegacyReportArtifact(payload.report, reportId)}<div class="message-meta"><span>Radar Laboral</span><span class="origin-badge saved">Guardado</span></div></div></div>
      </article>`;
  } catch (error) {
    setHeader(summary?.company.name || "Informe guardado", "No disponible", "paused");
    els.conversationTranscript.innerHTML = `<div class="empty-conversation"><div><div class="empty-mark" aria-hidden="true">!</div><h2>No se pudo abrir el informe</h2><p>${escapeHtml(error.message)}</p></div></div>`;
  }
}

function renderLegacyLoading() {
  return `<article class="message assistant"><div class="assistant-message"><span class="assistant-mark" aria-hidden="true"></span><div><div class="research-progress"><strong>Cargando informe guardado</strong><div class="progress-step running"><span class="progress-step-icon"></span><span><strong>Recuperando contenido</strong><small>Fuentes, evidencias y secciones</small></span></div></div></div></div></article>`;
}

function renderLegacyReportArtifact(report, reportId) {
  const sections = report.sections || [];
  const sources = report.sources || [];
  return `
    <section class="artifact" data-artifact-type="legacy-report" data-artifact-id="legacy-${escapeHtml(reportId)}">
      <header class="artifact-header"><div><h3>Informe: ${escapeHtml(report.company?.name || "Empresa")}</h3><p>${escapeHtml(formatDate(report.generated_at))}</p></div><span class="origin-badge saved">Guardado</span></header>
      <div class="artifact-body">${sections.map((section) => `
        <section class="artifact-section">
          <h4>${escapeHtml(section.title)}</h4>
          <p>${escapeHtml(section.summary)}</p>
          ${(section.claims || []).length ? `<ul>${section.claims.map((claim) => `<li>${escapeHtml(claim.text)}</li>`).join("")}</ul>` : ""}
        </section>`).join("") || '<section class="artifact-section"><p>El informe no contiene secciones disponibles.</p></section>'}</div>
      <footer class="artifact-footer"><div class="source-links">${renderSources(sources.slice(0, 6))}</div><span class="confidence-badge ${report.warnings?.length ? "medium" : "high"}">${sources.length} fuentes</span></footer>
    </section>`;
}

function renderSources(sources = []) {
  return sources.map((source) => `<a class="source-link" href="${escapeHtml(safeUrl(source.url))}" target="_blank" rel="noopener noreferrer">${escapeHtml(source.title || source.label)} ↗</a>`).join("");
}

async function handleComposerSubmit(event) {
  event.preventDefault();
  if (isTaskActive()) {
    await cancelActiveTask();
    return;
  }
  const content = els.composerText.value.trim();
  const draftKey = activeConversationId() || "new";
  const attachments = state.attachments[draftKey] || [];
  if (!content) {
    els.composerText.focus();
    return;
  }

  try {
    if (!state.activeConversation) {
      const created = await dataSource.createConversation();
      state.activeConversation = await dataSource.getConversation(created.conversation_id);
    }
    const conversationId = activeConversationId();
    const accepted = await dataSource.sendMessage(conversationId, content, attachments);
    state.drafts[draftKey] = "";
    state.attachments[draftKey] = [];
    if (draftKey === "new") {
      state.drafts[conversationId] = "";
      state.attachments[conversationId] = [];
    }
    els.composerText.value = "";
    state.activeConversation = await dataSource.getConversation(conversationId);
    await hydrateConversationArtifacts(state.activeConversation);
    state.progress = null;
    state.seenEventIds.clear();
    renderActiveView({ scrollToBottom: true });
    await loadConversations();
    openEventStream(accepted.events_url);
  } catch (error) {
    showToast(error.message);
  }
}

async function retryFailedTask() {
  const conversation = state.activeConversation;
  const taskId = conversation?.current_task?.task_run_id;
  if (!conversation || !taskId) return;
  try {
    const accepted = await dataSource.retryTask(taskId);
    state.activeConversation = await dataSource.getConversation(conversation.conversation_id);
    state.progress = null;
    state.seenEventIds.clear();
    renderActiveView({ scrollToBottom: true });
    await loadConversations();
    openEventStream(accepted.events_url);
  } catch (error) {
    showToast(error.message);
  }
}

function openEventStream(eventsUrl) {
  closeEventStream();
  state.seenEventIds.clear();
  const source = new EventSource(eventsUrl);
  state.eventSource = source;
  const eventTypes = [
    "task.started",
    "task.progress",
    "tool.started",
    "tool.completed",
    "artifact.created",
    "clarification.required",
    "approval.required",
    "review.required",
    "message.started",
    "message.delta",
    "message.completed",
    "task.completed",
    "task.cancelled",
    "task.failed",
    "heartbeat",
  ];
  eventTypes.forEach((type) => source.addEventListener(type, handleStreamEvent));
  state.eventPollTimer = window.setTimeout(pollActiveTaskState, 1000);
  source.onerror = async () => {
    if (state.eventSource !== source) return;
    await refreshActiveConversation();
    if (!isTaskActive()) {
      closeEventStream();
      return;
    }
    setHeader(state.activeConversation?.title || "Conversación", "Reconectando eventos…", "running");
  };
}

async function pollActiveTaskState() {
  const conversationId = activeConversationId();
  if (!conversationId || !isTaskActive()) return;
  try {
    const conversation = await dataSource.getConversation(conversationId);
    if (conversationId !== activeConversationId()) return;
    state.activeConversation = conversation;
    await hydrateConversationArtifacts(conversation);
    renderActiveView({ scrollToBottom: true });
    if (!isTaskActive()) {
      closeEventStream();
      await loadConversations();
      return;
    }
  } catch (error) {
    showToast(error.message);
  }
  state.eventPollTimer = window.setTimeout(pollActiveTaskState, 1000);
}

function handleStreamEvent(event) {
  if (event.lastEventId && state.seenEventIds.has(event.lastEventId)) return;
  if (event.lastEventId) state.seenEventIds.add(event.lastEventId);
  const payload = JSON.parse(event.data || "{}");
  const conversation = state.activeConversation;
  if (!conversation) return;

  if (event.type === "task.started") {
    conversation.current_task = { ...(conversation.current_task || {}), task_run_id: payload.task_run_id, status: "running" };
  } else if (["task.progress", "tool.started", "tool.completed"].includes(event.type)) {
    state.progress = payload;
  } else if (event.type === "artifact.created") {
    refreshActiveConversation();
  } else if (["clarification.required", "approval.required", "review.required"].includes(event.type)) {
    conversation.current_task = {
      ...(conversation.current_task || {}),
      task_run_id: payload.task_run_id,
      status: event.type === "clarification.required"
        ? "needs_clarification"
        : event.type === "approval.required"
          ? "awaiting_approval"
          : "awaiting_review",
      pause: payload,
    };
    state.progress = null;
    closeEventStream();
    refreshActiveConversation();
  } else if (event.type === "message.started" || event.type === "message.completed") {
    upsertMessage(payload.message);
  } else if (event.type === "message.delta") {
    const message = conversation.messages.find((item) => item.message_id === payload.message_id);
    if (message) message.content += payload.delta || "";
  } else if (["task.completed", "task.cancelled", "task.failed"].includes(event.type)) {
    conversation.current_task = { ...(conversation.current_task || {}), task_run_id: payload.task_run_id, status: payload.status };
    state.progress = null;
    closeEventStream();
    refreshActiveConversation();
  }
  renderActiveView({ scrollToBottom: true });
}

function upsertMessage(message) {
  if (!message || !state.activeConversation) return;
  const messages = state.activeConversation.messages;
  const index = messages.findIndex((item) => item.message_id === message.message_id);
  if (index >= 0) messages[index] = message;
  else messages.push(message);
}

function closeEventStream() {
  state.eventSource?.close();
  state.eventSource = null;
  window.clearTimeout(state.eventPollTimer);
  state.eventPollTimer = null;
}

async function refreshActiveConversation() {
  const conversationId = activeConversationId();
  if (!conversationId) return;
  try {
    state.activeConversation = await dataSource.getConversation(conversationId);
    await hydrateConversationArtifacts(state.activeConversation);
    renderActiveView({ scrollToBottom: true });
    await Promise.all([loadConversations(), loadLegacyReports()]);
  } catch (error) {
    showToast(error.message);
  }
}

async function hydrateConversationArtifacts(conversation) {
  const generated = (conversation?.artifacts || []).filter((item) => item.relationship_type === "generated");
  await Promise.all(generated.map(async (item) => {
    const key = `${item.type}:${item.artifact_id}`;
    if (state.artifactPayloads[key]) return;
    try {
      state.artifactPayloads[key] = item.type === "report"
        ? await dataSource.getLegacyReport(item.artifact_id)
        : item.type === "comparison"
          ? await dataSource.getComparison(item.artifact_id)
          : await dataSource.getCVRecommendation(item.artifact_id);
    } catch {
      state.artifactPayloads[key] = { unavailable: true };
    }
  }));
}

async function resumeActiveTask(payload) {
  const task = activeTask();
  if (!task) return;
  try {
    const accepted = await dataSource.resumeTask(task.task_run_id, payload);
    if (payload.response_type === "review" && payload.artifact_id) {
      delete state.artifactPayloads[`cv_recommendation:${payload.artifact_id}`];
    }
    state.activeConversation = await dataSource.getConversation(activeConversationId());
    state.progress = null;
    renderActiveView({ scrollToBottom: true });
    openEventStream(accepted.events_url);
  } catch (error) {
    showToast(error.message);
  }
}

async function cancelActiveTask() {
  const task = activeTask();
  if (!task) return;
  try {
    await dataSource.cancelTask(task.task_run_id);
    task.status = "cancelled";
    state.progress = null;
    renderActiveView({ scrollToBottom: true });
    await refreshActiveConversation();
  } catch (error) {
    showToast(error.message);
  }
}

function handleAttachmentAction(action) {
  closeMenus();
  if (action === "stored-cv") {
    if (!state.cvs.length) {
      showToast("Todavía no hay CV guardados. Subí uno primero.");
      return;
    }
    els.cvSelectInput.innerHTML = state.cvs.map((cv) => `<option value="${escapeHtml(cv.cv_id)}">${escapeHtml(cv.display_name)}${cv.is_default ? " (predeterminado)" : ""}</option>`).join("");
    els.cvSelectDialog.showModal();
    requestAnimationFrame(() => els.cvSelectInput.focus());
    return;
  }
  if (action === "upload-cv") {
    state.cvUploadTarget = "attach";
    els.cvFileInput.click();
    return;
  }
  if (action === "job") {
    els.jobDescriptionInput.value = "";
    els.jobDialog.showModal();
    requestAnimationFrame(() => els.jobDescriptionInput.focus());
    return;
  }
  if (action === "report") {
    const report = state.legacyReports[0];
    if (!report) {
      showToast("Todavía no hay informes guardados para adjuntar.");
      return;
    }
    const key = activeConversationId() || "new";
    state.attachments[key] ||= [];
    if (!state.attachments[key].some((item) => item.artifactId === report.report_id)) {
      state.attachments[key].push({
        id: `report-${report.report_id}`,
        type: "report",
        artifactId: report.report_id,
        label: `Informe de ${report.company.name}`,
      });
    }
    renderAttachmentChips(state.attachments[key]);
  }
}

function saveDraftAndScroll() {
  const conversationId = activeConversationId();
  const key = conversationId || "new";
  state.drafts[key] = els.composerText.value;
  if (conversationId) state.scrollPositions[conversationId] = els.conversationTranscript.scrollTop;
}

function showRenameDialog(conversationId, currentName) {
  state.pendingRename = conversationId;
  els.renameDialogTitle.textContent = "Renombrar conversación";
  els.renameDialogHint.textContent = "Usá un título que te ayude a encontrarla.";
  els.renameInput.value = currentName;
  els.renameDialog.showModal();
  requestAnimationFrame(() => els.renameInput.select());
}

function showConfirmation({ title, message, actionLabel = "Eliminar", action }) {
  state.pendingConfirmation = action;
  els.confirmDialogTitle.textContent = title;
  els.confirmDialogMessage.textContent = message;
  els.confirmAction.textContent = actionLabel;
  els.confirmDialog.showModal();
}

async function deleteConversation(conversationId) {
  try {
    await dataSource.deleteConversation(conversationId);
    closeEventStream();
    delete state.drafts[conversationId];
    delete state.attachments[conversationId];
    state.activeConversation = null;
    await loadConversations();
    if (state.conversations.length) await openConversation(state.conversations[0].conversation_id);
    else renderActiveView();
    showToast("Conversación eliminada.");
  } catch (error) {
    showToast(error.message);
  }
}

function toggleMenu(menu, button, force) {
  const shouldOpen = force ?? menu.classList.contains("hidden");
  menu.classList.toggle("hidden", !shouldOpen);
  button?.setAttribute("aria-expanded", String(shouldOpen));
}

function closeMenus() {
  toggleMenu(els.attachmentMenu, els.attachmentButton, false);
  toggleMenu(els.conversationMenu, els.conversationMenuButton, false);
}

function closeSidebar() {
  els.appShell.classList.remove("sidebar-open");
}

function resizeComposer() {
  els.composerText.style.height = "auto";
  els.composerText.style.height = `${Math.min(els.composerText.scrollHeight, 180)}px`;
}

function showToast(message) {
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  els.toastRegion.append(toast);
  setTimeout(() => toast.remove(), 3200);
}

function formatDate(value) {
  if (!value) return "Fecha no disponible";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("es-AR", { dateStyle: "medium" }).format(date);
}

async function loadConversations() {
  try {
    const payload = await dataSource.listConversations(state.search);
    state.conversations = payload.items || [];
  } catch (error) {
    state.conversations = [];
    showToast(error.message);
  }
  renderSidebar();
}

async function loadLegacyReports() {
  try {
    state.legacyReports = await dataSource.listLegacyReports();
    state.legacyStatus = "ready";
  } catch {
    state.legacyStatus = "error";
  }
  renderSidebar();
}

async function loadCVs() {
  state.cvStatus = "loading";
  if (state.activeView === "cv") renderActiveView();
  try {
    const payload = await dataSource.listCVs();
    state.cvs = payload.items || [];
    state.cvStatus = "ready";
    if (!state.cvs.some((cv) => cv.cv_id === state.selectedCVId)) {
      state.selectedCVId = state.cvs.find((cv) => cv.is_default)?.cv_id || state.cvs[0]?.cv_id || null;
    }
    if (state.selectedCVId) await loadCVDetail(state.selectedCVId, false);
    else state.cvDetail = null;
  } catch (error) {
    state.cvStatus = "error";
    state.cvs = [];
    state.cvDetail = null;
    showToast(error.message);
  }
  renderSidebar();
  if (state.activeView === "cv") renderActiveView();
}

async function loadCVDetail(cvId, rerender = true) {
  state.selectedCVId = cvId;
  state.cvViewedVersion = null;
  try {
    state.cvDetail = await dataSource.getCV(cvId);
  } catch (error) {
    state.cvDetail = null;
    showToast(error.message);
  }
  if (rerender) renderActiveView();
}

function attachStoredCV(cv) {
  const key = activeConversationId() || "new";
  state.attachments[key] ||= [];
  if (!state.attachments[key].some((item) => item.type === "cv" && item.artifactId === cv.cv_id)) {
    state.attachments[key].push({ id: `cv-${cv.cv_id}`, type: "cv", artifactId: cv.cv_id, label: cv.display_name });
  }
  renderAttachmentChips(state.attachments[key]);
}

async function uploadSelectedCV(file) {
  if (!file) return;
  const target = state.cvUploadTarget;
  try {
    const uploaded = await dataSource.uploadCV(file, target && target !== "attach" ? target : null);
    await loadCVs();
    if (target === "attach") attachStoredCV(state.cvs.find((cv) => cv.cv_id === uploaded.cv_id) || uploaded);
    if (state.activeView === "cv") await loadCVDetail(uploaded.cv_id);
    showToast(target && target !== "attach" ? "Nueva versión guardada." : "CV guardado localmente.");
  } catch (error) {
    showToast(error.message);
  } finally {
    state.cvUploadTarget = null;
    els.cvFileInput.value = "";
  }
}

async function handleCVWorkspaceAction(button) {
  const action = button.dataset.cvAction;
  const cv = state.cvDetail;
  if (action === "back") {
    state.activeView = "conversation";
    renderActiveView();
    requestAnimationFrame(() => els.composerText.focus());
    return;
  }
  if (action === "reload") return loadCVs();
  if (action === "upload") {
    state.cvUploadTarget = null;
    els.cvFileInput.click();
    return;
  }
  if (action === "select") return loadCVDetail(button.dataset.cvId);
  if (!cv) return;
  if (action === "current") {
    state.cvViewedVersion = null;
    renderActiveView();
  } else if (action === "view-version") {
    try {
      state.cvViewedVersion = await dataSource.getCVVersion(cv.cv_id, button.dataset.versionId);
      renderActiveView();
    } catch (error) { showToast(error.message); }
  } else if (action === "rename") {
    state.pendingCVRename = cv.cv_id;
    state.pendingRename = null;
    els.renameDialogTitle.textContent = "Renombrar CV";
    els.renameDialogHint.textContent = "Este nombre se usa solamente dentro de Radar Laboral.";
    els.renameInput.value = cv.display_name;
    els.renameDialog.showModal();
    requestAnimationFrame(() => els.renameInput.select());
  } else if (action === "default") {
    try { await dataSource.updateCV(cv.cv_id, { is_default: true }); await loadCVs(); showToast("CV predeterminado actualizado."); } catch (error) { showToast(error.message); }
  } else if (action === "replace") {
    state.cvUploadTarget = cv.cv_id;
    els.cvFileInput.click();
  } else if (action === "save-text") {
    const textValue = els.cvWorkspace.querySelector("[data-cv-text]")?.value.trim();
    if (!textValue) return showToast("El texto del CV no puede estar vacío.");
    try { await dataSource.updateCV(cv.cv_id, { extracted_text: textValue, source_version_id: cv.current_version_id }); await loadCVs(); showToast("Nueva versión de texto guardada."); } catch (error) { showToast(error.message); }
  } else if (action === "delete") {
    showConfirmation({ title: "Eliminar CV", message: `¿Eliminar ${cv.display_name} y todas sus versiones locales?`, action: async () => {
      try { await dataSource.deleteCV(cv.cv_id); state.selectedCVId = null; await loadCVs(); showToast("CV eliminado."); } catch (error) { showToast(error.message); }
    } });
  } else if (action === "delete-version") {
    const versionId = button.dataset.versionId;
    showConfirmation({ title: "Eliminar versión", message: "El archivo original de esta versión también se eliminará.", action: async () => {
      try { await dataSource.deleteCVVersion(cv.cv_id, versionId); await loadCVs(); showToast("Versión eliminada."); } catch (error) { showToast(error.message); }
    } });
  }
}

async function submitCVReview(button) {
  const artifact = button.closest("[data-artifact-id]");
  const artifactId = artifact?.dataset.artifactId;
  if (!artifactId) return;
  const decisions = [];
  for (const item of artifact.querySelectorAll("[data-review-suggestion]")) {
    const decision = item.querySelector("input[type=radio]:checked")?.value;
    if (!decision) {
      showToast("Elegí aceptar, editar o rechazar cada sugerencia.");
      return;
    }
    decisions.push({
      suggestion_id: item.dataset.reviewSuggestion,
      decision,
      edited_text: item.querySelector("[data-review-text]")?.value || null,
      truth_confirmed: Boolean(item.querySelector("[data-truth-confirmed]")?.checked),
    });
  }
  await resumeActiveTask({
    response_type: "review",
    artifact_id: artifactId,
    review_decisions: decisions,
    draft_text: artifact.querySelector("[data-review-draft]")?.value || null,
    save_as_cv_version: button.dataset.submitReview === "version",
  });
}

async function initialize() {
  await Promise.all([loadConversations(), loadLegacyReports(), loadCVs()]);
  if (state.conversations.length) await openConversation(state.conversations[0].conversation_id);
  else renderActiveView();
}

els.newChatButton.addEventListener("click", createNewConversation);
els.brandButton.addEventListener("click", createNewConversation);
els.cvLibraryButton.addEventListener("click", openCVLibraryV3);
els.settingsButton.addEventListener("click", () => showToast("La configuración se conectará en una etapa posterior."));
els.openSidebar.addEventListener("click", () => {
  els.appShell.classList.add("sidebar-open");
  requestAnimationFrame(() => els.closeSidebar.focus());
});
els.closeSidebar.addEventListener("click", closeSidebar);
els.sidebarBackdrop.addEventListener("click", closeSidebar);

els.conversationSearch.addEventListener("input", () => {
  state.search = els.conversationSearch.value;
  clearTimeout(state.searchTimer);
  state.searchTimer = setTimeout(loadConversations, 250);
});

els.conversationNav.addEventListener("click", (event) => {
  const conversationButton = event.target.closest("[data-conversation-id]");
  const legacyButton = event.target.closest("[data-legacy-report-id]");
  const menuButton = event.target.closest("[data-conversation-menu]");
  if (conversationButton) openConversation(conversationButton.dataset.conversationId);
  else if (legacyButton) openLegacyReport(legacyButton.dataset.legacyReportId);
  else if (menuButton) {
    const conversation = state.conversations.find((item) => item.conversation_id === menuButton.dataset.conversationMenu);
    if (conversation) showRenameDialog(conversation.conversation_id, conversation.title);
  }
});

els.conversationMenuButton.addEventListener("click", (event) => {
  event.stopPropagation();
  toggleMenu(els.conversationMenu, els.conversationMenuButton);
});

els.conversationMenu.addEventListener("click", (event) => {
  const button = event.target.closest("[data-header-action]");
  if (!button || !state.activeConversation) return;
  closeMenus();
  const conversation = state.activeConversation;
  if (button.dataset.headerAction === "rename") showRenameDialog(conversation.conversation_id, conversation.title);
  if (button.dataset.headerAction === "delete") {
    showConfirmation({
      title: "Eliminar conversación",
      message: `¿Eliminar “${conversation.title}”? Los informes adjuntos no se eliminarán.`,
      action: () => deleteConversation(conversation.conversation_id),
    });
  }
});

els.attachmentButton.addEventListener("click", (event) => {
  event.stopPropagation();
  toggleMenu(els.attachmentMenu, els.attachmentButton);
});

els.attachmentMenu.addEventListener("click", (event) => {
  const button = event.target.closest("[data-attachment-action]");
  if (button && !button.disabled) handleAttachmentAction(button.dataset.attachmentAction);
});

els.attachmentChips.addEventListener("click", (event) => {
  const button = event.target.closest("[data-remove-attachment]");
  if (!button) return;
  const key = activeConversationId() || "new";
  state.attachments[key] = (state.attachments[key] || []).filter((item) => item.id !== button.dataset.removeAttachment);
  renderAttachmentChips(state.attachments[key]);
});

els.composerForm.addEventListener("submit", handleComposerSubmit);
els.composerText.addEventListener("input", () => {
  const key = activeConversationId() || "new";
  state.drafts[key] = els.composerText.value;
  resizeComposer();
});
els.composerText.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    els.composerForm.requestSubmit();
  }
});

els.conversationTranscript.addEventListener("click", (event) => {
  const reviewButton = event.target.closest("[data-submit-review]");
  if (reviewButton) {
    submitCVReview(reviewButton);
    return;
  }
  if (event.target.closest("[data-retry-task]")) {
    retryFailedTask();
    return;
  }
  const resume = event.target.closest("[data-task-resume]");
  if (resume) {
    if (resume.dataset.taskResume === "approval") {
      resumeActiveTask({ response_type: "approval", decision: resume.dataset.decision, content: null, selected_option_ids: [] });
      return;
    }
    if (resume.dataset.taskResume === "clarification") {
      resumeActiveTask({ response_type: "clarification", decision: null, content: null, selected_option_ids: [resume.dataset.optionId] });
      return;
    }
    const input = els.conversationTranscript.querySelector("[data-clarification-input]");
    const content = input?.value.trim();
    if (!content) {
      input?.focus();
      return;
    }
    resumeActiveTask({ response_type: "clarification", decision: null, content, selected_option_ids: [] });
    return;
  }
  const starter = event.target.closest("[data-starter-prompt]");
  if (!starter) return;
  els.composerText.value = starter.dataset.starterPrompt;
  const key = activeConversationId() || "new";
  state.drafts[key] = els.composerText.value;
  resizeComposer();
  els.composerText.focus();
});

els.cvWorkspace.addEventListener("click", (event) => {
  const button = event.target.closest("[data-cv-action]");
  if (button) handleCVWorkspaceAction(button);
});

els.cvFileInput.addEventListener("change", () => uploadSelectedCV(els.cvFileInput.files?.[0]));

els.confirmCVSelection.addEventListener("click", (event) => {
  event.preventDefault();
  const cv = state.cvs.find((item) => item.cv_id === els.cvSelectInput.value);
  if (!cv) return;
  attachStoredCV(cv);
  els.cvSelectDialog.close();
  requestAnimationFrame(() => els.composerText.focus());
});

els.jobDialogForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const content = els.jobDescriptionInput.value.trim();
  if (!content) {
    els.jobDescriptionInput.focus();
    return;
  }
  const key = activeConversationId() || "new";
  state.attachments[key] ||= [];
  state.attachments[key] = state.attachments[key].filter((item) => item.type !== "job_description");
  state.attachments[key].push({ id: `job-${Date.now()}`, type: "job_description", content, title: "Descripción de puesto", label: "Descripción de puesto" });
  renderAttachmentChips(state.attachments[key]);
  els.jobDialog.close();
  requestAnimationFrame(() => els.composerText.focus());
});

els.confirmRename.addEventListener("click", async (event) => {
  event.preventDefault();
  const title = els.renameInput.value.trim();
  const conversationId = state.pendingRename;
  const cvId = state.pendingCVRename;
  if (!title || (!conversationId && !cvId)) {
    els.renameInput.focus();
    return;
  }
  try {
    if (cvId) {
      await dataSource.updateCV(cvId, { display_name: title });
      state.pendingCVRename = null;
      els.renameDialog.close();
      await loadCVs();
      return;
    }
    await dataSource.renameConversation(conversationId, title);
    state.pendingRename = null;
    els.renameDialog.close();
    await loadConversations();
    if (activeConversationId() === conversationId) state.activeConversation.title = title;
    renderActiveView();
    requestAnimationFrame(() => els.conversationMenuButton.focus());
  } catch (error) {
    showToast(error.message);
  }
});

els.confirmAction.addEventListener("click", (event) => {
  event.preventDefault();
  const action = state.pendingConfirmation;
  state.pendingConfirmation = null;
  els.confirmDialog.close();
  action?.();
});

document.addEventListener("click", (event) => {
  if (!event.target.closest("#attachmentMenu") && !event.target.closest("#attachmentButton")) toggleMenu(els.attachmentMenu, els.attachmentButton, false);
  if (!event.target.closest("#conversationMenu") && !event.target.closest("#conversationMenuButton")) toggleMenu(els.conversationMenu, els.conversationMenuButton, false);
});

document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLocaleLowerCase("es") === "k") {
    event.preventDefault();
    if (window.innerWidth < 900) els.appShell.classList.add("sidebar-open");
    requestAnimationFrame(() => els.conversationSearch.focus());
  }
  if (event.key === "Escape") {
    closeMenus();
    closeSidebar();
  }
});

window.addEventListener("beforeunload", closeEventStream);

initialize();
