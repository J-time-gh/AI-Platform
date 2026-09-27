const state = {
  token: window.localStorage.getItem("eai_access_token"),
  user: null,
  documents: [],
  polling: new Set(),
  // 记录本次页面会话里用户展开/收起了哪些文档流程，避免每次状态轮询后重置折叠状态。
  documentFlowOpen: new Map(),
  authMode: "login",
  chatMessages: [],
  chatSessions: [],
  activeChatSessionId: null,
  chatPending: false,
  chatSkillUpdating: false,
  highlightEvidence: true,
};

const $ = (selector) => document.querySelector(selector);
const authView = $("#auth-view");
const appView = $("#app-view");
const authForm = $("#auth-form");
const authEmail = $("#auth-email");
const authPassword = $("#auth-password");
const authTitle = $("#auth-title");
const authSubtitle = $("#auth-subtitle");
const authSubmit = $("#auth-submit");
const authError = $("#auth-error");
const uploadForm = $("#upload-form");
const documentFile = $("#document-file");
const dropZone = $("#drop-zone");
const fileName = $("#file-name");
const uploadSubmit = $("#upload-submit");
const uploadMessage = $("#upload-message");
const documentsList = $("#documents-list");
const chatForm = $("#chat-form");
const chatInput = $("#chat-input");
const chatSkill = $("#chat-skill");
const chatModeDescription = $("#chat-mode-description");
const chatRetrievalModeControl = $("#chat-retrieval-mode-control");
const chatTopKControl = $("#chat-top-k-control");
const chatMode = $("#chat-mode");
const chatTopK = $("#chat-top-k");
const chatSubmit = $("#chat-submit");
const chatMessage = $("#chat-message");
const chatMessages = $("#chat-messages");
const chatSessionsList = $("#chat-sessions-list");
const newChatButton = $("#new-chat-button");
const evidenceHighlightToggle = $("#evidence-highlight-toggle");
const toastRegion = $("#toast-region");

const RAG_SKILL_ID = "knowledge_base_qa";

const STATUS_LABELS = {
  pending: "等待处理",
  processing: "处理中",
  ready: "已就绪",
  failed: "处理失败",
};

// 这组阶段名与后端 DocumentInfo.ingest_stage 一一对应，是流程图的前后端契约。
// 顺序决定节点先后；新增/改名节点时，请同时调整后端 IngestStage 和对应 Alembic 迁移。
const INGEST_FLOW_STEPS = [
  { stage: "upload_received", label: "API 接收文件并落盘" },
  { stage: "queued", label: "Redis / RQ 任务队列" },
  { stage: "worker_start", label: "Worker 独立进程领取任务" },
  { stage: "parsing", label: "解析 PDF / Word / Excel / 文本" },
  { stage: "chunking", label: "切分文本块" },
  { stage: "embedding", label: "生成 Embedding 向量" },
  { stage: "saving", label: "写入文本块和向量" },
  { stage: "ready", label: "入库完成，可检索" },
];

async function api(path, options = {}) {
  const { timeoutMs = 15000, ...requestOptions } = options;
  const headers = new Headers(requestOptions.headers || {});
  if (state.token) headers.set("Authorization", `Bearer ${state.token}`);
  if (requestOptions.body && !(requestOptions.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }

  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(path, { ...requestOptions, headers, signal: controller.signal });
    const text = await response.text();
    let data = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = { detail: text };
      }
    }

    if (!response.ok) {
      if (response.status === 401) {
        clearSession();
        showAuth();
      }
      const detail = data && typeof data.detail === "string" ? data.detail : `请求失败（${response.status}）`;
      throw new Error(detail);
    }
    return data;
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error(`请求超过 ${Math.round(timeoutMs / 1000)} 秒未完成；首次请求可能正在加载模型，请稍后重试。`);
    }
    throw error;
  } finally {
    window.clearTimeout(timeoutId);
  }
}

function clearSession() {
  state.token = null;
  state.user = null;
  window.localStorage.removeItem("eai_access_token");
}

function showAuth() {
  authView.classList.remove("hidden");
  appView.classList.add("hidden");
  authEmail.focus();
}

function showApp() {
  authView.classList.add("hidden");
  appView.classList.remove("hidden");
  $("#user-email").textContent = state.user?.email || "";
}

function setMessage(element, text = "", type = "") {
  element.textContent = text;
  element.className = `form-message ${type ? `${type}-message` : ""}`.trim();
}

function syncChatSkillSelection(activeSkillId) {
  chatSkill.value = activeSkillId || "";
  updateChatModePresentation();
}

function updateChatModePresentation() {
  const ragEnabled = chatSkill.value === RAG_SKILL_ID;
  const controlsBusy = state.chatPending || state.chatSkillUpdating;

  chatRetrievalModeControl.classList.toggle("hidden", !ragEnabled);
  chatTopKControl.classList.toggle("hidden", !ragEnabled);
  chatModeDescription.textContent = ragEnabled
    ? "RAG 检索会查找当前账号已完成入库的资料，并展示可用引用。"
    : "普通对话直接根据问题和历史回答，不检索知识库。";
  chatInput.placeholder = ragEnabled
    ? "例如：北京普通员工住宿上限是多少？"
    : "向 AI 助手提问…";
  chatSkill.disabled = controlsBusy;
  chatMode.disabled = !ragEnabled || controlsBusy;
  chatTopK.disabled = !ragEnabled || controlsBusy;
  newChatButton.disabled = controlsBusy;
  chatSessionsList.querySelectorAll("button").forEach((button) => {
    button.disabled = controlsBusy;
  });
  renderChatMessages();
}

function currentChatPendingLabel() {
  return chatSkill.value === RAG_SKILL_ID
    ? "正在检索资料并生成回答…"
    : "正在生成回答…";
}

function setButtonBusy(button, busy, busyText) {
  if (busy) {
    button.dataset.originalText = button.textContent;
    button.disabled = true;
    button.textContent = busyText;
  } else {
    button.disabled = false;
    button.textContent = button.dataset.originalText || button.textContent;
  }
}

function switchAuthMode(mode) {
  state.authMode = mode;
  document.querySelectorAll("[data-auth-tab]").forEach((tab) => {
    const active = tab.dataset.authTab === mode;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  const registering = mode === "register";
  authTitle.textContent = registering ? "创建工作台账号" : "登录工作台";
  authSubtitle.textContent = registering ? "注册后即可开始构建你的知识库。" : "使用你的企业账号继续。";
  authSubmit.textContent = registering ? "创建账号" : "登录";
  authPassword.autocomplete = registering ? "new-password" : "current-password";
  setMessage(authError);
}

async function login(email, password) {
  const result = await api("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
  state.token = result.access_token;
  window.localStorage.setItem("eai_access_token", state.token);
  state.user = await api("/auth/me");
  showApp();
  await Promise.all([loadDocuments(), loadChatSessions()]);
}

async function handleAuth(event) {
  event.preventDefault();
  const email = authEmail.value.trim();
  const password = authPassword.value;
  setMessage(authError);
  setButtonBusy(authSubmit, true, state.authMode === "register" ? "创建中…" : "登录中…");

  try {
    if (state.authMode === "register") {
      await api("/auth/register", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });
      await login(email, password);
      showToast("账号创建成功，欢迎进入工作台");
    } else {
      await login(email, password);
    }
    authForm.reset();
  } catch (error) {
    setMessage(authError, error.message, "error");
  } finally {
    setButtonBusy(authSubmit, false);
  }
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(value) {
  try {
    return new Intl.DateTimeFormat("zh-CN", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(new Date(value));
  } catch {
    return value;
  }
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  })[character]);
}

function renderDocuments() {
  if (!state.documents.length) {
    documentsList.innerHTML = '<div class="empty-state">还没有文档，先上传一份资料吧。</div>';
    return;
  }

  documentsList.innerHTML = state.documents.map((doc) => {
    const status = STATUS_LABELS[doc.status] || doc.status;
    const error = doc.ingest_error
      ? `<p class="document-error">${escapeHtml(doc.ingest_error)}</p>`
      : "";
    // 每个文档卡片持有自己的流程图，状态更新时由 upsertDocument 重新绘制。
    const flow = renderIngestFlow(doc);
    return `
      <article class="document-card">
        <div class="document-main">
          <strong class="document-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</strong>
          <div class="document-meta">
            <span>${formatBytes(doc.size)}</span>
            <span>${formatDate(doc.uploaded_at)}</span>
          </div>
        </div>
        <span class="document-status status-${escapeHtml(doc.status)}">${escapeHtml(status)}</span>
        ${error}
        ${flow}
        <button class="document-delete" type="button" data-delete-document="${escapeHtml(doc.id)}">删除文档</button>
      </article>
    `;
  }).join("");

  // 把 details 的展开状态记在页面状态中；文档每 2 秒轮询重绘时仍尊重用户选择。
  documentsList.querySelectorAll("details[data-document-flow]").forEach((flow) => {
    flow.addEventListener("toggle", () => {
      state.documentFlowOpen.set(flow.dataset.documentFlow, flow.open);
    });
  });
}

function upsertDocument(document) {
  const index = state.documents.findIndex((item) => item.id === document.id);
  if (index === -1) state.documents.unshift(document);
  else state.documents[index] = document;
  state.documents.sort((a, b) => new Date(b.uploaded_at) - new Date(a.uploaded_at));
  renderDocuments();
}

// 根据 API 返回的持久化阶段推导节点状态：即使浏览器轮询刚好跳过几个阶段，
// 也会把之前的节点画为已完成，而不是只显示“当前阶段”一个点。
function ingestNodeState(document, stepIndex, currentIndex) {
  if (document.status === "ready") return "complete";

  if (document.status === "failed") {
    // 迁移前已失败的旧文档没有阶段历史，只显示通用失败提示，不猜测失败位置。
    if (document.ingest_stage === "unknown") return "pending";
    if (stepIndex < currentIndex) return "complete";
    if (stepIndex === currentIndex) return "failed";
    return "pending";
  }

  if (stepIndex < currentIndex) return "complete";
  if (stepIndex === currentIndex) return "active";
  return "pending";
}

// 流程图节点模板：只渲染固定的阶段标签和状态，不把文档正文/原始内容放进 HTML。
function renderIngestNode(label, state, detail = "") {
  const icon = {
    complete: "✓",
    active: "●",
    pending: "○",
    failed: "!",
  }[state] || "○";
  const stateLabel = {
    complete: "已完成",
    active: "进行中",
    pending: "等待中",
    failed: "失败",
  }[state] || "等待中";

  return `
    <div class="ingest-flow-node is-${state}" role="listitem" aria-label="${escapeHtml(label)}：${stateLabel}">
      <span class="ingest-flow-icon" aria-hidden="true">${icon}</span>
      <span class="ingest-flow-copy">
        <strong>${escapeHtml(label)}</strong>
        ${detail ? `<small>${escapeHtml(detail)}</small>` : ""}
      </span>
    </div>
  `;
}

// 每份上传文档独立渲染一张实例化流程图：队列之后分成 API 返回和后台 Worker 两条支路。
// Worker 是常驻的独立进程，不是每次上传时才启动；当前节点来自数据库轮询结果。
function renderIngestFlow(document) {
  const currentIndex = INGEST_FLOW_STEPS.findIndex((step) => step.stage === document.ingest_stage);
  const progress = Math.max(0, Math.min(100, Number(document.ingest_progress) || 0));
  const rememberedOpen = state.documentFlowOpen.get(document.id);
  const flowIsOpen = rememberedOpen ?? ["pending", "processing"].includes(document.status);
  const currentStep = currentIndex >= 0 ? INGEST_FLOW_STEPS[currentIndex] : null;
  const heading = document.status === "ready"
    ? "全部入库步骤已完成"
    : document.status === "failed"
      ? `流程中断${currentStep ? `：${currentStep.label}` : "：阶段信息不可用"}`
      : `当前：${currentStep?.label || "等待 Worker 领取"}`;
  const entryNodes = INGEST_FLOW_STEPS.slice(0, 2).map((step, index) => {
    const state = ingestNodeState(document, index, currentIndex);
    return `${index ? '<span class="ingest-flow-arrow" aria-hidden="true">→</span>' : ""}${renderIngestNode(step.label, state)}`;
  }).join("");
  const workerNodes = INGEST_FLOW_STEPS.slice(2).map((step, offset) => {
    const index = offset + 2;
    return renderIngestNode(
      step.label,
      ingestNodeState(document, index, currentIndex),
    );
  }).join("");
  const unknownFailure = document.status === "failed" && document.ingest_stage === "unknown"
    ? '<p class="ingest-flow-unknown">这是升级前的失败任务，系统没有保存失败阶段。</p>'
    : "";
  const apiResponseDetail = document.status === "failed" && document.ingest_stage === "queued"
    ? "接口已返回 202，但任务投递失败"
    : "接口返回 202，处理转入后台";

  return `
    <details class="ingest-flow" data-document-flow="${escapeHtml(document.id)}" ${flowIsOpen ? "open" : ""}>
      <summary class="ingest-flow-summary" aria-label="${escapeHtml(document.filename)} 的文档入库流程">
        <span class="ingest-flow-heading">
          <strong>本次入库流程 · 阶段估算</strong>
          <span>${escapeHtml(heading)}</span>
        </span>
        <span class="ingest-flow-chevron" aria-hidden="true">⌄</span>
      </summary>
      <div class="ingest-flow-content">
        <div class="ingest-progress-track" role="progressbar" aria-label="文档入库阶段进度估算"
          aria-valuemin="0" aria-valuemax="100" aria-valuenow="${progress}">
          <span style="width: ${progress}%"></span>
        </div>
        <div class="ingest-flow-entry" role="list" aria-label="上传 API 与任务队列">
          ${entryNodes}
        </div>
        <div class="ingest-flow-branches" aria-label="队列提交后的处理分支">
          <section class="ingest-flow-branch api-branch">
            <h4>上传 API 分支</h4>
            ${renderIngestNode("返回 202 Accepted", "complete", apiResponseDetail)}
          </section>
          <section class="ingest-flow-branch worker-branch">
            <h4>后台 Worker 分支 · 独立进程</h4>
            <div class="ingest-worker-steps" role="list" aria-label="Worker 处理步骤">
              ${workerNodes}
            </div>
            ${unknownFailure}
          </section>
        </div>
      </div>
    </details>
  `;
}

function sleep(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function watchDocument(documentId) {
  if (state.polling.has(documentId)) return;
  state.polling.add(documentId);
  try {
    for (let attempt = 0; attempt < 90; attempt += 1) {
      await sleep(2000);
      const document = await api(`/documents/${encodeURIComponent(documentId)}`);
      upsertDocument(document);
      if (["ready", "failed"].includes(document.status)) {
        showToast(document.status === "ready" ? `${document.filename} 已完成入库` : `${document.filename} 入库失败`, document.status === "failed" ? "error" : "");
        return;
      }
    }
    showToast("文档处理时间较长，请稍后刷新查看", "error");
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    state.polling.delete(documentId);
  }
}

async function loadDocuments() {
  documentsList.innerHTML = '<div class="loading-state">正在加载文档…</div>';
  try {
    state.documents = await api("/documents");
    renderDocuments();
    state.documents.filter((doc) => ["pending", "processing"].includes(doc.status)).forEach((doc) => watchDocument(doc.id));
  } catch (error) {
    documentsList.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  }
}

function renderChatSessions() {
  if (!state.chatSessions.length) {
    chatSessionsList.innerHTML = '<span class="session-empty">还没有历史对话</span>';
    return;
  }

  chatSessionsList.innerHTML = state.chatSessions.map((session) => `
    <div class="chat-session-item ${session.id === state.activeChatSessionId ? "active" : ""}">
      <button class="chat-session-select" type="button" data-select-chat-session="${escapeHtml(session.id)}" title="${escapeHtml(session.title || "新对话")}">
        <span>${escapeHtml(session.title || "新对话")}</span>
        <small>${formatDate(session.updated_at)}</small>
      </button>
      <button class="chat-session-delete" type="button" data-delete-chat-session="${escapeHtml(session.id)}" aria-label="删除对话" title="删除对话">×</button>
    </div>
  `).join("");
}

async function loadChatSessions(selectLatest = true) {
  chatSessionsList.innerHTML = '<span class="session-empty">正在加载对话…</span>';
  try {
    state.chatSessions = await api("/chat/sessions");
    if (selectLatest && state.chatSessions.length) {
      await selectChatSession(state.chatSessions[0].id);
    } else {
      renderChatSessions();
    }
  } catch (error) {
    chatSessionsList.innerHTML = `<span class="session-empty">${escapeHtml(error.message)}</span>`;
  }
}

async function selectChatSession(sessionId) {
  if (state.chatPending || state.chatSkillUpdating) return;
  state.activeChatSessionId = sessionId;
  const session = state.chatSessions.find((item) => item.id === sessionId);
  state.chatMessages = [];
  syncChatSkillSelection(session?.active_skill_id);
  state.chatPending = false;
  renderChatSessions();
  chatMessages.innerHTML = '<div class="loading-state">正在加载聊天记录…</div>';

  try {
    const messages = await api(`/chat/sessions/${encodeURIComponent(sessionId)}/messages`);
    if (state.activeChatSessionId !== sessionId) return;
    state.chatMessages = messages.map((message) => ({
      id: message.id,
      role: message.role,
      content: message.content,
      sources: Array.isArray(message.sources) ? message.sources : [],
      evidenceOpen: Array.isArray(message.sources) && message.sources.length > 0,
    }));
    renderChatMessages();
  } catch (error) {
    if (state.activeChatSessionId !== sessionId) return;
    setMessage(chatMessage, error.message, "error");
    renderChatMessages();
  }
}

function startNewChat() {
  if (state.chatPending || state.chatSkillUpdating) return;
  state.activeChatSessionId = null;
  state.chatMessages = [];
  state.chatPending = false;
  renderChatSessions();
  renderChatMessages();
  setMessage(chatMessage);
  chatInput.focus();
}

async function persistChatSessionSkill(sessionId, activeSkillId) {
  const session = await api(`/chat/sessions/${encodeURIComponent(sessionId)}/skill`, {
    method: "PATCH",
    body: JSON.stringify({ active_skill_id: activeSkillId }),
  });
  const sessionIndex = state.chatSessions.findIndex((item) => item.id === sessionId);
  if (sessionIndex >= 0) {
    state.chatSessions[sessionIndex] = session;
  } else {
    state.chatSessions.unshift(session);
  }
  renderChatSessions();
  return session;
}

async function handleChatSkillChange() {
  if (state.chatPending || state.chatSkillUpdating) return;

  const selectedSkillId = chatSkill.value || null;
  const sessionId = state.activeChatSessionId;
  const currentSession = state.chatSessions.find((item) => item.id === sessionId);
  const previousSkillId = currentSession?.active_skill_id ?? null;
  updateChatModePresentation();

  if (!sessionId || (currentSession && selectedSkillId === previousSkillId)) return;

  state.chatSkillUpdating = true;
  updateChatModePresentation();
  try {
    const updatedSession = await persistChatSessionSkill(sessionId, selectedSkillId);
    syncChatSkillSelection(updatedSession.active_skill_id);
    setMessage(
      chatMessage,
      selectedSkillId ? "本会话已切换为 RAG 检索。" : "本会话已切换为普通对话。",
      "success",
    );
  } catch (error) {
    syncChatSkillSelection(previousSkillId);
    setMessage(chatMessage, error.message, "error");
    showToast(error.message, "error");
  } finally {
    state.chatSkillUpdating = false;
    updateChatModePresentation();
  }
}

async function createChatSession(title) {
  const session = await api("/chat/sessions", {
    method: "POST",
    body: JSON.stringify({ title: title.slice(0, 80) }),
  });
  state.chatSessions.unshift(session);
  state.activeChatSessionId = session.id;
  renderChatSessions();

  const selectedSkillId = chatSkill.value || null;
  if (selectedSkillId) {
    try {
      const updatedSession = await persistChatSessionSkill(session.id, selectedSkillId);
      syncChatSkillSelection(updatedSession.active_skill_id);
    } catch (error) {
      syncChatSkillSelection(session.active_skill_id);
      throw error;
    }
  }

  return session.id;
}

async function deleteChatSession(sessionId) {
  if (state.chatPending || state.chatSkillUpdating) return;
  const session = state.chatSessions.find((item) => item.id === sessionId);
  if (!session || !window.confirm(`确认删除对话“${session.title || "新对话"}”及其聊天记录吗？`)) return;
  try {
    await api(`/chat/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
    state.chatSessions = state.chatSessions.filter((item) => item.id !== sessionId);
    if (state.activeChatSessionId === sessionId) {
      state.activeChatSessionId = null;
      state.chatMessages = [];
      renderChatMessages();
    }
    renderChatSessions();
    showToast("对话及聊天记录已删除");
  } catch (error) {
    showToast(error.message, "error");
  }
}

async function handleUpload(event) {
  event.preventDefault();
  const file = documentFile.files[0];
  if (!file) {
    setMessage(uploadMessage, "请先选择一个文件。", "error");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  setMessage(uploadMessage, "文件已发送，正在创建后台任务…");
  setButtonBusy(uploadSubmit, true, "提交中…");

  try {
    const document = await api("/documents/upload", { method: "POST", body: formData });
    upsertDocument(document);
    documentFile.value = "";
    fileName.textContent = "选择一个文件";
    setMessage(uploadMessage, "已进入队列，状态会自动更新。", "success");
    watchDocument(document.id);
  } catch (error) {
    setMessage(uploadMessage, error.message, "error");
  } finally {
    setButtonBusy(uploadSubmit, false);
  }
}

async function handleDelete(documentId) {
  const document = state.documents.find((item) => item.id === documentId);
  if (!document || !window.confirm(`确认删除“${document.filename}”吗？`)) return;
  try {
    await api(`/documents/${encodeURIComponent(documentId)}`, { method: "DELETE" });
    state.documents = state.documents.filter((item) => item.id !== documentId);
    renderDocuments();
    showToast("文档已删除");
  } catch (error) {
    showToast(error.message, "error");
  }
}

function highlightContent(content, highlightTerms) {
  const terms = [...new Set(
    (Array.isArray(highlightTerms) ? highlightTerms : [])
      .filter((term) => typeof term === "string" && term.length > 0),
  )].sort((left, right) => right.length - left.length);

  if (!terms.length) return escapeHtml(content);

  const expression = new RegExp(`(${terms.map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "gi");
  return String(content).split(expression).map((part) => {
    if (terms.some((term) => term.toLocaleLowerCase() === part.toLocaleLowerCase())) {
      return `<mark>${escapeHtml(part)}</mark>`;
    }
    return escapeHtml(part);
  }).join("");
}

function newMessageId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function renderEvidence(message) {
  const sources = Array.isArray(message.sources) ? message.sources : [];
  if (!sources.length) return "";

  const buttonLabel = message.evidenceOpen
    ? "收起检索片段 ▴"
    : `查看 ${sources.length} 条检索片段 ▾`;
  const cards = message.evidenceOpen ? `
    <div class="evidence-drawer">
      ${sources.map((source) => `
        <article class="evidence-card">
          <div class="result-top">
            <span class="result-file" title="${escapeHtml(source.filename)}">${escapeHtml(source.filename)}</span>
            <span class="result-score">相关度 ${(Number(source.score) || 0).toFixed(3)}</span>
          </div>
          <p class="result-content">${highlightContent(
            source.content,
            state.highlightEvidence ? source.highlight_terms : [],
          )}</p>
          <div class="result-foot">第 ${Number(source.chunk_index) + 1} 段 · 文档 ID ${escapeHtml(source.document_id)}</div>
        </article>
      `).join("")}
    </div>
  ` : "";

  return `
    <div class="evidence-block">
      <button class="evidence-toggle" type="button" data-toggle-evidence="${escapeHtml(message.id)}" aria-expanded="${message.evidenceOpen}">${buttonLabel}</button>
      ${cards}
    </div>
  `;
}

// 把后端 SSE 发来的真实 Agent/API 节点渲染为聊天消息下的可折叠流程时间线。
// 流程仅展示程序阶段和安全摘要，不展示模型隐藏推理、提示词或工具原始内容。
function renderChatFlow(message) {
  if (!Array.isArray(message.flowSteps) || !message.flowSteps.length) return "";

  const activeStep = [...message.flowSteps].reverse().find((step) => step.status === "active");
  const summary = message.flowStatus === "complete"
    ? "已完成"
    : message.flowStatus === "failed"
      ? "流程未完成"
      : activeStep?.label || "执行中";
  const steps = message.flowSteps.map((step) => {
    const icon = step.status === "complete" ? "✓" : step.status === "failed" ? "!" : "●";
    const detail = step.detail ? `<small>${escapeHtml(step.detail)}</small>` : "";
    return `
      <li class="chat-flow-step is-${escapeHtml(step.status)}">
        <span class="chat-flow-icon" aria-hidden="true">${icon}</span>
        <span class="chat-flow-copy">
          <strong>${escapeHtml(step.label)}</strong>
          ${detail}
        </span>
      </li>
    `;
  }).join("");

  return `
    <details class="chat-flow" data-chat-flow="${escapeHtml(message.id)}" ${message.flowOpen ? "open" : ""}>
      <summary class="chat-flow-summary">
        <strong>本轮问答流程</strong>
        <span>${escapeHtml(summary)}</span>
        <span class="chat-flow-chevron" aria-hidden="true">⌄</span>
      </summary>
      <ol class="chat-flow-steps" aria-label="本轮问答执行节点">${steps}</ol>
    </details>
  `;
}

function updateChatFlow(messageId, event) {
  const message = state.chatMessages.find((item) => item.id === messageId);
  if (!message || message.role !== "assistant" || event.kind !== "flow") return;

  message.flowSteps ||= [];
  if (event.id === "chat-failure") {
    message.flowSteps.forEach((step) => {
      if (step.status === "active") step.status = "failed";
    });
    message.flowStatus = "failed";
  }

  const step = {
    id: event.id,
    status: event.status,
    label: event.label,
    detail: event.detail || "",
  };
  const existingIndex = message.flowSteps.findIndex((item) => item.id === step.id);
  if (existingIndex === -1) message.flowSteps.push(step);
  else message.flowSteps[existingIndex] = step;
  renderChatMessages();
}

function renderChatMessages() {
  if (!state.chatMessages.length && !state.chatPending) {
    const emptyMessage = chatSkill.value === RAG_SKILL_ID
      ? "提出问题，我会先检索当前账号已完成入库的资料，再依据资料回答。"
      : "直接向 AI 助手提问，可进行通用对话。";
    chatMessages.innerHTML = `<div class="empty-chat"><span>✦</span><p>${emptyMessage}</p></div>`;
    return;
  }

  const messageMarkup = state.chatMessages.map((message) => {
    const isAssistant = message.role === "assistant";
    const metadata = isAssistant && message.model && message.model !== "none"
      ? `<span class="chat-model">${escapeHtml(message.model)}</span>`
      : "";
    return `
      <article class="chat-turn chat-turn-${escapeHtml(message.role)}">
        <div class="chat-turn-label">${isAssistant ? "AI 助手" : "你"}${metadata}</div>
        ${message.content ? `<div class="chat-bubble">${escapeHtml(message.content)}</div>` : ""}
        ${isAssistant ? renderChatFlow(message) : ""}
        ${isAssistant ? renderEvidence(message) : ""}
      </article>
    `;
  }).join("");

  const pendingMarkup = state.chatPending
    ? `<div class="chat-pending"><span></span><span></span><span></span>${currentChatPendingLabel()}</div>`
    : "";
  chatMessages.innerHTML = `${messageMarkup}${pendingMarkup}`;
  // 追踪用户收起/展开的选择，聊天节点每次到达都会重绘消息列表。
  chatMessages.querySelectorAll("details[data-chat-flow]").forEach((flow) => {
    flow.addEventListener("toggle", () => {
      const message = state.chatMessages.find((item) => item.id === flow.dataset.chatFlow);
      if (message) message.flowOpen = flow.open;
    });
  });
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function toggleEvidence(messageId) {
  const message = state.chatMessages.find((item) => item.id === messageId);
  if (!message || message.role !== "assistant" || !message.sources?.length) return;
  message.evidenceOpen = !message.evidenceOpen;
  renderChatMessages();
}

// 使用 fetch + ReadableStream 消费 POST SSE：原生 EventSource 不能携带此页面的 Bearer Token。
async function streamChat(payload, onFlowEvent) {
  const headers = new Headers({
    "Content-Type": "application/json",
    Accept: "text/event-stream",
  });
  if (state.token) headers.set("Authorization", `Bearer ${state.token}`);

  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), 180000);
  let reader = null;
  let finalResponse = null;
  let buffer = "";
  const decoder = new TextDecoder();

  const dispatchBlock = (block) => {
    const data = block.split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).replace(/^ /, ""))
      .join("\n");
    if (!data) return;

    const event = JSON.parse(data);
    if (event.kind === "flow") onFlowEvent(event);
    else if (event.kind === "complete") finalResponse = event.response;
    else if (event.kind === "error") throw new Error(event.detail || "聊天处理失败");
  };

  try {
    const response = await fetch("/chat/stream", {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!response.ok) {
      const text = await response.text();
      let data = null;
      try { data = text ? JSON.parse(text) : null; } catch { data = null; }
      if (response.status === 401) {
        clearSession();
        showAuth();
      }
      throw new Error(data?.detail || `请求失败（${response.status}）`);
    }
    if (!response.body) throw new Error("浏览器不支持读取实时聊天事件");

    reader = response.body.getReader();
    while (true) {
      const { value, done } = await reader.read();
      if (value) buffer += decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, "\n");

      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        dispatchBlock(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary + 2);
        boundary = buffer.indexOf("\n\n");
      }

      if (done) {
        buffer += decoder.decode();
        if (buffer.trim()) dispatchBlock(buffer);
        break;
      }
    }

    if (!finalResponse) throw new Error("聊天连接结束，但没有收到最终回答");
    return finalResponse;
  } catch (error) {
    if (reader) await reader.cancel().catch(() => {});
    if (error.name === "AbortError") {
      throw new Error("聊天处理超过 180 秒；模型或检索服务可能仍在运行，请稍后检查状态。");
    }
    throw error;
  } finally {
    window.clearTimeout(timeoutId);
  }
}

async function handleChat(event) {
  event.preventDefault();
  const content = chatInput.value.trim();
  if (!content || state.chatPending) return;

  const userMessageId = newMessageId();
  const ragEnabled = chatSkill.value === RAG_SKILL_ID;
  state.chatPending = true;
  chatInput.value = "";
  updateChatModePresentation();
  setMessage(chatMessage, currentChatPendingLabel());
  setButtonBusy(chatSubmit, true, ragEnabled ? "检索并回答…" : "回答中…");

  let traceMessage = null;
  try {
    const sessionId = state.activeChatSessionId || await createChatSession(content);
    state.chatMessages.push({ id: userMessageId, role: "user", content });
    traceMessage = {
      id: newMessageId(),
      role: "assistant",
      content: "",
      model: null,
      sources: [],
      evidenceOpen: false,
      flowSteps: [],
      flowStatus: "active",
      flowOpen: true,
    };
    state.chatMessages.push(traceMessage);
    renderChatMessages();

    const response = await streamChat(
      {
        message: content,
        session_id: sessionId,
        top_k: Number(chatTopK.value),
        mode: chatMode.value,
      },
      (flowEvent) => updateChatFlow(traceMessage.id, flowEvent),
    );
    const sources = Array.isArray(response.sources) ? response.sources : [];
    traceMessage.content = response.reply;
    traceMessage.model = response.model;
    traceMessage.sources = sources;
    traceMessage.evidenceOpen = ragEnabled && sources.length > 0;
    traceMessage.flowStatus = "complete";
    renderChatMessages();

    if (ragEnabled) {
      setMessage(
        chatMessage,
        sources.length
          ? `回答已生成，包含 ${sources.length} 条检索依据。`
          : "回答已生成，但本轮没有可展示的检索依据。",
        sources.length ? "success" : "",
      );
    } else {
      setMessage(chatMessage, "回答已生成。", "success");
    }
    await loadChatSessions(false);
  } catch (error) {
    const detail = error.message.includes("503") || error.message.includes("LLM")
      ? "模型服务不可用，请确认 LM Studio 已启动且已加载模型。"
      : error.message;
    if (traceMessage) {
      traceMessage.flowStatus = "failed";
      traceMessage.content = `本轮处理未完成：${detail}`;
      traceMessage.flowSteps.forEach((step) => {
        if (step.status === "active") step.status = "failed";
      });
      if (!traceMessage.flowSteps.some((step) => step.id === "chat-failure")) {
        traceMessage.flowSteps.push({
          id: "chat-failure",
          status: "failed",
          label: "处理失败",
          detail,
        });
      }
      renderChatMessages();
    } else {
      state.chatMessages = state.chatMessages.filter((message) => message.id !== userMessageId);
    }
    setMessage(chatMessage, detail, "error");
    showToast(detail, "error");
  } finally {
    state.chatPending = false;
    setButtonBusy(chatSubmit, false);
    updateChatModePresentation();
    chatInput.focus();
  }
}

function showToast(message, type = "") {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`.trim();
  toast.textContent = message;
  toastRegion.appendChild(toast);
  window.setTimeout(() => toast.remove(), 3600);
}

function bindDropZone() {
  ["dragenter", "dragover"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  }));
  ["dragleave", "drop"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  }));
  dropZone.addEventListener("drop", (event) => {
    const [file] = event.dataTransfer.files;
    if (!file) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    documentFile.files = transfer.files;
    fileName.textContent = file.name;
  });
}

async function bootstrap() {
  document.querySelectorAll("[data-auth-tab]").forEach((tab) => tab.addEventListener("click", () => switchAuthMode(tab.dataset.authTab)));
  authForm.addEventListener("submit", handleAuth);
  uploadForm.addEventListener("submit", handleUpload);
  chatForm.addEventListener("submit", handleChat);
  chatSkill.addEventListener("change", handleChatSkillChange);
  updateChatModePresentation();
  chatInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      chatForm.requestSubmit();
    }
  });
  chatMessages.addEventListener("click", (event) => {
    const button = event.target.closest("[data-toggle-evidence]");
    if (button) toggleEvidence(button.dataset.toggleEvidence);
  });
  newChatButton.addEventListener("click", startNewChat);
  chatSessionsList.addEventListener("click", (event) => {
    const deleteButton = event.target.closest("[data-delete-chat-session]");
    if (deleteButton) {
      deleteChatSession(deleteButton.dataset.deleteChatSession);
      return;
    }
    const selectButton = event.target.closest("[data-select-chat-session]");
    if (selectButton) selectChatSession(selectButton.dataset.selectChatSession);
  });
  evidenceHighlightToggle.addEventListener("change", () => {
    state.highlightEvidence = evidenceHighlightToggle.checked;
    renderChatMessages();
  });
  documentFile.addEventListener("change", () => {
    fileName.textContent = documentFile.files[0]?.name || "选择一个文件";
  });
  $("#refresh-documents").addEventListener("click", loadDocuments);
  documentsList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-delete-document]");
    if (button) handleDelete(button.dataset.deleteDocument);
  });
  $("#logout-button").addEventListener("click", () => {
    clearSession();
    state.documents = [];
    state.chatSessions = [];
    state.activeChatSessionId = null;
    state.chatMessages = [];
    state.chatPending = false;
    state.chatSkillUpdating = false;
    syncChatSkillSelection(null);
    renderChatSessions();
    showAuth();
    showToast("已退出登录");
  });
  bindDropZone();

  if (!state.token) {
    showAuth();
    return;
  }

  try {
    state.user = await api("/auth/me");
    showApp();
    await Promise.all([loadDocuments(), loadChatSessions()]);
  } catch {
    showAuth();
  }
}

bootstrap();
