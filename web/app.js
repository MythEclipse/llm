// Chat UI: sidebar of conversations, transcript in the middle, telemetry on the
// right. Plain ES modules, no build step — the gateway serves this directory.

import { render as renderMarkdown } from "/markdown.js";

const el = (id) => document.getElementById(id);

const state = {
  conversations: [],
  activeId: null,
  messages: [],
  streaming: false,
  abort: null,
  model: null,
};

const SUGGESTIONS = [
  ["Jelaskan CUDA graph", "kenapa satu replay per token lebih murah dari 169 peluncuran"],
  ["Ringkas arsitektur repo ini", "engine, api, gateway, cli — siapa memiliki apa"],
  ["Tulis tes Rust", "untuk perencana memori yang tidak boleh menumpuk buffer hidup"],
  ["Bandingkan Q8_0 dan Q4_K", "ukuran, akurasi, dan berapa bandwidth yang dihemat"],
];

// ------------------------------------------------------------------- plumbing

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "content-type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ error: response.statusText }));
    throw new Error(body.error ?? `HTTP ${response.status}`);
  }
  return response.status === 204 ? null : response.json();
}

function toast(message) {
  const node = el("toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => {
    node.hidden = true;
  }, 1600);
}

function formatBytes(bytes) {
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value < 10 && unit > 0 ? 1 : 0)} ${units[unit]}`;
}

function shortHash(hash) {
  return typeof hash === "string" ? hash.slice(0, 12) : "–";
}

// ---------------------------------------------------------------------- model

async function loadModel() {
  const info = await api("/api/model");
  state.model = info;
  el("brand-name").textContent = info.name;
  el("chat-meta").textContent = `${info.layers} layer · ${info.embedding}d · ctx ${info.context} · ${
    info.device ?? `${info.threads} thread`
  }`;

  el("model-info").innerHTML = [
    ["arsitektur", info.architecture],
    ["layer", info.layers],
    ["embedding", info.embedding],
    ["ffn", info.feed_forward],
    ["head q/kv", `${info.heads}/${info.kv_heads}`],
    ["vocab", info.vocab.toLocaleString("id-ID")],
    ["tensor", `${info.quantized_tensors} Q8_0 / ${info.tensors}`],
    ["bobot", formatBytes(info.weight_bytes)],
    ["KV cache", formatBytes(info.kv_bytes)],
    ["perangkat", info.device ?? "cpu"],
  ]
    .map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`)
    .join("");

  const build = info.build;
  const plan = info.plan;
  const rows = [
    ["arch", build.arch],
    ["nvcc", build.nvcc],
    ["kernel", build.kernels],
    ["sumber", shortHash(build.source_sha256)],
    ["ptx", shortHash(build.ptx_sha256)],
    ["build", shortHash(build.identity)],
    ["decode", plan.decode],
    ["prefill", plan.prefill],
    ["rencana", shortHash(plan.identity)],
    ["scratch", formatBytes(plan.scratch_bytes)],
  ];
  for (const [regime, reason] of plan.fallbacks ?? []) {
    rows.push([`fallback ${regime}`, reason]);
  }
  el("build-info").innerHTML = rows
    .map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`)
    .join("");
}

// -------------------------------------------------------------- conversations

async function loadConversations() {
  state.conversations = await api("/api/conversations");
  renderConversations();
}

function renderConversations() {
  const list = el("conversation-list");
  list.innerHTML = "";
  if (state.conversations.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "Belum ada percakapan.";
    list.append(empty);
    return;
  }
  for (const conversation of state.conversations) {
    const item = document.createElement("div");
    item.className = "conversation";
    item.classList.toggle("active", conversation.id === state.activeId);

    const open = document.createElement("button");
    open.type = "button";
    open.className = "title";
    open.textContent = conversation.title;
    open.title = conversation.title;
    open.addEventListener("click", () => openConversation(conversation.id));

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "remove";
    remove.textContent = "×";
    remove.title = "Hapus percakapan";
    remove.setAttribute("aria-label", `Hapus ${conversation.title}`);
    remove.addEventListener("click", () => deleteConversation(conversation.id));

    item.append(open, remove);
    list.append(item);
  }
}

async function openConversation(id) {
  state.activeId = id;
  const detail = await api(`/api/conversations/${id}`);
  state.messages = detail.messages;
  const title = detail.conversation.title || "Percakapan baru";
  el("chat-title").textContent = title;
  renderConversations();
  renderTranscript();
  closeDrawer();
}

async function deleteConversation(id) {
  await api(`/api/conversations/${id}`, { method: "DELETE" });
  if (state.activeId === id) {
    state.activeId = null;
    state.messages = [];
    el("chat-title").textContent = "Percakapan baru";
    renderTranscript();
  }
  await loadConversations();
}

async function createConversation() {
  const conversation = await api("/api/conversations", {
    method: "POST",
    body: JSON.stringify({ title: "" }),
  });
  await loadConversations();
  await openConversation(conversation.id);
  el("prompt").focus();
  return conversation;
}

function startNewChat() {
  state.activeId = null;
  state.messages = [];
  el("chat-title").textContent = "Percakapan baru";
  renderTranscript();
  el("prompt").focus();
  closeDrawer();
}

// ------------------------------------------------------------------ transcript

function thread() {
  return el("transcript").querySelector(".thread");
}

function renderTranscript() {
  const transcript = el("transcript");
  transcript.innerHTML = "";
  const container = document.createElement("div");
  container.className = "thread";
  transcript.append(container);

  if (!state.activeId || state.messages.length === 0) {
    container.append(hero());
    return;
  }
  for (const message of state.messages) {
    container.append(messageNode(message.role, message.content));
  }
  scrollToBottom();
}

function hero() {
  const node = document.createElement("section");
  node.className = "hero";
  const name = state.model?.name ?? "model lokal";
  const title = document.createElement("h2");
  title.textContent = `Halo. Mau dibahas apa?`;
  const lead = document.createElement("p");
  lead.textContent = `${name} berjalan di mesin ini — tidak ada permintaan yang keluar. Pilih satu contoh atau tulis sendiri.`;

  const grid = document.createElement("div");
  grid.className = "suggestions";
  for (const [label, hint] of SUGGESTIONS) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "suggestion";
    button.innerHTML = `<b></b><span></span>`;
    button.querySelector("b").textContent = label;
    button.querySelector("span").textContent = hint;
    button.addEventListener("click", () => {
      el("prompt").value = label;
      autoGrow();
      el("prompt").focus();
    });
    grid.append(button);
  }
  node.append(title, lead, grid);
  return node;
}

function messageNode(role, content) {
  const node = document.createElement("article");
  node.className = `message ${role}`;

  // Only the model gets a mark: a user badge is one more thing to read for
  // nothing, and ChatGPT/Gemini do not have one either.
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "AI";
  avatar.setAttribute("aria-hidden", "true");

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (role === "user") {
    bubble.textContent = content;
  } else {
    bubble.append(renderMarkdown(content));
  }

  const actions = document.createElement("div");
  actions.className = "message-actions";
  const copy = document.createElement("button");
  copy.type = "button";
  copy.textContent = "salin";
  copy.addEventListener("click", async () => {
    await copyText(node.dataset.raw ?? bubble.innerText);
    toast("tersalin");
  });
  actions.append(copy);

  const body = document.createElement("div");
  body.className = "body";
  body.append(bubble, actions);
  if (role === "user") {
    node.append(body);
  } else {
    node.append(avatar, body);
  }
  node.dataset.raw = content;
  return node;
}

function appendMessage(role, content) {
  const container = thread();
  if (container.querySelector(".hero")) {
    container.innerHTML = "";
  }
  const node = messageNode(role, content);
  container.append(node);
  scrollToBottom();
  return node;
}

function appendStreamingMessage() {
  const node = appendMessage("assistant", "");
  const bubble = node.querySelector(".bubble");
  bubble.innerHTML = `<span class="caret"></span>`;
  node.dataset.streaming = "true";
  return { node, bubble };
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    toast("salin gagal");
  }
}

function scrollToBottom() {
  const transcript = el("transcript");
  transcript.scrollTop = transcript.scrollHeight;
}

// --------------------------------------------------------------------- sending

async function send(event) {
  event.preventDefault();
  const prompt = el("prompt").value.trim();
  if (!prompt || state.streaming) return;

  if (!state.activeId) {
    await createConversation();
  }
  const conversationId = state.activeId;

  state.streaming = true;
  el("send").disabled = true;
  el("send").hidden = true;
  el("stop").hidden = false;
  el("live-pill").hidden = false;
  el("live-tps").textContent = "–";
  el("prompt").value = "";
  autoGrow();

  appendMessage("user", prompt);
  state.messages.push({ role: "user", content: prompt });
  const streamed = appendStreamingMessage();
  state.abort = new AbortController();
  let text = "";

  const paint = () => {
    streamed.bubble.innerHTML = "";
    streamed.bubble.append(renderMarkdown(text));
    const caret = document.createElement("span");
    caret.className = "caret";
    streamed.bubble.append(caret);
  };
  let frame = 0;
  const schedulePaint = () => {
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      paint();
      scrollToBottom();
    });
  };

  try {
    const response = await fetch(`/api/conversations/${conversationId}/messages`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      signal: state.abort.signal,
      body: JSON.stringify({
        content: prompt,
        temperature: Number(el("temperature").value),
        max_tokens: Number(el("max-tokens").value),
      }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.error ?? `HTTP ${response.status}`);
    }
    await consumeEvents(response, (name, payload) => {
      if (name === "start") {
        // History is trimmed to fit the window; say so instead of letting the
        // model look like it forgot on its own.
        const dropped = payload.dropped_turns ?? 0;
        el("run-tokens").textContent =
          dropped > 0
            ? `${payload.prompt_tokens} prompt · ${dropped} pesan lama dibuang`
            : `${payload.prompt_tokens} prompt`;
      } else if (name === "token") {
        text += payload.text;
        streamed.node.dataset.raw = text;
        schedulePaint();
      } else if (name === "stats") {
        applyRunStats(payload);
      } else if (name === "error") {
        text += `\n\n> error: ${payload.message}`;
        schedulePaint();
      }
    });
    streamed.node.classList.remove("error");
  } catch (error) {
    if (error.name === "AbortError") {
      text += text ? "\n\n_dihentikan_" : "_dihentikan_";
    } else {
      streamed.node.classList.add("error");
      text += text ? `\n\n> ${error.message}` : error.message;
    }
    paint();
  } finally {
    if (frame) cancelAnimationFrame(frame);
    paint();
    streamed.bubble.querySelector(".caret")?.remove();
    streamed.node.dataset.streaming = "";
    state.streaming = false;
    el("send").disabled = false;
    el("send").hidden = false;
    el("stop").hidden = true;
    state.abort = null;
    await loadConversations();
    if (state.activeId === conversationId) {
      await openConversation(conversationId);
    }
    await loadHistory();
  }
}

async function consumeEvents(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const chunk = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      dispatch(chunk, onEvent);
      boundary = buffer.indexOf("\n\n");
    }
  }
}

function dispatch(chunk, onEvent) {
  let name = "message";
  const data = [];
  for (const line of chunk.split("\n")) {
    if (line.startsWith("event:")) name = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trim());
  }
  if (data.length === 0) return;
  try {
    onEvent(name, JSON.parse(data.join("\n")));
  } catch {
    // A half-written frame is not worth failing the whole answer for.
  }
}

async function stopGeneration() {
  if (!state.activeId) return;
  await api(`/api/conversations/${state.activeId}/stop`, { method: "POST" });
  state.abort?.abort();
}

// ------------------------------------------------------------------ telemetry

function applyRunStats(run) {
  el("live-tps").textContent = run.tokens_per_second.toFixed(2);
  el("run-tps").textContent = run.tokens_per_second.toFixed(2);
  el("run-ttft").textContent = `${run.ttft_ms} ms`;
  el("run-prefill").textContent = `${run.prefill_ms} ms`;
  el("run-tokens").textContent = `${run.generated_tokens} / ${run.prompt_tokens}`;
  el("run-ctx").textContent = run.ctx_used;
  el("run-stop").textContent = run.stop_reason;
  drawBars(el("latency-chart"), run.token_latencies_ms, "ms");
}

async function loadHistory() {
  const [overview, daily, runs] = await Promise.all([
    api("/api/stats/overview"),
    api("/api/stats/timeseries?days=30"),
    api("/api/stats/runs?limit=15"),
  ]);
  el("stat-runs").textContent = overview.total_runs;
  el("stat-tokens").textContent = overview.total_tokens.toLocaleString("id-ID");
  el("stat-avg").textContent = overview.average_tokens_per_second.toFixed(2);
  el("stat-p50").textContent = overview.p50_tokens_per_second.toFixed(2);
  el("stat-p95").textContent = overview.p95_tokens_per_second.toFixed(2);
  el("stat-ttft").textContent = `${overview.average_ttft_ms.toFixed(0)} ms`;
  drawBars(el("daily-chart"), daily.map((bucket) => bucket.tokens), "token");

  el("runs-body").innerHTML = runs
    .map((run) => {
      const when = new Date(run.created_at).toLocaleString("id-ID", {
        day: "2-digit",
        month: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      });
      return `<div class="run-row"><span class="when">${when}</span><span>${run.generated_tokens} tok</span><span>${run.tokens_per_second.toFixed(1)} t/s</span></div>`;
    })
    .join("");
}

function drawBars(container, values, unit) {
  const series = (values ?? []).slice(-60);
  if (series.length === 0) {
    container.innerHTML = `<p class="empty" style="color:var(--text-faint);font-size:0.78rem">belum ada data</p>`;
    return;
  }
  const max = Math.max(...series, 1);
  const width = 300;
  const height = 96;
  // Keep at least 14 slots so one day of data reads as a bar on a timeline
  // rather than a solid block.
  const slots = Math.max(series.length, 14);
  const barWidth = width / slots;
  const originX = width - barWidth * series.length;
  const bars = series
    .map((value, index) => {
      const barHeight = Math.max(1, (value / max) * (height - 12));
      const x = originX + index * barWidth;
      const y = height - barHeight;
      return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(1, barWidth - 1).toFixed(1)}" height="${barHeight.toFixed(1)}" rx="1" style="fill:var(--accent);opacity:0.85"><title>${value.toFixed(2)} ${unit}</title></rect>`;
    })
    .join("");
  container.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="grafik ${unit}">${bars}</svg>`;
}

// --------------------------------------------------------------------- panels

function openDrawer() {
  el("app").classList.add("sidebar-open");
  el("scrim").hidden = false;
}

function closeDrawer() {
  el("app").classList.remove("sidebar-open");
  if (!el("app").classList.contains("telemetry-open")) {
    el("scrim").hidden = true;
  }
}

function toggleTelemetry() {
  const app = el("app");
  const open = app.classList.toggle("telemetry-open");
  if (open) {
    el("scrim").hidden = false;
    loadHistory().catch(() => {});
  } else if (!app.classList.contains("sidebar-open")) {
    el("scrim").hidden = true;
  }
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  el("theme-label").textContent = theme === "dark" ? "Tema terang" : "Tema gelap";
  localStorage.setItem("llm-theme", theme);
}

function autoGrow() {
  const prompt = el("prompt");
  prompt.style.height = "auto";
  prompt.style.height = `${Math.min(prompt.scrollHeight, 220)}px`;
}

function updateSettingsSummary() {
  const temperature = Number(el("temperature").value).toFixed(1);
  const tokens = el("max-tokens").value;
  el("temp-value").textContent = temperature;
  el("settings-summary").textContent = `temp ${temperature} · ${tokens} token`;
}

// ---------------------------------------------------------------------- wiring

function setupTabs() {
  for (const tab of document.querySelectorAll(".tabs button")) {
    tab.addEventListener("click", () => {
      for (const other of document.querySelectorAll(".tabs button")) {
        other.setAttribute("aria-selected", String(other === tab));
      }
      el("tab-live").hidden = tab.dataset.tab !== "live";
      el("tab-history").hidden = tab.dataset.tab !== "history";
    });
  }
}

async function main() {
  applyTheme(localStorage.getItem("llm-theme") ?? "dark");
  setupTabs();

  el("composer").addEventListener("submit", send);
  el("stop").addEventListener("click", stopGeneration);
  el("new-chat").addEventListener("click", startNewChat);
  el("toggle-theme").addEventListener("click", () => {
    applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  });
  el("open-sidebar").addEventListener("click", openDrawer);
  el("close-sidebar").addEventListener("click", closeDrawer);
  el("open-telemetry").addEventListener("click", toggleTelemetry);
  el("open-telemetry-top").addEventListener("click", toggleTelemetry);
  el("close-telemetry").addEventListener("click", toggleTelemetry);
  el("scrim").addEventListener("click", () => {
    closeDrawer();
    if (el("app").classList.contains("telemetry-open")) toggleTelemetry();
  });

  el("settings-toggle").addEventListener("click", () => {
    const panel = el("settings");
    panel.hidden = !panel.hidden;
    el("settings-toggle").setAttribute("aria-expanded", String(!panel.hidden));
  });
  el("temperature").addEventListener("input", updateSettingsSummary);
  el("max-tokens").addEventListener("input", updateSettingsSummary);

  el("prompt").addEventListener("input", autoGrow);
  el("prompt").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      el("composer").requestSubmit();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeDrawer();
      if (el("app").classList.contains("telemetry-open")) toggleTelemetry();
    }
  });

  updateSettingsSummary();
  renderTranscript();
  await loadModel();
  await loadConversations();
  await loadHistory();
  if (state.conversations.length > 0) {
    await openConversation(state.conversations[0].id);
  }
}

main().catch((error) => {
  el("transcript").innerHTML = `<p class="empty">Gagal memuat: ${error.message}</p>`;
});
