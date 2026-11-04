// Chat UI: conversations on the left, transcript in the middle, live telemetry
// on the right. No framework, no build step — plain ES modules.

const el = (id) => document.getElementById(id);

const state = {
  conversations: [],
  activeId: null,
  messages: [],
  streaming: false,
  abort: null,
};

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

// ---------------------------------------------------------------- model + meta

async function loadModel() {
  const info = await api("/api/model");
  el("model-name").textContent = info.name;
  el("model-meta").textContent =
    `${info.layers} layer · ${info.embedding}d · ctx ${info.context} · ${
      info.device ?? `${info.threads} thread`
    }`;
  const rows = [
    ["arsitektur", info.architecture],
    ["layer", info.layers],
    ["embedding", info.embedding],
    ["ffn", info.feed_forward],
    ["head q/kv", `${info.heads}/${info.kv_heads}`],
    ["vocab", info.vocab.toLocaleString("id-ID")],
    ["tensor", `${info.quantized_tensors} Q8_0 / ${info.tensors}`],
    ["bobot", formatBytes(info.weight_bytes)],
    ["KV cache", formatBytes(info.kv_bytes)],
  ];
  el("model-info").innerHTML = rows
    .map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`)
    .join("");
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

// ------------------------------------------------------------- conversations

async function loadConversations() {
  state.conversations = await api("/api/conversations");
  renderConversations();
}

function renderConversations() {
  const list = el("conversation-list");
  if (state.conversations.length === 0) {
    list.innerHTML = `<li class="empty" style="display:block;padding:0.5rem">Belum ada percakapan.</li>`;
    return;
  }
  list.innerHTML = "";
  for (const conversation of state.conversations) {
    const item = document.createElement("li");
    item.className = conversation.id === state.activeId ? "active" : "";
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
  renderConversations();
  renderTranscript();
}

async function deleteConversation(id) {
  await api(`/api/conversations/${id}`, { method: "DELETE" });
  if (state.activeId === id) {
    state.activeId = null;
    state.messages = [];
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
}

// ------------------------------------------------------------------ transcript

function renderTranscript() {
  const transcript = el("transcript");
  if (!state.activeId) {
    transcript.innerHTML = `<p class="empty">Pilih percakapan atau buat yang baru untuk mulai.</p>`;
    return;
  }
  transcript.innerHTML = "";
  for (const message of state.messages) {
    transcript.append(messageNode(message.role, message.content));
  }
  transcript.scrollTop = transcript.scrollHeight;
}

function messageNode(role, content) {
  const node = document.createElement("article");
  node.className = `message ${role}`;
  const label = document.createElement("span");
  label.className = "role";
  label.textContent = role === "user" ? "kamu" : "model";
  const body = document.createElement("div");
  body.textContent = content;
  node.append(label, body);
  return node;
}

function appendStreamingMessage() {
  const node = messageNode("assistant", "");
  node.classList.add("pending");
  node.dataset.streaming = "true";
  el("transcript").append(node);
  el("transcript").scrollTop = el("transcript").scrollHeight;
  return node.querySelector("div");
}

// --------------------------------------------------------------------- sending

async function send(event) {
  event.preventDefault();
  const prompt = el("prompt").value.trim();
  if (!prompt || state.streaming) return;

  if (!state.activeId) await createConversation();

  state.streaming = true;
  el("send").disabled = true;
  el("stop").hidden = false;
  el("prompt").value = "";

  const userNode = messageNode("user", prompt);
  el("transcript").append(userNode);
  const target = appendStreamingMessage();
  state.abort = new AbortController();

  try {
    const response = await fetch(`/api/conversations/${state.activeId}/messages`, {
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
    await consumeEvents(response, target);
  } catch (error) {
    if (error.name !== "AbortError") {
      target.textContent = `[gagal] ${error.message}`;
    }
  } finally {
    state.streaming = false;
    el("send").disabled = false;
    el("stop").hidden = true;
    state.abort = null;
    await loadConversations();
    await openConversation(state.activeId);
    await loadHistory();
  }
}

async function consumeEvents(response, target) {
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
      handleEvent(chunk, target);
      boundary = buffer.indexOf("\n\n");
    }
  }
}

function handleEvent(chunk, target) {
  let name = "message";
  const dataLines = [];
  for (const line of chunk.split("\n")) {
    if (line.startsWith("event:")) name = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (dataLines.length === 0) return;
  const payload = JSON.parse(dataLines.join("\n"));

  if (name === "start") {
    el("run-tokens").textContent = `${payload.prompt_tokens} prompt`;
  } else if (name === "token") {
    target.classList.remove("pending");
    target.textContent += payload.text;
    el("transcript").scrollTop = el("transcript").scrollHeight;
    const shown = target.textContent.length;
    el("live-tokens").textContent = shown;
  } else if (name === "stats") {
    applyRunStats(payload);
  } else if (name === "error") {
    target.textContent += `\n[error] ${payload.message}`;
  }
}

async function stopGeneration() {
  if (!state.activeId) return;
  await api(`/api/conversations/${state.activeId}/stop`, { method: "POST" });
  if (state.abort) state.abort.abort();
}

// ------------------------------------------------------------------ statistics

function applyRunStats(run) {
  el("live-tps").textContent = run.tokens_per_second.toFixed(2);
  el("live-ttft").textContent = run.ttft_ms;
  el("live-tokens").textContent = run.generated_tokens;
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
    .map(
      (run) => `<tr>
        <td>${new Date(run.created_at).toLocaleString("id-ID", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" })}</td>
        <td>${run.generated_tokens}</td>
        <td>${run.tokens_per_second.toFixed(2)}</td>
        <td>${run.ttft_ms}</td>
        <td>${run.stop_reason}</td>
      </tr>`,
    )
    .join("");
}

function drawBars(container, values, unit) {
  const series = (values ?? []).slice(-60);
  if (series.length === 0) {
    container.innerHTML = `<p class="muted">belum ada data</p>`;
    return;
  }
  const max = Math.max(...series, 1);
  const width = 300;
  const height = 120;
  // Keep at least 14 slots so a single day of data reads as one bar on a
  // timeline instead of a solid block.
  const slots = Math.max(series.length, 14);
  const barWidth = width / slots;
  const originX = width - barWidth * series.length;
  const bars = series
    .map((value, index) => {
      const barHeight = Math.max(1, (value / max) * (height - 16));
      const x = originX + index * barWidth;
      const y = height - barHeight;
      return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(1, barWidth - 1).toFixed(1)}" height="${barHeight.toFixed(1)}" fill="#6aa9ff" opacity="0.85"><title>${value.toFixed(2)} ${unit}</title></rect>`;
    })
    .join("");
  container.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="grafik ${unit}">${bars}</svg>`;
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
  setupTabs();
  el("composer").addEventListener("submit", send);
  el("stop").addEventListener("click", stopGeneration);
  el("new-chat").addEventListener("click", createConversation);
  el("prompt").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      el("composer").requestSubmit();
    }
  });

  renderTranscript();
  await loadModel();
  await loadConversations();
  await loadHistory();
}

main().catch((error) => {
  el("transcript").innerHTML = `<p class="empty">Gagal memuat: ${error.message}</p>`;
});
