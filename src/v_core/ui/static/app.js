"use strict";

const sessionToken = document.querySelector('meta[name="darklinger-session"]').content;
const headers = {"X-DARKLINGER-Session": sessionToken};
const jsonHeaders = {...headers, "Content-Type": "application/json"};

const byId = (id) => document.getElementById(id);

// Memory content is untrusted text. Never render it as HTML or execute it.
let memoryView = "sessions";
let memoryRequest = 0;
const expiredPanelMessage = "This panel belongs to a previous Darklinger run. Reload the page or open a new panel.";
function memoryConnectionMessage() {
  const state = shuttingDown
    ? "Darklinger is shut down."
    : "Cannot connect to Darklinger. The application may be shut down.";
  return `${state} To view saved memoirs, restart Darklinger and open a new panel.`;
}
async function memoryGet(query) {
  let response;
  try {
    response = await fetch(`/api/memory/browser?${new URLSearchParams(query)}`, {headers, cache: "no-store"});
  } catch (_error) {
    throw new Error(memoryConnectionMessage());
  }
  let data;
  try { data = await response.json(); }
  catch (_error) { throw new Error(`Could not read the memory panel response (HTTP ${response.status}).`); }
  if (response.status === 403 && data.error === "invalid local UI session") {
    throw new Error(expiredPanelMessage);
  }
  if (!response.ok) throw new Error(data.error || "Could not read memory.");
  return data;
}
function memorySection(title, text) {
  const section = document.createElement("section");
  const heading = document.createElement("h3");
  heading.textContent = title;
  const body = document.createElement("pre");
  body.textContent = text;
  section.append(heading, body);
  byId("memory-browser-content").append(section);
}
async function renderMemory(refreshList = true) {
  const request = ++memoryRequest;
  const status = byId("memory-browser-status");
  const content = byId("memory-browser-content");
  content.replaceChildren();
  status.textContent = "Loading…";
  byId("memory-session-label").hidden = memoryView !== "sessions";
  byId("memory-sessions").setAttribute("aria-pressed", String(memoryView === "sessions"));
  byId("memory-persona").setAttribute("aria-pressed", String(memoryView === "persona"));
  try {
    if (memoryView === "persona") {
      const data = await memoryGet({view: "persona"});
      if (request !== memoryRequest) return;
      status.textContent = data.notice;
      for (const section of data.sections) memorySection(section.title, section.text);
      if (!data.sections.length) memorySection("Persona", "Persona is not available in this runtime yet.");
      return;
    }
    const select = byId("memory-session-select");
    if (refreshList) {
      const data = await memoryGet({view: "sessions"});
      if (request !== memoryRequest) return;
      const previous = select.value;
      select.replaceChildren();
      for (const session of data.sessions) {
        const option = document.createElement("option");
        option.value = session.id;
        option.textContent = session.label;
        select.append(option);
      }
      if (data.sessions.some((s) => s.id === previous)) select.value = previous;
      select.dataset.scope = data.archive_allowed ? "Local profile archive." : "Public: current session only.";
      if (data.limited) select.dataset.scope += " List limited to 100 archives from 1,000 scanned entries.";
    }
    if (!select.value) {
      status.textContent = "Session storage is unavailable.";
      return;
    }
    const data = await memoryGet({view: "session", session: select.value});
    if (request !== memoryRequest) return;
    if (!data.available) { status.textContent = "Session unavailable."; return; }
    const record = data.record;
    const states = {idle: "not saved yet", saving: "saving", saved: "saved", error: "save error", empty: "empty session"};
    status.textContent = `${select.dataset.scope} Memoir: ${states[data.save_state] || data.save_state}.`;
    memorySection("Session memoir", record.memory || "No saved memoir. The current conversation transcript is shown below, not a summary.");
    memorySection("Provenance", `Session: ${record.session_id}\nStarted: ${record.started_at}\nSource: ${record.source}\nThe summary is not verified evidence.\nReferenced turns: ${record.turn_ids.join(", ") || "none"}${record.partial_excerpts ? "\nThe summary was generated from limited excerpts." : ""}${record.fallback_reason ? `\nFallback reason: ${record.fallback_reason}` : ""}`);
    if (record.omitted_turns) memorySection("View limit", `${record.omitted_turns} earlier turns omitted. Showing the last 50; long fields are marked as truncated.`);
    for (const turn of record.turns) {
      memorySection(`Turn ${turn.id ?? "?"} · ${turn.at} · ${turn.status}`, `You:\n${turn.user}\n\nV — recorded statement, not proof of execution:\n${turn.assistant}`);
    }
  } catch (error) {
    if (request === memoryRequest) status.textContent = error.message;
  }
}
byId("memory-browser-open").addEventListener("click", () => {
  byId("memory-browser-dialog").showModal();
  renderMemory();
});
byId("memory-sessions").addEventListener("click", () => { memoryView = "sessions"; renderMemory(); });
byId("memory-persona").addEventListener("click", () => { memoryView = "persona"; renderMemory(); });
byId("memory-refresh").addEventListener("click", () => renderMemory());
byId("memory-session-select").addEventListener("change", () => renderMemory(false));
byId("memory-browser-dialog").addEventListener("close", () => {
  ++memoryRequest;
  byId("memory-browser-content").replaceChildren();
});

const ui = {
  edition: byId("edition-badge"),
  runtime: byId("runtime-state"),
  entity: byId("entity-state"),
  orb: byId("v-orb"),
  modelAlias: byId("model-alias"),
  modelFile: byId("model-file"),
  modelContext: byId("model-context"),
  modelReasoning: byId("model-reasoning"),
  modelCache: byId("model-cache"),
  modelMeter: byId("model-meter"),
  uptime: byId("uptime"),
  toolCount: byId("tool-count"),
  toolList: byId("tool-list"),
  messages: byId("messages"),
  composer: byId("composer"),
  prompt: byId("prompt"),
  send: byId("send"),
  ptt: byId("ptt"),
  pttLabel: byId("ptt-label"),
  speak: byId("speak"),
  activity: byId("activity"),
  feed: byId("runtime-feed"),
  liveState: byId("live-state"),
  liveDetail: byId("live-detail"),
  livePhase: byId("live-phase"),
  liveOutput: byId("live-output"),
  liveSpeed: byId("live-speed"),
  livePromptSpeed: byId("live-prompt-speed"),
  liveElapsed: byId("live-elapsed"),
  liveStep: byId("live-step"),
  liveContext: byId("live-context"),
  liveContextMeter: byId("live-context-meter"),
  liveObjective: byId("live-objective"),
  liveAction: byId("live-action"),
  ownerDeck: byId("owner-deck"),
  capabilityAuditList: byId("capability-audit-list"),
  proposalList: byId("proposal-list"),
  shutdown: byId("shutdown"),
};

let working = false;
let recording = false;
let lastReady = null;
let requestInFlight = false;
let uploadingMedia = false;
let attachments = [];

function renderAttachments() {
  const list = byId("media-attachments");
  list.replaceChildren();
  attachments.forEach((file, index) => {
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "utility-button";
    remove.textContent = `${file.name} ×`;
    remove.disabled = working || uploadingMedia;
    remove.onclick = () => { attachments.splice(index, 1); renderAttachments(); };
    list.append(remove);
  });
}
byId("media-attach").onclick = () => byId("media-file").click();
byId("media-file").addEventListener("change", async (event) => {
  if (working || uploadingMedia) return;
  uploadingMedia = true;
  byId("media-attach").disabled = true;
  ui.send.disabled = true;
  const status = byId("media-status");
  status.hidden = false;
  try {
    for (const file of event.target.files) {
      if (file.size > 64 * 1024 * 1024) throw new Error("Attachment exceeds 64 MiB");
      status.textContent = `Attaching ${file.name}…`;
      const response = await fetch(`/api/perception/upload?name=${encodeURIComponent(file.name)}`, {
        method: "POST", headers: {...headers, "Content-Type": "application/octet-stream"}, body: file,
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Attachment failed");
      attachments.push(result);
    }
    status.textContent = "Attached locally. Send a message to inspect these files.";
  } catch (error) { status.textContent = error.message; }
  finally {
    uploadingMedia = false;
    event.target.value = "";
    byId("media-attach").disabled = working;
    ui.send.disabled = working;
    renderAttachments();
  }
});
byId("perception-open").onclick = async () => {
  byId("perception-dialog").showModal();
  const status = byId("perception-status");
  status.textContent = "Checking local configuration…";
  try {
    const result = await api("/api/perception/status");
    const vision = result.vision.managed_vision_configured || result.vision.external_local_endpoint_configured;
    status.textContent = `PDF text: available. Image model: ${vision ? "configured" : "not separately configured"}. Speech recognition: ${result.speech_configured ? "configured" : "not configured"}. Other sounds: ${result.sound_model_configured ? "configured; capability checked on use" : "not configured"}.`;
  } catch (error) { status.textContent = error.message; }
};

function clock() {
  return new Date().toLocaleTimeString("en-GB", {hour12: false});
}

function feed(text, key = "", timestamp = "") {
  if (key) {
    const existing = Array.from(ui.feed.children).find((item) => item.dataset.key === key);
    if (existing) return;
  }
  const row = document.createElement("li");
  const time = document.createElement("time");
  const body = document.createElement("span");
  row.dataset.key = key;
  time.textContent = timestamp
    ? new Date(timestamp).toLocaleTimeString("en-GB", {hour12: false})
    : clock();
  body.textContent = text;
  row.append(time, body);
  ui.feed.prepend(row);
  while (ui.feed.children.length > 16) ui.feed.lastElementChild.remove();
}

function formatCount(value) {
  return Number(value || 0).toLocaleString("en-US");
}

function formatDuration(total) {
  const seconds = Math.max(0, Number(total || 0));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60).toString().padStart(2, "0");
  const rest = Math.floor(seconds % 60).toString().padStart(2, "0");
  return hours ? `${hours}:${minutes}:${rest}` : `${minutes}:${rest}`;
}

function renderToolJobs(supervision) {
  const container = byId("tool-jobs");
  if (!container) return;
  const jobs = (supervision?.jobs || []).slice(-12);
  container.replaceChildren();
  if (!jobs.length) { container.textContent = "No jobs."; return; }
  const labels = {queued: "Queued", running: "Running", stopping: "Stopping", succeeded: "Completed", failed: "Failed", cancelled: "Cancelled"};
  const phases = {waiting_network: "Waiting for network", receiving_data: "Receiving data", process_started: "Process started", process_stopped: "Process exited", browser_starting: "Starting browser", waiting_for_page: "Waiting for page", observing_page: "Reading page", page_observed: "Page observed", sandbox_running: "Running in sandbox", sandbox_output: "Receiving sandbox output"};
  for (const job of jobs) {
    const row = document.createElement("article");
    row.className = "proposal";
    const text = document.createElement("p");
    text.textContent = `${job.tool}: ${labels[job.state] || job.state} · ${phases[job.phase] || labels[job.phase] || "Waiting for result"} · ${job.elapsed_seconds}s`;
    if (job.signals?.bytes_received !== undefined) text.textContent += ` · received ${job.signals.bytes_received} B`;
    row.append(text);
    if (["queued", "running"].includes(job.state)) {
      const stop = document.createElement("button");
      stop.type = "button";
      stop.textContent = "Stop job";
      stop.onclick = async () => {
        stop.disabled = true;
        try { await api(`/api/tool-jobs/${encodeURIComponent(job.job_id)}/cancel`, {}); }
        catch (error) { text.textContent = error.message; stop.disabled = false; }
      };
      row.append(stop);
    }
    container.append(row);
  }
}

function renderRuntimeActivity(activity) {
  if (!activity) return;
  const state = activity.state || "idle";
  const labels = {
    idle: "IDLE",
    prompt: "READING PROMPT",
    generating: "GENERATING",
    repairing: "REPAIRING",
    tool_running: "TOOL RUNNING",
    orchestrating: "ORCHESTRATING",
    stalled: "POSSIBLY STALLED",
  };
  ui.liveState.className = `live-state ${state}`;
  ui.liveState.innerHTML = `<i></i>${labels[state] || state.toUpperCase()}`;
  ui.liveDetail.textContent = activity.detail || "Runtime state unavailable";
  ui.livePhase.textContent = String(activity.phase || "idle").toUpperCase();
  ui.liveOutput.textContent = `${formatCount(activity.generated_tokens)} TOK`;
  const generationRate = activity.generation_tokens_per_second;
  const promptRate = activity.prompt_tokens_per_second;
  ui.liveSpeed.textContent = generationRate == null ? "— T/S" : `${Number(generationRate).toFixed(1)} T/S`;
  ui.livePromptSpeed.textContent = promptRate == null ? "— T/S" : `${Number(promptRate).toFixed(1)} T/S`;
  ui.liveElapsed.textContent = formatDuration(activity.task_elapsed_seconds);
  ui.liveStep.textContent = formatCount(activity.step);
  ui.liveContext.textContent = `${formatCount(activity.context_used)} / ${formatCount(activity.context_size)}`;
  ui.liveContextMeter.style.width = `${Math.max(0, Math.min(100, Number(activity.context_percent || 0)))}%`;
  ui.liveObjective.textContent = activity.objective || "Waiting for an objective.";
  const lastAction = activity.last_tool
    ? `${activity.last_tool} // ${activity.last_tool_status || "unknown"}`
    : "—";
  ui.liveAction.textContent = `LAST ACTION // ${lastAction}`;
  (activity.events || []).forEach((event) => {
    feed(`${event.label} // ${event.text}`, `runtime:${event.id}`, event.timestamp);
  });
}

function chatIsPinnedToBottom() {
  const remaining = ui.messages.scrollHeight - ui.messages.clientHeight - ui.messages.scrollTop;
  return remaining <= 72;
}

function scrollChatToBottom(force = false) {
  if (!force && !chatIsPinnedToBottom()) return;
  requestAnimationFrame(() => {
    ui.messages.scrollTop = ui.messages.scrollHeight;
  });
}

function message(role, text = "") {
  const followOutput = chatIsPinnedToBottom();
  const article = document.createElement("article");
  article.className = `message ${role === "user" ? "user-message" : "v-message"}`;
  const speaker = document.createElement("div");
  speaker.className = "speaker";
  speaker.textContent = role === "user" ? "B" : "V";
  const body = document.createElement("div");
  body.className = "message-body";
  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  body.append(paragraph);
  article.append(speaker, body);
  ui.messages.append(article);
  scrollChatToBottom(role === "user" || followOutput);
  return {article, paragraph};
}

function renderChips(container, values, maximum = 14) {
  container.replaceChildren();
  values.slice(0, maximum).forEach((value) => {
    const chip = document.createElement("span");
    chip.textContent = value;
    chip.title = value;
    container.append(chip);
  });
  if (values.length > maximum) {
    const more = document.createElement("span");
    more.textContent = `+${values.length - maximum} MORE`;
    container.append(more);
  }
}

function formatUptime(total) {
  const hours = Math.floor(total / 3600).toString().padStart(2, "0");
  const minutes = Math.floor((total % 3600) / 60).toString().padStart(2, "0");
  const seconds = Math.floor(total % 60).toString().padStart(2, "0");
  return `${hours}:${minutes}:${seconds}`;
}

function setWorking(value) {
  working = value;
  ui.prompt.disabled = value;
  ui.send.disabled = value || uploadingMedia;
  byId("media-attach").disabled = value || uploadingMedia;
  renderAttachments();
  ui.ptt.disabled = value && !recording;
  ui.orb.classList.toggle("busy", value);
  ui.runtime.className = `status-pill ${value ? "busy" : "ready"}`;
  ui.runtime.innerHTML = `<i></i> ${value ? "V IS WORKING" : "READY"}`;
  ui.entity.textContent = value ? "THINKING / EXECUTING" : "AWAKE & ARMED";
  ui.activity.textContent = value ? "TASK IN PROGRESS" : "NO ACTIVE TASK";
}

function renderOwner(owner) {
  if (!owner) {
    ui.ownerDeck.classList.add("hidden");
    return;
  }
  ui.ownerDeck.classList.remove("hidden");
  byId("proposal-count").textContent = `(${(owner.proposals || []).length})`;
  ui.capabilityAuditList.replaceChildren();
  (owner.capability_audits || []).forEach((audit) => {
    const row = document.createElement("article");
    row.className = `capability-audit ${audit.risk || "review"}`;
    const title = document.createElement("strong");
    const body = document.createElement("p");
    title.textContent = `${String(audit.risk || "review").toUpperCase()} // ${audit.name}`;
    const details = [...(audit.primitives || []), ...(audit.reasons || [])];
    body.textContent = `${audit.kind || "artifact"} · ${audit.origin || "unknown"}`
      + (details.length ? ` · ${details.join(", ")}` : "");
    row.append(title, body);
    ui.capabilityAuditList.append(row);
  });
  if (!(owner.capability_audits || []).length) {
    const empty = document.createElement("span");
    empty.className = "proposal-empty";
    empty.textContent = "NO SENSITIVE CAPABILITY CHANGES";
    ui.capabilityAuditList.append(empty);
  }
  ui.proposalList.replaceChildren();
  (owner.proposals || []).forEach((proposal) => {
    const row = document.createElement("article");
    row.className = "proposal";
    const title = document.createElement("strong");
    const body = document.createElement("p");
    const actions = document.createElement("div");
    title.textContent = proposal.title || "V suggestion";
    body.textContent = proposal.suggestion || "";
    ["approve", "reject"].forEach((decision) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = decision.toUpperCase();
      button.addEventListener("click", async () => {
        button.disabled = true;
        try {
          const response = await fetch(
            `/api/proposals/${encodeURIComponent(proposal.id)}/${decision}`,
            {method: "POST", headers: jsonHeaders, body: "{}"},
          );
          const result = await response.json();
          if (!response.ok) throw new Error(result.error || "proposal decision failed");
          feed(`Proposal ${decision === "approve" ? "approved" : "rejected"} by owner`);
          await refreshStatus();
        } catch (error) {
          feed(`Proposal failure: ${error.message}`);
          button.disabled = false;
        }
      });
      actions.append(button);
    });
    row.append(title, body, actions);
    ui.proposalList.append(row);
  });
  if (!(owner.proposals || []).length) {
    const empty = document.createElement("span");
    empty.className = "proposal-empty";
    empty.textContent = "NO PENDING CHANGES";
    ui.proposalList.append(empty);
  }
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status", {headers, cache: "no-store"});
    if (!response.ok) throw Object.assign(new Error(`status ${response.status}`), {httpStatus: response.status});
    const state = await response.json();
    if (state.closing) showShutdown(state.memoir);
    ui.edition.textContent = state.edition.toUpperCase();
    ui.modelAlias.textContent = state.model.alias;
    ui.modelFile.textContent = state.model.filename || state.model.state;
    ui.modelContext.textContent = state.model.context_size ? state.model.context_size.toLocaleString() : "EXTERNAL";
    ui.modelReasoning.textContent = state.model.reasoning.toUpperCase();
    ui.modelCache.textContent = `${state.model.cache_k}/${state.model.cache_v}`.toUpperCase();
    ui.modelMeter.style.width = state.model.state === "stopped" ? "8%" : "100%";
    ui.uptime.textContent = formatUptime(state.uptime_seconds);
    ui.toolCount.textContent = state.tools.count;
    renderChips(ui.toolList, state.tools.active || []);
    recording = Boolean(state.voice.recording);
    ui.ptt.classList.toggle("recording", recording);
    ui.orb.classList.toggle("recording", recording);
    ui.pttLabel.textContent = recording ? "STOP & TRANSCRIBE" : "F2 / PUSH TO TALK";
    renderOwner(state.owner);
    renderRuntimeActivity(state.activity);
    renderToolJobs(state.tool_supervision);
    if (!requestInFlight) setWorking(!state.ready);
    if (lastReady !== state.ready) {
      feed(state.ready ? "Runtime ready" : "Runtime entered active task");
      lastReady = state.ready;
    }
  } catch (error) {
    ui.runtime.className = "status-pill";
    ui.runtime.innerHTML = shuttingDown ? "<i></i> STOPPED" : "<i></i> LINK LOST";
    ui.entity.textContent = shuttingDown ? "CLOSED" : "CONNECTION LOST";
    const explanation = error.httpStatus === 403 ? expiredPanelMessage
      : error.httpStatus ? `The Darklinger panel returned HTTP ${error.httpStatus}. Try reloading the page.`
      : memoryConnectionMessage();
    if (shuttingDown && !error.httpStatus) {
      const notice = byId("shutdown-notice");
      notice.classList.remove("hidden");
      const saved = lastMemoirState.state === "saved"
        ? (lastMemoirState.fallback ? "Fallback note saved. " : "Session memoir saved. ")
        : "";
      notice.textContent = saved + memoryConnectionMessage();
    }
    if (byId("memory-browser-dialog").open) {
      byId("memory-browser-status").textContent = explanation;
    }
  }
}

async function sendPrompt(text) {
  if (working || uploadingMedia) return;
  const selected = attachments.slice();
  const prompt = [text.trim(), ...selected.map(file => file.uri)].filter(Boolean).join("\n");
  if (!prompt) return;
  message("user", prompt);
  const output = message("v", "");
  let showingDraft = false;
  const draftStatus = document.createElement("small");
  draftStatus.hidden = true;
  output.article.appendChild(draftStatus);
  output.paragraph.classList.add("cursor");
  ui.prompt.value = "";
  requestInFlight = true;
  setWorking(true);
  feed("Prompt accepted by runtime");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({message: prompt, speak: ui.speak.checked}),
    });
    if (!response.ok) {
      const body = await response.json();
      throw new Error(body.error || `request failed: ${response.status}`);
    }
    attachments = attachments.filter(file => !selected.includes(file));
    renderAttachments();
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === "draft_start") {
          showingDraft = true;
          output.paragraph.textContent = "";
          draftStatus.hidden = false;
          draftStatus.textContent = "Draft — generating, not yet verified";
        } else if (event.type === "draft_token") {
          const followOutput = chatIsPinnedToBottom();
          output.paragraph.textContent += event.text;
          scrollChatToBottom(followOutput);
        } else if (event.type === "draft_validating") {
          draftStatus.textContent = "Draft — checking against tool results";
        } else if (event.type === "token") {
          if (showingDraft) {
            output.paragraph.textContent = "";
            showingDraft = false;
            draftStatus.hidden = true;
          }
          const followOutput = chatIsPinnedToBottom();
          output.paragraph.textContent += event.text;
          scrollChatToBottom(followOutput);
        } else if (event.type === "speech") {
          feed(event.state === "speaking" ? "Local voice synthesis started" : `Voice: ${event.state}`);
        } else if (event.type === "error") {
          throw new Error(event.error || "V runtime failed");
        } else if (event.type === "done") {
          output.paragraph.textContent = event.answer || output.paragraph.textContent;
          draftStatus.hidden = true;
          feed("Runtime completed the response");
        }
      }
      if (done) break;
    }
  } catch (error) {
    draftStatus.hidden = true;
    output.article.classList.add("error-message");
    output.paragraph.textContent = `Runtime error: ${error.message}`;
    feed(`Failure: ${error.message}`);
  } finally {
    requestInFlight = false;
    output.paragraph.classList.remove("cursor");
    setWorking(false);
    ui.prompt.focus();
    refreshStatus();
  }
}

async function togglePTT() {
  if (working && !recording) return;
  ui.ptt.disabled = true;
  try {
    const endpoint = recording ? "/api/voice/ptt/stop" : "/api/voice/ptt/start";
    const response = await fetch(endpoint, {method: "POST", headers: jsonHeaders, body: "{}"});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "voice request failed");
    recording = Boolean(result.recording);
    ui.ptt.classList.toggle("recording", recording);
    ui.orb.classList.toggle("recording", recording);
    if (recording) {
      ui.pttLabel.textContent = "STOP & TRANSCRIBE";
      feed("Microphone recording started");
    } else {
      ui.pttLabel.textContent = "F2 / PUSH TO TALK";
      feed("Speech transcribed locally");
      if (result.transcript) await sendPrompt(result.transcript);
    }
  } catch (error) {
    feed(`Voice failure: ${error.message}`);
    message("v", `Voice channel failed: ${error.message}`).article.classList.add("error-message");
  } finally {
    ui.ptt.disabled = false;
  }
}

ui.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  sendPrompt(ui.prompt.value);
});

ui.prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendPrompt(ui.prompt.value);
  }
});

ui.ptt.addEventListener("click", togglePTT);
window.addEventListener("keydown", (event) => {
  if (event.key === "F2" && !event.repeat) {
    event.preventDefault();
    togglePTT();
  }
});

let shuttingDown = false;
let lastMemoirState = {};
function showShutdown(memoir = {}) {
  shuttingDown = true;
  ui.shutdown.disabled = true;
  lastMemoirState = {...lastMemoirState, ...memoir};
  const notice = byId("shutdown-notice");
  notice.classList.remove("hidden");
  if (memoir.state === "saved") {
    notice.textContent = `${memoir.fallback ? "Fallback note saved" : "Memoir saved"}: ${memoir.file}. Closing…`;
  } else if (memoir.state === "error") notice.textContent = memoir.message;
  else notice.textContent = "V is saving this session memoir (up to 60 seconds). Emergency stop skips this step.";
}
async function shutdown(emergency = false) {
  if (shuttingDown && !emergency) return;
  shuttingDown = true;
  ui.shutdown.disabled = true;
  try {
    const result = await api(`/api/shutdown${emergency ? "?emergency=1" : ""}`, {});
    showShutdown();
    if (result.status === "shutting_down") byId("shutdown-notice").textContent = "Closing Darklinger and the model…";
  } catch (error) {
    feed(`Could not shut down: ${error.message}`);
    shuttingDown = false;
    ui.shutdown.disabled = false;
  }
}
ui.shutdown.addEventListener("click", () => shutdown());
byId("emergency-stop").addEventListener("click", () => {
  if (window.confirm("Stop immediately without saving a session memoir?")) shutdown(true);
});

async function api(path, body) {
  const response = await fetch(path, body === undefined
    ? {headers, cache: "no-store"}
    : {method: "POST", headers: jsonHeaders, body: JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
  return result;
}

let logPaused = false;
let logBusy = false;
byId("log-pause").addEventListener("click", () => {
  logPaused = !logPaused;
  byId("log-pause").textContent = logPaused ? "Resume" : "Pause";
  byId("log-pause").setAttribute("aria-pressed", String(logPaused));
  if (!logPaused) refreshLog();
});
async function refreshLog() {
  if (logPaused || logBusy) return;
  logBusy = true;
  try {
    const result = await api("/api/model/log");
    byId("log-source").textContent = result.source || "Backend has no local log";
    const console = byId("model-log");
    if (console.textContent !== result.text) {
      const position = console.scrollTop;
      console.textContent = result.text || "The log is still empty.";
      console.scrollTop = byId("log-follow").checked ? console.scrollHeight : position;
    }
  } catch (error) { byId("log-source").textContent = `Log unavailable: ${error.message}`; }
  finally { logBusy = false; }
}

const voiceDialog = byId("voice-settings-dialog");
let voiceBusy = false;
let voiceInstalling = false;
let installerWasBusy = false;
function voiceMessage(text) { byId("voice-settings-message").textContent = text; }
function options(id, items, value) {
  const select = byId(id);
  select.replaceChildren();
  if (!items.length) items = [{id: "", label: "None — install or configure first"}];
  else if (!items.some((item) => item.id === value)) items = [{id: "", label: "Current selection unavailable — choose from the list"}, ...items];
  for (const item of items) select.add(new Option(item.label, item.id));
  select.value = items.some((item) => item.id === value) ? value : "";
}
function renderVoice(state, replace = false) {
  voiceInstalling = state.installing;
  if (replace) {
    options("voice-choice", state.voices, state.selected.voice);
    options("whisper-engine", state.engines, state.selected.engine);
    options("whisper-model", state.models, state.selected.model);
    byId("whisper-language").value = state.selected.language;
    byId("whisper-threads").value = state.selected.threads;
    const downloads = byId("whisper-downloads");
    downloads.replaceChildren();
    for (const item of state.downloads) {
      const row = document.createElement("div");
      row.className = "download-row";
      const label = document.createElement("span");
      label.textContent = `${item.label} · ${Math.round(item.bytes / 1024 / 1024)} MiB`;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "utility-button";
      button.textContent = item.installed ? "Installed" : "Download";
      button.dataset.installed = String(item.installed);
      button.addEventListener("click", () => installVoice("model", item.id,
        `Download ${item.label} (${Math.round(item.bytes / 1024 / 1024)} MiB) from Hugging Face?`));
      row.append(label, button);
      downloads.append(row);
    }
  }
  byId("voice-save").disabled = state.installing;
  byId("whisper-install-engine").disabled = state.installing || state.cpu_engine_installed;
  byId("whisper-install-engine").textContent = state.cpu_engine_installed ? "whisper.cpp CPU installed" : "Install whisper.cpp CPU";
  byId("whisper-downloads").querySelectorAll("button").forEach((button) => {
    button.disabled = state.installing || button.dataset.installed === "true";
  });
  const job = state.job || {};
  const progress = byId("voice-install-progress");
  progress.classList.toggle("hidden", !state.installing);
  byId("voice-install-cancel").classList.toggle("hidden", !state.installing);
  if (job.total) progress.value = 100 * job.bytes / job.total;
  else progress.removeAttribute("value");
  byId("voice-install-status").textContent = job.state === "idle" ? "" :
    `${job.state}: ${job.message || ""}${job.total ? `\n${Math.round(job.bytes / 1024 / 1024)} / ${Math.round(job.total / 1024 / 1024)} MiB` : ""}${job.detail ? `\n${job.detail}` : ""}`;
}
async function refreshVoice(replace = false) {
  if (voiceBusy) return;
  voiceBusy = true;
  try {
    const state = await api("/api/voice/settings");
    renderVoice(state, replace || (installerWasBusy && !state.installing));
    installerWasBusy = state.installing;
  } catch (error) { voiceMessage(error.message); }
  finally { voiceBusy = false; }
}
byId("voice-settings-open").addEventListener("click", () => {
  voiceDialog.showModal();
  voiceMessage("");
  refreshVoice(true);
});
byId("voice-settings-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = {language: byId("whisper-language").value, threads: Number(byId("whisper-threads").value)};
  for (const [field, id] of [["voice", "voice-choice"], ["engine", "whisper-engine"], ["model", "whisper-model"]]) {
    if (byId(id).value) payload[field] = byId(id).value;
  }
  try {
    renderVoice(await api("/api/voice/settings", payload), true);
    voiceMessage("Saved. New settings apply to the next recording or spoken response.");
  } catch (error) { voiceMessage(error.message); }
});
byId("voice-preview").addEventListener("click", async () => {
  byId("voice-preview").disabled = true;
  voiceMessage("Playing saved voice…");
  try { await api("/api/voice/preview", {}); voiceMessage("Preview complete."); }
  catch (error) { voiceMessage(error.message); }
  finally { byId("voice-preview").disabled = false; }
});
async function installVoice(kind, id, question) {
  if (voiceInstalling || !window.confirm(question)) return;
  try {
    await api("/api/voice/install", {kind, id});
    voiceMessage("Installation started. Current settings are unchanged.");
    await refreshVoice();
  } catch (error) { voiceMessage(error.message); }
}
byId("whisper-install-engine").addEventListener("click", () => installVoice("engine", "cpu",
  "Download whisper.cpp v1.8.3 source from GitHub and build the CPU engine locally? Requires at least 1 GB of free space and may take several minutes."));
byId("voice-install-cancel").addEventListener("click", async () => {
  try { await api("/api/voice/install/cancel", {}); await refreshVoice(true); }
  catch (error) { voiceMessage(error.message); }
});
setInterval(() => { if (voiceDialog.open || voiceInstalling) refreshVoice(); }, 1500);
refreshLog();
setInterval(refreshLog, 1500);

refreshStatus();
setInterval(refreshStatus, 2000);
ui.prompt.focus();


const modelTestsDialog = byId("model-tests-dialog");
let modelTestsBusy = false;
let modelTestsSelected = "";
let modelTestsSignature = "";
function testNode(tag, text, parent) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = String(text);
  if (parent) parent.append(node);
  return node;
}
function renderModelTests(run) {
  const container = byId("model-tests-content");
  if (!run) {
    byId("model-tests-status").textContent = "No transcripts yet. Earlier qualification cards contain scores only; questions and replies are saved from the next qualification.";
    container.replaceChildren();
    modelTestsSignature = "";
    return;
  }
  const status = run.state === "completed" ? `Completed · ${run.overall_score}/100`
    : run.state === "unfinished" ? "In progress / no final result recorded" : run.state;
  byId("model-tests-status").textContent = `${run.alias || run.model_path} · ${status} · ${run.completed}/${run.total} tests finished · Last update: ${new Date(run.updated_at).toLocaleString()}`
    + (run.error ? ` · ${run.error}` : "")
    + (run.truncated || run.incomplete_line ? " · Some log data is incomplete or omitted." : "");
  const signature = JSON.stringify([run.run_id, run.updated_at, run.events.length, run.incomplete_line]);
  for (const waiting of container.querySelectorAll(".model-test-waiting")) {
    const elapsed = Math.max(0, Math.floor((Date.now() - Date.parse(waiting.dataset.started)) / 1000));
    waiting.textContent = `Waiting for model reply… ${elapsed} s`;
  }
  if (signature === modelTestsSignature) return;
  const known = new Set([...container.querySelectorAll("details")].map(node => node.dataset.key));
  const sameRun = container.dataset.runId === run.run_id;
  const opened = new Set([...container.querySelectorAll("details[open]")].map(node => node.dataset.key));
  container.replaceChildren();
  container.dataset.runId = run.run_id;
  modelTestsSignature = signature;
  const probes = new Map();
  const requests = new Map();
  const results = new Map(run.events.filter(event => event.event === "probe_finished").map(event => [event.probe, event.result]));
  for (const event of run.events) {
    if (event.event === "probe_started") {
      const section = testNode("details", undefined, container);
      section.dataset.key = `probe:${event.probe}`;
      section.open = sameRun && known.has(section.dataset.key) ? opened.has(section.dataset.key) : !results.has(event.probe);
      const result = results.get(event.probe);
      testNode("summary", `${event.index}/${event.total} · ${event.probe} · ${result ? `${result.score}/100 (${result.passed ? "passed" : "failed"})` : "waiting"}`, section);
      if (result) testNode("p", `${result.detail} · ${(result.latency_ms / 1000).toFixed(1)} s`, section);
      probes.set(event.probe, section);
    } else if (event.event === "request_started") {
      const parent = probes.get(event.probe) || container;
      const request = testNode("section", undefined, parent);
      requests.set(event.request_id, request);
      testNode("h3", `Request ${event.request_id} · attempt ${event.attempt}`, request);
      if (event.truncated) testNode("p", event.detail, request);
      for (const message of event.messages || []) {
        testNode("h4", message.role || "message", request);
        testNode("pre", typeof message.content === "string" ? message.content : JSON.stringify(message.content, null, 2), request);
      }
      if ((event.tools || []).length) {
        const schemas = testNode("details", undefined, request);
        schemas.dataset.key = `tools:${event.request_id}`;
        schemas.open = sameRun && opened.has(schemas.dataset.key);
        testNode("summary", "Available simulated tools", schemas);
        testNode("pre", JSON.stringify(event.tools, null, 2), schemas);
      }
      const waiting = testNode("p", "Waiting for model reply…", request);
      waiting.className = "model-test-waiting";
      waiting.dataset.started = event.timestamp;
    } else if (event.event === "response_received") {
      const request = requests.get(event.request_id) || probes.get(event.probe) || container;
      request.querySelector(".model-test-waiting")?.remove();
      testNode("h4", "Model reply", request);
      if (event.truncated) { testNode("p", event.detail, request); continue; }
      testNode("pre", event.response.content || "(No prose reply)", request);
      if (event.response.tool_calls.length) {
        testNode("h4", "Requested simulated actions", request);
        testNode("pre", JSON.stringify(event.response.tool_calls, null, 2), request);
      }
      testNode("p", `Finish: ${event.response.finish_reason || "unknown"} · ${(event.latency_ms / 1000).toFixed(1)} s`, request);
    } else if (event.event === "request_failed") {
      const request = requests.get(event.request_id) || probes.get(event.probe) || container;
      request.querySelector(".model-test-waiting")?.remove();
      testNode("p", `Request failed: ${event.error || event.detail}`, request);
    }
  }
}
async function refreshModelTests() {
  if (modelTestsBusy || !modelTestsDialog.open) return;
  modelTestsBusy = true;
  try {
    const result = await api(`/api/model/tests${modelTestsSelected ? `?run_id=${encodeURIComponent(modelTestsSelected)}` : ""}`);
    const select = byId("model-tests-run");
    select.replaceChildren();
    const latest = testNode("option", "Latest run (follow)", select);
    latest.value = "";
    for (const run of result.runs) {
      const option = testNode("option", `${run.alias || run.model_path} · ${new Date(run.started_at).toLocaleString()} · ${run.state}`, select);
      option.value = run.run_id;
    }
    select.value = modelTestsSelected;
    renderModelTests(result.selected);
  } catch (error) { byId("model-tests-status").textContent = `Test preview unavailable: ${error.message}`; }
  finally { modelTestsBusy = false; }
}
byId("model-tests-open").addEventListener("click", () => {
  modelTestsDialog.showModal();
  refreshModelTests();
});
byId("model-tests-run").addEventListener("change", event => {
  modelTestsSelected = event.target.value;
  refreshModelTests();
});
setInterval(refreshModelTests, 1500);
