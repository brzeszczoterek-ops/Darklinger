"use strict";

const sessionToken = document.querySelector('meta[name="paladyn-session"]').content;
const headers = {"X-PALADYN-Session": sessionToken};
const jsonHeaders = {...headers, "Content-Type": "application/json"};

const byId = (id) => document.getElementById(id);
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
  ownerTitle: byId("owner-title"),
  ownerSubtitle: byId("owner-subtitle"),
  ownerCards: byId("owner-cards"),
  ownerCapabilities: byId("owner-capabilities"),
  foundry: byId("foundry-state"),
  capabilityAuditList: byId("capability-audit-list"),
  proposalList: byId("proposal-list"),
  shutdown: byId("shutdown"),
};

let working = false;
let recording = false;
let lastReady = null;
let requestInFlight = false;

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
  ui.send.disabled = value;
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
  ui.ownerTitle.textContent = owner.title || "OWNER DECK";
  ui.ownerSubtitle.textContent = owner.subtitle || "Private operational surface";
  ui.ownerCards.replaceChildren();
  (owner.cards || []).forEach((item) => {
    const card = document.createElement("div");
    card.className = "owner-card";
    const label = document.createElement("span");
    const value = document.createElement("strong");
    label.textContent = item.label;
    value.textContent = item.value;
    card.append(label, value);
    ui.ownerCards.append(card);
  });
  renderChips(ui.ownerCapabilities, owner.capabilities || [], 10);
  ui.foundry.textContent = `FOUNDRY // ${owner.foundry || "unavailable"}`;
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
    if (!response.ok) throw new Error(`status ${response.status}`);
    const state = await response.json();
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
    if (!requestInFlight) setWorking(!state.ready);
    if (lastReady !== state.ready) {
      feed(state.ready ? "Runtime ready" : "Runtime entered active task");
      lastReady = state.ready;
    }
  } catch (error) {
    ui.runtime.className = "status-pill";
    ui.runtime.innerHTML = "<i></i> LINK LOST";
    ui.entity.textContent = "CONNECTION LOST";
  }
}

async function sendPrompt(text) {
  const prompt = text.trim();
  if (!prompt || working) return;
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

let killTimer = null;
function disarmKill() {
  clearTimeout(killTimer);
  killTimer = null;
  ui.shutdown.classList.remove("arming");
}
function armKill() {
  if (killTimer) return;
  ui.shutdown.classList.add("arming");
  killTimer = setTimeout(async () => {
    feed("Shutdown requested — stopping model and V");
    await fetch("/api/shutdown", {method: "POST", headers: jsonHeaders, body: "{}"});
    ui.shutdown.textContent = "STOPPING";
  }, 1000);
}
ui.shutdown.addEventListener("pointerdown", armKill);
ui.shutdown.addEventListener("pointerup", disarmKill);
ui.shutdown.addEventListener("pointerleave", disarmKill);
ui.shutdown.addEventListener("pointercancel", disarmKill);

refreshStatus();
setInterval(refreshStatus, 2000);
ui.prompt.focus();
