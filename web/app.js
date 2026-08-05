// Tutore front-end: push-to-talk voice loop.
// Record (hold mic / Space) -> POST /api/turn -> render transcript + corrections
// + Marco's reply, and auto-play the returned audio.

const $ = (sel) => document.querySelector(sel);
const log = $("#log");
const micBtn = $("#mic");
const micLabel = $("#micLabel");
const statusEl = $("#status");
const turnTpl = $("#turnTpl");

const state = {
  sessionId: Math.floor(Math.random() * 1e9),
  mediaRecorder: null,
  chunks: [],
  recording: false,
  busy: false,
};

// ---- init ----
init();
async function init() {
  try {
    const info = await (await fetch("/api/backend")).json();
    $("#backend").textContent = `${info.backend} · ${info.model}`;
  } catch {
    $("#backend").textContent = "backend offline";
  }
  try {
    const p = await (await fetch("/api/profile")).json();
    $("#level").value = p.level || "A2";
    $("#speed").value = p.tts_speed ?? 0.9;
    $("#speedVal").textContent = (p.tts_speed ?? 0.9).toFixed(2) + "×";
  } catch {}
}

// ---- speed slider persists to profile ----
$("#speed").addEventListener("input", (e) => {
  $("#speedVal").textContent = (+e.target.value).toFixed(2) + "×";
});
$("#speed").addEventListener("change", (e) => {
  fetch("/api/profile", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tts_speed: +e.target.value }),
  }).catch(() => {});
});
$("#level").addEventListener("change", (e) => {
  fetch("/api/profile", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ level: e.target.value }),
  }).catch(() => {});
});

// ---- push-to-talk wiring ----
micBtn.addEventListener("pointerdown", startRecording);
micBtn.addEventListener("pointerup", stopRecording);
micBtn.addEventListener("pointerleave", stopRecording);

window.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !e.repeat && !isTyping(e)) {
    e.preventDefault();
    startRecording();
  }
});
window.addEventListener("keyup", (e) => {
  if (e.code === "Space" && !isTyping(e)) {
    e.preventDefault();
    stopRecording();
  }
});
function isTyping(e) {
  const t = e.target.tagName;
  return t === "INPUT" || t === "TEXTAREA" || t === "SELECT";
}

async function startRecording() {
  if (state.recording || state.busy) return;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    setStatus("Microphone access denied.");
    return;
  }
  state.chunks = [];
  const mr = new MediaRecorder(stream);
  mr.ondataavailable = (e) => e.data.size && state.chunks.push(e.data);
  mr.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    handleRecording();
  };
  state.mediaRecorder = mr;
  mr.start();
  state.recording = true;
  micBtn.classList.add("recording");
  micLabel.textContent = "Listening…";
  setStatus("");
}

function stopRecording() {
  if (!state.recording) return;
  state.recording = false;
  micBtn.classList.remove("recording");
  micLabel.textContent = "Hold to speak";
  try {
    state.mediaRecorder && state.mediaRecorder.stop();
  } catch {}
}

async function handleRecording() {
  const blob = new Blob(state.chunks, { type: "audio/webm" });
  if (blob.size < 1200) {
    setStatus("Too short — hold the button while you speak.");
    return;
  }
  setBusy(true, "Transcribing & thinking…");

  const fd = new FormData();
  fd.append("audio", blob, "speech.webm");
  fd.append("mode", $("#mode").value);
  fd.append("level", $("#level").value);
  fd.append("session_id", String(state.sessionId));

  try {
    const res = await fetch("/api/turn", { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    renderTurn(data);
    setStatus(timingLine(data.timings));
  } catch (err) {
    setStatus("Error: " + (err.message || err));
  } finally {
    setBusy(false);
  }
}

function renderTurn(data) {
  $("#hint")?.remove();
  const node = turnTpl.content.cloneNode(true);

  node.querySelector(".said").textContent = data.transcript || "(nothing heard)";
  const conf = node.querySelector(".conf");
  const pct = Math.round((data.stt_confidence || 0) * 100);
  conf.textContent = `speech clarity ~${pct}%`;
  if (pct < 55) {
    conf.classList.add("low");
    conf.textContent += " — try enunciating a touch more";
  }

  const corr = node.querySelector(".corrections");
  (data.corrections || []).forEach((c) => corr.appendChild(correctionEl(c)));

  node.querySelector(".reply").textContent = data.reply || "";

  const vocab = node.querySelector(".vocab");
  (data.new_vocab || []).forEach((v) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.innerHTML = `<b>${escapeHtml(v.italian)}</b> — ${escapeHtml(v.english || "")}`;
    if (v.example) chip.title = v.example;
    vocab.appendChild(chip);
  });

  const audio = playAudio(data.audio_b64);
  node.querySelector(".replay").addEventListener("click", () => audio && audio.play());

  log.appendChild(node);
  log.scrollTop = log.scrollHeight;
}

function correctionEl(c) {
  const el = document.createElement("div");
  el.className = "correction";
  const was = c.your_text ? `<span class="was">${escapeHtml(c.your_text)}</span> → ` : "";
  el.innerHTML =
    `<div>${was}<span class="fix">${escapeHtml(c.correction || "")}</span></div>` +
    (c.explanation ? `<div class="why">${escapeHtml(c.explanation)}</div>` : "") +
    (c.tag ? `<span class="tag">${escapeHtml(c.tag)}</span>` : "");
  return el;
}

function playAudio(b64) {
  if (!b64) return null;
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const url = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
  const audio = new Audio(url);
  audio.play().catch(() => {});
  return audio;
}

function setBusy(busy, msg) {
  state.busy = busy;
  micBtn.classList.toggle("busy", busy);
  micBtn.disabled = busy;
  micLabel.textContent = busy ? "Thinking…" : "Hold to speak";
  if (msg !== undefined) setStatus(msg);
}
function setStatus(msg) {
  statusEl.textContent = msg;
}
function timingLine(t) {
  if (!t) return "";
  const s = (ms) => (ms / 1000).toFixed(1) + "s";
  const total = (t.stt_ms || 0) + (t.llm_ms || 0) + (t.tts_ms || 0);
  return `⏱ ${s(total)} total — hear ${s(t.stt_ms)} · think ${s(t.llm_ms)} · speak ${s(t.tts_ms)}`;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

// ---- Anki panel ----
const ankiPanel = $("#ankiPanel");
$("#ankiBtn").addEventListener("click", async () => {
  ankiPanel.showModal();
  await refreshAnki();
});

async function refreshAnki() {
  const dot = $("#ankiDot");
  const statusEl = $("#ankiStatus");
  try {
    const s = await (await fetch("/api/anki/status")).json();
    const deckSel = $("#deck");
    deckSel.innerHTML = "";
    (s.decks || []).forEach((d) => {
      const o = document.createElement("option");
      o.value = o.textContent = d;
      deckSel.appendChild(o);
    });
    // Ensure the default deck is selectable even if it doesn't exist yet.
    if (![...deckSel.options].some((o) => o.value === s.default_deck)) {
      const o = document.createElement("option");
      o.value = o.textContent = s.default_deck + " (will be created)";
      o.dataset.real = s.default_deck;
      deckSel.appendChild(o);
    }
    if (s.available) {
      dot.className = "dot on";
      statusEl.className = "anki-status on";
      statusEl.textContent = "✓ Anki connected";
    } else {
      dot.className = "dot off";
      statusEl.className = "anki-status off";
      statusEl.textContent = "Anki not detected — open Anki desktop with the AnkiConnect add-on to push cards.";
    }
  } catch {
    dot.className = "dot off";
    statusEl.textContent = "Could not reach the server.";
  }
}
// Reflect Anki status on the topbar dot at load.
refreshAnki();

function chosenDeck() {
  const sel = $("#deck");
  const typed = $("#deckNew").value.trim();
  if (typed) return typed;
  const opt = sel.selectedOptions[0];
  return opt ? (opt.dataset.real || opt.value) : "";
}

$("#genCards").addEventListener("click", async () => {
  const btn = $("#genCards");
  const out = $("#genResult");
  btn.disabled = true;
  out.textContent = "Handing off to the big model (qwen3:14b) to craft cards… this can take a bit.";
  try {
    const res = await fetch("/api/cards/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ deck: chosenDeck() || undefined, session_id: state.sessionId }),
    });
    const data = await res.json();
    out.textContent = data.message || `Generated ${data.generated}, pushed ${data.pushed}.`;
    await refreshAnki();
  } catch (err) {
    out.textContent = "Error: " + (err.message || err);
  } finally {
    btn.disabled = false;
  }
});

$("#importVocab").addEventListener("click", async () => {
  const out = $("#importResult");
  const fd = new FormData();
  fd.append("text", $("#knownVocab").value);
  try {
    const data = await (await fetch("/api/vocab/import", { method: "POST", body: fd })).json();
    out.textContent = `Imported ${data.added} new, ${data.skipped} already known.`;
    $("#knownVocab").value = "";
  } catch (err) {
    out.textContent = "Error: " + (err.message || err);
  }
});
