const $ = (id) => document.getElementById(id);
let backends = [];
let current = null;
const prompts = {}; // key -> edited prompt; absent means the backend default
let busy = false;
let output = "";
let raw = "";
let examples = [];
let retrieved = ""; // `${backend}\n${query}` the cached examples belong to

function status(text, kind = "") {
  $("status").textContent = text;
  $("status").className = `status ${kind}`;
}

async function request(path, data) {
  const response = await fetch(path, data === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "The request failed. Please try again.");
  return result;
}

function setBusy(value) {
  busy = value;
  for (const id of ["generate", "retrieve", "query", "prompt", "reset-prompt", "known"]) $(id).disabled = value;
  for (const card of document.querySelectorAll(".backend-card")) card.disabled = value;
  $("query-form").setAttribute("aria-busy", String(value));
}

// Highlight using text nodes: catalog and model content never become HTML.
function showJSON(element, text) {
  element.replaceChildren();
  const tokens = /"(?:\\.|[^"\\])*"\s*:|"(?:\\.|[^"\\])*"|\b(?:true|false|null|\d+(?:\.\d+)?)\b/g;
  let last = 0;
  for (const match of text.matchAll(tokens)) {
    element.append(document.createTextNode(text.slice(last, match.index)));
    const span = document.createElement("span");
    span.className = match[0].endsWith(":") ? "json-key" : match[0].startsWith('"') ? "json-string" : "json-number";
    span.textContent = match[0];
    element.append(span);
    last = match.index + match[0].length;
  }
  element.append(document.createTextNode(text.slice(last)));
}

function card(rank, text, meta, tag, tagClass = "") {
  const node = $("example-template").content.cloneNode(true);
  node.querySelector(".example-rank").textContent = rank;
  node.querySelector(".similarity").textContent = tag;
  node.querySelector(".similarity").className = `similarity ${tagClass}`;
  node.querySelector(".example-text").textContent = text;
  showJSON(node.querySelector("pre"), JSON.stringify(meta, null, 2));
  for (const layer of meta.layers || []) {
    const chip = document.createElement("span");
    chip.className = "layer-chip";
    chip.textContent = layer;
    node.querySelector(".example-layers").append(chip);
  }
  return node;
}

function clearResult() {
  output = raw = "";
  $("result-content").hidden = true;
  $("result-empty").hidden = false;
  $("result-badge").textContent = "OUTPUT";
  $("result-badge").className = "small-tag";
  $("toggle-raw").setAttribute("aria-pressed", "false");
  $("result-footer-text").textContent = "Schema-compiled. Strictly judged when a record exists.";
  $("gold").replaceChildren();
  $("gold-empty").hidden = false;
  $("verdict").textContent = "NO RECORD";
  $("verdict").className = "count-pill";
}

function renderExamples() {
  $("examples").replaceChildren();
  $("examples-empty").hidden = examples.length > 0;
  $("example-count").textContent = `${examples.length} / 5 RETRIEVED`;
  examples.forEach((example, i) => {
    const score = typeof example.score === "number" ? `${example.score.toFixed(3)} similarity` : "sent";
    $("examples").append(card(String(i + 1).padStart(2, "0"), example.text, example.meta, score));
  });
}

function renderGold(query, result) {
  if (result.expected === null || result.expected === undefined) return;
  const verdict = result.exact ? "match" : "mismatch";
  $("gold").replaceChildren(card("GOLD", query, result.expected, `strict comparator: ${verdict}`, verdict));
  $("gold").querySelector("details").open = true;
  $("gold-empty").hidden = true;
  $("verdict").textContent = result.exact ? "EXACT MATCH" : "MISMATCH";
  $("verdict").className = `count-pill ${verdict}`;
}

function showResult(result) {
  $("result-empty").hidden = true;
  $("result-content").hidden = false;
  if (!result.valid) {
    output = raw = result.error;
    $("result-code").textContent = result.error;
    $("copy").textContent = "Copy message";
    if (result.clarification) {
      $("result-badge").textContent = "NEEDS CLARIFICATION";
      $("result-badge").className = "small-tag";
      $("result-description").textContent = "QUESTION FOR YOU";
      $("result-summary").textContent = "The backend will not guess. Rephrase with the intended field.";
      $("result-footer-text").textContent = `No model call · ${current.label}`;
      status("Rephrase the question and generate again.", "error");
      return;
    }
    $("result-badge").textContent = "INVALID OUTPUT";
    $("result-badge").className = "small-tag invalid";
    $("result-description").textContent = "DECODING ERROR";
    $("result-summary").textContent = "The backend did not produce a schema-valid FELN.";
    $("result-footer-text").textContent = `Failed after ${result.seconds.toFixed(2)} s · ${current.label}`;
    status("Decoding stopped before a valid FELN was produced.", "error");
    return;
  }
  output = JSON.stringify(result.feln, null, 2);
  raw = result.raw;
  showJSON($("result-code"), output);
  $("result-badge").textContent = "VALID FELN";
  $("result-badge").className = "small-tag valid";
  $("result-description").textContent = "FELN / JSON · schema-compiled";
  $("copy").textContent = "Copy JSON";
  $("result-summary").textContent = `${result.feln.layers.join(" → ")} · ${result.feln.relations.join(" · ") || "No spatial join"}`;
  const bits = [`${result.seconds.toFixed(2)} s`];
  if (result.generated_tokens != null) bits.unshift(`${result.generated_tokens} tokens`);
  if (result.first_token_seconds != null) bits.push(`first token ${result.first_token_seconds.toFixed(2)} s`);
  if (result.prompt_tokens != null) bits.push(`${result.prompt_tokens} prompt tokens`);
  if (result.attempts > 1) bits.push(`${result.attempts} attempts`);
  bits.push(current.label);
  $("result-footer-text").textContent = bits.join(" · ");
}

async function retrieve(query) {
  const stamp = `${current.key}\n${query}`;
  if (retrieved === stamp && examples.length === 5) return;
  status("Finding the five closest examples… The first query may take a moment to load the encoder.", "busy");
  examples = (await request("/api/retrieve", { query, backend: current.key })).examples;
  retrieved = stamp;
  renderExamples();
}

async function run(generate) {
  if (busy || !current || !$("query").reportValidity()) return;
  const query = $("query").value.trim();
  if (!query) return status("Enter a question to get started.", "error");
  const prompt = $("prompt").value;
  if (generate && !prompt.trim()) {
    $("prompt").closest("details").open = true;
    $("prompt").focus();
    return status("Enter a prompt before generating.", "error");
  }
  setBusy(true);
  clearResult();
  try {
    if (current.retrieves) await retrieve(query);
    if (!generate) return status("Five examples ready. Expand any card to inspect its FELN, or generate your answer.");
    $("result-badge").textContent = "GENERATING";
    status(`Decoding with ${current.label}…`, "busy");
    const body = { query, backend: current.key };
    if (prompt !== current.prompt) body.prompt = prompt;
    if (current.retrieves) body.ids = examples.map((example) => example.id);
    const result = await request("/api/generate", body);
    showResult(result);
    if (!result.valid) return;
    renderGold(query, result);
    status(result.exact === null || result.exact === undefined
      ? "Your FELN is ready. Validated against the catalog."
      : result.exact ? "Your FELN matches the recorded answer." : "Your FELN differs from the recorded answer. Compare the two.",
      result.exact === false ? "error" : "");
  } catch (error) {
    $("result-badge").textContent = "TRY AGAIN";
    status(error.message || "Could not connect to the server. Please try again.", "error");
  } finally {
    setBusy(false);
  }
}

function choose(key) {
  current = backends.find((backend) => backend.key === key) || backends[0];
  for (const card of document.querySelectorAll(".backend-card")) {
    card.setAttribute("aria-checked", String(card.dataset.key === current.key));
  }
  $("prompt").value = prompts[current.key] ?? current.prompt;
  $("prompt-label").textContent = current.prompt_label;
  $("backend-info").textContent = current.info;
  $("backend-decoding").textContent = current.decoding;
  $("retrieve").hidden = !current.retrieves;
  $("examples-section").hidden = !current.retrieves;
  $("action-note").textContent = current.retrieves
    ? "Generation sends the prompt, five examples, and your question to the configured provider."
    : "Greedy decoding on this Mac. Nothing leaves it.";
  examples = [];
  retrieved = "";
  renderExamples();
  clearResult();
  status(current.healthy
    ? `${current.label} selected. Generate a new FELN.`
    : `${current.label} is not answering. Start its server, then reload.`, current.healthy ? "" : "error");
}

function renderBackends() {
  $("backends").replaceChildren(...backends.map((backend) => {
    const node = $("backend-template").content.cloneNode(true);
    const button = node.querySelector(".backend-card");
    button.dataset.key = backend.key;
    button.querySelector(".backend-family").textContent = backend.family.toUpperCase();
    button.querySelector(".health").className = `health ${backend.healthy ? "" : "down"}`;
    button.querySelector(".backend-label").textContent = backend.label;
    button.querySelector(".backend-decoding").textContent = backend.decoding;
    button.addEventListener("click", () => { if (!busy) choose(backend.key); });
    return node;
  }));
  const healthy = backends.filter((backend) => backend.healthy).length;
  $("rail-count").textContent = `${healthy} / ${backends.length} ONLINE`;
}

$("query-form").addEventListener("submit", (event) => { event.preventDefault(); run(true); });
$("retrieve").addEventListener("click", () => run(false));
document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") { event.preventDefault(); run(true); }
});
$("query").addEventListener("input", () => {
  $("character-count").textContent = `${$("query").value.length} characters`;
  if ($("known").value !== $("query").value) $("known").value = "";
  examples = [];
  retrieved = "";
  renderExamples();
  clearResult();
  status("Query changed. Generate a new FELN.");
});
$("known").addEventListener("change", () => {
  if (!$("known").value) return;
  $("query").value = $("known").value;
  $("query").dispatchEvent(new Event("input"));
  $("known").value = $("query").value;
});
$("prompt").addEventListener("input", () => {
  prompts[current.key] = $("prompt").value;
  clearResult();
  status("Prompt changed. Generate to see the effect.");
});
$("reset-prompt").addEventListener("click", () => {
  delete prompts[current.key];
  $("prompt").value = current.prompt;
  clearResult();
  status("Default prompt restored.");
});
$("toggle-raw").addEventListener("click", () => {
  const showRaw = $("toggle-raw").getAttribute("aria-pressed") !== "true";
  $("toggle-raw").setAttribute("aria-pressed", String(showRaw));
  if (showRaw) $("result-code").textContent = raw; else showJSON($("result-code"), output);
  $("result-description").textContent = showRaw ? "RAW MODEL OUTPUT" : "FELN / JSON · schema-compiled";
  $("copy").textContent = showRaw ? "Copy raw" : "Copy JSON";
});
$("copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("toggle-raw").getAttribute("aria-pressed") === "true" ? raw : output);
    $("copy").textContent = "Copied!";
  } catch {
    status("Clipboard unavailable. Select and copy the output directly.", "error");
  }
});

async function init() {
  $("character-count").textContent = `${$("query").value.length} characters`;
  try {
    const config = await request("/api/config");
    backends = config.backends;
    renderBackends();
    $("known").append(...config.questions.map((question) => new Option(question, question)));
    $("corpus-info").textContent = `${backends.length} backend${backends.length === 1 ? "" : "s"} · ${config.questions.length} recorded questions · ${config.catalog}`;
    setBusy(false);
    choose((backends.find((backend) => backend.healthy) || backends[0]).key);
    status("Workspace ready. Pick a backend, ask, or choose a recorded question.");
  } catch {
    $("backend-info").textContent = "Disconnected";
    $("rail-count").textContent = "OFFLINE";
    status("Could not connect to the local server. Start python -m feln_studio.server and reload this page.", "error");
  }
}
init();
