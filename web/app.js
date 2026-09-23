/* llm-bench WebUI */
const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const api = async (path, opts = {}) => {
  const init = { ...opts };
  if (init.body != null) {
    // fetch() defaults string bodies to text/plain, which FastAPI rejects;
    // every POST here sends JSON.
    init.headers = { "Content-Type": "application/json", ...(init.headers || {}) };
  }
  const r = await fetch(path, init);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const body = await r.json();
      let d = body.detail;
      if (Array.isArray(d)) d = d.map(e => `${(e.loc || []).join(".")}: ${e.msg}`).join("; ");
      else if (d && typeof d === "object") d = JSON.stringify(d);
      if (d) msg = String(d);
    } catch {}
    throw new Error(msg);
  }
  return r.status === 204 ? null : r.json();
};
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 1) => v == null ? "" : Number(v).toFixed(d);

/* ---------- tabs ---------- */
let pollTimer = null;
let pendingPreselect = null;
$$(".tab").forEach(b => b.onclick = () => {
  $$(".tab").forEach(x => x.classList.toggle("active", x === b));
  $$(".tab-page").forEach(p => p.classList.add("hidden"));
  const tab = b.dataset.tab;
  $(`#tab-${tab}`).classList.remove("hidden");
  clearInterval(pollTimer);
  if (tab === "sweeps") { renderSweeps(); pollTimer = setInterval(renderSweeps, 1500); }
  if (tab === "results") renderResultPicker(pendingPreselect);
  if (tab === "logs") { renderLogs(); pollTimer = setInterval(renderLogs, 3000); }
  if (tab === "engines") renderEngines();
  if (tab === "new") loadEngineSelect();
  pendingPreselect = null;
});

/* ---------- sweeps ---------- */
async function renderSweeps() {
  const sweeps = await api("/api/sweeps");
  const el = $("#sweep-list");
  if (!sweeps.length) { el.innerHTML = `<div class="card">No sweeps yet. <button onclick="$('[data-tab=new]').click()">Create one</button></div>`; return; }
  el.innerHTML = sweeps.map(s => {
    const p = s.status === "running" ? progressCard(s) : "";
    const models = (s.spec.models && s.spec.models.length ? s.spec.models : (s.spec.model ? [s.spec.model] : []));
    const modelStr = models.length ? models.map(m => m.split("/").pop()).join(", ") : "(none)";
    return `<div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <b>#${s.id} ${esc(s.name)}</b> <span class="badge ${esc(s.status)}">${esc(s.status)}</span>
      </div>
      <div class="hint">${esc(s.spec.engine)} · model ${esc(modelStr)} ·
        ${s.spec.workloads.length} workload(s) · ${s.spec.repetitions} reps · ${esc(s.created_at)}</div>
      ${p}
      <div class="actions">
        ${s.status === "running"
          ? `<button class="danger small" onclick="cancelSweep(${s.id})">Cancel</button>`
          : `<button class="small" onclick="runSweep(${s.id})">Run</button>`}
        <button class="small ghost" onclick="sweepDetail(${s.id})">Variants</button>
        <button class="small ghost" onclick="viewResults(${s.id})">Results</button>
        <button class="small ghost" onclick="cloneSweep(${s.id})">Clone</button>
        <button class="small ghost" onclick="deleteSweep(${s.id})">Delete</button>
      </div>
    </div>`;
  }).join("");
}
function viewResults(id) {
  pendingPreselect = id;
  $('[data-tab="results"]').click();
}
async function cloneSweep(id) {
  try {
    const s = await api(`/api/sweeps/${id}`);
    const sp = s.spec;
    $("#sf-name").value = (sp.name || "sweep") + " (copy)";
    await loadEngineSelect();
    $("#sf-engine").value = sp.engine;
    // models
    $("#models-table tbody").innerHTML = "";
    const models = (sp.models && sp.models.length ? sp.models : (sp.model ? [sp.model] : []));
    (models.length ? models : [""]).forEach(m => addModel(m));
    $("#sf-base").value = sp.base_args || "";
    $("#sf-baseenv").value = Object.entries(sp.base_env || {}).map(([k, v]) => `${k}=${v}`).join("\n");
    // workloads
    $("#wl-table tbody").innerHTML = "";
    (sp.workloads && sp.workloads.length ? sp.workloads : [{ kind: "pg", n_prompt: 512, n_gen: 128 }])
      .forEach(w => addWl(w.kind, w.n_prompt, w.n_gen));
    // dimensions
    $("#dim-table tbody").innerHTML = "";
    (sp.dimensions && sp.dimensions.length ? sp.dimensions : [])
      .forEach(d => addDim(d.name || "", d.args || "", d.values || "", d.type || "arg"));
    $("#sf-reps").value = sp.repetitions ?? 3;
    $("#sf-warmup").checked = sp.warmup !== false;
    $("#sf-cooldown").value = sp.cooldown_s ?? 2;
    $("#sf-startup").value = sp.startup_timeout_s ?? 300;
    $("#plan-preview").textContent = "";
    $('[data-tab="new"]').click();
  } catch (e) { alert(e.message); }
}
function progressCard(s) {
  const pr = s.progress || {};
  const pct = pr.total ? Math.round(100 * (pr.done || 0) / pr.total) : 0;
  return `<div class="progress"><div style="width:${pct}%"></div></div>
    <div class="hint">${pr.done || 0}/${pr.total || "?"} variants — ${esc(pr.variant_label || "")} ${esc(pr.workload || "")} ${pr.rep != null ? `rep ${pr.rep}` : ""} ${esc(pr.message || "")}</div>`;
}
async function runSweep(id) { try { await api(`/api/sweeps/${id}/run`, { method: "POST" }); } catch (e) { alert(e.message); } renderSweeps(); }
async function cancelSweep(id) { await api(`/api/sweeps/${id}/cancel`, { method: "POST" }); }
async function deleteSweep(id) { if (confirm(`Delete sweep ${id}?`)) { await api(`/api/sweeps/${id}`, { method: "DELETE" }); renderSweeps(); } }

/* ---------- new sweep form ---------- */
function addWl(kind = "pg", pp = 512, tg = 128) {
  const tr = document.createElement("tr");
  tr.innerHTML = `<td><select class="wl-kind"><option value="pp">pp (prefill)</option><option value="tg">tg (decode)</option><option value="pg">pg (both)</option></select></td>
    <td><input class="wl-pp" type="number" value="${pp}"></td>
    <td><input class="wl-tg" type="number" value="${tg}"></td>
    <td><button type="button" class="small danger" onclick="this.closest('tr').remove()">×</button></td>`;
  tr.querySelector(".wl-kind").value = kind;
  $("#wl-table tbody").appendChild(tr);
}
function addModel(path = "") {
  const tr = document.createElement("tr");
  tr.innerHTML = `<td><input class="model-path" value="${esc(path)}" placeholder="/models/Qwen2.5-7B-Q4_K_M.gguf"></td>
    <td><button type="button" class="small danger" onclick="this.closest('tr').remove()">×</button></td>`;
  $("#models-table tbody").appendChild(tr);
}
function addDim(name = "", args = "", values = "", type = "arg") {
  const tr = document.createElement("tr");
  tr.innerHTML = `<td><input class="dim-name" value="${esc(name)}" placeholder="flash-attn"></td>
    <td><select class="dim-type">
      <option value="arg" ${type === "arg" ? "selected" : ""}>arg</option>
      <option value="env" ${type === "env" ? "selected" : ""}>env</option>
    </select></td>
    <td><input class="dim-args" value="${esc(args)}" placeholder="empty = --flash-attn value"></td>
    <td><input class="dim-values" value="${esc(values)}" placeholder="arg: on,off,auto · env: CUDA_VISIBLE_DEVICES=0;=1"></td>
    <td><button type="button" class="small danger" onclick="this.closest('tr').remove()">×</button></td>`;
  $("#dim-table tbody").appendChild(tr);
}
// Parse "KEY=value" lines (bare KEY -> "1") into an object.
function parseEnvLines(text) {
  const o = {};
  for (const line of String(text || "").split("\n")) {
    const t = line.trim();
    if (!t) continue;
    const i = t.indexOf("=");
    if (i >= 0) o[t.slice(0, i).trim()] = t.slice(i + 1).trim();
    else o[t] = "1";
  }
  return o;
}
$("#wl-add").onclick = () => addWl();
$("#dim-add").onclick = () => addDim();
$("#models-add").onclick = () => addModel();

function formSpec() {
  // NOTE: never read fields via form.<name> — "name" collides with the
  // HTMLFormElement.name built-in. Always use explicit ids.
  const models = $$("#models-table tbody tr").map(tr => $(".model-path", tr).value.trim()).filter(Boolean);
  return {
    name: $("#sf-name").value.trim(),
    engine: $("#sf-engine").value,
    model: models[0] || "",
    models: models,
    base_args: $("#sf-base").value.trim(),
    base_env: parseEnvLines($("#sf-baseenv").value),
    repetitions: +$("#sf-reps").value || 3,
    warmup: $("#sf-warmup").checked,
    cooldown_s: +$("#sf-cooldown").value || 0,
    startup_timeout_s: +$("#sf-startup").value || 300,
    workloads: $$("#wl-table tbody tr").map(tr => ({
      kind: $(".wl-kind", tr).value,
      n_prompt: +$(".wl-pp", tr).value || 0,
      n_gen: +$(".wl-tg", tr).value || 0,
    })),
    dimensions: $$("#dim-table tbody tr").map(tr => ({
      name: $(".dim-name", tr).value.trim(),
      type: $(".dim-type", tr).value,
      args: $(".dim-args", tr).value,
      values: $(".dim-values", tr).value.trim(),
    })).filter(d => d.name && d.values),
  };
}
$("#btn-plan").onclick = async () => {
  try {
    const p = await api("/api/sweeps/plan", { method: "POST", body: JSON.stringify(formSpec()) });
    let html = `→ ${p.n_variants} variant(s) × workloads × reps = ${p.n_requests_total} measured requests. ` +
      `Variants: ${esc(p.variants.slice(0, 8).join(" | "))}${p.variants.length > 8 ? " …" : ""}`;
    if (p.example_command) {
      html += `<br>Exact command for the first variant (port is auto-assigned at run time):<br><code class="cmd-preview">${esc(p.example_command)}</code>`;
    }
    if (p.example_env && Object.keys(p.example_env).length) {
      const envStr = Object.entries(p.example_env).map(([k, v]) => `${k}=${v}`).join("  ");
      html += `<br>Env for the first variant:<br><code class="cmd-preview">${esc(envStr)}</code>`;
    }
    $("#plan-preview").innerHTML = html;
  } catch (e) { $("#plan-preview").textContent = `⚠ ${e.message}`; }
};
$("#sweep-form").onsubmit = async ev => {
  ev.preventDefault();
  try {
    const { id } = await api("/api/sweeps", { method: "POST", body: JSON.stringify(formSpec()) });
    await api(`/api/sweeps/${id}/run`, { method: "POST" });
    $('[data-tab="sweeps"]').click();
  } catch (e) { alert(e.message); }
};
async function loadEngineSelect() {
  const engines = await api("/api/engines");
  const cur = $("#sf-engine").value;
  $("#sf-engine").innerHTML = engines.map(e => `<option value="${esc(e.name)}">${esc(e.name)}</option>`).join("");
  // Preserve the current selection across reloads (e.g. when cloning a sweep
  // and the tab switch re-populates the list).
  if (cur && [...$("#sf-engine").options].some(o => o.value === cur)) $("#sf-engine").value = cur;
}

/* ---------- results ---------- */
let selectedSweeps = new Set();
async function renderResultPicker(preselect) {
  if (preselect != null) selectedSweeps.add(preselect);
  const sweeps = await api("/api/sweeps");
  const el = $("#result-pick");
  // Checkboxes reflect the actual selection so the view never drifts from it.
  el.innerHTML = sweeps.map(s =>
    `<label class="checkbox"><input type="checkbox" value="${s.id}" ${selectedSweeps.has(s.id) ? "checked" : ""}>
      #${s.id} ${esc(s.name)} <span class="badge ${esc(s.status)}">${esc(s.status)}</span></label>`).join("")
    || `<em>No sweeps.</em>`;
  $$("input", el).forEach(cb => cb.onchange = () => { cb.checked ? selectedSweeps.add(+cb.value) : selectedSweeps.delete(+cb.value); renderResults(); });
  // Refresh the results to exactly match what is checked (on every visit).
  renderResults();
}
function selectSweep(id) { const cb = $(`#result-pick input[value="${id}"]`); if (cb) { cb.checked = true; cb.onchange(); } }

const METRICS = [
  // [key, header, dir]  dir: 1 = higher is better, -1 = lower is better
  ["pp_tps", "Prefill t/s", 1], ["tg_tps", "Decode t/s", 1], ["ttft_ms", "TTFT ms", -1],
  ["e2e_ms", "End-to-end ms", -1], ["itl_p50_ms", "ITL p50 ms", -1], ["itl_p90_ms", "ITL p90 ms", -1], ["itl_p99_ms", "ITL p99 ms", -1],
];
// t: 0 = worst (red), 0.5 = mid (yellow), 1 = best (green)
function heatColor(t) {
  t = Math.max(0, Math.min(1, t));
  let r, g, b;
  if (t <= 0.5) {
    const u = t / 0.5;
    r = 235; g = Math.round(80 + u * 140); b = 70;
  } else {
    const u = (t - 0.5) / 0.5;
    r = Math.round(235 - u * 165); g = 220; b = Math.round(70 + u * 20);
  }
  return `rgba(${r},${g},${b},0.28)`;
}
// Render the light markdown the backend uses (**bold**) as HTML.
const md = s => esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
let glossary = null;
async function loadGlossary() {
  if (!glossary) { try { glossary = await api("/api/glossary"); } catch { glossary = []; } }
  return glossary;
}
let chart = null;

async function renderResults() {
  const ids = [...selectedSweeps];
  if (!ids.length) { $("#result-body").innerHTML = ""; return; }
  const rows = await api(`/api/results?sweep_ids=${ids.join(",")}`);
  const qs = ids.join(",");
  const recs = await api(`/api/results/recommend?sweep_ids=${qs}`).catch(() => []);
  const gl = await loadGlossary();
  // Per-metric min/max across all rows, for the good/bad color gradient.
  const ranges = {};
  for (const [k] of METRICS) {
    const vals = rows.map(r => r[`${k}_mean`]).filter(v => v != null);
    if (vals.length) ranges[k] = [Math.min(...vals), Math.max(...vals)];
  }
  const heat = (k, v) => {
    const m = METRICS.find(x => x[0] === k);
    if (v == null || !ranges[k] || !m) return "";
    const [lo, hi] = ranges[k];
    if (hi === lo) return ` style="background:${heatColor(0.5)}"`;
    const t = m[2] === 1 ? (v - lo) / (hi - lo) : (hi - v) / (hi - lo);
    return ` style="background:${heatColor(t)}"`;
  };
  const recHtml = recs.length ? `
    <div class="card rec-card">
      <h3>Best per workload <span class="hint">— fastest by total request time, with caveats</span></h3>
      ${recs.map(rec => `
        <div class="rec-wl">
          <div><b>${esc(rec.workload)}:</b> ${esc(rec.main)}${rec.main_value != null ? ` <span class="hint">— ${fmt(rec.main_value, 1)} ${esc(rec.main_unit)}</span>` : ""}</div>
          ${rec.caveats.map(c => `<div class="rec-caveat">⚠ ${md(c)}</div>`).join("")}
          ${rec.unstable.length ? `<div class="rec-caveat warn">◦ high run-to-run variance in: ${rec.unstable.map(esc).join(", ")} — more repetitions would firm this up.</div>` : ""}
        </div>`).join("")}
    </div>` : "";
  $("#result-body").innerHTML = `
    <div class="card">
      <h3>Export</h3>
      <div class="export-btns">
        <a href="/api/export/markdown?sweep_ids=${qs}"><button class="small">⬇ Markdown summary</button></a>
        <a href="/api/export/markdown?sweep_ids=${qs}&title=comparison"><button class="small">⬇ Markdown (multi-sweep)</button></a>
        <a href="/api/export/csv?sweep_ids=${qs}"><button class="small ghost">⬇ CSV</button></a>
        <a href="/api/export/json?sweep_ids=${qs}"><button class="small ghost">⬇ JSON</button></a>
        <a href="/api/export/sql?sweep_ids=${qs}"><button class="small ghost">⬇ SQL (SQLite)</button></a>
      </div>
    </div>
    ${recHtml}
    <div class="card">
      <label>Chart metric
        <select id="chart-metric">${METRICS.map(([k, h]) => `<option value="${k}">${h}</option>`).join("")}</select>
      </label>
      <div class="chart-box"><canvas id="chart" height="110"></canvas></div>
      <p class="chart-caption">Bars are the mean across repetitions; the vertical whiskers are ±1 standard
      deviation, so the spread of the repeated runs is visible at a glance.</p>
    </div>
    <div class="card" style="overflow-x:auto">
      <table id="results-table"><thead><tr>
        ${["Sweep", "Variant", "Test", "n", ...METRICS.map(([, h]) => h)].map(h => `<th>${h}</th>`).join("")}
      </tr></thead><tbody>
      ${rows.map(r => `<tr>
        <td>${r.sweep_id}</td><td>${esc(r.label)}</td><td>${esc(r.workload)}</td><td>${r.n}</td>
        ${METRICS.map(([k]) => {
          const m = r[`${k}_mean`], sd = r[`${k}_std`];
          return `<td${heat(k, m)}>${m == null ? "" : fmt(m, 2) + (sd ? ` <span class="hint">±${fmt(sd, 2)}</span>` : "")}</td>`;
        }).join("")}
      </tr>`).join("")}</tbody></table>
      <p class="hint">Cell color per column:
        <span class="swatch" style="background:${heatColor(1)}"></span> best ·
        <span class="swatch" style="background:${heatColor(0.5)}"></span> middle ·
        <span class="swatch" style="background:${heatColor(0)}"></span> worst.
        t/s columns reward higher; latency columns reward lower.</p>
      ${rows.some(r => r.errors) ? `<p class="hint">⚠ Some samples errored — see variant logs on the sweep page.</p>` : ""}
    </div>
    ${gl.length ? `<details class="card glossary">
      <summary>How to read these numbers</summary>
      ${gl.map(([k, v]) => `<div class="gloss-item"><b>${esc(k)}</b> — ${esc(v)}</div>`).join("")}
    </details>` : ""}
  `;
  const draw = () => drawChart(rows);
  $("#chart-metric").onchange = draw;
  draw();
}
// Chart.js plugin: draw ±stddev whiskers (mean ± 1 std across repetitions)
// on top of each bar, so the spread of the repeated runs is visible.
const errorBarPlugin = {
  id: "errorBars",
  afterDatasetsDraw(chart) {
    const ds = chart.data.datasets[0];
    if (!ds || !ds.error) return;
    const meta = chart.getDatasetMeta(0);
    const y = chart.scales.y;
    if (!y || !meta) return;
    const { ctx } = chart;
    ctx.save();
    ctx.strokeStyle = "rgba(230,233,237,0.85)";
    ctx.lineWidth = 1.5;
    meta.data.forEach((bar, i) => {
      const m = ds.data[i], e = ds.error[i];
      if (m == null || e == null || e <= 0) return;
      const x = bar.x;
      const yTop = y.getPixelForValue(m + e);
      const yBot = y.getPixelForValue(Math.max(0, m - e));
      ctx.beginPath();
      ctx.moveTo(x, yTop); ctx.lineTo(x, yBot);
      ctx.moveTo(x - 4, yTop); ctx.lineTo(x + 4, yTop);
      ctx.moveTo(x - 4, yBot); ctx.lineTo(x + 4, yBot);
      ctx.stroke();
    });
    ctx.restore();
  },
};

function drawChart(rows) {
  const metric = $("#chart-metric").value;
  const workloads = [...new Set(rows.map(r => r.workload))];
  const key = r => r.label + (workloads.length > 1 ? ` · ${r.workload}` : "");
  const labels = [...new Set(rows.map(key))];
  const rowFor = lab => rows.find(x => key(x) === lab);
  const data = labels.map(lab => { const r = rowFor(lab); return r ? r[`${metric}_mean`] : null; });
  const error = labels.map(lab => { const r = rowFor(lab); return r ? r[`${metric}_std`] : null; });
  if (chart) chart.destroy();
  chart = new Chart($("#chart"), {
    type: "bar",
    data: { labels, datasets: [{ label: metric, data, error, backgroundColor: "#4aa3ff" }] },
    options: {
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true } },
    },
    plugins: [errorBarPlugin],
  });
}

/* ---------- engines ---------- */
let editingEngine = null;
async function renderEngines() {
  const engines = await api("/api/engines");
  $("#engine-list").innerHTML = engines.map(e => `
    <div class="card" style="display:flex;justify-content:space-between;align-items:center">
      <div><b>${esc(e.name)}</b><br>
        <span class="pill">${esc(e.executable)} ${esc((e.args || []).join(" "))}</span><br>
        <span class="hint">${esc(e.note || "")}</span>
        ${e.docs ? ` <a class="doc-link" href="${esc(e.docs)}" target="_blank" rel="noopener">argument docs ↗</a>` : ""}</div>
      <div style="display:flex;gap:8px">
        <button class="small ghost" onclick="checkEngine('${esc(e.name)}')">Check</button>
        <button class="small ghost" onclick="editEngine('${esc(e.name)}')">Edit</button>
        <button class="small danger" onclick="deleteEngine('${esc(e.name)}')">Delete</button>
      </div>
    </div>`).join("") || `<em>No engines.</em>`;
}
async function checkEngine(name) {
  $("#engine-check").textContent = "checking…";
  try {
    const r = await api(`/api/system/check-engine/${encodeURIComponent(name)}`);
    $("#engine-check").textContent = r.found ? `✓ found at ${r.path}` : "✗ executable not found on this machine";
    if (r.version) $("#engine-check").textContent += ` — ${r.version.split("\n")[0]}`;
  } catch (e) { $("#engine-check").textContent = `⚠ ${e.message}`; }
}
function engineFormFill(e = {}) {
  $("#ef-name").value = e.name || "";
  $("#ef-executable").value = e.executable || "";
  $("#ef-args").value = (e.args || ["--model", "{model}", "--host", "127.0.0.1", "--port", "{port}"]).join("\n");
  $("#ef-ready").value = e.ready_path || "/health";
  $("#ef-timeout").value = e.ready_timeout_s || 300;
  $("#ef-timing").value = e.timing || "llamacpp";
  $("#ef-modelreq").checked = e.model_required !== false;
  $("#ef-env").value = e.env && Object.keys(e.env).length ? JSON.stringify(e.env, null, 1) : "";
  $("#ef-headers").value = e.headers && Object.keys(e.headers).length ? JSON.stringify(e.headers, null, 1) : "";
  $("#ef-docs").value = e.docs || "";
  $("#ef-note").value = e.note || "";
}
async function editEngine(name) {
  const e = (await api("/api/engines")).find(x => x.name === name);
  if (!e) return;
  engineFormFill(e);
  editingEngine = name;
  $("#engine-form-title").textContent = `Edit engine: ${name}`;
  $("#engine-cancel").classList.remove("hidden");
  $("#engine-check").textContent = "";
  $("#engine-form").scrollIntoView({ behavior: "smooth", block: "start" });
}
$("#engine-cancel").onclick = () => {
  editingEngine = null;
  $("#engine-form").reset();
  $("#engine-form-title").textContent = "Add engine";
  $("#engine-cancel").classList.add("hidden");
};
$("#engine-form").onsubmit = async ev => {
  ev.preventDefault();
  let eng;
  try {
    eng = {
      name: $("#ef-name").value.trim(),
      executable: $("#ef-executable").value.trim(),
      args: $("#ef-args").value.split("\n").map(s => s.trimEnd()).filter(s => s.trim() !== ""),
      ready_path: $("#ef-ready").value.trim() || "/health",
      ready_timeout_s: +$("#ef-timeout").value || 300,
      timing: $("#ef-timing").value,
      model_required: $("#ef-modelreq").checked,
      env: JSON.parse($("#ef-env").value.trim() || "{}"),
      headers: JSON.parse($("#ef-headers").value.trim() || "{}"),
      docs: $("#ef-docs").value.trim(),
      note: $("#ef-note").value.trim(),
    };
    if (!eng.name) throw new Error("Name is required");
    if (!eng.executable) throw new Error("Executable path is required");
  } catch (e) {
    alert(`Check the form: ${e.message}`);
    return;
  }
  try {
    await api("/api/engines", { method: "POST", body: JSON.stringify(eng) });
    if (editingEngine && editingEngine !== eng.name) {
      // renamed: remove the old entry so the list doesn't keep a stale copy
      await api(`/api/engines/${encodeURIComponent(editingEngine)}`, { method: "DELETE" }).catch(() => {});
    }
    editingEngine = null;
    $("#engine-form").reset();
    $("#engine-form-title").textContent = "Add engine";
    $("#engine-cancel").classList.add("hidden");
    renderEngines();
    loadEngineSelect();
  } catch (e) { alert(e.message); }
};
async function deleteEngine(name) { if (confirm(`Delete engine ${name}?`)) { await api(`/api/engines/${encodeURIComponent(name)}`, { method: "DELETE" }); renderEngines(); } }

/* ---------- logs (one place for all server output) ---------- */
let currentLogId = null;
let currentLogTimer = null;
async function renderLogs() {
  let rows;
  try { rows = await api("/api/logs"); } catch { return; }
  const el = $("#log-list");
  if (!rows.length) { el.innerHTML = `<div class="card">No logs yet — run a sweep first.</div>`; return; }
  const isLive = r => r.status === "starting" || r.status === "measuring";
  el.innerHTML = `<table><thead><tr>
      <th>Sweep</th><th>Variant</th><th>Status</th><th>Started</th><th>Size</th><th></th>
    </tr></thead><tbody>` +
    rows.map(r => `<tr>
      <td>${r.sweep_id}: ${esc(r.sweep_name)}</td>
      <td>${esc(r.label)}</td>
      <td><span class="badge ${esc(r.status)}">${esc(r.status)}</span></td>
      <td class="hint">${esc(r.started_at || "")}</td>
      <td class="hint">${r.log_bytes ? (r.log_bytes/1024).toFixed(1) + " KB" : "—"}</td>
      <td><button class="small ghost" onclick="showLog(${r.id}, '${esc(r.sweep_name)} · ${esc(r.label)}', ${isLive(r)})">View</button></td>
    </tr>`).join("") + "</tbody></table>";
}
async function showLog(vid, label, live = false) {
  currentLogId = vid;
  clearInterval(currentLogTimer);
  $("#log-title").textContent = `Log: ${label}`;
  await refreshLog();
  if (live) currentLogTimer = setInterval(refreshLog, 2000);
  $("#log-download").style.display = "inline";
  $("#log-download").href = `/api/variants/${vid}/log`;
  $("#log-modal").classList.remove("hidden");
}
async function refreshLog() {
  if (currentLogId == null) return;
  try { $("#log-body").textContent = await (await fetch(`/api/variants/${currentLogId}/log`)).text(); }
  catch {}
}
$("#log-refresh").onclick = refreshLog;
$("#log-close").onclick = () => {
  $("#log-modal").classList.add("hidden");
  clearInterval(currentLogTimer);
  currentLogId = null;
};

/* ---------- sweep detail with variant table + logs ---------- */
async function renderSweepsWithVariants() { await renderSweeps(); }
async function sweepDetail(id) {
  const s = await api(`/api/sweeps/${id}`);
  const rows = (s.variants || []).map(v => `<tr>
    <td>${v.idx + 1}</td><td>${esc(v.label)}</td>
    <td><span class="badge ${esc(v.status)}">${esc(v.status)}</span></td>
    <td class="hint">${esc(v.error || "")}</td>
    <td>${v.n_samples}</td>
    <td><button class="small ghost" onclick="showLog(${v.id}, '${esc(v.label)}')">Log</button></td>
  </tr>`).join("");
  $("#sweep-detail").innerHTML = `<div class="card" style="overflow-x:auto">
    <h3>Sweep ${id}: ${esc(s.name)} <span class="badge ${esc(s.status)}">${esc(s.status)}</span></h3>
    ${progressCard(s)}
    <table><thead><tr><th>#</th><th>Variant</th><th>Status</th><th>Error</th><th>Samples</th><th></th></tr></thead>
    <tbody>${rows}</tbody></table></div>`;
}

/* init */
addModel();
addWl("pg", 512, 128);
addDim();
renderSweeps();
pollTimer = setInterval(renderSweeps, 1500);
