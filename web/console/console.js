/* CoatShield Live Console: replays precomputed batches. No network, no server. */
"use strict";

const C = { gate: "#E8A33D", batch: "#3B8ED0", thick: "#3FA35B", comp: "#8A63C9", stop: "#C62839",
            dot: "#a9b4be", muted: "#9fb0bf", text: "#eef2f5", line: "#2f3b48", panel: "#19212a" };
const SPEEDS = [1, 10, 30, 60, 120, 200, 600];
const LANE = { agglomerate: 0.7, undecided: 1.7 };   // rows near the axis for objects with no thickness
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? "–" : Number(v).toFixed(d);
const hms = (s) => { if (s === null || s === undefined || Number.isNaN(s)) return "–";
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60); return `${h} h ${String(m).padStart(2, "0")} min`; };

const state = { id: null, data: null, t: 0, playing: false, speed: 60, lastFrame: 0, lastDraw: 0,
                tab: "monitoring", signed: {}, pinned: null, selected: null, embedded: false,
                args: {}, pendingSign: null, localAudit: [], stopShown: {} };

/* ---------- data ------------------------------------------------------------------- */
function loadScript(src) {
  return new Promise((resolve, reject) => { const s = document.createElement("script");
    s.src = src; s.onload = resolve; s.onerror = () => reject(new Error("cannot load " + src));
    document.head.appendChild(s); });
}
async function loadScenario(id) {
  window.CONSOLE_DATA = window.CONSOLE_DATA || {};
  if (!window.CONSOLE_DATA[id]) await loadScript(`data/${id}/data.js`);
  return prepare(window.CONSOLE_DATA[id]);
}
function prepare(d) {
  if (d.prepared) return d;
  const e = d.events, acc = { x: [], y: [], i: [] }, agg = { x: [], i: [] }, und = { x: [], i: [] };
  for (let i = 0; i < e.t_s.length; i++) {
    const h = e.t_s[i] / 3600;
    if (e.accepted[i]) { acc.x.push(h); acc.y.push(e.thickness_um[i]); acc.i.push(i); }
    else if (e.undecided[i]) { und.x.push(h); und.i.push(i); }
    else { agg.x.push(h); agg.i.push(i); }
  }
  d.acc = acc; d.agg = agg; d.und = und;
  d.hours = d.timeline.t_s.map((s) => s / 3600);
  d.stopByRule = Object.fromEntries(d.stops.map((s) => [s.rule, s]));
  let top = 0;
  for (const v of d.timeline.raw_mean) if (v > top) top = v;
  d.yMax = Math.ceil(top * 1.25 + 2);
  d.prepared = true;
  return d;
}
const upto = (arr, x) => { let lo = 0, hi = arr.length; while (lo < hi) { const m = (lo + hi) >> 1;
  if (arr[m] <= x) lo = m + 1; else hi = m; } return lo; };
const stepIndex = () => Math.min(state.data.timeline.t_s.length - 1,
                                 Math.floor(state.t / state.data.meta.step_s) - 1);

/* ---------- monitoring chart ----------------------------------------------------- */
const layout = () => ({
  paper_bgcolor: C.panel, plot_bgcolor: C.panel, font: { color: C.muted, size: 12 },
  margin: { l: 54, r: 16, t: 12, b: 44 }, showlegend: false, hovermode: "closest", uirevision: "keep",
  xaxis: { title: "Batch time (h)", range: [0, state.data.meta.duration_s / 3600], gridcolor: C.line, zeroline: false },
  yaxis: { title: "Coating thickness (µm)", range: [0, state.data.yMax], gridcolor: C.line, zeroline: false },
  shapes: [], annotations: [],
});
function chartTraces() {
  const d = state.data, tl = d.timeline, h = state.t / 3600, k = stepIndex() + 1;
  const nA = upto(d.acc.x, h), nG = upto(d.agg.x, h), nU = upto(d.und.x, h);
  const xs = d.hours.slice(0, k), cut = (a) => a.slice(0, k);
  const traces = [
    { type: "scattergl", mode: "markers", x: d.acc.x.slice(0, nA), y: d.acc.y.slice(0, nA),
      marker: { color: C.dot, size: 4, opacity: 0.55 }, hovertemplate: "%{y:.1f} µm<extra>accepted pellet</extra>" },
    { type: "scattergl", mode: "markers", x: d.agg.x.slice(0, nG), y: new Array(nG).fill(LANE.agglomerate),
      marker: { color: C.gate, size: 6, symbol: "x" }, hovertemplate: "held back by the gate<extra></extra>" },
    { type: "scattergl", mode: "markers", x: d.und.x.slice(0, nU), y: new Array(nU).fill(LANE.undecided),
      marker: { color: "rgba(0,0,0,0)", size: 7, line: { color: C.dot, width: 1.5 } },
      hovertemplate: "undecided scan<extra></extra>" },
    { type: "scatter", mode: "lines", x: xs, y: cut(tl.corr_d10_hi), line: { width: 0 }, hoverinfo: "skip" },
    { type: "scatter", mode: "lines", x: xs, y: cut(tl.corr_d10_lo), line: { width: 0 }, fill: "tonexty",
      fillcolor: "rgba(59,142,208,0.28)", hoverinfo: "skip" },
    { type: "scatter", mode: "lines", x: xs, y: cut(tl.corr_d10), line: { color: C.batch, width: 3 },
      hovertemplate: "%{y:.2f} µm<extra>corrected d10</extra>" },
    { type: "scatter", mode: "lines", x: xs, y: cut(tl.raw_d10), line: { color: C.stop, width: 2, dash: "dash" },
      hovertemplate: "%{y:.2f} µm<extra>raw d10</extra>" },
    { type: "scatter", mode: "lines", x: xs, y: cut(tl.raw_mean), line: { color: C.muted, width: 1 },
      hovertemplate: "%{y:.2f} µm<extra>raw mean</extra>" },
    { type: "scatter", mode: "lines", x: $("reveal").checked ? xs : [], y: $("reveal").checked ? cut(tl.true_d10) : [],
      line: { color: C.thick, width: 2, dash: "dot" }, hovertemplate: "%{y:.2f} µm<extra>true d10 (hidden from the system)</extra>" },
  ];
  // Trend projection of the corrected d10 to the spec.
  const i = k - 1, eta = i >= 0 ? tl.eta_to_spec_s[i] : null, spec = d.meta.spec_d10_min_um;
  traces.push(eta > 0 && tl.corr_d10[i] !== null
    ? { type: "scatter", mode: "lines", x: [d.hours[i], d.hours[i] + eta / 3600], y: [tl.corr_d10[i], spec],
        line: { color: C.batch, width: 1.5, dash: "dot" }, hoverinfo: "skip" }
    : { type: "scatter", mode: "lines", x: [], y: [] });
  return traces;
}
function chartLayout() {
  const d = state.data, L = layout(), spec = d.meta.spec_d10_min_um, h = state.t / 3600;
  L.shapes.push({ type: "line", xref: "paper", x0: 0, x1: 1, y0: spec, y1: spec, line: { color: C.stop, width: 1.5 } });
  L.annotations.push({ xref: "paper", x: 1, y: spec, text: `spec ${spec} µm`, showarrow: false, xanchor: "right",
                       yanchor: "bottom", font: { color: C.stop, size: 11 } });
  L.annotations.push({ xref: "paper", x: 0, y: LANE.agglomerate, text: "held by gate", showarrow: false, xanchor: "left",
                       yanchor: "bottom", font: { color: C.gate, size: 10 } });
  L.annotations.push({ xref: "paper", x: 0, y: LANE.undecided, text: "undecided", showarrow: false, xanchor: "left",
                       yanchor: "bottom", font: { color: C.muted, size: 10 } });
  if ($("ghost").checked) {
    const colour = { raw_d10: C.stop, raw_mean: C.muted, gravimetric: C.gate, coatshield: C.batch };
    let level = 0;
    for (const s of d.stops) {
      if (!s.stopped || s.t_s / 3600 > h) continue;
      const x = s.t_s / 3600, y = d.yMax * (0.97 - 0.075 * (level++ % 4));
      L.shapes.push({ type: "line", x0: x, x1: x, y0: 0, y1: d.yMax, line: { color: colour[s.rule], width: 1.5, dash: "dashdot" } });
      L.annotations.push({ x, y, text: `<b>${s.label}</b> stops: ${s.true_below_spec_pct.toFixed(1)}% below spec`,
        showarrow: false, xanchor: "right", xshift: -4, font: { color: colour[s.rule], size: 11 }, bgcolor: C.panel });
    }
  }
  return L;
}
function drawChart(force) {
  const now = performance.now();
  if (!force && now - state.lastDraw < 110) return;
  state.lastDraw = now;
  Plotly.react("chart", chartTraces(), chartLayout(), { displayModeBar: false, responsive: true });
}

/* ---------- KPI tiles ------------------------------------------------------------- */
const TILES = [
  ["d10", "Thinnest 10% (corrected d10)", "batch wide"], ["p", "P(d10 ≥ spec)", "batch wide"],
  ["d50", "Median thickness", "thick"], ["cv", "Batch variation (CV)", "thick"],
  ["n", "Refractive index", "thick"], ["eta", "Est. time to stop", "batch"],
  ["gate", "Gate", "gate wide"], ["elapsed", "Elapsed time", "wide"],
];
function buildTiles() {
  $("tiles").innerHTML = TILES.map(([id, label, cls]) =>
    `<div class="tile ${cls}"><div class="k">${label}</div><div class="v" id="t-${id}">–</div>` +
    `<div class="s" id="s-${id}"></div>${id === "p" ? '<div class="gauge" id="g-p"><div></div></div>' : ""}</div>`).join("");
}
function updateTiles() {
  const d = state.data, tl = d.timeline, i = stepIndex(), set = (id, v, s = "") => { $("t-" + id).textContent = v; $("s-" + id).textContent = s; };
  set("elapsed", hms(state.t), `of ${hms(d.meta.duration_s)} simulated`);
  if (i < 0 || tl.corr_d10[i] === null) {
    for (const id of ["d10", "p", "d50", "cv", "n", "eta", "gate"]) set(id, "–", id === "d10" ? "waiting for enough pellets" : "");
    $("g-p").firstChild.style.width = "0"; return;
  }
  const half = tl.corr_d10_hi[i] !== null ? (tl.corr_d10_hi[i] - tl.corr_d10_lo[i]) / 2 : null;
  set("d10", `${fmt(tl.corr_d10[i], 2)} µm`, `± ${fmt(half, 2)} µm · raw sample reads ${fmt(tl.raw_d10[i], 2)} µm`);
  const p = tl.p_d10_ge_spec[i];
  set("p", p === null ? "–" : `${(100 * p).toFixed(0)} %`, `stop needs more than ${(100 * d.meta.p_stop).toFixed(0)} %`);
  $("g-p").firstChild.style.width = `${100 * (p || 0)}%`;
  $("g-p").classList.toggle("ok", p > d.meta.p_stop);
  set("d50", `${fmt(tl.corr_d50[i], 2)} µm`, `d90 ${fmt(tl.corr_d90[i], 2)} µm`);
  set("cv", `${fmt(100 * tl.cv[i], 1)} %`);
  set("n", fmt(tl.n_pooled[i], 3), `± ${fmt(tl.n_se[i], 4)} · solved, not assumed`);
  const eta = tl.eta_to_spec_s[i];
  set("eta", eta === 0 ? "at spec" : hms(eta), "corrected d10 trend to the spec");
  const tot = tl.n_accepted[i] + tl.n_rejected_agglom[i] + tl.n_held_back[i] + tl.n_undecided[i] + tl.n_fines[i];
  set("gate", `${tl.n_accepted[i].toLocaleString()} accepted`,
      `${tl.n_rejected_agglom[i].toLocaleString()} agglomerates rejected · ` +
      `${(100 * tl.n_undecided[i] / Math.max(tot, 1)).toFixed(1)} % undecided (now ${fmt(100 * tl.undecided_rate[i], 1)} %)`);
}

/* ---------- pellet card ----------------------------------------------------------- */
function eventCard(i, pinned) {
  const d = state.data, e = d.events, lib = window.CONSOLE_SCANS, card = $("card");
  const sid = e.scan_id[i], cls = e.undecided[i] ? "undecided" : e.gate_class[i];
  const verdict = { single: "single pellet: passed", agglomerate: "agglomerate: held back", fines: "fines: held back",
                    held_back: "unclear image: held back", undecided: "single pellet: scan undecided" }[cls];
  const scan = sid >= 0 ? `<img class="scan" alt="B-scan with both surfaces" src="data/scans/scan_${String(sid).padStart(4, "0")}.jpg">` :
    `<div class="fine" style="width:200px">No scan: the gate held this object back before the OCT measurement.</div>`;
  const cam = sid >= 0 ? `scan_${String(sid).padStart(4, "0")}_camera.jpg` : `camera_${e.gate_class[i]}.jpg`;
  card.innerHTML = `${scan}<img class="cam" alt="camera frame" src="data/scans/${cam}">
    <div><h3>Pellet at ${hms(e.t_s[i])} ${pinned ? "(pinned)" : ""}</h3>
    <dl><dt>Gate verdict</dt><dd><span class="tag ${cls}">${verdict}</span> confidence ${fmt(e.gate_conf[i], 2)}</dd>
    <dt>Thickness</dt><dd>${e.thickness_um[i] === null ? "not reported" : fmt(e.thickness_um[i], 2) + " µm"}</dd>
    <dt>Refractive index</dt><dd>${fmt(e.n_est[i], 3)}</dd>
    <dt>Confidence</dt><dd>${fmt(e.conf[i], 2)}</dd>
    <dt>Camera size</dt><dd>${fmt(e.size_um[i], 0)} µm</dd></dl>
    <p class="fine">${lib.note}: the picture is the stored scan closest to this pellet in thickness, size and window state.</p></div>`;
  card.hidden = false;
  state.selected = i;
  if (state.tab === "signal" && TAB_DRAW.signal) TAB_DRAW.signal(true);
}
function wireChart() {
  const pick = (ev, pinned) => { const p = ev.points && ev.points[0]; if (!p) return;
    const d = state.data, src = [d.acc, d.agg, d.und][p.curveNumber]; if (!src) return;
    if (!pinned && state.pinned !== null) return;
    if (pinned) state.pinned = src.i[p.pointNumber];
    eventCard(src.i[p.pointNumber], pinned); };
  $("chart").on("plotly_hover", (ev) => pick(ev, false));
  $("chart").on("plotly_click", (ev) => pick(ev, true));
  $("chart").on("plotly_doubleclick", () => { state.pinned = null; });
}

/* ---------- stop recommendation and signature ------------------------------------ */
function checkStop() {
  const d = state.data, cs = d.stopByRule.coatshield, banner = $("stopBanner"), i = stepIndex();
  const key = state.id;
  if (cs.stopped && state.t >= cs.t_s) {
    banner.classList.remove("hold");
    banner.querySelector("strong").textContent = state.signed[key] ? "STOP signed" : "Recommend STOP";
    $("stopText").textContent = state.signed[key]
      ? `Signed by ${state.signed[key].user} (${state.signed[key].meaning.replace(/_/g, " ")}).`
      : `P(d10 ≥ spec) passed ${(100 * d.meta.p_stop).toFixed(0)} % at ${hms(cs.t_s)}. The operator decides.`;
    $("signOpen").hidden = !!state.signed[key];
    banner.hidden = false;
    if (!state.stopShown[key]) { state.stopShown[key] = true; pause(); openSign(); }
  } else if (!cs.stopped && i >= 0 && d.timeline.undecided_rate[i] > d.meta.undecided_limit) {
    banner.classList.add("hold");
    banner.querySelector("strong").textContent = "Holding: no stop recommended";
    $("stopText").textContent = `${fmt(100 * d.timeline.undecided_rate[i], 0)} % of scans are undecided (limit ` +
      `${(100 * d.meta.undecided_limit).toFixed(0)} %). CoatShield will not recommend a stop on readings it cannot trust.`;
    $("signOpen").hidden = true; banner.hidden = false;
  } else banner.hidden = true;
}
function openSign() {
  const d = state.data, cs = d.stopByRule.coatshield, a = state.args, i = stepIndex();
  const fill = (id, list) => { $(id).innerHTML = list.map((v) => `<option value="${v}">${String(v).replace(/_/g, " ")}</option>`).join(""); };
  fill("signUser", a.users || ["operator1", "qa1"]);
  fill("signMeaning", a.meanings || ["approve_stop", "reject_recommendation"]);
  fill("signReason", a.reasons || ["target_reached", "process_fault", "quality_hold", "other"]);
  $("signSummary").textContent = `Recommendation: stop the spray at ${hms(cs.t_s)}. Corrected d10 ` +
    `${fmt(d.timeline.corr_d10[i], 2)} µm against a spec of ${d.meta.spec_d10_min_um} µm.`;
  $("signNote").textContent = state.embedded
    ? "Your signature is checked and written to the hash-chained audit trail."
    : "Standalone file: the signature is kept in this browser only. Open the console from the dashboard to write to the audit trail.";
  $("signResult").textContent = ""; $("signResult").className = "result"; $("signPin").value = "";
  $("signModal").hidden = false;
}
async function sha256(text) {
  if (!(window.crypto && crypto.subtle)) return "unavailable";
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
async function submitSign(ev) {
  ev.preventDefault();
  const d = state.data, cs = d.stopByRule.coatshield;
  const req = { nonce: `${Date.now()}-${Math.round(performance.now())}`, scenario: state.id, user: $("signUser").value,
                meaning: $("signMeaning").value, reason: $("signReason").value, comment: $("signComment").value,
                pin: $("signPin").value, stop_h: cs.t_s / 3600, config_hash: d.meta.config_hash };
  if (state.embedded) {
    state.pendingSign = req;
    $("signResult").textContent = "Checking the signature…";
    toStreamlit("streamlit:setComponentValue", { value: { sign: req }, dataType: "json" });
    return;
  }
  const prev = state.localAudit.length ? state.localAudit[state.localAudit.length - 1].hash : "0".repeat(64);
  const entry = { seq: state.localAudit.length + 1, time_utc: new Date().toISOString(), user: req.user, action: req.meaning,
                  object: `replay ${state.id}`, reason: req.reason + (req.comment ? ": " + req.comment : ""), prev_hash: prev };
  entry.hash = await sha256(JSON.stringify(entry));
  state.localAudit.push(entry);
  signDone({ ok: true, user: req.user, meaning: req.meaning, time_utc: entry.time_utc, hash: entry.hash, local: true });
}
function signDone(res) {
  const out = $("signResult");
  if (res.ok) {
    state.signed[state.id] = res;
    out.className = "result ok";
    out.textContent = `Signed by ${res.user} at ${res.time_utc}. Entry hash ${String(res.hash).slice(0, 12)}…` +
      (res.local ? " (browser only)" : " (audit trail)");
    setTimeout(() => { $("signModal").hidden = true; }, 1400);
  } else { out.className = "result bad"; out.textContent = `Signature refused: ${res.problem}. The attempt was logged.`; }
  checkStop(); if (!$("drawer").hidden) HOOKS.drawer();
}

/* ---------- replay clock ---------------------------------------------------------- */
function setTime(t) {
  state.t = Math.max(0, Math.min(t, state.data.meta.duration_s));
  $("scrub").value = Math.round(1000 * state.t / state.data.meta.duration_s);
  $("clockVal").textContent = hms(state.t);
}
function render(force) {
  updateTiles(); checkStop();
  if (state.tab === "monitoring") drawChart(force);
  else if (TAB_DRAW[state.tab]) TAB_DRAW[state.tab](force);
}
function frame(now) {
  if (state.playing) {
    const dt = Math.min(0.25, (now - state.lastFrame) / 1000);
    setTime(state.t + dt * state.speed);
    if (state.t >= state.data.meta.duration_s) pause();
    render(false);
  }
  state.lastFrame = now;
  requestAnimationFrame(frame);
}
function play() { if (state.t >= state.data.meta.duration_s) setTime(0); state.playing = true; $("playBtn").textContent = "Playing"; }
function pause() { state.playing = false; $("playBtn").textContent = state.t > 0 ? "Resume" : "Start"; }
function reset() { pause(); state.signed = {}; state.stopShown = {}; state.pinned = null; state.selected = null;
  $("card").hidden = true; setTime(0); $("playBtn").textContent = "Start"; render(true); }

async function useScenario(id, keepClock) {
  const t = state.t;
  state.data = await loadScenario(id); state.id = id;
  $("scenario").value = id; $("spec").value = `${state.data.meta.spec_d10_min_um} µm`;
  $("assumption").textContent = `${state.data.meta.pellet_assumption}; the window measures about ` +
    `${state.data.meta.window_share_pct}% of them.`;
  document.querySelectorAll("button.fault").forEach((b) => b.classList.toggle("on", b.dataset.fault === id && id !== "default"));
  setTime(keepClock ? t : 0);
  if (!keepClock) { state.pinned = null; state.selected = null; $("card").hidden = true; }
  render(true);
  drawFoot();
}
function drawFoot() {
  const m = state.data.meta, models = Object.entries(m.models).map(([k, v]) => `${k} ${v.version} (${v.sha256.slice(0, 12)})`).join(", ");
  $("foot").textContent = `${m.note} Scenario: ${m.label}. Seed ${m.seed}, config ${m.config_hash}. Models: ${models}. ` +
    (m.headline.ratio ? `Validation averages: raw d10 rule ${m.headline.raw_d10_below_spec_pct}% below spec, ` +
      `CoatShield ${m.headline.coatshield_below_spec_pct}% (${m.headline.ratio}×).` : "");
}

/* ---------- tabs (filled in below) ------------------------------------------------ */
const TAB_DRAW = {};
function showTab(name) {
  state.tab = name;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === name));
  document.querySelectorAll(".tabpane").forEach((p) => p.classList.toggle("on", p.id === "tab-" + name));
  render(true);
  if (name === "monitoring") Plotly.Plots.resize("chart");
}
const HOOKS = { drawer: () => {}, record: () => {} };   // filled in by console_tabs.js

/* ---------- Streamlit bridge ---------------------------------------------------- */
function toStreamlit(type, data) { window.parent.postMessage(Object.assign({ isStreamlitMessage: true, type }, data), "*"); }
window.addEventListener("message", (ev) => {
  if (!ev.data || ev.data.type !== "streamlit:render") return;
  state.embedded = true; state.args = ev.data.args || {};
  const res = state.args.sign_result;
  if (res && state.pendingSign && res.nonce === state.pendingSign.nonce) { state.pendingSign = null; signDone(res); }
  if (state.args.record) HOOKS.record(state.args.record);
  if (!$("drawer").hidden) HOOKS.drawer();
  toStreamlit("streamlit:setFrameHeight", { height: Math.max(900, document.body.scrollHeight + 10) });
});

/* ---------- start ----------------------------------------------------------------- */
async function start() {
  buildTiles();
  $("scenario").innerHTML = window.CONSOLE_SCENARIOS.map((s) => `<option value="${s.id}">${s.label}</option>`).join("");
  $("legend").innerHTML =
    `<span><i class="dot" style="background:${C.dot}"></i>accepted pellet</span>` +
    `<span><i class="x" style="color:${C.gate}">✕</i> held by gate</span>` +
    `<span><i class="ring" style="border-color:${C.dot}"></i>undecided scan</span>` +
    `<span><i style="border-color:${C.batch}"></i>corrected d10 with interval</span>` +
    `<span><i class="dash" style="border-color:${C.stop}"></i>raw d10</span>` +
    `<span><i style="border-color:${C.muted};border-top-width:1px"></i>raw mean</span>` +
    `<span><i style="border-color:${C.stop};border-top-width:1.5px"></i>spec</span>`;
  const setSpeed = () => { state.speed = SPEEDS[$("speed").value]; $("speedVal").textContent = `${state.speed}×`; };
  $("speed").addEventListener("input", setSpeed); setSpeed();
  $("playBtn").onclick = play; $("pauseBtn").onclick = pause; $("resetBtn").onclick = reset;
  $("scrub").addEventListener("input", () => { setTime($("scrub").value / 1000 * state.data.meta.duration_s); render(true); });
  $("scenario").addEventListener("change", () => { pause(); useScenario($("scenario").value, false); });
  $("reveal").addEventListener("change", () => render(true));
  $("ghost").addEventListener("change", () => render(true));
  document.querySelectorAll("button.fault").forEach((b) => { b.onclick = () => useScenario(b.dataset.fault, true); });
  document.querySelectorAll("#tabs button").forEach((b) => { b.onclick = () => showTab(b.dataset.tab); });
  $("signOpen").onclick = openSign; $("signCancel").onclick = () => { $("signModal").hidden = true; };
  $("signForm").addEventListener("submit", submitSign);
  await useScenario(window.CONSOLE_SCENARIOS[0].id, false);
  wireChart();
  // Preload the other scenarios so switching never waits.
  for (const s of window.CONSOLE_SCENARIOS.slice(1)) loadScenario(s.id).catch(() => {});
  toStreamlit("streamlit:componentReady", { apiVersion: 1 });
  toStreamlit("streamlit:setFrameHeight", { height: 940 });
  requestAnimationFrame(frame);
  if (typeof window.consoleExtras === "function") window.consoleExtras();
}
window.addEventListener("DOMContentLoaded", start);
window.CS = { state, play, pause, reset, setTime, render, useScenario, showTab, TAB_DRAW, C, $, fmt, hms, stepIndex, upto,
              eventCard, openSign, signDone };
