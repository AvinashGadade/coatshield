/* CoatShield Live Console: the Process, Image, Signal and Batch tabs, the compliance drawer
   and the guided tour. All of them read the same replay clock (state.t) as the Monitoring tab. */
"use strict";

const pad4 = (n) => String(n).padStart(4, "0");
const plotBase = (xTitle, yTitle) => ({
  paper_bgcolor: C.panel, plot_bgcolor: C.panel, font: { color: C.muted, size: 12 },
  margin: { l: 54, r: 14, t: 10, b: 42 }, showlegend: false,
  xaxis: { title: xTitle, gridcolor: C.line, zeroline: false },
  yaxis: { title: yTitle, gridcolor: C.line, zeroline: false },
});
const PLOT_CFG = { displayModeBar: false, responsive: true };
const VERDICT = { single: "single pellet: passed", agglomerate: "agglomerate: held back", fines: "fines: held back",
                  held_back: "unclear image: held back", undecided: "single pellet: scan undecided" };

/* ---------- Process ---------------------------------------------------------------- */
const BED = { w: 520, h: 460, cx: 260, particles: [], visual: 0, last: 0 };
function buildProcess() {
  $("tab-process").innerHTML = `<div class="grid2">
    <div class="box"><h3>Wurster coater (schematic)</h3>
      <canvas id="bed" width="${BED.w}" height="${BED.h}" style="max-width:100%"></canvas>
      <p class="fine">Schematic only: the dots show the circulation pattern, not individual simulated pellets.</p></div>
    <div class="box"><h3>At the probe window</h3>
      <div class="counters">
        <div class="tile"><div class="k">Detections per minute</div><div class="v" id="p-det">–</div></div>
        <div class="tile thick"><div class="k">Accepted</div><div class="v" id="p-acc">–</div></div>
        <div class="tile gate"><div class="k">Agglomerates per minute</div><div class="v" id="p-agg">–</div></div>
      </div>
      <h3 style="margin-top:16px">Window health</h3>
      <div class="health"><div id="healthBar" style="width:100%"></div></div>
      <p class="fine" id="healthText"></p>
      <h3 style="margin-top:16px">What the window sees</h3>
      <p class="fine" id="shareText"></p></div></div>`;
  // Deterministic layout of the dots (no random source: same picture every run).
  for (let i = 0; i < 170; i++) {
    const a = (i * 0.61803398875) % 1, b = (i * 0.41421356237) % 1, c = (i * 0.73205080757) % 1;
    BED.particles.push({ u0: a, side: i % 2 ? 1 : -1, r: b, j: 2 * c - 1, speed: 0.055 + 0.03 * b, big: i % 9 === 0, twin: i % 37 === 0 });
  }
}
function bedPoint(p, u) {
  const cx = BED.cx, jx = p.j * 18, top = 240, out = 95 + p.r * 30, low = 45 + p.r * 35;
  if (u < 0.12) { const s = u / 0.12; return [cx + jx, 385 - 145 * s]; }
  if (u < 0.30) { const s = (u - 0.12) / 0.18;
    return [cx + jx + p.side * s * out, top - Math.sin(Math.PI * s) * (95 + 55 * p.r)]; }
  if (u < 0.94) { const s = (u - 0.30) / 0.64;
    return [cx + jx * (1 - s) * 0.3 + p.side * (out + (low - out) * s), top + 148 * s]; }
  const s = (u - 0.94) / 0.06;
  return [cx + p.side * low * (1 - s) + jx * s, 388 - 3 * s];
}
function drawProcess() {
  const d = state.data, tl = d.timeline, i = stepIndex(), now = performance.now();
  if (state.playing) BED.visual += Math.min(0.1, (now - BED.last) / 1000);
  BED.last = now;
  const ctx = $("bed").getContext("2d"), cx = BED.cx;
  ctx.clearRect(0, 0, BED.w, BED.h);
  ctx.lineWidth = 2; ctx.strokeStyle = C.muted; ctx.fillStyle = "#141b23";
  ctx.beginPath(); ctx.moveTo(120, 30); ctx.lineTo(120, 230); ctx.lineTo(170, 400); ctx.lineTo(350, 400);
  ctx.lineTo(400, 230); ctx.lineTo(400, 30); ctx.fill(); ctx.stroke();
  ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(170, 400); ctx.lineTo(350, 400); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = C.text; ctx.beginPath(); ctx.moveTo(cx - 25, 250); ctx.lineTo(cx - 25, 380);
  ctx.moveTo(cx + 25, 250); ctx.lineTo(cx + 25, 380); ctx.stroke();
  // Spray cone and nozzle.
  ctx.fillStyle = "rgba(59,142,208,0.22)"; ctx.beginPath(); ctx.moveTo(cx, 402); ctx.lineTo(cx - 22, 300); ctx.lineTo(cx + 22, 300); ctx.fill();
  ctx.fillStyle = C.batch; ctx.beginPath(); ctx.moveTo(cx - 8, 418); ctx.lineTo(cx + 8, 418); ctx.lineTo(cx, 400); ctx.fill();
  for (const p of BED.particles) {
    const [x, y] = bedPoint(p, (p.u0 + BED.visual * p.speed) % 1);
    ctx.fillStyle = p.twin ? C.gate : C.dot; ctx.globalAlpha = 0.85;
    ctx.beginPath(); ctx.arc(x, y, p.big ? 4 : 2.8, 0, 6.2832); ctx.fill();
    if (p.twin) { ctx.beginPath(); ctx.arc(x + 4.5, y + 1, 2.8, 0, 6.2832); ctx.fill(); }
  }
  ctx.globalAlpha = 1;
  // Probe window on the downbed wall; it browns as the window fouls.
  const foul = i >= 0 ? tl.fouling[i] : 0, wy = 300, wx = 400 - 50 * (wy - 230) / 170;
  const pulse = state.playing ? 0.55 + 0.45 * Math.sin(now / 90) : 0.6;
  ctx.fillStyle = `rgba(63,163,91,${0.35 + 0.5 * pulse * (1 - foul)})`; ctx.fillRect(wx - 5, wy - 22, 8, 44);
  ctx.fillStyle = `rgba(120,84,40,${foul})`; ctx.fillRect(wx - 5, wy - 22, 8, 44);
  ctx.strokeStyle = C.thick; ctx.strokeRect(wx - 5, wy - 22, 8, 44);
  ctx.fillStyle = C.panel; ctx.strokeStyle = C.muted; ctx.fillRect(wx + 8, wy - 14, 62, 28); ctx.strokeRect(wx + 8, wy - 14, 62, 28);
  ctx.fillStyle = C.muted; ctx.font = "12px system-ui, sans-serif";
  ctx.fillText("probe", wx + 22, wy + 4); ctx.fillText("OCT + camera window", wx + 8, wy + 34);
  ctx.fillText("fountain", cx - 24, 70); ctx.fillText("spray nozzle", cx + 14, 432);
  // Labels left of the vessel, each with a leader line.
  ctx.strokeStyle = C.line; ctx.lineWidth = 1;
  for (const [text, y, x1] of [["draft tube", 268, cx - 27], ["downbed", 330, 150]]) {
    ctx.fillText(text, 8, y + 4); ctx.beginPath(); ctx.moveTo(74, y); ctx.lineTo(x1, y); ctx.stroke();
  }
  if (i < 0) return;
  const stepMin = d.meta.step_s / 60, prev = Math.max(i - 1, 0), det = tl.detections_per_min[i];
  const dAcc = (tl.n_accepted[i] - (i ? tl.n_accepted[prev] : 0)) / stepMin;
  const dAgg = (tl.n_rejected_agglom[i] - (i ? tl.n_rejected_agglom[prev] : 0)) / stepMin;
  $("p-det").textContent = fmt(det, 0);
  $("p-acc").textContent = `${fmt(100 * dAcc / Math.max(det, 1), 1)} %`;
  $("p-agg").textContent = fmt(dAgg, 1);
  $("healthBar").style.width = `${100 * (1 - foul)}%`;
  $("healthBar").style.background = foul > 0.5 ? C.stop : foul > 0.2 ? C.gate : C.thick;
  $("healthText").textContent = `${fmt(100 * (1 - foul), 0)} % clear. Undecided scans in the last window: ${fmt(100 * tl.undecided_rate[i], 1)} %.`;
  $("shareText").textContent = `${tl.n_accepted[i].toLocaleString()} pellets measured so far, out of about ` +
    `${(d.meta.n_real_pellets / 1e6).toFixed(0)} million in the bed (${d.meta.pellet_assumption}).`;
}

/* ---------- Image ------------------------------------------------------------------ */
const IMG = { key: "" };
function scannedList(d) {
  if (!d.scanned) { d.scanned = { t: [], i: [] };
    d.events.scan_id.forEach((sid, i) => { if (sid >= 0) { d.scanned.t.push(d.events.t_s[i]); d.scanned.i.push(i); } }); }
  return d.scanned;
}
function buildImage() {
  $("tab-image").innerHTML = `<div class="grid2" style="grid-template-columns:minmax(0,2.2fr) minmax(0,1fr)">
    <div class="box"><h3>B-scans as they arrive
        <label class="check" style="float:right;font-weight:400"><input type="checkbox" id="camToggle"> Grad-CAM: where the model looks</label></h3>
      <div class="strip" id="strip"></div>
      <p class="fine" id="stripNote"></p></div>
    <div class="box"><h3>Camera frame and gate</h3>
      <img id="camNow" alt="camera frame" style="width:192px;border-radius:8px;background:#000">
      <div id="camVerdict" style="margin-top:8px"></div></div></div>`;
  $("camToggle").addEventListener("change", () => drawImage(true));
}
function drawImage(force) {
  const d = state.data, e = d.events, sc = scannedList(d), lib = window.CONSOLE_SCANS;
  const n = upto(sc.t, state.t), last = upto(e.t_s, state.t) - 1, cam = $("camToggle").checked && lib.gradcam;
  const key = `${state.id}:${n}:${last}:${cam}`;
  if (!force && key === IMG.key) return;
  IMG.key = key;
  const shown = sc.i.slice(Math.max(0, n - 5), n).reverse();
  $("strip").innerHTML = shown.length ? shown.map((i) => {
    const sid = e.scan_id[i], what = e.undecided[i] ? "undecided" : `${fmt(e.thickness_um[i], 1)} µm`;
    return `<figure><img alt="B-scan" src="data/scans/scan_${pad4(sid)}${cam ? "_gradcam" : ""}.jpg">` +
      `<figcaption>${hms(e.t_s[i])} · ${what}</figcaption></figure>`; }).join("")
    : `<p class="fine">Press Start: scans appear here as pellets pass the window.</p>`;
  $("stripNote").innerHTML = `${lib.note}. <span style="color:#e2525f">Red</span>: coating surface. ` +
    `<span style="color:#28bed4">Cyan</span>: coating–core surface, both drawn by the trained boundary finder. ` +
    `One in ${d.meta.event_stride} detections is shown.`;
  if (last >= 0) {
    const sid = e.scan_id[last], cls = e.undecided[last] ? "undecided" : e.gate_class[last];
    $("camNow").src = sid >= 0 ? `data/scans/scan_${pad4(sid)}_camera.jpg` : `data/scans/camera_${e.gate_class[last]}.jpg`;
    $("camVerdict").innerHTML = `<span class="tag ${cls}">${VERDICT[cls]}</span>` +
      `<p class="fine" style="margin-top:6px">Gate confidence ${fmt(e.gate_conf[last], 2)} · camera size ${fmt(e.size_um[last], 0)} µm · ${hms(e.t_s[last])}</p>`;
  }
}

/* ---------- Signal ---------------------------------------------------------------- */
const SIG = { key: "", loading: false, last: 0 };
function buildSignal() {
  $("tab-signal").innerHTML = `<p class="fine" id="sigWho" style="margin:0 0 10px"></p>
    <div class="grid2">
      <div class="box"><h3>1 · Raw spectrum from the detector <span class="fine">(dotted: background)</span></h3><div class="plot" id="sig1"></div></div>
      <div class="box"><h3>2 · Background removed, resampled to even wavenumber</h3><div class="plot" id="sig2"></div></div>
      <div class="box"><h3>3 · After the FFT: one A-scan, two peaks</h3><div class="plot" id="sig3"></div></div>
      <div class="box"><h3>4 · Thickness</h3><div class="formula" id="sigFormula"></div>
        <h3 style="margin-top:14px">Refractive index: three independent estimates</h3>
        <div class="counters" id="sigN"></div><p class="fine" id="sigNote"></p></div></div>`;
}
function drawSignal(force) {
  const d = state.data, e = d.events, sc = scannedList(d), lib = window.CONSOLE_SCANS;
  if (!window.CONSOLE_SIGNALS) {
    if (!SIG.loading) { SIG.loading = true; $("sigWho").textContent = "Loading the signal traces…";
      loadScript("data/scans/signals.js").then(() => drawSignal(true)); }
    return;
  }
  let i = state.selected !== null && e.scan_id[state.selected] >= 0 ? state.selected : null;
  if (i === null) { const n = upto(sc.t, state.t); i = n ? sc.i[n - 1] : sc.i[0]; }
  const sid = e.scan_id[i], now = performance.now();
  if (!force && (sid === SIG.key || now - SIG.last < 500)) return;
  SIG.key = sid; SIG.last = now;
  const s = window.CONSOLE_SIGNALS[String(sid)], row = lib.scans.find((r) => r.scan_id === sid);
  $("sigWho").textContent = `${lib.note}. Showing the stored scan closest to the ` +
    `${state.selected === i ? "pellet you picked on the Monitoring chart" : "latest scanned pellet"} (${hms(e.t_s[i])}).`;
  const nm = s.spectrum.map((_, k) => s.wavelength_nm[0] + (s.wavelength_nm[1] - s.wavelength_nm[0]) * k / (s.spectrum.length - 1));
  Plotly.react("sig1", [
    { x: nm, y: s.spectrum, type: "scatter", mode: "lines", line: { color: C.muted, width: 1 }, hovertemplate: "%{x:.1f} nm<extra>detector</extra>" },
    { x: nm, y: s.background, type: "scatter", mode: "lines", line: { color: C.gate, width: 1.5, dash: "dot" }, hovertemplate: "%{x:.1f} nm<extra>background (sample blocked)</extra>" },
  ], plotBase("Wavelength (nm), centre of the detector line", "Detector signal"), PLOT_CFG);
  Plotly.react("sig2", [{ y: s.fringes_k, type: "scatter", mode: "lines", line: { color: C.thick, width: 1 } }],
    plotBase("Sample on the even-wavenumber grid", "Fringes"), PLOT_CFG);
  const z = s.ascan_db.map((_, k) => k * s.depth_px_um), L = plotBase("Depth (µm of optical path)", "Signal (dB)");
  const zo = s.outer_row * s.depth_px_um, zi = s.inner_row * s.depth_px_um, dz = zi - zo;
  L.xaxis.range = [Math.max(0, zo - 12), Math.min(z[z.length - 1], zi + 30)];
  L.shapes = [zo, zi].map((x, k) => ({ type: "line", x0: x, x1: x, yref: "paper", y0: 0, y1: 1,
                                       line: { color: k ? "#28bed4" : "#e2525f", width: 1.5, dash: "dot" } }));
  L.annotations = [{ x: (zo + zi) / 2, yref: "paper", y: 0.97, text: `Δz = ${dz.toFixed(2)} µm`, showarrow: false, font: { color: C.text } }];
  Plotly.react("sig3", [{ x: z, y: s.ascan_db, type: "scatter", mode: "lines", line: { color: C.thick, width: 1.5 } }], L, PLOT_CFG);
  const k = stepIndex(), n = (k >= 0 && d.timeline.n_pooled[k]) || lib.index_methods.reflectance || d.meta.true_n;
  $("sigFormula").innerHTML = `d = Δz ÷ n = ${dz.toFixed(2)} ÷ ${n.toFixed(3)} = <b>${(dz / n).toFixed(2)} µm</b>` +
    `<p class="fine">Centre A-scan only. The chain uses every A-scan of the pellet and corrects for refraction at the curved surface: ` +
    `it reported ${row && row.measured_um !== null ? row.measured_um.toFixed(2) + " µm" : "no value"} for this scan.</p>`;
  const m = lib.index_methods;
  $("sigN").innerHTML = [["Reflectance", m.reflectance], ["Camera–OCT fusion", m.fusion], ["Microscopy anchor", m.anchor]]
    .map(([name, v]) => `<div class="tile thick"><div class="k">${name}</div><div class="v">${fmt(v, 3)}</div></div>`).join("");
  $("sigNote").textContent = `Set in the simulation: ${fmt(m.true_n, 2)}. Mean results at normal signal, from ${m.source}.`;
}

/* ---------- Batch ------------------------------------------------------------------ */
const BAT = { key: "" };
function buildBatch() {
  $("tab-batch").innerHTML = `<p class="fine" id="batWho" style="margin:0 0 10px"></p><div class="grid2">
    <div class="box"><h3>Size: what the window sees against the bed</h3><div class="plot" id="bat1" style="height:360px"></div>
      <p class="fine">Weight of a size class: w = share in the bed ÷ share in the window. Small pellets are rare in the window, so each one counts for more.</p></div>
    <div class="box"><h3>Thickness: raw sample against the corrected batch</h3><div class="plot" id="bat2" style="height:360px"></div>
      <p class="fine" id="batTail"></p></div></div>`;
}
function drawBatch(force) {
  const d = state.data, b = d.batch, spec = d.meta.spec_d10_min_um;
  let j = 0; while (j + 1 < b.frames.length && b.frames[j + 1].t_s <= state.t) j++;
  const f = b.frames[j], reveal = $("reveal").checked, key = `${state.id}:${j}:${reveal}`;
  if (!force && key === BAT.key) return;
  BAT.key = key;
  if (!f.f_obs || f.t_s > state.t) {
    $("batWho").textContent = "Waiting for enough accepted pellets in the window.";
    Plotly.react("bat1", [], plotBase("Camera size (µm)", "Share"), PLOT_CFG);
    Plotly.react("bat2", [], plotBase("Coating thickness (µm)", "Share of pellets"), PLOT_CFG);
    $("batTail").textContent = ""; return;
  }
  const ref = { atline: "an at-line sizing sample", coa: "the certificate of analysis", oracle: "the true bed" }[b.reference];
  $("batWho").textContent = `Snapshot at ${hms(f.t_s)}: ${f.n_window.toLocaleString()} accepted pellets in the window. Bed sizes come from ${ref}.`;
  const L1 = plotBase("Camera size (µm)", "Share of pellets"); L1.barmode = "group"; L1.showlegend = true;
  L1.legend = { orientation: "h", y: 1.12, x: 0 }; L1.margin.r = 50; L1.margin.t = 28;
  L1.yaxis2 = { title: "Weight w", overlaying: "y", side: "right", showgrid: false, rangemode: "tozero" };
  Plotly.react("bat1", [
    { type: "bar", x: f.size_um, y: f.f_obs, name: "window sample", marker: { color: C.dot } },
    { type: "bar", x: f.size_um, y: f.f_true, name: "bed", marker: { color: C.batch } },
    { type: "scatter", mode: "lines+markers", x: f.size_um, y: f.weight, name: "weight w", yaxis: "y2", line: { color: C.text, width: 1.5 } },
  ], L1, PLOT_CFG);
  const x = b.thickness_um, tail = x.map((v, k) => (v <= spec ? f.corrected[k] : null));
  const L2 = plotBase("Coating thickness (µm)", "Share of pellets"); L2.showlegend = true;
  L2.legend = { orientation: "h", y: 1.12, x: 0 }; L2.margin.t = 28; L2.xaxis.range = [0, d.yMax];
  L2.shapes = [{ type: "line", x0: spec, x1: spec, yref: "paper", y0: 0, y1: 1, line: { color: C.stop, width: 1.5 } }];
  const traces = [
    { type: "scatter", mode: "lines", x, y: f.corrected, name: "corrected", line: { color: C.batch, width: 2.5 }, fill: "tozeroy", fillcolor: "rgba(59,142,208,0.15)" },
    { type: "scatter", mode: "lines", x, y: tail, name: "below spec", line: { width: 0 }, fill: "tozeroy", fillcolor: "rgba(198,40,57,0.55)", connectgaps: false },
    { type: "scatter", mode: "lines", x, y: f.raw, name: "raw sample", line: { color: C.stop, width: 2, dash: "dash" } },
  ];
  if (reveal) traces.push({ type: "scatter", mode: "lines", x, y: f.truth, name: "true batch", line: { color: C.thick, width: 2, dash: "dot" } });
  Plotly.react("bat2", traces, L2, PLOT_CFG);
  const share = (arr) => 100 * arr.reduce((a, v, k) => a + (x[k] <= spec ? v : 0), 0);
  $("batTail").textContent = `Below the ${spec} µm spec: ${share(f.corrected).toFixed(1)} % by the corrected estimate, ` +
    `${share(f.raw).toFixed(1)} % by the raw sample` + (reveal ? `, ${share(f.truth).toFixed(1)} % in truth.` : ".");
}

/* ---------- Compliance drawer ------------------------------------------------------ */
const REC = { nonce: null, done: {} };
function drawDrawer() {
  const d = state.data, a = state.args, i = stepIndex(), drawer = $("drawer");
  const reg = a.registry || null;
  const models = Object.entries(d.meta.models).map(([name, m]) => {
    const live = reg && reg[name];
    const mark = live ? (live.ok ? "✓ file matches" : "✗ MISMATCH") : "recorded at export";
    return `<tr><td>${name} ${m.version}</td><td><code>${m.sha256.slice(0, 24)}…</code></td><td>${m.status}</td><td>${mark}</td></tr>`; }).join("");
  const drift = i >= 0 ? d.timeline.drift_state[i] : "OK";
  const entries = state.embedded ? (a.audit ? a.audit.entries : []) : state.localAudit.slice(-10);
  const chain = state.embedded ? (a.audit ? (a.audit.ok ? `Chain check passed (${a.audit.count} entries).` : `Chain check FAILED: ${a.audit.problem}.`) : "")
    : "Standalone file: entries are kept in this browser only.";
  const rows = entries.slice().reverse().map((e) =>
    `<tr><td>${e.seq}</td><td>${String(e.time_utc).slice(11, 19)}</td><td>${e.user}</td><td>${String(e.action).replace(/_/g, " ")}</td>` +
    `<td>${e.object || ""}</td><td><code>${String(e.hash).slice(0, 10)}</code></td></tr>`).join("");
  drawer.innerHTML = `<button id="drawerClose" style="float:right">Close</button><h2>Compliance</h2>
    <p class="fine">Demonstration on simulated data. Not a validated GMP system.</p>
    <h3>Model registry</h3><table><tr><th>Model</th><th>SHA-256</th><th>Status</th><th>Check</th></tr>${models}</table>
    <p class="fine">Combined hash <code>${d.meta.model_hash.slice(0, 24)}…</code></p>
    <h3>Drift monitor</h3><span class="state ${drift}">${drift}</span>
    <p class="fine">Undecided scans in the last window: ${i >= 0 ? fmt(100 * d.timeline.undecided_rate[i], 1) : "–"} % ` +
      `(warning at ${(100 * d.meta.drift.undecided_warning).toFixed(0)} %, alarm at ${(100 * d.meta.drift.undecided_alarm).toFixed(0)} %). ` +
      `This is the monitor's undecided-rate channel; the full monitor on the Drift page also tracks scan statistics.</p>
    <h3>Audit trail: last ${Math.min(10, entries.length)} entries</h3>
    <table><tr><th>#</th><th>UTC</th><th>User</th><th>Action</th><th>Object</th><th>Hash</th></tr>${rows || '<tr><td colspan="6">No entries yet.</td></tr>'}</table>
    <p class="fine">${chain}</p>
    <h3>Batch record</h3><button id="recordBtn" class="primary" ${state.embedded ? "" : "disabled"}>Export batch record (PDF)</button>
    <p class="fine" id="recordNote">${state.embedded ? "Built by the existing exporter from this scenario, the model hashes and the signed actions."
      : "Available when the console is opened from the dashboard (it needs the Python exporter)."}</p>`;
  $("drawerClose").onclick = () => { drawer.hidden = true; };
  $("recordBtn").onclick = () => { REC.nonce = `${Date.now()}`; $("recordNote").textContent = "Building the record…";
    toStreamlit("streamlit:setComponentValue", { value: { record: { nonce: REC.nonce, scenario: state.id } }, dataType: "json" }); };
}
function gotRecord(rec) {
  if (!rec || rec.nonce !== REC.nonce || REC.done[rec.nonce]) return;
  REC.done[rec.nonce] = true;
  const bytes = Uint8Array.from(atob(rec.pdf_b64), (ch) => ch.charCodeAt(0));
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([bytes], { type: "application/pdf" }));
  link.download = rec.name; document.body.appendChild(link); link.click(); link.remove();
}

/* ---------- Guided tour ------------------------------------------------------------ */
const TOUR = [
  { title: "1 · Process", tab: "process", anchor: "tab-process",
    text: "Pellets circulate through the spray. A probe window on the downbed sees a small, size-biased share of them." },
  { title: "2 · Gate", tab: "monitoring", anchor: "t-gate",
    text: "A camera looks first. Fused pellets are held back (orange ✕) so they never reach the thickness measurement." },
  { title: "3 · Scan", tab: "image", anchor: "tab-image",
    text: "Each accepted pellet gets an OCT scan. The trained boundary finder draws both surfaces; unclear scans turn undecided." },
  { title: "4 · Correction", tab: "batch", anchor: "tab-batch",
    text: "The window favours big pellets, and big pellets coat faster. Reweighting by size gives the batch's real thinnest tenth." },
  { title: "5 · Stop + sign", tab: "monitoring", anchor: "chart", run: "stop",
    text: "The raw rule stops early. CoatShield waits until the corrected d10 clears the spec, then asks an operator to sign." },
  { title: "6 · Faults", tab: "monitoring", anchor: "faultGroup",
    text: "Inject a fault. With a fouled window CoatShield refuses to stop; with spray-drying the gravimetric rule ships far too many thin pellets." },
];
let tourAt = -1;
function tour(n) {
  const box = $("callout");
  if (n < 0 || n >= TOUR.length) { tourAt = -1; box.hidden = true; return; }
  tourAt = n;
  const step = TOUR[n];
  showTab(step.tab);
  if (step.run === "stop") {
    const cs = state.data.stopByRule.coatshield;
    if (cs.stopped) { state.stopShown[state.id] = false; delete state.signed[state.id];
      setTime(Math.max(0, cs.t_s - 900)); state.speed = 60; $("speed").value = 3; $("speedVal").textContent = "60×"; play(); }
  } else if (n === 0 && state.t === 0) play();
  box.innerHTML = `<h4>${step.title}</h4><p>${step.text}</p>` +
    `<button id="tourNext">${n + 1 < TOUR.length ? "Next" : "Finish"}</button> ` +
    (n ? `<button class="skip" id="tourBack">Back</button> ` : "") + `<button class="skip" id="tourEnd">End tour</button>`;
  box.hidden = false;
  const r = $(step.anchor).getBoundingClientRect();
  const left = Math.max(12, Math.min(window.innerWidth - 360, r.left + 16));
  const top = Math.max(70, Math.min(window.innerHeight - 190, r.top + Math.min(r.height, 320) - 40));
  box.style.left = `${left}px`; box.style.top = `${top}px`;
  $("tourNext").onclick = () => tour(n + 1);
  $("tourEnd").onclick = () => tour(-1);
  if (n) $("tourBack").onclick = () => tour(n - 1);
  render(true);
}

window.consoleExtras = function () {
  buildProcess(); buildImage(); buildSignal(); buildBatch();
  TAB_DRAW.process = drawProcess; TAB_DRAW.image = drawImage; TAB_DRAW.signal = drawSignal; TAB_DRAW.batch = drawBatch;
  HOOKS.drawer = drawDrawer; HOOKS.record = gotRecord;
  $("complianceBtn").onclick = () => { $("drawer").hidden = !$("drawer").hidden; if (!$("drawer").hidden) drawDrawer(); };
  $("tourBtn").onclick = () => tour(tourAt < 0 ? 0 : -1);
  window.CS.tour = tour; window.CS.drawDrawer = drawDrawer;
  // The drift state moves with the replay clock, so an open drawer is refreshed while playing.
  setInterval(() => { if (state.playing && !$("drawer").hidden) drawDrawer(); }, 1000);
};
