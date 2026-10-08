# CoatShield Live Console: build brief for Claude Code

**Goal:** add one new demo screen, the **Live Console**. It replays a simulated batch the way an in-line OCT operator screen does: per-pellet thickness dots streaming in over time, live KPI tiles, and hovering on a dot to see its scan. It must also show what no plain OCT screen shows:

- the agglomerate gate
- the size-bias correction
- the d10 stop rule
- the solved refractive index
- the "undecided" state
- the operator e-signature

**Rules:**

- Reuse the existing CoatShield prototype: the twin, the estimators, the measurement-chain error model, the synthetic OCT generator, the U-Net and graph search, and the audit trail. Do **not** retrain anything, and do not change locked models (v1.0).
- Every screen says **"Simulated replay"**. No claim of real-pellet accuracy.
- Our own UI, colours and wording. Do not copy any vendor's software layout, names or branding.
- Same seed gives the same replay. Add tests for every exporter and every number shown.

**Colours:** these match the deck and the video.

| Use | Colour |
|---|---|
| Gate | orange `#E8A33D` |
| Batch correction | blue `#3B8ED0` |
| Thickness | green `#3FA35B` |
| Compliance | purple `#8A63C9` |
| Stop / fail | red `#C62839` |
| Raw estimate | red dashed |
| Corrected estimate | solid blue with band |

**Recommended tech:**

- A static single-page app at `web/console/index.html` using Plotly.js or ECharts, vanilla JS or Vite + React.
- It reads precomputed JSON and PNG files from `web/console/data/<scenario>/`.
- No server is needed, it runs offline, and it can be hosted anywhere later.
- Add a Streamlit page `app/pages/00_Live_Console.py` that embeds it, so the existing story mode can link to it.

---

## Phase 1: Replay data + the main screen (core; must finish)

### 1a. Exporter (`coatshield/console_export.py` + CLI `make console-data`)

For each chosen scenario (start with: default bias, no bias, window fouling, spray-drying), run the twin with the fitted measurement-chain error model and write the files below.

**`events.json`**: one row per window detection, downsampled to about 3,000–6,000 per batch. Fields:

- `t_s`
- `size_um`
- `gate_class`: single / agglomerate / partial / fouled / defocused / empty
- `gate_conf`
- `accepted` (bool)
- `undecided` (bool)
- `thickness_um` (measured)
- `n_est`
- `conf`
- `scan_id`

**`timeline.json`**: one row per estimator update. Fields:

- `t_s`
- `raw_mean`
- `raw_d10`
- `corr_d10`, `corr_d10_lo`, `corr_d10_hi`
- `corr_d50`, `corr_d90`
- `p_d10_ge_spec`
- `cv`
- `n_accepted`, `n_rejected_agglom`, `n_undecided`
- `true_d10` (hidden by default; it is for the "reveal" toggle)

**`stops.json`**: when each rule would stop, with the true % below spec at that moment. One entry for each rule: raw mean, raw d10, gravimetric, and CoatShield.

**`scans/scan_XXXX.png`**: about 300 representative synthetic B-scans, generated with the Phase 4 simulator and segmented by the Phase 5 boundary finder, with the two surfaces drawn on.

- Each event's `scan_id` points to the nearest scan by thickness and size.
- The tooltip says **"representative simulated scan"**.
- Also save a matching camera-frame thumbnail per gate class.

**`meta.json`**: scenario name, spec, seed, model hash, the pellet-size assumption, and the source file for each number.

### 1b. Main screen: "Monitoring" tab

**Left rail:**

- Start / Pause / Reset buttons.
- Replay speed: 1× to 200×.
- Scenario selector.
- Spec input (read-only from meta).
- "Reveal true d10" toggle.

**Centre chart:** thickness (µm) against batch time.

- Accepted pellets are grey dots.
- Rejected agglomerates are small orange ✕ marks.
- Undecided scans are hollow grey circles.
- Red horizontal line at the spec.
- Red dashed line for raw d10.
- Blue line with a shaded band for the corrected d10.
- Thin grey line for the raw mean.
- A trend projection gives the "estimated time to spec".

**Right column, KPI tiles:**

| Tile | Shows |
|---|---|
| Thinnest 10% (corrected d10) | value ± interval |
| P(d10 ≥ spec) | gauge, turns green above 0.95 |
| Median thickness | value |
| Batch variation | CV |
| Refractive index | n ± |
| Gate | accepted / agglomerates rejected / undecided % |
| Elapsed time | — |
| Est. time to stop | — |

**Hover or click on a dot** opens a side card showing:

- the B-scan with both surfaces drawn
- the camera thumbnail and gate verdict
- thickness, n, confidence and timestamp

**Stop event:** when P(d10 ≥ spec) crosses 0.95:

1. A red banner appears: **"Recommend STOP"**.
2. An operator e-signature modal opens, asking for name, reason and password.
3. Signing writes an entry to the existing hash-chained audit trail.
4. Replay pauses.

**Done when:**

- The default scenario plays smoothly at 60× with no frame drops.
- The KPI values match `timeline.json`.
- Hover shows the scan.
- Signing creates a valid audit entry.
- The tests pass.

---

## Phase 2: The other tabs (what makes it look like a real instrument)

**Process overview:**

- A Wurster schematic with the draft tube, fountain and downbed. The probe window sits at the downbed.
- Animated dots move at the scenario's rate.
- Live counters: detections per minute, accepted %, agglomerates per minute.
- A window-health bar that drops during fouling.

**Image:**

- A live strip of streaming B-scans, with the U-Net surfaces drawn on (red top, cyan bottom).
- Next to it, the camera frame with the gate verdict and confidence.
- A Grad-CAM toggle, using the existing explainability output.

**Signal:**

- For the selected pellet: raw spectrum → resampled → FFT → A-scan with two peaks and Δz marked.
- Then `d = Δz ÷ n`, with the three n estimates shown side by side (reflectance, camera–OCT fusion, microscopy anchor).

**Batch:**

- Window size histogram vs true bed size histogram.
- The weights `w = f_true ÷ f_obs`.
- Raw vs corrected thickness distributions, with the failing tail shaded.

**Done when:** all four tabs use the same replay clock, and switching tabs never resets the replay.

---

## Phase 3: The wow moments + demo hardening

**Ghost-run comparison:** on the Monitoring chart, show vertical markers where each rule would have stopped. Each marker shows the true % below spec from `stops.json`:

- Raw rule: stops early, 23.1% below spec
- CoatShield: 9.6% below spec

This is the main moment of the demo.

**Fault buttons:** inject faults during replay, switching to the precomputed fault scenario:

- **Window fouling:** undecided % rises, and CoatShield refuses to stop.
- **Spray-drying:** the gravimetric rule would ship 44% below spec.

**Compliance drawer:**

- Model registry hash ✓.
- Drift monitor state.
- The last 10 audit entries.
- "Export batch record (PDF)" button, using the existing exporter.

**Story mode:** add a 6-step guided tour with "Next" callouts:

1. Process
2. Gate
3. Scan
4. Correction
5. Stop + sign
6. Faults

**Hardening:**

- Offline build.
- Preload all data.
- A 90-second screen-recorded backup of the full demo.
- Quick-verify still reproduces the headline numbers.
- Re-tag as v1.1.

---

## Numbers to keep consistent everywhere (deck, video, console)

Use the **final prototype numbers**, from the realistic measurement chain over 2,140 simulated batches:

- Raw d10 rule: **23.1%** below spec
- CoatShield: **9.6%** below spec
- That is **2.4×** the out-of-spec pellets.

**Window share and pellet count:**

- The twin assumes about 96 million pellets in 30 kg and a 0.28% window share.
- The deck says about 2.6 billion pellets and 0.005%.
- The difference comes from the core size assumed. Pick one and state it on screen, for example: *"assumes ~700 µm cores"*.
