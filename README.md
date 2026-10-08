# CoatShield prototype

Software prototype for in-line pellet coating measurement on a Wurster fluid bed.

**Claim.** A raw in-line sample overstates coating quality, because the measurement window
sees big pellets more often and big pellets coat faster. Correcting the size bias and
stopping on d10 (the thinnest tenth of the batch), not on the mean, fixes it. The OCT
modules show that the measurement itself is buildable.

This is a demonstration on simulated and public data. It is not a validated GMP system and
makes no claim about accuracy on real coated pellets.

## Run it

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                   # the test suite
streamlit run app/Home.py                # the dashboard
```

On the dashboard's home page press **Start the demo**. Story mode walks the nine demo steps:
each **Next step** sets the sliders and opens the right page. The default scenarios load
from `app/assets/` (no computation, no network, no torch); any other slider setting is
simulated live.

If the dashboard cannot be shown, open `web/index.html` in any browser. It is a single
file with no dependencies that runs a reduced version of the batch simulation in the page,
with the window-bias slider.

## Verify the headline

```bash
python scripts/run_validation.py --quick   # reduced grid, a few minutes on a laptop
python scripts/run_validation.py           # full grid: 2,140 simulated batches
python scripts/build_final_report.py       # every reported number with its source file
```

Each run writes CSVs, figures and a summary to `reports/`, named with the hash of the
configuration that produced them. `reports/final/final_report.md` collects every number
used in the deck.

## What is in the box

| Path | What it holds |
| --- | --- |
| `configs/default.yaml` | Every parameter, each with its source. Presets in `configs/presets/` override it |
| `coatshield/twin/` | Mass-conserving, pass-based batch digital twin with six fault scenarios |
| `coatshield/estimate/` | Raw, IPW, model-based and hybrid estimators; four stopping rules; diagnosis |
| `coatshield/oct/` | Spectral-domain OCT simulator for coated pellets and the instrument-style processing chain |
| `coatshield/seg/` | Boundary finder: compact U-Net (2.2 M parameters) plus a graph search for two ordered surfaces |
| `coatshield/solve/` | Surface fit, refraction correction, three refractive-index methods |
| `coatshield/gate/` | Agglomerate gate: dark-field silhouettes, classical shape rule, small CNN (not trained) |
| `coatshield/chain/` | The whole chain for one object, and the error model the twin uses in its place |
| `coatshield/compliance/` | Confidence and undecided state, drift monitor, audit trail, e-signature, model registry, batch record |
| `app/` | Streamlit dashboard, 15 pages |
| `web/index.html` | Server-free browser fallback |
| `models/` | `unet.onnx`, `error_model.json`, `manifest.json` (hashes, metrics), `CHANGELOG.md` |
| `scripts/` | Data downloads, dataset builders, validation, calibration, exports |
| `reports/` | Validation CSVs, figures and summaries |
| `tests/` | One test file per module |

## Rebuilding from scratch

```bash
OCT5K_IMAGES_PASSWORD=... python scripts/download_oct5k.py   # password: see the script's docstring
python scripts/download_zenodo_skin.py
python scripts/make_synthetic_oct.py        # 10,000 + 1,000 synthetic scans
bash scripts/train_on_gpu.sh <folder with the zips from scripts/package_for_kaggle.py>
python scripts/calibrate_chain.py           # fit the chain's error model
python scripts/build_gallery.py             # scans for the dashboard
python scripts/precompute_scenarios.py      # dashboard scenarios
python scripts/build_web.py                 # browser fallback
```

## Compliance demonstration

`coatshield/compliance/` shows the design a GMP regulator expects of an AI system, at
demonstration level: a confidence score with an "undecided" state, a drift monitor with
alarms, a hash-chained audit trail (`AuditTrail.verify_chain()`), operator e-signatures, a
model registry that refuses a file whose SHA-256 differs from `models/manifest.json`,
deterministic CPU inference (`python scripts/check_determinism.py`), and a batch record as
JSON and PDF. The demo accounts are `operator1` (PIN 2468) and `qa1` (PIN 1357); they exist
only so the sign-off flow can be shown.

The locked test set is built and evaluated by its owner only
(`scripts/make_locked_testset.py`, `scripts/run_locked_eval.py`); no training code may
reference it, and a test enforces that.

**Safety separation.** The system recommends; an operator decides and signs. It never
writes a setpoint by itself, and no code path connects to any safety interlock.

## Limits

- No real coated-pellet scans were used; nothing here claims real-world accuracy.
- The headline depends on two assumptions: the window favours big pellets (never measured)
  and big pellets coat faster (published, 1.08-1.81x within one batch). With either absent,
  the raw rule does not ship extra out-of-spec pellets.
- The correction only undoes selection by size, which the camera sees.
- The estimator relies on a known measurement noise; overstating it makes d10 read high.
- The boundary finder was trained on synthetic scans only. Pretraining on real retinal OCT
  was tried and did not help. Pigmented coatings are not readable: such scans end undecided.
- The gate was tested on synthetic silhouettes only; how fused pellets misread is unknown.
- Maldistribution cannot be told from a healthy batch at the simulated measurement noise.
- The reflectance method for the refractive index is only as accurate as its calibration.
- The dissolution projection is illustrative: a line through three published points.

## Data and licences

Datasets are downloaded into `data/raw/` (git-ignored) with SHA-256 checksums in
`data/raw/checksums.json`. None of them is redistributed here.

| Dataset | Script | Licence and use |
| --- | --- | --- |
| OCT5k retinal OCT labels | `scripts/download_oct5k.py` | CC0 on the UCL record (doi:10.5522/04/22128671); research use per the paper |
| OCT5k scan images | same script | From Rasti, Rabbani, Mehri, Hajizadeh, IEEE TMI 37(4):1024-1034, 2018; cite the paper |
| In-vivo skin OCT, Zenodo 18095266 | `scripts/download_zenodo_skin.py` | CC BY 4.0; used only to compare speckle statistics |
| Duke Chiu 2015 | `scripts/download_duke.py` | Research and education only; fallback, never redistributed, not used |
| Synthetic pellet scans, batches, silhouettes | built by this code | Ours |

The parameter sources (publications behind every number) are listed on the dashboard's
"Limits and sources" page and beside each value in `configs/default.yaml`.
