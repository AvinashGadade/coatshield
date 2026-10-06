# CoatShield prototype

Software prototype for in-line pellet coating measurement on a Wurster fluid bed.

**Claim.** A raw in-line sample overstates coating quality, because the measurement window
sees big pellets more often and big pellets coat faster. Correcting the size bias and
stopping on d10 (not the mean) fixes it. The OCT modules show the measurement is buildable.

This is a demonstration built on simulated and public data. It is not a validated GMP
system and makes no claim about accuracy on real coated pellets.

## Run it

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                  # all tests
python scripts/run_twin.py --hours 8    # one batch of the digital twin, with its check numbers
```

## Dashboard

```bash
streamlit run app/Home.py
```

Press **Start the demo** on the home page: story mode sets the sliders and opens each page.
The default scenarios load from `app/assets/bundles/` (built by
`python scripts/precompute_scenarios.py`); any other slider setting is simulated live.

## Verify the headline

```bash
python scripts/run_validation.py --quick   # reduced grid, minutes on a laptop
python scripts/run_validation.py           # full grid: 2,140 batches, hours; resumes if stopped
```

Both write CSVs, figures and `validation_summary_<config hash>.md` to `reports/`.

## Layout

| Path | What it holds |
| --- | --- |
| `configs/default.yaml` | Every parameter, each with its source. Presets in `configs/presets/` override it |
| `coatshield/config.py`, `seeds.py` | Typed config loading; the single source of randomness |
| `coatshield/twin/` | M1: mass-conserving, pass-based batch digital twin |
| `coatshield/estimate/` | M6: raw, IPW, model-based and hybrid estimators, four stopping rules, diagnosis |
| `coatshield/oct/` | M2: spectral-domain OCT simulator for coated pellets and its processing chain |
| `coatshield/solve/` | M4: surface fit, refraction correction, three refractive-index methods |
| `coatshield/bundle.py` | Everything the dashboard shows for one scenario, as small files |
| `app/` | Streamlit dashboard (story mode, Batch, Distributions, Controllers, Sensitivity) |
| `scripts/` | Data downloads, check scripts, validation runs |
| `reports/` | Validation CSVs, figures and summaries, named by config hash |
| `tests/` | One test file per module |

## Limits

- No real coated-pellet scans were used; nothing here claims real-world accuracy.
- The headline depends on two assumptions: the window favours big pellets (never measured)
  and big pellets coat faster (published, 1.08-1.81x).
- The correction only undoes selection by size, which the camera sees.
- The system recommends; an operator decides. No code path connects to a safety interlock.

## Data

Datasets are downloaded into `data/raw/` (git-ignored) with SHA-256 checksums in
`data/raw/checksums.json`. None of them is redistributed here.

| Dataset | Script | Licence |
| --- | --- | --- |
| OCT5k retinal OCT labels + scans | `scripts/download_oct5k.py` | Labels CC0 (UCL record); research use per the paper. Scans from Rasti et al. 2018, cite the paper |
| In-vivo skin OCT, Zenodo 18095266 | `scripts/download_zenodo_skin.py` | CC BY 4.0 |
| Duke Chiu 2015 (fallback only) | `scripts/download_duke.py` | Research and education only; never redistributed |
