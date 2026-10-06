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

## Layout

| Path | What it holds |
| --- | --- |
| `configs/default.yaml` | Every parameter, each with its source. Presets in `configs/presets/` override it |
| `coatshield/config.py`, `seeds.py` | Typed config loading; the single source of randomness |
| `coatshield/twin/` | M1: mass-conserving, pass-based batch digital twin |
| `scripts/` | Data downloads, check scripts, validation runs |
| `tests/` | One test file per module |

## Data

Datasets are downloaded into `data/raw/` (git-ignored) with SHA-256 checksums in
`data/raw/checksums.json`. None of them is redistributed here.

| Dataset | Script | Licence |
| --- | --- | --- |
| OCT5k retinal OCT labels + scans | `scripts/download_oct5k.py` | Labels CC0 (UCL record); research use per the paper. Scans from Rasti et al. 2018, cite the paper |
| In-vivo skin OCT, Zenodo 18095266 | `scripts/download_zenodo_skin.py` | CC BY 4.0 |
| Duke Chiu 2015 (fallback only) | `scripts/download_duke.py` | Research and education only; never redistributed |
