"""Write web/index.html: the server-free browser fallback, with its parameters from the config.

The page is a dependency-free JavaScript port of the twin with the raw and corrected d10,
two stopping rules and the window-bias slider. Open the file directly in a browser.
"""

from __future__ import annotations

import json

from _common import REPO_ROOT

from coatshield.config import load_config

WEB = REPO_ROOT / "web"


def web_params(cfg) -> dict:
    wc = cfg.web
    return {
        "seed": cfg.seed % 2**32,
        "nPellets": wc.n_pellets,
        "stepS": wc.step_s,
        "smoothSteps": wc.smooth_steps,
        "durationH": cfg.batch.duration_h,
        "coreMedian": cfg.pellet.core_median_um,
        "sizeSigma": cfg.pellet.size_sigma_log,
        "cycleTime": cfg.wurster.cycle_time_s,
        "cycleRsd": cfg.wurster.cycle_time_rsd,
        "depositNm": cfg.wurster.deposit_nm_per_pass,
        "b": cfg.wurster.cycle_size_exponent_b,
        "k": cfg.wurster.size_growth_exponent_k,
        "m": cfg.window.size_bias_m,
        "objectsPerMin": cfg.window.objects_per_min,
        "windowMin": cfg.estimator.window_min,
        "sizeBins": cfg.estimator.n_size_bins,
        "minBinCount": cfg.estimator.min_bin_count,
        "cameraNoise": cfg.camera.size_noise_um,
        "measureSigma": cfg.measurement.sigma_um,
        "spec": cfg.spec.d10_min_um,
        "configHash": cfg.analysis_hash(),
    }


def main() -> None:
    cfg = load_config()
    template = (WEB / "template.html").read_text()
    page = template.replace("/*PARAMS*/", json.dumps(web_params(cfg)))
    (WEB / "index.html").write_text(page)
    print(f"wrote {WEB / 'index.html'} ({len(page) / 1000:.0f} kB)")


if __name__ == "__main__":
    main()
