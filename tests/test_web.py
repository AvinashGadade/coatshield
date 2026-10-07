"""The browser fallback: built from the config, and its simulation shows the headline."""

import json
import shutil
import subprocess
import sys

import pytest

from coatshield.config import REPO_ROOT, load_config

sys.path.insert(0, str(REPO_ROOT / "scripts"))

RUNNER = """
const fs = require("fs");
const html = fs.readFileSync(process.argv[1], "utf8");
const src = html.split("<script>")[1].split("</script>")[0];
const m = { exports: {} };
new Function("module", src)(m);
const out = {};
for (const bias of [0, 3]) {
  const r = m.exports.simulate(bias);
  out[bias] = { rawBelow: r.below[r.rawStop], corBelow: r.below[r.corStop],
                gap: r.raw[r.rawStop] - r.truth[r.rawStop],
                corError: r.corrected[r.corStop] - r.truth[r.corStop] };
}
console.log(JSON.stringify(out));
"""


def test_page_is_built_from_the_config_and_has_no_dependencies(tmp_path):
    from build_web import web_params

    cfg = load_config()
    page = (REPO_ROOT / "web" / "index.html").read_text()
    params = web_params(cfg)
    assert json.dumps(params) in page, "run scripts/build_web.py after changing the config"
    assert params["nPellets"] == 50000 and params["m"] == cfg.window.size_bias_m
    assert "http://" not in page and "https://" not in page and "<script src" not in page


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_simulation_reproduces_the_raw_against_corrected_gap(tmp_path):
    page = (REPO_ROOT / "web" / "index.html").read_text().replace('"nPellets": 50000',
                                                                 '"nPellets": 12000')
    path = tmp_path / "index.html"
    path.write_text(page)
    run = subprocess.run(["node", "-e", RUNNER, str(path)], capture_output=True, text=True,
                         check=True, timeout=600)
    out = json.loads(run.stdout)
    biased, fair = out["3"], out["0"]
    assert biased["gap"] > 0.6  # the raw sample overstates d10
    assert biased["rawBelow"] > 1.4 * biased["corBelow"]
    assert biased["corBelow"] < 13 and abs(biased["corError"]) < 0.4
    assert fair["gap"] < 0.2  # no window bias: no overstatement
    assert abs(fair["rawBelow"] - fair["corBelow"]) < 6
