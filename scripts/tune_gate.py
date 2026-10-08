"""Tune the classical gate's twin rule and compare gate methods on held-out images.

Thresholds are tuned on one synthetic set (stream "tune") and scored on another ("test").
Writes the comparison table and confusion matrix to reports/. The CNN row is filled only
if a trained checkpoint exists; otherwise it is marked "not run" with the reason.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from _common import REPORTS_DIR

from coatshield.config import load_config
from coatshield.gate.classical import classify, measure, tune_twin_rule
from coatshield.gate.metrics import confusion_matrix, gate_summary
from coatshield.gate.silhouettes import CLASSES, make_set


def main() -> None:
    cfg = load_config()
    gv = cfg.gate_vision
    tag = cfg.hash(include=("seed", "gate_vision"))

    images, labels, _ = make_set(cfg, "tune", gv.n_tune)
    tuned = tune_twin_rule([measure(im, cfg) for im in images], labels, cfg)
    print("tuned on", gv.n_tune, "images:", json.dumps(tuned))
    if (tuned["solidity_min"], tuned["defect_max"]) != (gv.solidity_min, gv.defect_max):
        print(f"NOTE: configs/default.yaml has solidity_min={gv.solidity_min}, "
              f"defect_max={gv.defect_max}; update it to the tuned values to use them.")

    test_images, test_labels, _ = make_set(cfg, "test", gv.n_test)
    start = time.perf_counter()
    verdicts = [classify(im, cfg) for im in test_images]
    latency_ms = (time.perf_counter() - start) / len(test_images) * 1000.0
    predicted = np.array([v.label for v in verdicts])
    confidence = np.array([v.confidence for v in verdicts])
    summary = gate_summary(test_labels, predicted, confidence, gv.single_confidence_min,
                           gv.ece_bins)
    summary["cpu_ms_per_image"] = latency_ms
    matrix = pd.DataFrame(confusion_matrix(test_labels, predicted), index=CLASSES,
                          columns=CLASSES)
    matrix.to_csv(REPORTS_DIR / f"gate_confusion_classical_{tag}.csv")

    from coatshield.gate.train import CHECKPOINT

    cnn_row = ("| Small CNN | not run: no trained checkpoint (needs a GPU session; "
               "`python -m coatshield.gate.train`) | | | | | |")
    meets = (summary["twin_recall"] >= gv.twin_recall_min
             and summary["false_twin_rate"] <= gv.false_twin_max)
    lines = [
        "# Agglomerate gate: method comparison", "",
        f"Config `{tag}` · thresholds tuned on {gv.n_tune} images, scored on {gv.n_test} "
        f"held-out synthetic images, {len(CLASSES)} balanced classes", "",
        "| Method | Twin recall | False twins | Twins leaking as single | Accuracy (8 classes) "
        "| ECE | CPU ms per image |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        f"| Classical (solidity < {gv.solidity_min:g} or defect > {gv.defect_max:g}) | "
        f"{100 * summary['twin_recall']:.1f}% | {100 * summary['false_twin_rate']:.1f}% | "
        f"{100 * summary['twin_leak']:.1f}% | {100 * summary['accuracy']:.1f}% | "
        f"{summary['ece']:.3f} | {latency_ms:.1f} |",
        cnn_row if not CHECKPOINT.exists() else "| Small CNN | see gate_cnn report | | | | | |",
        "",
        f"Target: twin recall >= {100 * gv.twin_recall_min:.0f}% with false twins <= "
        f"{100 * gv.false_twin_max:.0f}%: **{'met' if meets else 'NOT met'}** by the classical "
        "method.",
        "",
        f"Chosen method: classical. {100 * summary['single_pass_rate']:.1f}% of single pellets "
        f"pass to OCT; {100 * summary['non_single_leak']:.2f}% of all other objects pass.",
        "",
        "Limits: the images are synthetic dark-field silhouettes; how real fused pellets look "
        "and misread is unknown, so the gate's benefit is shown, not quantified.", "",
        "## Confusion matrix (rows: true class)", "",
        "| | " + " | ".join(CLASSES) + " |",
        "| --- |" + " --- |" * len(CLASSES),
        *[f"| {name} | " + " | ".join(str(v) for v in matrix.loc[name]) + " |"
          for name in CLASSES],
        "",
    ]
    text = "\n".join(lines)
    (REPORTS_DIR / f"gate_comparison_{tag}.md").write_text(text)
    on_tuning_set = {f"tuning_{k}": v for k, v in tuned.items()
                     if k in ("twin_recall", "false_twin_rate")}
    thresholds = {k: tuned[k] for k in ("solidity_min", "defect_max")}
    pd.DataFrame([{"method": "classical", **summary, **thresholds, **on_tuning_set}]).to_csv(
        REPORTS_DIR / f"gate_comparison_{tag}.csv", index=False)
    print(text)


if __name__ == "__main__":
    main()
