"""Build the locked test set. To be run by the test owner only, never by training code.

The scans come from a seed range that training and validation never use
(oct_dataset.seed_offsets.locked). Every build is appended to the access log.
This script and scripts/run_locked_eval.py are the only code allowed to touch
data/test_locked/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import json
import os

from _common import REPO_ROOT
from make_synthetic_oct import build_split

from coatshield.config import load_config

LOCKED = REPO_ROOT / "data" / "test_locked"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--owner", default=getpass.getuser(), help="name of the test owner")
    parser.add_argument("--n", type=int, default=None, help="scans (default: config)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--confirm", action="store_true",
                        help="required: confirms you are the test owner and do not train models")
    args = parser.parse_args()
    if not args.confirm:
        raise SystemExit("Refusing to run without --confirm. Only the test owner builds this set.")
    cfg = load_config()
    manifest = build_split(cfg, "locked", args.n or cfg.oct_dataset.n_locked,
                           LOCKED / "oct", args.workers)
    entry = {"action": "build", "who": args.owner,
             "when_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), **manifest}
    with open(LOCKED / "access_log.jsonl", "a") as log:
        log.write(json.dumps(entry) + "\n")
    print(json.dumps(entry))


if __name__ == "__main__":
    main()
