"""Drive the Live Console in headless Chrome and check what the brief asks of it.

Checks: the page loads with no script errors; the KPI tiles match timeline.json at several
batch times; a pellet card shows its scan; every tab draws and switching tabs keeps the
replay clock; replay at 60x holds its frame rate. `--shots DIR` also saves a screenshot per
tab. Needs google-chrome on PATH; no network is used.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets
from _common import REPO_ROOT

PORT = 9334
PAGE = REPO_ROOT / "web" / "console" / "index.html"
DATA = REPO_ROOT / "web" / "console" / "data"
TABS = ("monitoring", "process", "image", "signal", "batch")
CHECK_HOURS = (2.0, 5.0, 8.0, 9.5)
PLAY_SECONDS = 6.0
MIN_FPS = 30.0  # "smooth" in the brief; headless Chrome draws with software rendering


class Browser:
    def __init__(self, ws) -> None:
        self.ws, self.counter, self.errors = ws, 0, []

    async def call(self, method: str, **params):
        self.counter += 1
        mine = self.counter
        await self.ws.send(json.dumps({"id": mine, "method": method, "params": params}))
        while True:
            message = json.loads(await self.ws.recv())
            if message.get("method") == "Runtime.exceptionThrown":
                details = message["params"]["exceptionDetails"]
                self.errors.append(details.get("exception", {}).get("description")
                                   or details.get("text"))
            if message.get("id") == mine:
                return message.get("result", {})

    async def js(self, expression: str):
        result = await self.call("Runtime.evaluate", expression=expression, returnByValue=True,
                                 awaitPromise=True)
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"].get("exception", {})
                               .get("description", "script error"))
        return result["result"].get("value")

    async def shot(self, path: Path) -> None:
        data = await self.call("Page.captureScreenshot", format="png")
        path.write_bytes(base64.b64decode(data["data"]))


async def run(shots: Path | None, width: int, height: int) -> list[str]:
    problems: list[str] = []
    profile = tempfile.mkdtemp(prefix="chrome-console-")
    chrome = subprocess.Popen(
        ["google-chrome", "--headless=new", "--no-sandbox", f"--user-data-dir={profile}",
         f"--remote-debugging-port={PORT}", f"--window-size={width},{height}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        tabs = []
        for _ in range(60):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://localhost:{PORT}/json"))
                break
            except OSError:
                time.sleep(0.5)
        target = next(t for t in tabs if t["type"] == "page")["webSocketDebuggerUrl"]
        async with websockets.connect(target, max_size=None) as ws:
            b = Browser(ws)
            await b.call("Runtime.enable")
            await b.call("Page.enable")
            await b.call("Page.navigate", url=PAGE.as_uri())
            for _ in range(60):
                await asyncio.sleep(0.5)
                if await b.js("!!(window.CS && CS.state.data && window.CONSOLE_DATA && "
                              "Object.keys(CONSOLE_DATA).length === CONSOLE_SCENARIOS.length)"):
                    break
            else:
                return ["the console did not finish loading"]

            # 1. KPI tiles against timeline.json.
            timeline = json.loads((DATA / "default" / "timeline.json").read_text())
            meta = json.loads((DATA / "default" / "meta.json").read_text())
            for hours in CHECK_HOURS:
                await b.js(f"CS.setTime({hours * 3600}); CS.render(true); 0")
                i = min(len(timeline["t_s"]) - 1, int(hours * 3600 // meta["step_s"]) - 1)
                tiles = await b.js("Object.fromEntries(['d10','p','d50','cv','n','gate']"
                                   ".map(k => [k, document.getElementById('t-'+k).textContent]))")
                want = {"d10": f"{timeline['corr_d10'][i]:.2f} µm",
                        "p": f"{100 * timeline['p_d10_ge_spec'][i]:.0f} %",
                        "d50": f"{timeline['corr_d50'][i]:.2f} µm",
                        "cv": f"{100 * timeline['cv'][i]:.1f} %",
                        "n": f"{timeline['n_pooled'][i]:.3f}",
                        "gate": f"{timeline['n_accepted'][i]:,} accepted"}
                for key, value in want.items():
                    if tiles[key] != value:
                        problems.append(f"tile {key} at {hours} h shows {tiles[key]!r}, "
                                        f"timeline says {value!r}")

            # 2. A pellet card shows its scan. (Reset first: the last KPI time is past the stop.)
            await b.js("CS.reset(); document.getElementById('signModal').hidden = true; "
                       "CS.setTime(5 * 3600); CS.render(true); 0")
            await b.js("CS.eventCard(CS.state.data.acc.i[CS.upto(CS.state.data.acc.x, 5) - 1], "
                       "true); 0")
            await asyncio.sleep(1.0)
            card = await b.js("(() => { const im = document.querySelector('#card img.scan'); "
                              "return im ? im.naturalWidth : 0; })()")
            if not card:
                problems.append("the pellet card did not show a scan")
            if shots:
                await b.shot(shots / "console_card.png")

            # 3. Every tab draws; switching tabs keeps the clock.
            await b.js("CS.setTime(6 * 3600); CS.render(true); 0")
            for tab in TABS:
                before = await b.js("CS.state.t")
                await b.js(f"CS.showTab('{tab}'); 0")
                await asyncio.sleep(2.5 if tab == "signal" else 1.2)
                if await b.js("CS.state.t") != before:
                    problems.append(f"switching to the {tab} tab moved the replay clock")
                if shots:
                    await b.shot(shots / f"console_{tab}.png")

            # 4. Replay at 60x: frame rate while the Monitoring chart streams.
            await b.js("CS.showTab('monitoring'); CS.setTime(3 * 3600); CS.state.speed = 60; 0")
            frames = await b.js(
                "new Promise(done => { const t = []; CS.play(); const end = performance.now() + "
                f"{PLAY_SECONDS * 1000};" " (function f(now) { t.push(now); if (now < end) "
                "requestAnimationFrame(f); else { CS.pause(); done(t); } })(performance.now()); })")
            gaps = [b_ - a for a, b_ in zip(frames, frames[1:], strict=False)]
            fps = 1000.0 * len(gaps) / sum(gaps)
            slow = sum(g > 50.0 for g in gaps)
            advanced = await b.js("CS.state.t") - 3 * 3600
            print(f"replay at 60x: {fps:.0f} frames/s, {slow} of {len(gaps)} frames over 50 ms, "
                  f"clock advanced {advanced:.0f} s in {PLAY_SECONDS:.0f} s")
            if fps < MIN_FPS:
                problems.append(f"replay at 60x ran at {fps:.0f} frames/s (need {MIN_FPS:.0f})")
            if abs(advanced - 60 * PLAY_SECONDS) > 0.1 * 60 * PLAY_SECONDS:
                problems.append(f"the replay clock advanced {advanced:.0f} s, not "
                                f"{60 * PLAY_SECONDS:.0f} s")

            # 5. The stop recommendation opens the signature form.
            stop = await b.js("CS.state.data.stopByRule.coatshield.t_s")
            await b.js(f"CS.setTime({stop} + 60); CS.render(true); 0")
            await asyncio.sleep(0.5)
            if await b.js("document.getElementById('signModal').hidden"):
                problems.append("the signature form did not open at the stop recommendation")
            if shots:
                await b.shot(shots / "console_stop.png")
            problems += [f"script error: {e}" for e in b.errors]
    finally:
        chrome.terminate()
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shots", type=Path, help="directory for one screenshot per tab")
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=1000)
    args = parser.parse_args()
    if args.shots:
        args.shots.mkdir(parents=True, exist_ok=True)
    problems = asyncio.run(run(args.shots, args.width, args.height))
    for p in problems:
        print("PROBLEM:", p)
    print("console check:", "FAILED" if problems else "passed")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
