"""Screenshot web pages with headless Chrome: screenshot_pages.py OUT_DIR WAIT_S URL [URL...]

Used to check dashboard pages and to export deck figures. Needs google-chrome on PATH.
"""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.request

import websockets

PORT = 9333
WIDTH = 1500


async def capture(out_dir: str, wait_s: float, urls: list[str]) -> None:
    profile = tempfile.mkdtemp(prefix="chrome-shot-")
    chrome = subprocess.Popen(
        ["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu",
         f"--user-data-dir={profile}", f"--remote-debugging-port={PORT}",
         f"--window-size={WIDTH},1300", "about:blank"],
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
            counter = 0

            async def call(method: str, **params):
                nonlocal counter
                counter += 1
                await ws.send(json.dumps({"id": counter, "method": method, "params": params}))
                while True:
                    message = json.loads(await ws.recv())
                    if message.get("id") == counter:
                        return message.get("result", {})

            height_js = ("Math.max(document.documentElement.scrollHeight, ...[...document."
                         "querySelectorAll('[data-testid=stMain]')].map(e => e.scrollHeight), 900)")
            for url in urls:
                name = url.rstrip("/").split("/")[-1].split("?")[0] or "Home"
                await call("Page.navigate", url=url)
                await asyncio.sleep(wait_s)
                result = await call("Runtime.evaluate", expression=height_js)
                height = int(min(result["result"].get("value", 1300) + 80, 4000))
                await call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=height,
                           deviceScaleFactor=1, mobile=False)
                await asyncio.sleep(1.5)
                shot = await call("Page.captureScreenshot", format="png")
                path = f"{out_dir}/shot_{name}.png"
                with open(path, "wb") as fh:
                    fh.write(base64.b64decode(shot["data"]))
                await call("Emulation.clearDeviceMetricsOverride")
                print("saved", path)
    finally:
        chrome.terminate()


if __name__ == "__main__":
    asyncio.run(capture(sys.argv[1], float(sys.argv[2]), sys.argv[3:]))
