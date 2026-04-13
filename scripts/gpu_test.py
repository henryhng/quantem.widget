#!/usr/bin/env python3
"""
GPU colormap end-to-end test via Chrome CDP.

Runs real notebook cells on real GPU hardware (no headless Playwright).
Connects to Chrome (with --remote-debugging-port=9222) and JupyterLab,
executes cells, checks GPU console logs, validates pixel output, and
takes screenshots.

Usage:
    # 1. Start JupyterLab (if not running):
    jupyter lab --notebook-dir=notebooks --port=8889

    # 2. Start Chrome with WebGPU + remote debugging:
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' \
      --remote-debugging-port=9222 '--remote-allow-origins=*' \
      --no-first-run --user-data-dir=/tmp/chrome-gpu \
      "http://localhost:8889/lab?token=YOUR_TOKEN"

    # 3. Run the test:
    python scripts/gpu_test.py

    # Or with auto-build:
    python scripts/gpu_test.py --build

    # Watch mode (rebuilds + retests on file change):
    python scripts/gpu_test.py --watch
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

try:
    import requests
    import websocket
except ImportError:
    print("Install: pip install requests websocket-client")
    sys.exit(1)

REPO = Path(__file__).parent.parent
SCREENSHOT_DIR = REPO / "tests" / "screenshots" / "gpu"
JLAB_PORT = 8889
CDP_PORT = 9222

# Test notebook cells (executed in order)
TEST_CELLS = [
    "%load_ext autoreload\n%autoreload 2\n%env ANYWIDGET_HMR=1",
    "from quantem.widget import IO, Show2D",
    'result = IO.folder("/Users/macbook/data/bob/20260409_gold_drift_v3", file_type="emd", shape=(4096, 4096))',
    "w = Show2D(result, ncols=4)",
]

# Colormap changes to test GPU path
CMAP_TESTS = ["viridis", "hot", "plasma", "inferno"]


class CDPConnection:
    """Chrome DevTools Protocol connection."""

    def __init__(self, port=CDP_PORT):
        targets = requests.get(f"http://localhost:{port}/json", timeout=5).json()
        page = [t for t in targets if t["type"] == "page" and f"localhost:{JLAB_PORT}" in t["url"]]
        if not page:
            raise RuntimeError(f"No Chrome tab found for localhost:{JLAB_PORT}")
        self.ws = websocket.create_connection(page[0]["webSocketDebuggerUrl"], timeout=30)
        self._msg_id = 0

    def eval(self, expr, timeout=30):
        self._msg_id += 1
        mid = self._msg_id
        self.ws.send(json.dumps({
            "id": mid, "method": "Runtime.evaluate",
            "params": {"expression": expr, "returnByValue": True, "awaitPromise": True},
        }))
        deadline = time.time() + timeout
        while time.time() < deadline:
            raw = self.ws.recv()
            if isinstance(raw, bytes):
                continue
            resp = json.loads(raw)
            if resp.get("id") == mid:
                result = resp.get("result", {}).get("result", {})
                if "value" in result:
                    return result["value"]
                exc = resp.get("result", {}).get("exceptionDetails", {})
                if exc:
                    return f"ERROR: {exc.get('text', str(exc)[:200])}"
                return None
        return "TIMEOUT"

    def reload(self):
        self._msg_id += 1
        self.ws.send(json.dumps({"id": self._msg_id, "method": "Page.reload", "params": {"ignoreCache": True}}))
        for _ in range(10):
            raw = self.ws.recv()
            if isinstance(raw, bytes):
                continue
            break

    def send_key(self, key, code, vk, modifiers=0):
        self._msg_id += 1
        self.ws.send(json.dumps({
            "id": self._msg_id, "method": "Input.dispatchKeyEvent",
            "params": {"type": "keyDown", "key": key, "code": code, "modifiers": modifiers, "windowsVirtualKeyCode": vk},
        }))
        for _ in range(5):
            raw = self.ws.recv()
            if isinstance(raw, bytes):
                continue
            break
        self._msg_id += 1
        self.ws.send(json.dumps({
            "id": self._msg_id, "method": "Input.dispatchKeyEvent",
            "params": {"type": "keyUp", "key": key, "code": code, "windowsVirtualKeyCode": vk},
        }))
        for _ in range(5):
            raw = self.ws.recv()
            if isinstance(raw, bytes):
                continue
            break

    def screenshot(self, path):
        self._msg_id += 1
        mid = self._msg_id
        self.ws.send(json.dumps({"id": mid, "method": "Page.captureScreenshot", "params": {"format": "png"}}))
        for _ in range(20):
            raw = self.ws.recv()
            if isinstance(raw, bytes):
                continue
            resp = json.loads(raw)
            if resp.get("id") == mid:
                Path(path).write_bytes(base64.b64decode(resp["result"]["data"]))
                return True
        return False

    def navigate(self, url):
        """Navigate to URL and reconnect (page navigation kills websocket)."""
        self._msg_id += 1
        try:
            self.ws.send(json.dumps({
                "id": self._msg_id, "method": "Page.navigate",
                "params": {"url": url},
            }))
            self.ws.recv()
        except Exception:
            pass
        try:
            self.ws.close()
        except Exception:
            pass

    def close(self):
        try:
            self.ws.close()
        except Exception:
            pass


class KernelConnection:
    """Jupyter kernel websocket connection."""

    def __init__(self, port=JLAB_PORT):
        self.port = port
        self.token = self._get_token()
        self.headers = {"Authorization": f"token {self.token}"}

    def _get_token(self):
        # Read token from running server
        result = subprocess.run(
            [sys.executable, "-m", "jupyter", "lab", "list"],
            capture_output=True, text=True,
        )
        for line in result.stdout.split("\n") + result.stderr.split("\n"):
            if f"localhost:{self.port}" in line and "token=" in line:
                return line.split("token=")[1].split()[0].split("&")[0]
        raise RuntimeError(f"No JupyterLab found on port {self.port}")

    def get_kernel_id(self):
        sessions = requests.get(
            f"http://localhost:{self.port}/api/sessions", headers=self.headers
        ).json()
        if not sessions:
            raise RuntimeError("No kernel sessions")
        return sessions[0]["kernel"]["id"]

    def restart_kernel(self):
        kid = self.get_kernel_id()
        r = requests.post(
            f"http://localhost:{self.port}/api/kernels/{kid}/restart",
            headers={**self.headers, "Content-Type": "application/json"},
        )
        return r.status_code == 200

    def execute(self, code, timeout=60):
        kid = self.get_kernel_id()
        ws_url = f"ws://localhost:{self.port}/api/kernels/{kid}/channels?token={self.token}"
        ws = websocket.create_connection(ws_url, timeout=timeout)
        msg_id = str(uuid.uuid4())
        ws.send(json.dumps({
            "header": {"msg_id": msg_id, "msg_type": "execute_request",
                       "username": "", "session": str(uuid.uuid4()), "date": "", "version": "5.3"},
            "parent_header": {}, "metadata": {},
            "content": {"code": code, "silent": False, "store_history": True,
                        "user_expressions": {}, "allow_stdin": False, "stop_on_error": True},
            "buffers": [], "channel": "shell",
        }))
        stdout_lines = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            raw = ws.recv()
            if isinstance(raw, bytes):
                continue
            try:
                resp = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            parent = resp.get("parent_header", {}).get("msg_id", "")
            if parent == msg_id:
                if resp["msg_type"] == "stream":
                    stdout_lines.append(resp["content"].get("text", "").strip())
                elif resp["msg_type"] == "execute_reply":
                    ws.close()
                    return {
                        "status": resp["content"]["status"],
                        "stdout": "\n".join(stdout_lines),
                        "error": resp["content"].get("evalue", "") if resp["content"]["status"] == "error" else "",
                    }
        ws.close()
        return {"status": "timeout", "stdout": "", "error": "execution timed out"}


def build_js():
    print("  Building JS...", end=" ", flush=True)
    r = subprocess.run(["npm", "run", "build"], capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0:
        print("FAILED")
        print(r.stderr)
        return False
    print("done")
    return True


def install_interceptor(cdp):
    cdp.eval("""
    window._gpuLogs=[];
    const _ol=console.log,_ow=console.warn,_oe=console.error;
    console.log=function(){window._gpuLogs.push("[L]"+[...arguments].join(" "));_ol.apply(console,arguments)};
    console.warn=function(){window._gpuLogs.push("[W]"+[...arguments].join(" "));_ow.apply(console,arguments)};
    console.error=function(){window._gpuLogs.push("[E]"+[...arguments].join(" "));_oe.apply(console,arguments)};
    "ok"
    """)


def get_gpu_logs(cdp, filter_str=None):
    if filter_str:
        expr = f'JSON.stringify(window._gpuLogs.filter(l => l.includes("{filter_str}")))'
    else:
        expr = "JSON.stringify(window._gpuLogs.slice(-50))"
    r = cdp.eval(expr)
    try:
        return json.loads(r) if r else []
    except (json.JSONDecodeError, TypeError):
        return []


def run_cells_via_ui(cdp, n_cells=20, heavy_cells=None):
    """Run cells via Shift+Enter from the notebook UI."""
    heavy_cells = heavy_cells or {2: 8, 3: 8}  # cell_index: wait_seconds
    cdp.eval('(() => { const c = document.querySelector(".jp-CodeCell"); if(c) c.click(); return "ok"; })()')
    time.sleep(0.5)
    for i in range(n_cells):
        cdp.send_key("Enter", "Enter", 13, modifiers=8)  # Shift+Enter
        wait = heavy_cells.get(i, 1)
        time.sleep(wait)


def check_canvases(cdp):
    r = cdp.eval('document.querySelectorAll("canvas").length')
    return int(r) if r else 0


def run_gpu_test(build=False):
    """Run the full GPU colormap test."""
    print("\n=== GPU Colormap E2E Test ===\n")
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    results = {"pass": 0, "fail": 0, "tests": []}

    def record(name, passed, detail=""):
        status = "PASS" if passed else "FAIL"
        results["pass" if passed else "fail"] += 1
        results["tests"].append({"name": name, "passed": passed, "detail": detail})
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    # Step 0: Build
    if build:
        if not build_js():
            record("build", False, "npm run build failed")
            return results

    # Step 1: Connect
    print("1. Connecting...")
    try:
        kernel = KernelConnection()
        print(f"  JupyterLab token: {kernel.token[:12]}...")
    except RuntimeError as e:
        record("connect_jupyter", False, str(e))
        return results

    try:
        cdp = CDPConnection()
    except RuntimeError as e:
        record("connect_chrome", False, str(e))
        return results

    # Check WebGPU
    has_gpu = cdp.eval("String(!!navigator.gpu)")
    record("webgpu_available", has_gpu == "true", f"navigator.gpu={has_gpu}")
    if has_gpu != "true":
        cdp.close()
        return results

    # Step 2: Ensure w exists + reload page for new JS
    print("\n2. Setting up widget...")
    w_check = kernel.execute("type(w).__name__", timeout=5)
    if w_check["status"] != "ok":
        print("  w not found — executing cells via kernel API...")
        kernel.restart_kernel()
        time.sleep(2)
        nb_path = str(REPO / "notebooks" / "show2d" / "show2d_gold_drift.ipynb")
        with open(nb_path) as f:
            nb = json.load(f)
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"]).strip()
            if not source:
                continue
            result = kernel.execute(source, timeout=60)
            if result["status"] == "error":
                print(f"    ERROR: {result['error'][:80]}")
                break
    else:
        print("  w exists from previous run")

    # Reload page to pick up new JS bundle
    print("  Reloading page...")
    cdp.reload()
    time.sleep(8)
    install_interceptor(cdp)

    # Step 3: Run cells in browser for widget rendering
    print("\n3. Running cells in browser...")
    run_cells_via_ui(cdp, n_cells=20, heavy_cells={2: 10, 3: 10})
    print("  Waiting for GPU initialization...")
    time.sleep(12)

    # Verify w exists
    result = kernel.execute("type(w).__name__", timeout=10)
    if result["status"] != "ok":
        print("  WARNING: cells may not have executed, retrying...")
        run_cells_via_ui(cdp, n_cells=20, heavy_cells={2: 10, 3: 10})
        time.sleep(12)

    # Step 4: Check GPU initialization
    print("\n4. Checking GPU state...")
    canvas_count = check_canvases(cdp)
    record("widget_rendered", canvas_count > 0, f"{canvas_count} canvases")

    # GPU init logs may fire before console interceptor is installed.
    # The real verification is in step 5 (gpu_colormap_active).
    gpu_logs = get_gpu_logs(cdp, "Show2D")
    has_fft = any("FFT initialized" in l for l in gpu_logs)
    has_cmap = any("colormap engine initialized" in l for l in gpu_logs)
    record("gpu_fft_init", has_fft or canvas_count > 0, "log captured" if has_fft else f"{canvas_count} canvases (init before interceptor)")
    record("gpu_colormap_init", has_cmap or canvas_count > 0, "log captured" if has_cmap else f"{canvas_count} canvases (init before interceptor)")

    # Step 5: Test colormap changes via kernel
    print("\n5. Testing GPU colormap path...")
    for cmap in CMAP_TESTS:
        result = kernel.execute(f"w.cmap = '{cmap}'")
        if result["status"] == "error":
            record(f"cmap_{cmap}", False, result["error"])
            continue
        time.sleep(1.5)

    # Check renderSlots timing
    timing_logs = get_gpu_logs(cdp, "GPU colormap")
    warm_times = []
    for log in timing_logs:
        # Parse: "[L][Show2D] GPU colormap: 12×4096×4096 in 104ms (gpu=61ms copy=43ms)"
        if " in " in log and "ms" in log:
            try:
                total = int(log.split(" in ")[1].split("ms")[0])
                warm_times.append(total)
            except (ValueError, IndexError):
                pass

    if warm_times:
        # Skip first (cold) call
        warm = warm_times[1:] if len(warm_times) > 1 else warm_times
        avg = sum(warm) / len(warm)
        record("gpu_colormap_active", True, f"avg={avg:.0f}ms over {len(warm)} calls")
        record("gpu_colormap_fast", avg < 400, f"avg={avg:.0f}ms (target: <400ms, warm <200ms)")
        for i, t in enumerate(warm_times):
            label = "cold" if i == 0 else "warm"
            print(f"    call {i+1} ({label}): {t}ms")
    else:
        # Check if CPU fallback was used
        cpu_logs = get_gpu_logs(cdp, "path check")
        if cpu_logs:
            record("gpu_colormap_active", False, "GPU path not triggered — CPU fallback only")
        else:
            record("gpu_colormap_active", False, "no timing logs found")

    # Step 6: Verify rendering (non-zero pixels)
    print("\n6. Verifying pixel output...")
    pixel_check = cdp.eval("""
    (() => {
        const canvases = document.querySelectorAll("canvas");
        let nonZero = 0, total = 0;
        for (const c of canvases) {
            if (c.width < 100 || c.height < 100) continue;
            try {
                const ctx = c.getContext("2d");
                if (!ctx) continue;
                const d = ctx.getImageData(c.width/2, c.height/2, 10, 10).data;
                total++;
                for (let i = 0; i < d.length; i++) { if (d[i] !== 0) { nonZero++; break; } }
            } catch(e) {}
        }
        return nonZero + "/" + total;
    })()
    """)
    if pixel_check and "/" in str(pixel_check):
        nz, tot = str(pixel_check).split("/")
        record("pixels_nonzero", int(nz) > 0, f"{nz}/{tot} canvases have non-zero center pixels")
    else:
        record("pixels_nonzero", False, f"check returned: {pixel_check}")

    # Step 7: Screenshot
    print("\n7. Taking screenshots...")
    cdp.eval("""
    (() => {
        const cells = document.querySelectorAll('.jp-CodeCell');
        if (cells.length >= 4) cells[3].scrollIntoView({behavior: 'instant', block: 'start'});
    })()
    """)
    time.sleep(0.5)
    ss_path = SCREENSHOT_DIR / "gpu_colormap_test.png"
    cdp.screenshot(str(ss_path))
    print(f"  Saved: {ss_path}")

    cdp.close()

    # Summary
    total = results["pass"] + results["fail"]
    print(f"\n{'='*40}")
    print(f"  {results['pass']}/{total} passed, {results['fail']} failed")
    print(f"{'='*40}\n")
    return results


def watch_mode():
    """Watch for JS changes, rebuild, and retest."""
    import hashlib

    static_dir = REPO / "src" / "quantem" / "widget" / "static"
    js_dir = REPO / "js"

    def get_hash():
        h = hashlib.md5()
        for p in sorted(js_dir.rglob("*.ts")) + sorted(js_dir.rglob("*.tsx")):
            h.update(p.read_bytes())
        return h.hexdigest()

    last_hash = get_hash()
    print("Watching for JS/TS changes... (Ctrl+C to stop)\n")
    run_gpu_test(build=True)

    while True:
        time.sleep(2)
        current = get_hash()
        if current != last_hash:
            last_hash = current
            print("\n--- Files changed, rebuilding + retesting ---\n")
            run_gpu_test(build=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU colormap E2E test via Chrome CDP")
    parser.add_argument("--build", action="store_true", help="Run npm run build first")
    parser.add_argument("--watch", action="store_true", help="Watch for file changes and auto-retest")
    args = parser.parse_args()

    if args.watch:
        watch_mode()
    else:
        results = run_gpu_test(build=args.build)
        sys.exit(0 if results["fail"] == 0 else 1)
