"""
Shared Chrome CDP + Jupyter kernel infrastructure for GPU tests.

Usage:
    from tests.gpu.cdp import CDPConnection, KernelConnection, install_interceptor, get_gpu_logs
"""

import base64
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests
import websocket

REPO = Path(__file__).parent.parent.parent
JLAB_PORT = 8889
CDP_PORT = 9222


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

    def navigate(self, url):
        self._msg_id += 1
        try:
            self.ws.send(json.dumps({"id": self._msg_id, "method": "Page.navigate", "params": {"url": url}}))
            self.ws.recv()
        except Exception:
            pass
        try:
            self.ws.close()
        except Exception:
            pass

    def send_shift_enter(self):
        self._msg_id += 1
        self.ws.send(json.dumps({
            "id": self._msg_id, "method": "Input.dispatchKeyEvent",
            "params": {"type": "keyDown", "key": "Enter", "code": "Enter", "modifiers": 8, "windowsVirtualKeyCode": 13},
        }))
        for _ in range(5):
            raw = self.ws.recv()
            if isinstance(raw, bytes):
                continue
            break
        self._msg_id += 1
        self.ws.send(json.dumps({
            "id": self._msg_id, "method": "Input.dispatchKeyEvent",
            "params": {"type": "keyUp", "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13},
        }))
        for _ in range(5):
            raw = self.ws.recv()
            if isinstance(raw, bytes):
                continue
            break

    def screenshot(self, path):
        self._msg_id += 1
        mid = self._msg_id
        try:
            self.ws.send(json.dumps({"id": mid, "method": "Page.captureScreenshot", "params": {"format": "png"}}))
            for _ in range(20):
                raw = self.ws.recv()
                if isinstance(raw, bytes):
                    continue
                resp = json.loads(raw)
                if resp.get("id") == mid:
                    data = resp.get("result", {}).get("data")
                    if data:
                        Path(path).write_bytes(base64.b64decode(data))
                        return True
                    return False
        except Exception:
            pass
        return False

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


def install_interceptor(cdp):
    """Install console.log interceptor for GPU log capture."""
    cdp.eval("""
    window._gpuLogs=[];
    const _ol=console.log,_ow=console.warn,_oe=console.error;
    console.log=function(){window._gpuLogs.push("[L]"+[...arguments].join(" "));_ol.apply(console,arguments)};
    console.warn=function(){window._gpuLogs.push("[W]"+[...arguments].join(" "));_ow.apply(console,arguments)};
    console.error=function(){window._gpuLogs.push("[E]"+[...arguments].join(" "));_oe.apply(console,arguments)};
    "ok"
    """)


def get_gpu_logs(cdp, filter_str=None):
    """Read GPU-related console logs."""
    if filter_str:
        expr = f'JSON.stringify(window._gpuLogs.filter(l => l.includes("{filter_str}")))'
    else:
        expr = "JSON.stringify(window._gpuLogs.slice(-50))"
    r = cdp.eval(expr)
    try:
        return json.loads(r) if r else []
    except (json.JSONDecodeError, TypeError):
        return []


def clear_gpu_logs(cdp):
    cdp.eval("window._gpuLogs=[]")


def run_cells_via_ui(cdp, n_cells=20, heavy_cells=None):
    """Run cells via Shift+Enter from the notebook UI."""
    heavy_cells = heavy_cells or {2: 8, 3: 8}
    cdp.eval('(() => { const c = document.querySelector(".jp-CodeCell"); if(c) c.click(); return "ok"; })()')
    time.sleep(0.5)
    for i in range(n_cells):
        cdp.send_shift_enter()
        wait = heavy_cells.get(i, 1)
        time.sleep(wait)


def check_canvases(cdp):
    r = cdp.eval('document.querySelectorAll("canvas").length')
    return int(r) if r else 0


def build_js():
    """Run npm run build."""
    r = subprocess.run(["npm", "run", "build"], capture_output=True, text=True, cwd=REPO)
    return r.returncode == 0


def parse_gpu_timing(logs):
    """Parse GPU colormap timing from console logs. Returns list of (total, gpu, copy)."""
    times = []
    for l in logs:
        if "gpu=" in l and "copy=" in l and " in " in l:
            try:
                total = int(l.split(" in ")[1].split("ms")[0])
                gpu = int(l.split("gpu=")[1].split("ms")[0])
                copy = int(l.split("copy=")[1].split("ms")[0].rstrip(")"))
                times.append((total, gpu, copy))
            except (ValueError, IndexError):
                pass
    return times


def bench_colormap(cdp, kernel, n_images, n_changes=5, wait=1.5):
    """Benchmark colormap changes. Returns avg warm total ms, or 0 on failure."""
    cmaps = ["hot", "viridis", "plasma", "magma", "inferno"][:n_changes]
    clear_gpu_logs(cdp)
    for cmap in cmaps:
        kernel.execute(f"w.cmap = '{cmap}'")
        time.sleep(wait)
    time.sleep(1)
    logs = get_gpu_logs(cdp, "GPU colormap")
    times = parse_gpu_timing(logs)
    if len(times) >= 2:
        warm = times[1:]
        return sum(t[0] for t in warm) // len(warm)
    elif times:
        return times[0][0]
    return 0
