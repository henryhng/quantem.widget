"""
E2E Screenshot Capture for Live Widget

Captures screenshots of Live widget with real and synthetic data:
- 2D mode with thumbnail panel
- 3D mode with frame slider
- Real DM3 data (4K)
- Real EMD data
- Keyboard navigation

Usage:
    python tests/capture_live.py

Screenshots are saved to tests/screenshots/live/
"""

import json
import subprocess
import sys
import time
import socket
from pathlib import Path

from playwright.sync_api import sync_playwright

JUPYTER_PORT = 8899
SCREENSHOT_DIR = Path(__file__).parent / "screenshots" / "live"
NOTEBOOK_PATH = Path(__file__).parent / "_test_live_capture.ipynb"


def create_test_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import numpy as np\n",
                    "from quantem.widget import Live, IO\n",
                ]
            },
            # Cell 1: Synthetic 2D mode — use constructor (batch)
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "np.random.seed(42)\n",
                    "images = []\n",
                    "for i in range(6):\n",
                    "    y, x = np.mgrid[0:256, 0:256]\n",
                    "    r = np.sqrt((x - 128)**2 + (y - 128)**2)\n",
                    "    img = (np.sin(r / (3 + i)) * np.exp(-r / (30 + i * 5)) + np.random.rand(256, 256) * 0.2).astype(np.float32)\n",
                    "    images.append(img)\n",
                    "w_2d = Live(images, title='2D Synthetic', cmap='inferno')\n",
                    "w_2d",
                ]
            },
            # Cell 2: 3D mode — use constructor (batch)
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "stack = np.stack([\n",
                    "    (np.sin(np.sqrt((np.mgrid[0:128, 0:128][1] - 64)**2 + (np.mgrid[0:128, 0:128][0] - 64)**2) / (4 + i * 0.5)) * np.exp(-np.sqrt((np.mgrid[0:128, 0:128][1] - 64)**2 + (np.mgrid[0:128, 0:128][0] - 64)**2) / 40)).astype(np.float32)\n",
                    "    for i in range(10)\n",
                    "])\n",
                    "w_3d = Live(stack, title='3D Stack Mode', mode='3d', cmap='viridis')\n",
                    "w_3d",
                ]
            },
            # Cell 3: Real DM3 data (4K)
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "w_dm3 = Live.from_folder(\n",
                    "    '/Users/macbook/data/korean/20251209_KoreanSample/C1_TEM_Bob/Bob_Lee_Sangjoon/',\n",
                    "    pattern='*.dm3', title='TEM 4K (DM3)', cmap='gray'\n",
                    ")\n",
                    "w_dm3",
                ]
            },
            # Cell 4: Real EMD data
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "w_emd = Live.from_folder(\n",
                    "    '/Users/macbook/data/korean/20251209_KoreanSample/D1_STEM_Karen/',\n",
                    "    pattern='*.emd', title='STEM HAADF (EMD)', cmap='inferno', log_scale=True\n",
                    ")\n",
                    "w_emd",
                ]
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(notebook, f, indent=2)
    print(f"Created test notebook: {NOTEBOOK_PATH}")


def cleanup_test_notebook():
    if NOTEBOOK_PATH.exists():
        NOTEBOOK_PATH.unlink()
        print(f"Cleaned up: {NOTEBOOK_PATH}")


def start_jupyter():
    print("Starting JupyterLab...")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "jupyter", "lab",
            f"--port={JUPYTER_PORT}", "--no-browser",
            "--NotebookApp.token=''", "--NotebookApp.password=''",
            "--ServerApp.disable_check_xsrf=True",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=Path(__file__).parent.parent,
    )
    print("Waiting for JupyterLab to start...")
    for _ in range(30):
        time.sleep(1)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', JUPYTER_PORT))
            sock.close()
            if result == 0:
                print("JupyterLab is ready!")
                time.sleep(2)
                return proc
        except Exception:
            pass
    raise RuntimeError("JupyterLab failed to start within 30 seconds")


def stop_jupyter(proc):
    print("Stopping JupyterLab...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()


def save(element, name, out_dir):
    element.screenshot(path=str(out_dir / f"{name}.png"))
    print(f"  Saved: {name}.png")


def run_captures(page):
    out_dir = SCREENSHOT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dismiss dialogs
    try:
        page.keyboard.press("Escape")
        time.sleep(0.5)
    except Exception:
        pass

    # Run All Cells via command palette
    page.keyboard.press("Meta+Shift+C")
    time.sleep(0.3)
    page.keyboard.type("Run All Cells")
    time.sleep(0.3)
    page.keyboard.press("Enter")
    print("Executing all cells...")
    time.sleep(25)  # Wait for all widgets to render

    # Capture 2D mode
    widgets = page.locator(".live-widget")
    count = widgets.count()
    print(f"Found {count} Live widgets")

    names = ["live_2d_synthetic", "live_3d_stack", "live_dm3_4k", "live_emd_stem"]
    for i in range(min(count, 4)):
        w = widgets.nth(i)
        try:
            w.scroll_into_view_if_needed(timeout=5000)
        except Exception:
            pass
        time.sleep(1)
        save(w, names[i], out_dir)

    # Test keyboard navigation on first widget
    if count >= 1:
        w = widgets.nth(0)
        try:
            w.scroll_into_view_if_needed(timeout=5000)
        except Exception:
            pass
        w.click()
        time.sleep(0.5)
        page.keyboard.press("ArrowLeft")
        time.sleep(0.3)
        page.keyboard.press("ArrowLeft")
        time.sleep(0.3)
        save(w, "live_2d_after_arrow_left", out_dir)


def main():
    create_test_notebook()
    jupyter_proc = None
    try:
        jupyter_proc = start_jupyter()
        nb_name = NOTEBOOK_PATH.name
        url = f"http://localhost:{JUPYTER_PORT}/lab/tree/tests/{nb_name}"

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(viewport={"width": 1400, "height": 900})
            page = ctx.new_page()
            page.goto(url, timeout=60000)
            time.sleep(8)

            # Close any dialogs
            for _ in range(3):
                accept = page.locator("button:has-text('Ok'), button:has-text('Accept'), button:has-text('Don\\'t save')")
                if accept.count() > 0:
                    accept.first.click()
                    time.sleep(0.5)

            # Select kernel if dialog appears
            kernel_btn = page.locator("button:has-text('Select'), button:has-text('Python 3')")
            if kernel_btn.count() > 0:
                kernel_btn.first.click()
                time.sleep(2)

            run_captures(page)
            browser.close()

        print(f"\nScreenshots saved to: {SCREENSHOT_DIR}")
    finally:
        if jupyter_proc:
            stop_jupyter(jupyter_proc)
        cleanup_test_notebook()


if __name__ == "__main__":
    main()
