"""
Quick visual capture of MetricExplorer using Playwright.

Usage:
    python tests/capture_metric_explorer.py

Starts JupyterLab, opens a notebook with synthetic data, screenshots the widget.
Output: tests/screenshots/smoke/show_metric_explorer.png
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).parent
PROJECT_DIR = TESTS_DIR.parent
SCREENSHOT_DIR = TESTS_DIR / "screenshots" / "smoke"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

NOTEBOOK_PATH = TESTS_DIR / "_capture_metric_explorer.ipynb"

CELLS = [
    {"source": [
        "import numpy as np\n",
        "from quantem.widget import MetricExplorer\n",
    ]},
    {"source": [
        "# Generate synthetic SSB sweep: 3 groups x 11 defocus = 33 points\n",
        "H, W = 64, 64\n",
        "np.random.seed(42)\n",
        "defocus_values = np.linspace(-50, 50, 11)\n",
        "bf_radii = [20, 25, 30]\n",
        "points = []\n",
        "for r in bf_radii:\n",
        "    for c10 in defocus_values:\n",
        "        # Simulate phase with defocus-dependent blur\n",
        "        y, x = np.meshgrid(np.linspace(0, 8*np.pi, H), np.linspace(0, 8*np.pi, W), indexing='ij')\n",
        "        phase = np.sin(x) * np.sin(y) + 0.5 * np.sin(2*x + y)\n",
        "        sigma = 1.0 + 0.05 * abs(c10)\n",
        "        from scipy.ndimage import gaussian_filter\n",
        "        phase = gaussian_filter(phase, sigma=sigma).astype(np.float32)\n",
        "        phase += np.random.randn(H, W).astype(np.float32) * (0.05 + 0.002 * abs(c10))\n",
        "        var_loss = float(np.var(phase))\n",
        "        tv = float(np.sum(np.abs(np.diff(phase, axis=0))) + np.sum(np.abs(np.diff(phase, axis=1))))\n",
        "        fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(phase)))\n",
        "        c = np.array(fft_mag.shape) // 2\n",
        "        lo = np.zeros_like(fft_mag, dtype=bool)\n",
        "        lo[c[0]-3:c[0]+3, c[1]-3:c[1]+3] = True\n",
        "        fft_snr = float(fft_mag[~lo].mean() / (fft_mag[lo].mean() + 1e-10))\n",
        "        points.append({'image': phase, 'label': f'r={r} C10={c10:+.0f}nm',\n",
        "            'metrics': {'variance_loss': var_loss, 'tv': tv, 'fft_snr': fft_snr},\n",
        "            'params': {'C10_nm': c10, 'bf_radius': r}})\n",
        "bf = np.sin(x) * np.sin(y).astype(np.float32)\n",
        "dpc = np.gradient(bf, axis=0).astype(np.float32)\n",
        "print(f'{len(points)} points generated')\n",
    ]},
    {"source": [
        "explorer = MetricExplorer(\n",
        "    points,\n",
        "    x_key='C10_nm',\n",
        "    x_label='Defocus C10 (nm)',\n",
        "    group_key='bf_radius',\n",
        "    reference_images=[bf, dpc],\n",
        "    reference_labels=['BF', 'DPC'],\n",
        "    metric_labels={'tv': 'Total Variation', 'fft_snr': 'FFT SNR'},\n",
        "    metric_directions={'variance_loss': 'max', 'tv': 'min', 'fft_snr': 'max'},\n",
        ")\n",
        "explorer\n",
    ]},
]


def write_notebook():
    nb = {
        "cells": [
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c["source"]}
            for c in CELLS
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(nb, f, indent=2)


def port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex(("localhost", port)) == 0


def main():
    from playwright.sync_api import sync_playwright

    write_notebook()

    port = 8899
    while port_open(port):
        port += 1

    log = open("/tmp/capture_jupyter.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "jupyter", "lab",
         f"--port={port}", "--ServerApp.port_retries=0",
         "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''",
         "--ServerApp.disable_check_xsrf=True"],
        stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT_DIR,
    )

    print(f"Waiting for JupyterLab on port {port}...")
    deadline = time.time() + 90
    while time.time() < deadline:
        if proc.poll() is not None:
            print("JupyterLab exited early!")
            sys.exit(1)
        if port_open(port):
            break
        time.sleep(1)
    else:
        print("Timeout waiting for JupyterLab")
        proc.kill()
        sys.exit(1)

    time.sleep(3)
    print("JupyterLab ready. Launching browser...")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1600, "height": 1000})
            page = context.new_page()

            rel = NOTEBOOK_PATH.relative_to(PROJECT_DIR)
            url = f"http://localhost:{port}/lab/tree/{rel}"
            print(f"Opening {url}")
            page.goto(url, timeout=60000)
            time.sleep(8)

            # Dismiss dialogs
            try:
                page.keyboard.press("Escape")
                time.sleep(0.5)
            except Exception:
                pass

            # Click on the notebook tab to ensure it's focused
            time.sleep(1)

            # Run All Cells via menu: Run → Run All Cells
            page.click('text=Run')
            time.sleep(0.5)
            page.click('text=Run All Cells')
            print("Running cells... waiting 30s for widget render")
            time.sleep(30)

            # Scroll down to see the widget output
            page.keyboard.press("End")
            time.sleep(2)

            # Screenshot full page
            out = SCREENSHOT_DIR / "show_metric_explorer.png"
            page.screenshot(path=str(out), full_page=True)
            print(f"Screenshot saved: {out}")

            # Screenshot just the widget area (light theme)
            widgets = page.query_selector_all(".show-metric-explorer-root")
            if widgets:
                out2 = SCREENSHOT_DIR / "show_metric_explorer_widget.png"
                widgets[0].screenshot(path=str(out2))
                print(f"Widget screenshot (light): {out2}")
            else:
                print("Could not find widget element for focused screenshot")

            # Toggle FFT on by clicking the FFT switch in the header
            fft_switch = widgets[0].query_selector("input[type='checkbox']")
            if fft_switch:
                fft_switch.click()
                time.sleep(3)
                # Re-query widget since it may have grown
                w2 = page.query_selector_all(".show-metric-explorer-root")
                if w2:
                    out_fft = SCREENSHOT_DIR / "show_metric_explorer_fft.png"
                    w2[0].screenshot(path=str(out_fft))
                    print(f"Widget screenshot (FFT): {out_fft}")
                # Toggle FFT off
                fft_switch2 = w2[0].query_selector("input[type='checkbox']") if w2 else None
                if fft_switch2:
                    fft_switch2.click()
                    time.sleep(1)

            # Switch to dark theme
            page.evaluate("""
                document.body.setAttribute('data-jp-theme-light', 'false');
                document.body.setAttribute('data-jp-theme-name', 'JupyterLab Dark');
                document.body.style.backgroundColor = '#111';
                document.body.style.color = '#d4d4d4';
            """)
            time.sleep(3)

            # Screenshot dark theme
            widgets = page.query_selector_all(".show-metric-explorer-root")
            if widgets:
                out3 = SCREENSHOT_DIR / "show_metric_explorer_dark.png"
                widgets[0].screenshot(path=str(out3))
                print(f"Widget screenshot (dark): {out3}")

            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        # Clean up temp notebook
        NOTEBOOK_PATH.unlink(missing_ok=True)

    print("Done!")


if __name__ == "__main__":
    main()
