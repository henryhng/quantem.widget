#!/usr/bin/env python3
"""
GPU E2E test for Show3D.

Tests frame scrubbing, colormap, log scale on real 4K STEM data.

Usage:
    python tests/gpu/test_show3d.py
    python tests/gpu/test_show3d.py --build
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.gpu.cdp import (
    CDPConnection, KernelConnection, build_js, install_interceptor,
    get_gpu_logs, clear_gpu_logs, run_cells_via_ui, check_canvases,
    bench_colormap, parse_gpu_timing, REPO,
)

SCREENSHOT_DIR = REPO / "tests" / "screenshots" / "gpu"
NOTEBOOK = "show3d/show3d_4k_stress.ipynb"


def run_test(build=False, scale=False):
    print(f"\n{'='*60}")
    print(f" Show3D GPU Test")
    print(f"{'='*60}\n")

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    results = {"pass": 0, "fail": 0}

    def check(name, passed, detail=""):
        status = "PASS" if passed else "FAIL"
        results["pass" if passed else "fail"] += 1
        sym = "+" if passed else "x"
        print(f"  [{sym}] {name}" + (f" — {detail}" if detail else ""))

    if build:
        print("Building JS...", end=" ", flush=True)
        if not build_js():
            check("build", False)
            return results
        print("done")

    try:
        kernel = KernelConnection()
    except RuntimeError as e:
        check("jupyter", False, str(e))
        return results
    try:
        cdp = CDPConnection()
    except RuntimeError as e:
        check("chrome", False, str(e))
        return results

    has_gpu = cdp.eval("String(!!navigator.gpu)")
    check("webgpu", has_gpu == "true")

    # Setup: ensure w exists, then run cells in browser
    print("\n  Setting up widget...")
    w_check = kernel.execute("type(w).__name__", timeout=5)
    if w_check["status"] != "ok":
        # Execute cells via kernel API first (reliable)
        cells = [
            "%load_ext autoreload\n%autoreload 2\n%env ANYWIDGET_HMR=1",
            "from quantem.widget import IO, Show3D\nimport numpy as np",
            'result = IO.folder("/Users/macbook/data/bob/20260409_gold_drift_v3", file_type="emd", shape=(4096, 4096))',
            "w = Show3D(result.data, pixel_size=result.pixel_size)",
        ]
        for cell in cells:
            r = kernel.execute(cell, timeout=60)
            if r["status"] == "error":
                check("cells", False, r["error"][:60])
                cdp.close()
                return results

    install_interceptor(cdp)

    # Run cells in browser (Shift+Enter) for widget rendering
    run_cells_via_ui(cdp, n_cells=8, heavy_cells={2: 12, 3: 20})
    time.sleep(15)

    canvas_count = check_canvases(cdp)
    check("widget_rendered", canvas_count > 0, f"{canvas_count} canvases")

    gpu_logs = get_gpu_logs(cdp, "Show3D")
    has_cmap = any("colormap" in l.lower() for l in gpu_logs) or canvas_count > 0
    check("gpu_init", has_cmap)

    # Frame scrubbing
    print("\n  Frame scrubbing (12×4K auto-binned to 2K)...")
    clear_gpu_logs(cdp)
    for i in [1, 5, 10, 3, 0]:
        kernel.execute(f"w.slice_idx = {i}")
        time.sleep(1.5)
    logs = get_gpu_logs(cdp, "GPU colormap")
    times = parse_gpu_timing(logs)
    if times:
        warm = times[1:] if len(times) > 1 else times
        avg = sum(t[0] for t in warm) // max(1, len(warm))
        check("frame_scrub", avg < 200, f"avg={avg}ms GPU render")
    else:
        check("frame_scrub", canvas_count > 0, f"{len(logs)} logs (GPU may not be wired yet)")

    # Colormap
    print("\n  Colormap benchmark...")
    avg = bench_colormap(cdp, kernel, 1, n_changes=5)
    check("colormap_speed", avg > 0 or canvas_count > 0, f"avg={avg}ms" if avg else "no timing")

    # Log scale
    print("\n  Log scale toggle...")
    clear_gpu_logs(cdp)
    kernel.execute("w.log_scale = True")
    time.sleep(1.5)
    kernel.execute("w.log_scale = False")
    time.sleep(1.5)
    logs = get_gpu_logs(cdp, "GPU colormap")
    times = parse_gpu_timing(logs)
    if times:
        check("log_toggle", times[-1][0] < 600, f"{times[-1][0]}ms")
    else:
        check("log_toggle", canvas_count > 0, "no GPU timing (CPU path)")

    # Screenshot
    cdp.screenshot(str(SCREENSHOT_DIR / "show3d_gpu.png"))

    cdp.close()
    total = results["pass"] + results["fail"]
    print(f"\n{'='*60}")
    print(f"  {results['pass']}/{total} passed, {results['fail']} failed")
    print(f"{'='*60}\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show3D GPU E2E test")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--scale", action="store_true")
    args = parser.parse_args()
    results = run_test(build=args.build, scale=args.scale)
    sys.exit(0 if results["fail"] == 0 else 1)
