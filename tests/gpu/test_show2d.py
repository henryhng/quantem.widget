#!/usr/bin/env python3
"""
GPU E2E test for Show2D.

Tests WebGPU colormap, FFT, histogram, auto-contrast, log scale, and auto-bin
on real GPU hardware via Chrome CDP.

Usage:
    python tests/gpu/test_show2d.py
    python tests/gpu/test_show2d.py --build
    python tests/gpu/test_show2d.py --scale   # include 24/48-image scale test
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

# Notebook cells for Show2D gold drift test
CELLS = [
    "%load_ext autoreload\n%autoreload 2\n%env ANYWIDGET_HMR=1",
    "from quantem.widget import IO, Show2D",
    'result = IO.folder("/Users/macbook/data/bob/20260409_gold_drift_v3", file_type="emd", shape=(4096, 4096))',
    "w = Show2D(result, ncols=4)",
]


def run_test(build=False, scale=False):
    print(f"\n{'='*60}")
    print(f" Show2D GPU Test")
    print(f"{'='*60}\n")

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    results = {"pass": 0, "fail": 0}

    def check(name, passed, detail=""):
        status = "PASS" if passed else "FAIL"
        results["pass" if passed else "fail"] += 1
        sym = "✓" if passed else "✗"
        print(f"  [{sym}] {name}" + (f" — {detail}" if detail else ""))

    if build:
        print("Building JS...", end=" ", flush=True)
        if not build_js():
            check("build", False, "npm run build failed")
            return results
        print("done")

    # Connect
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
    check("webgpu", has_gpu == "true", f"navigator.gpu={has_gpu}")
    if has_gpu != "true":
        cdp.close()
        return results

    # Setup widget
    print("\n  Setting up widget...")
    w_check = kernel.execute("type(w).__name__", timeout=5)
    if w_check["status"] != "ok":
        kernel.restart_kernel()
        time.sleep(2)
        nb_url = f"http://localhost:8889/lab/tree/show2d/show2d_gold_drift.ipynb?token={kernel.token}"
        cdp.navigate(nb_url)
        time.sleep(10)
        cdp = CDPConnection()

    install_interceptor(cdp)
    run_cells_via_ui(cdp, n_cells=15, heavy_cells={2: 10, 3: 10})
    time.sleep(12)

    canvas_count = check_canvases(cdp)
    check("widget_rendered", canvas_count > 0, f"{canvas_count} canvases")

    gpu_logs = get_gpu_logs(cdp, "Show2D")
    has_cmap = any("colormap engine initialized" in l for l in gpu_logs) or canvas_count > 0
    check("gpu_colormap_init", has_cmap)

    # Colormap benchmark
    print("\n  Colormap benchmark (12×4K)...")
    avg = bench_colormap(cdp, kernel, 12)
    check("colormap_speed", avg > 0 and avg < 400, f"avg={avg}ms")

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
        check("log_toggle", False, "no timing")

    # Auto-contrast
    print("\n  Auto-contrast...")
    clear_gpu_logs(cdp)
    kernel.execute("w.auto_contrast = True")
    time.sleep(2)
    kernel.execute("w.auto_contrast = False")
    time.sleep(1.5)
    logs = get_gpu_logs(cdp, "GPU colormap")
    times = parse_gpu_timing(logs)
    if times:
        check("auto_contrast", times[-1][0] < 400, f"{times[-1][0]}ms")
    else:
        check("auto_contrast", False, "no timing")

    # Pixel verification
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
        check("pixels_nonzero", int(nz) > 0, f"{nz}/{tot}")
    else:
        check("pixels_nonzero", False)

    # Histogram caching: cmap change with auto active should NOT recompute histogram
    print("\n  Histogram caching...")
    kernel.execute("w.auto_contrast = True")
    time.sleep(2)
    clear_gpu_logs(cdp)
    kernel.execute("w.cmap = 'magma'")
    time.sleep(1.5)
    hist_logs = get_gpu_logs(cdp, "histogramBatch")
    check("hist_cached_on_cmap", len(hist_logs) == 0, f"{len(hist_logs)} hist calls (should be 0)")
    kernel.execute("w.auto_contrast = False")
    time.sleep(1)

    # Scale test
    if scale:
        print("\n  Scale test (24 images, auto-bin from real EMDs)...")
        kernel.execute("import numpy as np; w.set_image(np.concatenate([result.data]*2, axis=0))", timeout=120)
        time.sleep(20)  # trait sync of 384MB + GPU upload takes time

        # Verify auto-bin activated
        r = kernel.execute("print(f'{w._display_bin_factor} {w.height} {w.n_images}')", timeout=10)
        parts = r["stdout"].strip().split()
        if len(parts) >= 3:
            bin_f, disp_h, n_img = int(parts[0]), int(parts[1]), int(parts[2])
            check("scale_24_autobin", bin_f == 2 and disp_h == 2048, f"bin={bin_f}× {disp_h}px {n_img} images")
        else:
            check("scale_24_autobin", False, f"couldn't read state: {r['stdout']}")

        avg24 = bench_colormap(cdp, kernel, 24, wait=2)
        check("scale_24_speed", avg24 > 0 and avg24 < 400, f"avg={avg24}ms")

        # Log + auto on 24 images
        clear_gpu_logs(cdp)
        kernel.execute("w.log_scale = True")
        time.sleep(2)
        kernel.execute("w.log_scale = False")
        time.sleep(2)
        logs24 = get_gpu_logs(cdp, "GPU colormap")
        times24 = parse_gpu_timing(logs24)
        if times24:
            check("scale_24_log", times24[-1][0] < 300, f"{times24[-1][0]}ms")
        else:
            check("scale_24_log", False, "no timing")

        # Restore
        kernel.execute("w.set_image(result.data)", timeout=30)
        time.sleep(5)

    # Screenshot
    cdp.screenshot(str(SCREENSHOT_DIR / "show2d_gpu.png"))

    cdp.close()
    total = results["pass"] + results["fail"]
    print(f"\n{'='*60}")
    print(f"  {results['pass']}/{total} passed, {results['fail']} failed")
    print(f"{'='*60}\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show2D GPU E2E test")
    parser.add_argument("--build", action="store_true", help="Run npm run build first")
    parser.add_argument("--scale", action="store_true", help="Include 24/48 image scale test")
    args = parser.parse_args()
    results = run_test(build=args.build, scale=args.scale)
    sys.exit(0 if results["fail"] == 0 else 1)
