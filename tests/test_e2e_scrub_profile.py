"""
Concrete profiler: how long does a single Show3D scrub take, end-to-end?

We don't speculate about "SSH latency is 100-150 ms" — we measure the actual
per-scrub wall clock from programmatic Shift+slider press to first-pixel-change
on the canvas.  Multiple stack sizes reveal the crossover between
per-scrub-comm-bound and browser-paint-bound.

Also: what's the browser's actual RAM ceiling?  We try progressively larger
stacks and record which sizes crash the page.  Laptop-safe limits matter.

This runs locally (kernel and browser on the same Linux machine).  To compare
with VS Code Remote-SSH:

  1. Run this test on the Linux kernel box (what we're doing now) → baseline.
  2. Re-run on Mac with a local kernel → upper bound for SSH to "beat".
  3. Re-run on Mac with the kernel on Linux over SSH → the number you care
     about.  Delta vs #2 = SSH transport cost per scrub.

Run with:

    python -m pytest tests/test_e2e_scrub_profile.py -v -s
"""

from __future__ import annotations

import statistics
import time

import pytest

from conftest import TESTS_DIR, JUPYTER_PORT, _write_notebook

NOTEBOOK_PATH = TESTS_DIR / "_scrub_profile.ipynb"


# Realistic microscope scenarios, small → large.  We stop before 12×4K because
# headless Chromium OOMs there (already measured).  The user's real MacBook
# browser may do better, but this is the honest headless floor.
SCENARIOS = [
    # (n_slices, h, w, label) — progressively larger, stop before browser OOM.
    (30, 512,  512,  "SSB 30×512² (30 MB)"),
    (20, 1024, 1024, "stack 20×1K² (80 MB)"),
    (10, 2048, 2048, "stack 10×2K² (160 MB)"),
]
SCRUBS_PER_SCENARIO = 15


def _build_notebook() -> None:
    cells = [
        {"source": [
            "import numpy as np\n",
            "import gc\n",
            "from quantem.widget import Show3D\n",
        ]},
    ]
    for i, (n, h, w, label) in enumerate(SCENARIOS):
        # Free previous scenario's memory to avoid compounding pressure.
        cleanup = "" if i == 0 else (
            f"try: del w_{i-1}\n"
            f"except NameError: pass\n"
            f"try: del data_{i-1}\n"
            f"except NameError: pass\n"
            f"gc.collect()\n"
        )
        cells.append({"source": [
            cleanup,
            f"# {label}\n",
            f"data_{i} = np.random.default_rng(0).standard_normal(({n},{h},{w})).astype(np.float32)\n",
            f"w_{i} = Show3D(data_{i}, title='{label}')\n",
            f"w_{i}\n",
        ]})
    _write_notebook(NOTEBOOK_PATH, cells)


def _canvas_pixel_hash(page, cell_idx: int) -> int:
    """Hash a small center patch of the first visible canvas.  Used to detect
    a repaint: when the frame changes, so does the hash."""
    js = """
    (cellIdx) => {
        const cells = document.querySelectorAll('.jp-Cell');
        const cell = cells[cellIdx];
        if (!cell) return 0;
        const canvases = cell.querySelectorAll('canvas');
        for (const c of canvases) {
            if (!c.width || !c.height) continue;
            try {
                const ctx = c.getContext('2d');
                if (!ctx) continue;
                const data = ctx.getImageData(
                    Math.floor(c.width/2), Math.floor(c.height/2), 8, 8
                ).data;
                let h = 0;
                for (let i = 0; i < data.length; i++) h = (h * 31 + data[i]) | 0;
                return h;
            } catch (e) { /* CORS */ }
        }
        return 0;
    }
    """
    return page.evaluate(js, cell_idx)


def _canvas_painted(page, cell_idx: int) -> bool:
    return _canvas_pixel_hash(page, cell_idx) != 0


def _set_slice_idx(page, cell_idx: int, new_idx: int) -> None:
    """Set slice_idx on the widget in this cell via its ipywidget model.

    Traverses the DOM to find the widget's model-id, then writes slice_idx
    directly through the Jupyter widget manager.  This simulates programmatic
    scrubbing with one comm round-trip (JS→Python→JS), same path as dragging
    the slider — but deterministic.
    """
    js = """
    (args) => {
        const [cellIdx, newIdx] = args;
        const cells = document.querySelectorAll('.jp-Cell');
        const cell = cells[cellIdx];
        if (!cell) return 'no-cell';
        // Find the slice_idx slider in this cell's output; it's an <input type=range>.
        const range = cell.querySelector('input[type=range]');
        if (!range) return 'no-slider';
        const setter = Object.getOwnPropertyDescriptor(
            window.HTMLInputElement.prototype, 'value').set;
        setter.call(range, String(newIdx));
        range.dispatchEvent(new Event('input', { bubbles: true }));
        range.dispatchEvent(new Event('change', { bubbles: true }));
        return 'ok';
    }
    """
    page.evaluate(js, [cell_idx, new_idx])


def test_show3d_scrub_latency(browser_context):
    """Measure per-scrub wall clock across realistic stack sizes."""
    _build_notebook()
    page = browser_context.new_page()

    rel = NOTEBOOK_PATH.relative_to(TESTS_DIR.parent)
    # `?reset` nukes JupyterLab's saved workspace so the notebook opens in the
    # foreground tab every time — otherwise a previous session's open notebooks
    # persist and our cells render in a hidden background tab (width=0 height=0).
    url = f"http://localhost:{JUPYTER_PORT}/lab/tree/{rel}?reset"
    page.goto(url, timeout=60000)
    # 12 s — see test_e2e_timing_truthful.py for why.
    time.sleep(12)
    try:
        page.keyboard.press("Escape")
    except Exception:
        pass

    # Cell 0 imports — same pattern as tests/test_e2e_timing_truthful.py which works.
    page.locator(".jp-Cell").nth(0).click()
    time.sleep(0.3)
    page.keyboard.press("Shift+Enter")
    time.sleep(2.0)

    results = []

    for i, (n, h, w, label) in enumerate(SCENARIOS):
        cell_idx = 1 + i
        cell = page.locator(".jp-Cell").nth(cell_idx)
        try:
            cell.click(timeout=10000)
        except Exception as exc:
            results.append({"label": label, "error": f"cell click failed: {exc}"})
            continue
        time.sleep(0.3)

        # Construct + wait for first paint
        payload_mb = n * h * w * 4 / (1 << 20)
        t_construct = time.perf_counter()
        page.keyboard.press("Shift+Enter")
        paint_deadline = t_construct + max(30.0, 10.0 + payload_mb * 0.2)
        painted = False
        while time.perf_counter() < paint_deadline:
            if _canvas_painted(page, cell_idx):
                painted = True
                break
            time.sleep(0.1)
        construct_ms = int(1000 * (time.perf_counter() - t_construct))

        if not painted:
            results.append({
                "label": label, "n": n, "construct_ms": construct_ms,
                "error": "canvas never painted (likely browser OOM)",
            })
            # Page may be crashed; try to reload before the next scenario.
            try:
                page.reload(timeout=30000)
                time.sleep(5)
            except Exception:
                pass
            continue

        # Programmatic scrubs — jump to random-ish frames, measure per-scrub wall clock.
        # We wait for the canvas hash to change to confirm the frame actually repainted;
        # that's the real "user sees new frame" signal.
        per_scrub_ms = []
        rng_order = [(i_ * 37 + 7) % n for i_ in range(SCRUBS_PER_SCENARIO)]
        prev_hash = _canvas_pixel_hash(page, cell_idx)
        for target in rng_order:
            t0 = time.perf_counter()
            _set_slice_idx(page, cell_idx, target)
            # Poll for hash change (frame actually repainted with new pixels)
            changed = False
            poll_deadline = t0 + 3.0
            while time.perf_counter() < poll_deadline:
                cur = _canvas_pixel_hash(page, cell_idx)
                if cur != prev_hash and cur != 0:
                    changed = True
                    prev_hash = cur
                    break
                time.sleep(0.005)  # 5 ms poll
            elapsed_ms = 1000 * (time.perf_counter() - t0)
            if changed:
                per_scrub_ms.append(elapsed_ms)

        if not per_scrub_ms:
            results.append({
                "label": label, "n": n, "construct_ms": construct_ms,
                "error": f"no frame change detected after {SCRUBS_PER_SCENARIO} scrubs",
            })
            continue

        results.append({
            "label": label,
            "n": n, "h": h, "w": w,
            "payload_mb": round(payload_mb),
            "construct_ms": construct_ms,
            "scrub_count": len(per_scrub_ms),
            "scrub_median_ms": round(statistics.median(per_scrub_ms)),
            "scrub_p95_ms":    round(sorted(per_scrub_ms)[int(0.95 * len(per_scrub_ms))]),
            "scrub_min_ms":    round(min(per_scrub_ms)),
            "scrub_max_ms":    round(max(per_scrub_ms)),
        })

    page.close()

    # ---- Report ----
    print()
    print("=" * 100)
    print(f"{'scenario':<26} {'MB':>5} {'construct':>11} "
          f"{'scrub median':>14} {'p95':>6} {'min':>6} {'max':>6} {'N':>4}")
    print("-" * 100)
    for r in results:
        if "error" in r:
            print(f"{r['label']:<26} {r.get('payload_mb','-'):>5} "
                  f"{'-':>11} {r['error']}")
            continue
        print(f"{r['label']:<26} {r['payload_mb']:>3} MB "
              f"{r['construct_ms']:>9} ms "
              f"{r['scrub_median_ms']:>12} ms "
              f"{r['scrub_p95_ms']:>4} ms "
              f"{r['scrub_min_ms']:>4} ms "
              f"{r['scrub_max_ms']:>4} ms "
              f"{r['scrub_count']:>4}")
    print("=" * 100)
    print()
    print("Note: all numbers above are for kernel + browser on the SAME machine")
    print("(localhost Comm transport).  To compare against VS Code Remote-SSH,")
    print("re-run this same test with the kernel remote and UI on the Mac —")
    print("the delta in scrub_median_ms is the actual SSH round-trip cost per scrub.")

    # ---- Regression gate ----
    # Budgets are LOOSE — these are for catching someone accidentally doubling the
    # per-scrub comm traffic (e.g. introducing an extra observer round-trip), not
    # for chasing every millisecond.  Localhost measurements today:
    #   SSB 30×512²:  11 ms median (budget 100 ms → 9× headroom)
    #   stack 20×1K²: 17 ms median (budget 120 ms)
    #   stack 10×2K²: 28 ms median (budget 150 ms)
    #   stack 4×4K²:  32 ms median (budget 200 ms)
    # Construct time is informational; don't gate on it (cold/warm JS bundle variance).
    BUDGETS_MS = {
        (30, 512,  512):  100,
        (20, 1024, 1024): 120,
        (10, 2048, 2048): 200,
    }
    failed = []
    for r in results:
        if "error" in r:
            failed.append(f"{r['label']}: {r['error']}")
            continue
        key = (r["n"], r["h"], r["w"])
        budget = BUDGETS_MS.get(key)
        if budget is None:
            continue
        if r["scrub_median_ms"] > budget:
            failed.append(
                f"{r['label']}: scrub median {r['scrub_median_ms']} ms > "
                f"{budget} ms budget — per-scrub comm traffic likely doubled"
            )
    if failed:
        raise AssertionError("Scrub regression:\n  " + "\n  ".join(failed))
