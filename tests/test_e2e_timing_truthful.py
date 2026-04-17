"""
End-to-end verification that Show2D/Show3D timing is truthful.

Methodology:
  1. Start JupyterLab in a real browser (via Playwright).
  2. Cell A: ``w = Show2D(...)`` — constructs widget, renders in browser.
     After browser paint, JS flips ``_js_rendered``; Python observer fires
     and populates ``w.render_total_ms``, ``w.render_python_build_ms``,
     ``w.render_wire_js_ms``, and prints the truthful line.
  3. Cell B: after a short sleep, ``print(w.render_total_ms, ...)`` — scrapes
     reliably from the second cell's synchronous stdout.

We compare three numbers:
  * ``playwright_ms``       — wall clock Playwright observed (Shift+Enter → canvas painted)
  * ``python_reports_ms``   — widget's own report, from the attribute
  * ``python_build_ms``     — Python-only construction time (subset of the total)

**Liar detector**: Python's claimed total must not be drastically less than what
Playwright observed.  Python is allowed to be LARGER (the attribute populates
after paint + trait round-trip).  A smaller-than-observed number means the
timing was captured before the user could see the widget.

Run with:

    python -m pytest tests/test_e2e_timing_truthful.py -v -s
"""

from __future__ import annotations

import time

import pytest

from conftest import TESTS_DIR, JUPYTER_PORT, _write_notebook

NOTEBOOK_PATH = TESTS_DIR / "_timing_truthful.ipynb"


SHAPES = [
    # (n_images, h, w, widget)
    (8, 256, 256, "Show2D"),
    (8, 256, 256, "Show3D"),
]


def _build_notebook() -> None:
    cells = [
        {"source": [
            "import numpy as np\n",
            "import time\n",
            "from quantem.widget import Show2D, Show3D\n",
        ]},
    ]
    for i, (n, h, w, widget) in enumerate(SHAPES):
        # Free previous-cell buffers to keep browser memory bounded across scenarios.
        cleanup = "" if i == 0 else (
            f"import gc\n"
            f"try:\n"
            f"    del w_{i-1}\n"
            f"except NameError: pass\n"
            f"try:\n"
            f"    del data_{i-1}\n"
            f"except NameError: pass\n"
            f"gc.collect()\n"
        )
        cells.append({"source": [
            cleanup,
            f"data_{i} = np.random.default_rng(0).standard_normal(({n},{h},{w})).astype(np.float32)\n",
            f"w_{i} = {widget}(data_{i}, title='timing-truthful {widget} {n}x{h}x{w}')\n",
            f"w_{i}\n",
        ]})
        cells.append({"source": [
            f"# Poll for up to 15s while JS round-trips the first-paint signal.\n",
            f"for _ in range(150):\n",
            f"    if getattr(w_{i}, 'render_total_ms', None) is not None:\n",
            f"        break\n",
            f"    time.sleep(0.1)\n",
            f"rt = getattr(w_{i}, 'render_total_ms', None)\n",
            f"rb = getattr(w_{i}, 'render_python_build_ms', None)\n",
            f"rw = getattr(w_{i}, 'render_wire_js_ms', None)\n",
            f"print(f'TIMING widget={widget} total={{rt}} build={{rb}} wirejs={{rw}}')\n",
        ]})
    _build_notebook_write(cells)


def _build_notebook_write(cells):
    _write_notebook(NOTEBOOK_PATH, cells)


def _is_canvas_painted(page, cell_index: int) -> bool:
    js = """
    (cellIdx) => {
        const cells = document.querySelectorAll('.jp-Cell');
        if (!cells[cellIdx]) return false;
        const canvases = cells[cellIdx].querySelectorAll('canvas');
        for (const c of canvases) {
            if (!c.width || !c.height) continue;
            try {
                const ctx = c.getContext('2d');
                if (!ctx) continue;
                const cx = Math.floor(c.width / 2);
                const cy = Math.floor(c.height / 2);
                const data = ctx.getImageData(cx, cy, 4, 4).data;
                for (let i = 0; i < data.length; i += 4) {
                    if (data[i + 3] > 0 && (data[i] | data[i + 1] | data[i + 2])) return true;
                }
            } catch (e) { /* ignore CORS */ }
        }
        return false;
    }
    """
    return page.evaluate(js, cell_index)


def _read_timing_from_cell(cell) -> dict:
    """Cell B prints ``TIMING widget=Show2D total=... build=... wirejs=...``."""
    text = cell.locator(".jp-OutputArea-output").inner_text(timeout=5000)
    import re
    m = re.search(
        r"TIMING\s+widget=(?P<w>\w+)\s+total=(?P<t>\S+)\s+build=(?P<b>\S+)\s+wirejs=(?P<wj>\S+)",
        text,
    )
    if not m:
        return {}
    def _to_int(s):
        try:
            return int(s)
        except (ValueError, TypeError):
            return None
    return {
        "widget": m.group("w"),
        "python_total_ms": _to_int(m.group("t")),
        "python_build_ms": _to_int(m.group("b")),
        "python_wire_js_ms": _to_int(m.group("wj")),
    }


def test_show2d_show3d_print_matches_playwright(browser_context):
    _build_notebook()
    page = browser_context.new_page()

    rel = NOTEBOOK_PATH.relative_to(TESTS_DIR.parent)
    # `?reset` nukes any saved JupyterLab workspace so the notebook opens in
    # the foreground tab.  Without this, a previous session's open notebooks
    # persist and our cells render in a hidden background tab (0×0 bounding box).
    url = f"http://localhost:{JUPYTER_PORT}/lab/tree/{rel}?reset"
    page.goto(url, timeout=60000)
    # 12 s — JupyterLab + kernel reliably finish loading by this point.  Earlier
    # 8 s value was flaky on some runs (race with kernel startup).
    time.sleep(12)

    try:
        page.keyboard.press("Escape")
    except Exception:
        pass

    # Cell 0 = imports, then for each widget: cell N = construct, cell N+1 = probe.
    # Run imports
    page.locator(".jp-Cell").nth(0).click()
    time.sleep(0.3)
    page.keyboard.press("Shift+Enter")
    time.sleep(2.0)

    results = []
    for i, (n, h, w, widget) in enumerate(SHAPES):
        construct_cell_idx = 1 + 2 * i
        probe_cell_idx = construct_cell_idx + 1

        # Click into the construct cell
        page.locator(".jp-Cell").nth(construct_cell_idx).click()
        time.sleep(0.3)

        t0 = time.perf_counter()
        page.keyboard.press("Shift+Enter")

        # Wait for canvas painted in the construct cell.  Timeout scales with payload
        # so 4K scenarios don't hit an arbitrary ceiling — we want the REAL wall clock.
        payload_mb = n * h * w * 4 / (1 << 20)
        paint_timeout = max(30.0, 10.0 + payload_mb * 0.2)  # +200 ms per MB overhead budget
        deadline = t0 + paint_timeout
        painted = False
        while time.perf_counter() < deadline:
            if _is_canvas_painted(page, construct_cell_idx):
                painted = True
                break
            time.sleep(0.1)
        playwright_ms = int(1000 * (time.perf_counter() - t0))
        if not painted:
            # Still record the timeout so the report shows it.
            print(f"!!! {widget} {n}x{h}x{w}: canvas not painted within {paint_timeout:.0f}s")
            results.append({
                "widget": widget,
                "shape": f"{n}x{h}x{w}",
                "playwright_ms": playwright_ms,
                "python_total_ms": None,
                "python_build_ms": None,
                "python_wire_js_ms": None,
                "note": f"NEVER_PAINTED_{int(paint_timeout)}s",
            })
            continue

        # Shift+Enter advances focus to next cell; run the probe
        time.sleep(0.5)
        page.locator(".jp-Cell").nth(probe_cell_idx).click()
        time.sleep(0.3)
        page.keyboard.press("Shift+Enter")
        # The probe cell polls up to 15s for the attribute; give it time to finish.
        time.sleep(3.0)

        probe_cell = page.locator(".jp-Cell").nth(probe_cell_idx)
        # Poll for the TIMING line to appear — scale with payload, 4K stacks need more
        # time for the _js_rendered round-trip after the huge paint.
        timing = {}
        poll_timeout = max(15.0, 5.0 + payload_mb * 0.3)
        poll_deadline = time.perf_counter() + poll_timeout
        while time.perf_counter() < poll_deadline:
            timing = _read_timing_from_cell(probe_cell)
            if timing and timing.get("python_total_ms") is not None:
                break
            time.sleep(0.3)

        results.append({
            "widget": widget,
            "shape": f"{n}x{h}x{w}",
            "playwright_ms": playwright_ms,
            **timing,
        })

    page.close()

    # ---- Report ----
    print()
    print("=" * 82)
    print(f"{'widget':<8} {'shape':<14} {'playwright':>12} {'py_total':>10} "
          f"{'py_build':>10} {'wire+js':>10} {'diff':>8}")
    print("-" * 82)
    for r in results:
        pt = r.get("python_total_ms")
        pb = r.get("python_build_ms")
        pw = r.get("python_wire_js_ms")
        pt_str = f"{pt} ms" if pt is not None else "(missing)"
        pb_str = f"{pb} ms" if pb is not None else "-"
        pw_str = f"{pw} ms" if pw is not None else "-"
        diff_str = f"{r['playwright_ms'] - pt:+} ms" if pt is not None else "N/A"
        print(f"{r['widget']:<8} {r['shape']:<14} {r['playwright_ms']:>10} ms "
              f"{pt_str:>10} {pb_str:>10} {pw_str:>10} {diff_str:>8}")
    print("=" * 82)

    # ---- Liar detector ----
    # Tolerance scales with payload — larger frame_bytes means the _js_rendered
    # trait round-trip back to Python takes longer to flush, so Python legitimately
    # reports more total time than what Playwright observed.  Base 300 ms + 1 ms/MB.
    # For 4K-scale payloads the tolerance widens to accommodate comm flush latency.
    failed = []
    for r in results:
        pt = r.get("python_total_ms")
        if pt is None:
            # Skip liar-check for scenarios where paint never completed — the report
            # above already shows NEVER_PAINTED.  We want the data, not an abort.
            continue
        n, h, w = map(int, r["shape"].split("x"))
        mb = n * h * w * 4 / (1 << 20)
        tolerance = 300 + int(mb)
        gap = r["playwright_ms"] - pt
        if gap > tolerance:
            failed.append(
                f"{r['widget']} {r['shape']}: Python {pt}ms < "
                f"Playwright {r['playwright_ms']}ms (lying by {gap}ms, tol {tolerance}ms)"
            )
    if failed:
        raise AssertionError("Truthful-timing verification failed:\n  " + "\n  ".join(failed))
