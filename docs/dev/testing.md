# Testing

quantem.widget uses three testing layers, each catching different classes of bugs.
All three are needed — no single layer is sufficient.

| Layer | What it tests | Speed | GPU? |
|-------|--------------|-------|------|
| **Unit tests** | Python traits, data pipeline, state persistence | ~20s | No |
| **Smoke tests** | UI layout, controls, interactions (CPU fallback) | ~4min | No |
| **GPU E2E tests** | WebGPU shaders, GPU colormap, FFT, real rendering | ~2min | **Yes** |

## Unit tests

Fast Python-side tests. No browser, no rendering. Tests trait handling, data
conversion, state persistence, ROI math, statistics, and array compatibility.

```bash
# All unit tests
python -m pytest tests/ -v --ignore-glob='tests/test_e2e_*.py'

# Single widget
python -m pytest tests/test_widget_show2d.py -v

# Quick check during development
python -m pytest tests/test_widget_show2d.py -v -x --tb=short
```

### What unit tests cover

- Widget construction with NumPy, PyTorch, CuPy, and Dataset objects
- `state_dict()` / `load_state_dict()` roundtrip (3 tests per widget)
- `save()` / load from file (versioned envelope JSON)
- `summary()` output content
- `set_image()` data replacement
- ROI creation, stats computation, clear
- Line profile extraction
- `save_image()` export (PNG, PDF)
- JS computation parity (FFT windowing, coordinate transforms)

### Test file naming

| Pattern | Purpose |
|---------|---------|
| `test_widget_*.py` | Per-widget unit tests |
| `test_io.py` | IO module tests |
| `test_widget_dataset_integration.py` | Dataset duck-typing tests |
| `test_e2e_smoke.py` | E2E smoke tests (separate layer) |

## Smoke tests (Playwright)

Headless browser tests using Playwright. Renders widgets in real JupyterLab,
captures screenshots, tests interactions (clicks, toggles, drag).

```bash
# All smoke tests (~4 minutes, 137 tests)
python -m pytest tests/test_e2e_smoke.py -v

# Single widget (~60 seconds)
python -m pytest tests/test_e2e_smoke.py -v -k show2d

# Multiple widgets
python -m pytest tests/test_e2e_smoke.py -v -k "show2d or show3d"
```

### What smoke tests cover

- Widget root element renders
- Canvas has non-zero pixels
- Default screenshot (light theme)
- Dark theme screenshot
- Control interactions (colormap change, scale toggle, FFT toggle)
- ROI FFT
- Profile draw and drag
- Playback controls

### Screenshot verification

After running smoke tests, visually inspect screenshots in
`tests/screenshots/smoke/`:

```
show2d.png           # Default state
show2d_fft.png       # FFT enabled
show2d_viridis.png   # Colormap changed
show2d_dark.png      # Dark theme
```

**Check for:** black/blank panels, missing images, wrong colormaps, broken
overlays, corrupted FFT, missing controls.

### Important limitation

**Headless Playwright has NO WebGPU.** Smoke tests always exercise the CPU
fallback path. They prove nothing about GPU shaders, GPU colormap, or GPU FFT.
If you change GPU code and only run smoke tests, you will miss bugs.

## GPU E2E tests (Chrome CDP)

Tests WebGPU shaders and GPU rendering on real hardware by connecting to Chrome
via Chrome DevTools Protocol. This is the only way to test GPU code paths.

### Setup

You need three terminals:

**Terminal 1 — JupyterLab:**
```bash
jupyter lab --notebook-dir=notebooks --port=8889
```

**Terminal 2 — Chrome with WebGPU + remote debugging:**
```bash
'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' \
  --remote-debugging-port=9222 \
  '--remote-allow-origins=*' \
  --no-first-run \
  --user-data-dir=/tmp/chrome-gpu \
  "http://localhost:8889/lab?token=YOUR_TOKEN"
```

Note: quote `'--remote-allow-origins=*'` for zsh (prevents glob expansion).

**Terminal 3 — Run the test:**
```bash
# Build JS + run GPU test
python scripts/gpu_test.py --build

# Watch mode (auto-retest when JS/TS files change)
python scripts/gpu_test.py --watch
```

### What GPU tests check

1. **WebGPU available** — `navigator.gpu` exists in Chrome
2. **Widget rendered** — canvases appear in the DOM
3. **GPU FFT initialized** — console log confirms WebGPU FFT pipeline
4. **GPU colormap initialized** — console log confirms colormap engine
5. **GPU colormap active** — `renderSlots` timing logs on colormap changes
6. **GPU colormap fast** — warm calls < 200ms for 12 x 4K images
7. **Pixels non-zero** — canvases have actual rendered content

### Console log protocol

All GPU-enabled widgets emit structured console logs for automated verification:

```
[Show2D] WebGPU FFT initialized — GPU
[Show2D] WebGPU colormap engine initialized
[Show2D] GPU colormap: 12x4096x4096 in 104ms (gpu=61ms copy=43ms)
```

### How it works

`scripts/gpu_test.py` connects to Chrome via CDP (`localhost:9222`) and
JupyterLab via the kernel websocket API (`localhost:8889`). It:

1. Restarts the kernel and reloads the page (picks up latest JS build)
2. Installs a console log interceptor via `Runtime.evaluate`
3. Runs notebook cells by simulating Shift+Enter via `Input.dispatchKeyEvent`
4. Waits for GPU initialization (~10s)
5. Changes colormaps via kernel execution (`w.cmap = 'hot'`)
6. Reads console logs to verify GPU path activation and timing
7. Checks canvas pixels for non-zero content
8. Takes a screenshot

### Writing GPU tests for new widgets

When adding GPU support to a widget:

1. Add console logs following the protocol:
   ```typescript
   console.log(`[WidgetName] WebGPU colormap engine initialized`);
   console.log(`[WidgetName] GPU colormap: ${n}x${w}x${h} in ${ms}ms (gpu=${gpuMs}ms copy=${copyMs}ms)`);
   ```

2. Add test cells to `scripts/gpu_test.py` `TEST_CELLS` or create a
   widget-specific test notebook

3. Add colormap change tests to `CMAP_TESTS`

## When to run which tests

### After Python changes
```bash
python -m pytest tests/ -v --ignore-glob='tests/test_e2e_*.py'
```

### After JS/TS UI changes (no GPU)
```bash
npm run build
python -m pytest tests/test_e2e_smoke.py -v -k <widget>
# Then visually verify screenshots
```

### After GPU/shader/colormap/FFT changes
```bash
python scripts/gpu_test.py --build
# Also run smoke tests for CPU fallback:
python -m pytest tests/test_e2e_smoke.py -v -k <widget>
```

### Before committing
```bash
# All unit tests
python -m pytest tests/ -v --ignore-glob='tests/test_e2e_*.py'
# Full smoke suite (if JS changed)
python -m pytest tests/test_e2e_smoke.py -v
# GPU test (if GPU code changed)
python scripts/gpu_test.py --build
```

## Visual catalog

For full QA review across all widgets:

```bash
# Capture every widget in key states
python scripts/generate_catalog.py

# Light theme only (faster)
python scripts/generate_catalog.py --light-only

# Start local server for review
python scripts/generate_catalog.py --serve
# Open http://localhost:8321
```

The catalog generates a self-contained HTML page at
`tests/screenshots/catalog/index.html` with filter-by-widget, light/dark theme
comparison, and sidebar navigation.

## Test data

- **Synthetic data** in unit tests: OK for CI portability
- **Real data** in notebooks: always (never synthetic for-loops)
- **Arina fixture**: `tests/data/arina_fixture/` — real binned 4D-STEM data,
  committed to repo, must not be skipped
- **Gold drift data**: `~/data/bob/20260409_gold_drift_v3/` — 12 x 4K EMD files
  for GPU stress testing (not committed, local only)
