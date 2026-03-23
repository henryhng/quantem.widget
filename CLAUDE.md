# CLAUDE.md

## Project

quantem.widget — Interactive Jupyter widgets for electron microscopy (anywidget + React). Maintained and led by Sangjoon Bob Lee ([@bobleesj](https://github.com/bobleesj)) — reach out anytime.

## Response Protocol

- Always include a `What’s next` section at the end of substantive updates.
- `What’s next` must be a numbered list with concrete, buildable next actions (not generic ideas).
- For each next action, include one line on microscopy experiment value (how it improves acquisition safety, QC confidence, throughput, or reproducibility during live runs).
- Prioritize next actions that reduce experiment risk first (e.g., bad-file ingestion, wrong calibration, failed long-running merges) before convenience features.
- Always end with a `Files modified` section listing each file path and the number of lines changed (e.g., `notebooks/show3d/show3d_simple.ipynb (+12 −3)`).

## Widgets

- **Show1D** — 1D viewer for spectra, profiles, and time series (multi-trace, zoom/pan, log scale, auto-contrast percentile clipping, peak detection, export)
- **Show2D** — 2D image viewer with gallery, ROI, line profiles, FFT, export (PNG/GIF/ZIP)
- **Show3D** — 3D stack viewer with playback, ROI, FFT, export (PNG/GIF/ZIP)
- **Show3DVolume** — Orthogonal slice viewer (XY, XZ, YZ) with FFT, playback, export
- **Show4D** — General 4D dataset explorer (nav + signal panels), ROI masking, path animation
- **Show4DSTEM** — 4D-STEM diffraction pattern viewer with virtual imaging, ROI presets, 5D time/tilt series support
- **ShowComplex2D** — Complex-valued 2D viewer (amplitude/phase/HSV/real/imag), phase colorwheel
- **Mark2D** — Interactive 2D image annotation (points, ROIs, profiles, snap-to-peak, undo)
- **Edit2D** — Interactive crop/pad/mask editor with brush tool
- **Align2D** — Image alignment overlay with FFT-based auto-align, opacity blending
- **Bin2D** — Interactive 2D image binning with side-by-side original/binned comparison, gallery support, calibration tracking
- **Bin4D** — 4D-STEM binning tool with scan/detector bin factors, edge modes, BF/ADF/mean-DP previews, export (PNG/ZIP/GIF)
- **MetricExplorer** — Interactive metric explorer for parameter sweeps (SSB aberrations, denoising hyperparams) with grouped line charts, crosshair sync, best-point markers, and click-to-inspect images
- **Browse** — File browser widget for exploring directories of EM data

## Stack

- Python: `src/quantem/widget/` (anywidget + traitlets)
- JS/TS: `js/` (React + MUI, bundled with esbuild)
- Build: `npm run build` → `src/quantem/widget/static/*.js`

## Environment

quantem.widget is designed for personal laptops (e.g., MacBook Pro M5 24 GB RAM). All widgets run in the browser via JupyterLab — no GPU required for visualization.

- Use a dedicated conda env for development and testing (e.g., `widget-env`). Use `mamba` internally for env creation (faster solver), but use `conda` in all public-facing docs (README, getting-started, etc.) since users may not have mamba installed.
- HMR (hot module reload): requires `watchfiles` (`pip install watchfiles`). Run `npm run dev` in a separate terminal and set `%env ANYWIDGET_HMR=1` in notebooks. Without `watchfiles`, anywidget prints a warning and live-reload is disabled.

## Commands

```bash
# Install
npm install && npm run build && pip install -e .

# Run unit tests (fast, no browser)
python -m pytest tests/ -v --ignore-glob='tests/test_e2e_*.py'

# Run E2E tests (launches JupyterLab + Playwright, ~4 min)
# All E2E/Playwright tests must be named test_e2e_*.py
python -m pytest tests/test_e2e_smoke.py -v

# Run all tests
python -m pytest tests/ -v

# Build for PyPI
python -m build

# TypeScript check
npm run typecheck

# Start JupyterLab (notebooks folder)
jupyter lab --notebook-dir=notebooks
```

## Kernel Crash Recovery

If the Jupyter kernel crashes after frontend or widget changes:

```bash
# Reinstall editable Python package in the active env
pip install -e .

# Rebuild/watch JS bundles during notebook development
npm run dev
```

Use the kernel matching your development conda env for notebooks.

## Testing Rules

- All widgets must accept NumPy arrays, PyTorch tensors, CuPy arrays, and quantem Dataset objects as input.
- quantem Dataset integration: Show2D, Mark2D, and Edit2D accept `Dataset2d`, Show3D and Show3DVolume accept `Dataset3d`, Show4D and Show4DSTEM accept `Dataset4dstem`, Align2D accepts `Dataset2d` for both images, ShowComplex2D accepts Dataset (generic). Metadata (title, pixel_size, units) must be auto-extracted via duck typing.
- Run `python -m pytest tests/ -v --ignore-glob='tests/test_e2e_*.py'` after every change. All tests must pass.
- No unused variables or imports. Run `npm run typecheck` for JS/TS.
- **Never skip tests for missing dependencies that are in `pyproject.toml`.** If `h5py`, `hdf5plugin`, `torch`, or any hard dependency is missing, the install is broken — tests must fail loudly, not skip silently. Use `pytest.skip` only for truly optional features (e.g., CuPy on machines without NVIDIA GPUs).
- **Real arina test fixture:** `tests/data/arina_fixture/` contains a binned real arina dataset (16 frames, 24×24 detector, bitshuffle+LZ4). Tests using this fixture must not be skipped — the data is committed to the repo.
- **Visual screenshot verification is mandatory for all JS/UI changes.** Do not rely on Python unit tests for widget UI correctness — they only test Python-side logic, not rendering. After running smoke tests, always read and visually inspect the screenshots in `tests/screenshots/smoke/`. This is the ground truth for UI correctness.
- After modifying any widget UI (JS/TS):
  - Always build first: `npm run build`.
  - Run smoke tests **only for the modified widget** using `-k`: `python -m pytest tests/test_e2e_smoke.py -v -k show2d` (filters to that widget's tests — rendering, interactions, and dark theme). No need to run all 69 tests for every change.
  - Run the full suite (`python -m pytest tests/test_e2e_smoke.py -v`) only before committing or when changes affect shared modules (theme, colormaps, scalebar, etc.).
  - Visually verify the screenshots: default state (`show2d.png`), interactions (`show2d_fft.png`, `show2d_viridis.png`), and dark theme (`show2d_dark.png`).
  - Screenshots go in `tests/screenshots/smoke/` (E2E smoke tests) and `tests/screenshots/<widget>/<theme>/` (capture scripts).
- **Visual catalog for full QA review:** `python scripts/generate_catalog.py` captures every widget in key states via Playwright, then assembles a self-contained HTML page at `tests/screenshots/catalog/index.html`. Supports filter-by-widget-type, light/dark theme comparison, and sidebar navigation. Use `--light-only` for faster runs, `--serve` to start a local server at `localhost:8321` for easy refresh-based review.
- **JS computation validation:** When implementing mathematical operations in JavaScript (FFT, windowing, statistics, coordinate transforms), always port the exact JS code to Python and validate against NumPy/SciPy/PyTorch ground truth in unit tests. Do not hardcode formulas separately — port the JS line-by-line so the test catches any JS-side bugs. See `test_widget_show2d.py` `_js_hann_1d` / `_js_apply_hann_window_2d` for the pattern.
- **IO testing must use real data.** When testing IO features (progress bars, benchmark, loading), write notebooks that load real files already on disk — never generate synthetic data with for-loops and tempfiles. Use paths from existing IO notebooks (`notebooks/io/io_file.ipynb`, `io_folder.ipynb`) as reference for where data lives. Supported formats: EMD (Velox), DM3/DM4, TIFF, Arina 4D-STEM (see `notebooks/io/io_arina_file.ipynb` and `io_arina_folder.ipynb`). Unit tests (`tests/test_io.py`) may use synthetic data for CI portability, but notebooks must always use real files.

## Shared Modules

- **`js/theme.ts`** — Shared theme detection and color system. Exports `detectTheme()`, `useTheme()` hook, `ThemeColors` type, `DARK_COLORS`/`LIGHT_COLORS`. Used by all widgets for automatic light/dark theme support.
- **`js/webgpu-fft.ts`** — Shared FFT module used by all FFT-capable widgets (Show2D, Show3D, Show3DVolume, Show4D, Show4DSTEM, ShowComplex2D, Align2D). Provides `WebGPUFFT` class (GPU-accelerated), `fft2d` (CPU fallback, synchronous), `fft2dAsync` (CPU fallback in Web Worker — non-blocking), `fftshift`, `computeMagnitude` (complex→magnitude), `autoEnhanceFFT` (DC mask + 99.9% percentile), `getWebGPUFFT` (singleton), `getGPUInfo`. **Do not debounce FFT recomputation** — use the **rAF generation counter** pattern instead: each FFT trigger increments a generation counter and yields via `requestAnimationFrame` before computing. After the yield, stale generations are discarded. This coalesces rapid drag events to ≤1 FFT per display frame without artificial delay. For the CPU fallback path, use `fft2dAsync` (Web Worker) so the main thread is never blocked. See Show2D's FFT effect for the reference implementation.
- **`js/colormaps.ts`** — Shared colormap LUTs and rendering. Exports `COLORMAPS`, `COLORMAP_NAMES`, `applyColormap` (float data → RGBA via LUT), `renderToOffscreen` (creates colormapped offscreen canvas).
- **`js/scalebar.ts`** — Shared scale bar, colorbar, and figure export. Exports `roundToNiceValue`, `formatScaleLabel`, `drawScaleBarHiDPI`, `drawFFTScaleBarHiDPI`, `drawColorbar`, `exportFigure`. HiDPI-aware rendering with automatic unit conversion (Å→nm, mrad→rad). `exportFigure()` creates publication-quality canvas with title, scale bar, colorbar, and annotation callback.
- **`js/format.ts`** — Data conversion, formatting, and file download. Exports `extractBytes` (DataView→Uint8Array), `extractFloat32` (DataView→Float32Array), `formatNumber` (exponential notation), `downloadBlob` (save Blob as file), `downloadDataView` (save DataView as file).
- **`js/histogram.ts`** — Histogram computation. Exports `computeHistogramFromBytes` (Float32Array→normalized bins).
- **`js/stats.ts`** — Data statistics and transforms. Exports `findDataRange` (min/max), `computeStats` (mean/min/max/std), `applyLogScale` (log1p transform), `applyLogScaleInPlace` (in-place variant), `percentileClip` (percentile-based vmin/vmax), `sliderRange` (slider % → data values). Note: `sliderRange(dataMin, dataMax, vminPct, vmaxPct)` returns `{ vmin, vmax }` — always destructure the result.
- **`js/webgpu-volume.ts`** — WebGPU ray-casting volume renderer used by Show3DVolume. Exports `VolumeRenderParams`, `CameraState`, `DEFAULT_CAMERA`, full matrix math (mat4 multiply/inverse/perspective/lookAt), WGSL shader, texture upload, and render loop. Uses shared `GPUDevice` via `getGPUDevice()` from `webgpu-fft.ts`. Self-contained module (no external 3D library dependency).

### Shared Python modules

- **`src/quantem/widget/array_utils.py`** — `to_numpy()` (NumPy/CuPy/PyTorch→NumPy), `bin2d()`, `_resize_image()`, `apply_shift()`.
- **`src/quantem/widget/detector.py`** — 4D-STEM detector utilities. `detect_bf_disk(data)` auto-detects BF disk center and radius (same algorithm as `Show4DSTEM.auto_detect_center` but pure NumPy). `make_virtual_masks(det_rows, det_cols, center_row, center_col, bf_radius)` creates BF/ADF/HAADF annular masks. `virtual_images(data)` is the high-level one-liner that auto-detects and computes all three. Used by `IO.arina_folder(virtual_only=True)` and available as `from quantem.widget import virtual_images`.

### What stays per-widget (intentionally NOT shared)
- Style constants: `SPACING`, `typography`, `controlRow`, `compactButton`, `switchStyles`, `sliderStyles`, `container`, `upwardMenuProps`, `MIN_ZOOM`, `MAX_ZOOM`, `DPR`
- React components: `Histogram`, `InfoTooltip`, resize handle JSX
- React hooks: wheel scroll prevention, canvas resize drag
- Zoom/pan mouse handlers (two conventions: simple and center-based, coupled to widget state)

## Notebooks

- Demo notebooks live in `notebooks/<widget>/` (gitignored), one folder per widget.
- Two notebooks per widget: `<widget>_simple.ipynb` (minimal demo) and `<widget>_all_features.ipynb` (comprehensive, shows ALL widget capabilities). Widget name prefix ensures filenames are meaningful when shared individually.
- **Published docs (`docs/examples/`) include only `simple` + `all_features` per widget — no extra notebooks.** Specialized notebooks (file loaders, benchmarks, advanced workflows) stay in `docs/examples/` directories for internal use but are excluded from the Sphinx toctree and build via `conf.py exclude_patterns`.
- All notebooks are interactive anywidget demos — they must display live widgets, not static images.
- First code cell must enable autoreload and HMR:
  ```python
  %load_ext autoreload
  %autoreload 2
  %env ANYWIDGET_HMR=1
  ```
- Synthetic data must look like realistic electron microscopy: crystal lattices, atomic columns, diffraction patterns with Bragg spots, focal series with Fresnel fringes, core-shell nanoparticles for tomography, etc. No plain random noise or trivial Gaussians.
- Synthetic data generators must use **PyTorch with GPU acceleration** (MPS on macOS, CUDA on Linux/Windows) for compute-heavy operations. Use `torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")` with automatic fallback. Use vectorized broadcasting, precompute templates, and batch-process — never use nested Python for-loops over pixel/position indices. Convert to NumPy at the end with `.cpu().numpy()`.
- Notebooks must be runnable with `jupyter nbconvert --execute` for CI validation.
- **Keep widget state in notebooks** — the `metadata.widgets` field stores serialized widget models so docs render previews without re-execution. Do NOT strip widget state before committing.
- **Exception:** if a notebook's widget state pushes it over GitHub's 100 MB file limit, strip only that notebook (`del nb['metadata']['widgets']`). Currently applies to `show3d_all_features.ipynb` (~102 MB with state).
- **Real-data showcase notebooks** (e.g. `notebooks/<user>/`) are gitignored (private data). When generating notebooks for a user's real data:
  - Enable features a microscopist would actually use for that data type — don't just show bare minimum single-feature demos. For TEM images: enable `log_scale=True`, `show_fft=True`, `roi_active=True`, place ROIs, draw line profiles, etc.
  - Pair 2D viewers with Show1D for quantitative line profiles — extract `profile_values` from Show2D/Show3D and display in Show1D. Compare the same profile across multiple frames to check drift or dose damage.
  - Use `add_trace()` to overlay profiles from different regions or frames in a single Show1D plot.
  - Show realistic workflows: image → ROI FFT → line profile → multi-frame comparison → annotation → alignment.

## JS/TS Architecture

- **Single-file widgets (~2,000 lines each) are intentional.** Each widget is one `index.tsx` with all hooks, effects, handlers, and JSX in a single file. This is optimized for AI-assisted development: all context in one file, no cross-file navigation, grep finds everything. Do not split widgets into smaller files.

- Theme detection and colors are shared via `js/theme.ts`. All widgets support automatic light/dark mode.
- Functional utilities are shared: `js/webgpu-fft.ts` (FFT), `js/colormaps.ts` (colormap LUTs), `js/theme.ts` (theme).
- Widget-specific UI (typography, spacing, control layout) is inlined per widget.
- Show3D is the "model" widget — all other widgets follow its layout patterns (header row, bordered canvas box with resize handle, stats bar, control rows + histogram).
- All widgets must use the same font sizes, control styling, and layout as Show3D/Show4DSTEM. Controls use fontSize 10, labels use fontSize 11, stats use monospace. Histogram rendering, switch styles, and dropdown selects must be visually identical across widgets. Buttons use MUI default uppercase text (no `textTransform: "none"`).
- All canvas-based widgets must prevent page scrolling when using mouse scroll to zoom. Use `addEventListener("wheel", preventDefault, { passive: false })` on the canvas container element.
- All canvas-based widgets must have a resize handle (triangular gradient in bottom-right corner) for drag-resizing the canvas.
- Scale bars use the HiDPI pattern from Show4DSTEM's `drawScaleBarHiDPI`:
  - Render on a **separate HiDPI UI canvas** (`width={cssW * DPR}`, `style={{ width: cssW }}`) for crisp text at all zoom levels.
  - Fixed CSS pixel sizes: `targetBarPx = 60`, `barThickness = 5`, `fontSize = 16`, `margin = 12`.
  - Always **white** with drop shadow (`shadowColor: rgba(0,0,0,0.5)`, `shadowBlur: 2`).
  - **Centered label** above bar, system font stack (`-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`).
  - **Zoom indicator** bottom-left (e.g., `1.0×`).
  - Physical length = `(targetBarPx / effectiveZoom) * pixelSize`, rounded to nice value via `roundToNiceValue`.
  - Unit conversion: Å → nm at ≥10 Å, px for uncalibrated. Use `formatScaleLabel(value, unit)`.
  - Not user-configurable — no Python traits for bar size/thickness/font. Show2D, Show3D, and Show4DSTEM all use this pattern.
- All text/number overlays on canvas (marker labels, point numbers, etc.) must be **pixel-independent** — rendered on the HiDPI UI canvas, not on the main image canvas. This ensures text stays crisp at any zoom level, just like the scale bar. Coordinates are transformed from image space to screen space before drawing on the UI canvas at `DPR` resolution.
- No `borderRadius` on control rows, stats bars, or container elements. Keep edges sharp.
- **Avoid global ArrowUp/ArrowDown keyboard shortcuts.** These conflict with Jupyter's cell navigation (moving cursor between cells). Only use ArrowLeft/ArrowRight for horizontal navigation (prev/next frame, slice, image, or nudge).
- **Show4DSTEM / Show4D exception:** ArrowUp/ArrowDown are allowed for row navigation only when key handling is focus-scoped to the widget root (`tabIndex` + `onKeyDown`), focus is acquired via canvas interaction, typing targets are ignored (`input`/`textarea`/`select`/`[role=textbox]`/contenteditable), and there is no global `window` key listener.
- Keyboard shortcut tooltips use the `KeyboardShortcuts` component (per-widget, renders a two-column table with monospace keys and descriptions). InfoTooltip accepts `React.ReactNode` for structured content.
- **Common keyboard shortcuts** (all canvas-based widgets must implement these):
  - `R` — reset zoom to 1× and pan to origin
  - `Scroll` — zoom in/out centered on cursor
  - `Double-click` — reset view (same as R)
  - `← / →` — navigate (prev/next frame, image, or nudge ROI/crop depending on widget)
  - `↑ / ↓` — vertical nudge (crop region, ROI position, alignment offset) for widgets with spatial editing. Focus-scoped only (tabIndex + onKeyDown). Currently: Edit2D, Mark2D, Show4D, Show4DSTEM, Align2D.
- **Common widget traits** (all widgets should support):
  - `title` — displayed in the header row, with a sensible default (e.g. `"Image"`, `"Mark2D"`)
  - `show_stats` — toggle the statistics bar (mean, min, max, std) below the canvas

### Tool visibility/locking pattern (reusable across widgets)

- Use two orthogonal APIs for feature gating:
  - **Disable (lock)**: controls remain visible but non-interactive (`disabled_tools` + `disable_*` flags).
  - **Hide**: controls are not rendered (`hidden_tools` + `hide_*` flags).
- Canonical source of truth:
  - Registry file: `src/quantem/widget/tool_parity.json`
  - Python helpers/runtime API binding: `src/quantem/widget/tool_parity.py`
  - Frontend helpers: `js/tool-parity.ts`
  - Frontend dropdown UI: `js/control-customizer.tsx`
- **Parity requirement**: when this pattern is added or changed in one widget, propagate the same API shape to other applicable widgets in the same change set (or document a short-term gap explicitly in docs/changelog).
- Hidden tools should also be behavior-locked (hiding implies disable for interactions).
- Keep naming consistent across widgets:
  - list traits: `disabled_tools`, `hidden_tools`
  - boolean convenience flags: `disable_<tool>`, `hide_<tool>` (for each supported tool key in that widget)
  - global flags: `disable_all`, `hide_all`
- Supported tool-group keys should be canonical and shared where possible.
  - Baseline keys for most viewers/editors: `display`, `histogram`, `stats`, `navigation`, `view`, `export`, `all`
  - Advanced viewer keys (when applicable): `roi`, `profile`, `fft`, `playback`, `virtual`, `frame`, `volume`
  - Mark2D-specific keys: `points`, `roi`, `profile`, `marker_style`, `snap`
  - Edit2D-specific keys: `mode`, `edit`
  - Bin4D-specific keys: `binning`, `mask`, `preview`
  - Bin2D-specific keys: `binning`, `histogram`, `stats`, `navigation`, `export`
- Precedence rules:
  - `hide_*`/`hidden_tools` take precedence over `disable_*`/`disabled_tools`.
  - `*_all` takes precedence over individual keys.
- Validate keys strictly in Python (`ValueError` on unknown key). Use canonical keys only (no aliases); casing may be normalized.
- Persist lock/hide settings via `state_dict()`/`save()` and restore through `state=` or `load_state_dict()` for share/reload reproducibility.
- Enforce lock/hide in both UI and behavior paths: buttons/toggles disabled/hidden and keyboard/mouse handlers blocked.
- **Common style patterns** (extract into variables, don't duplicate inline):
  - `switchStyles.small` — consistent switch sizing: `{ '& .MuiSwitch-thumb': { width: 12, height: 12 }, '& .MuiSwitch-switchBase': { padding: '4px' } }`
  - **Switch placement**: Label text always comes **before** the Switch toggle, never after. Pattern: `<Typography>Label:</Typography> <Switch />`. Examples: `FFT: <Switch>`, `Auto: <Switch>`, `Snap: <Switch>`, `Loop <Switch>`.
  - `upwardMenuProps` — menu positioning constant (outside render)
  - `themedSelect` + `themedMenuProps` — themed dropdown styling (inside render, depends on theme colors)

## Coordinate Convention

**All user-facing coordinates use `(row, col)` — never `(x, y)`.** This matches the electron microscopy convention and NumPy array indexing (`array[row, col]`). The first value is always the row (vertical, top-to-bottom), the second is the column (horizontal, left-to-right).

This applies to **all widgets** (Show2D, Show3D, Show3DVolume, Show4D, Show4DSTEM, ShowComplex2D, Mark2D, Edit2D, Align2D, Bin2D, Bin4D) and covers:
- Python API: ROI/point dicts (`{"row": r, "col": c}`), tuple inputs `(row, col)`, and method parameters.
- JS/TS types: `Point = { row, col, ... }`, `ROI = { row, col, ... }`, hover/cursor state `{ row, col, ... }`.
- Display: coordinates shown as `(row, col)` in stats bars, point lists, tooltips, and readouts.

### Mapping to other coordinate systems

Different systems use different conventions. When interfacing with them, convert internally but always expose `(row, col)` to the user:

| System | First axis | Second axis | Notes |
|--------|-----------|-------------|-------|
| **Our API** | `row` (vertical) | `col` (horizontal) | User-facing, everywhere |
| **NumPy** | `array[row, col]` | — | Same — `row` = axis 0, `col` = axis 1 |
| **JS flat array** | `data[row * width + col]` | — | Same — `row` strides by `width` |
| **Matplotlib** | `y` (vertical) | `x` (horizontal) | `plt.scatter(col, row)` — axes are swapped |
| **JS Canvas** | vertical (`canvasH`) | horizontal (`canvasW`) | `row/height * canvasH`, `col/width * canvasW` |
| **DOM/browser** | `clientY` | `clientX` | Internal only, never exposed to user |

### What uses (row, col) vs what keeps x/y

- **User-facing** (MUST be row/col): trait names, point/ROI dicts synced to Python, hover readout display, `selected_points`, `roi_list`, method parameters, `summary()` output, docstrings.
- **Internal drawing** (keeps x/y): canvas screen positions in `drawROI(ctx, x, y, ...)`, `drawMarker(ctx, x, y, ...)`, DOM event coords (`e.clientX`), pan/zoom state (`panX, panY`), resize handles, `imgX/imgY` in mouse handlers. These are pixel positions on screen, not image coordinates.

### Current conversion status

- **Mark2D**: fully converted (Python + JS + tests).
- **Show2D**: fully converted (profile_line, cursorInfo use row/col).
- **Show3D**: fully converted (`roi_list` entries store `row`/`col`, cursorInfo uses row/col).
- **Show3D ROI API**: breaking changes are allowed. Do not preserve old single-ROI traits (`roi_row`, `roi_col`, `roi_radius`, etc.). Use `roi_list` and `roi_selected_idx`.
- **File loading**: use `IO.file()` / `IO.folder()` for all formats. No `from_*` classmethods on widgets.
- **Show3DVolume**: fully converted (slice_x/y/z axis indices are correct, cursorInfo uses row/col).
- **Show4DSTEM**: fully converted. Scan-space: `pos_row`/`pos_col`, `shape_rows`/`shape_cols`, `det_rows`/`det_cols`, `vi_roi_center_row`/`vi_roi_center_col`. K-space: `center_row`/`center_col`, `roi_center_row`/`roi_center_col`. CursorInfo uses row/col. `center=(row, col)` tuple input.
- **Show4D**: fully converted. `roi_center_row`/`roi_center_col`, `pos_row`/`pos_col`.
- **ShowComplex2D**: fully converted (cursorInfo uses row/col).
- **Edit2D**: fully converted (crop uses top/left/bottom/right, not row/col — but display shows row/col).
- **Align2D**: `dx`/`dy` kept as domain-specific (image registration convention).

## Rotation Convention

Per-image rotation uses `np.rot90` convention (CCW-positive). The `image_rotations` trait stores the `k` value directly:

| `image_rotations[i]` | `np.rot90` k | Direction | `rotate()` angle |
|-----------------------|-------------|-----------|-----------------|
| `0` | `k=0` | identity | `0` or `360` |
| `1` | `k=1` | 90° CCW | `90` |
| `2` | `k=2` | 180° | `180` |
| `3` | `k=3` | 270° CCW (= 90° CW) | `-90` |

- **Keyboard**: `]` = CW (intuitive: right bracket = rotate right), `[` = CCW.
- **Python API**: `w.rotate(idx, angle)` — positive angle = CCW, negative = CW.
- **Trait**: `image_rotations` list of ints (0–3), one per image. Directly usable as `np.rot90(img, k=val)`.
- **Non-square images**: after rotation, all gallery images are center-padded to `max(H) × max(W)`.
- **Widgets**: Show2D only. Show3D does not need per-frame rotation (same instrument/session).

## Data Pipeline

- All image widgets send **raw float32** data to JS — normalization and colormap application happen in JavaScript for instant interactivity.
  - **Show2D**: `frame_bytes` = raw float32 for all images.
  - **Show3D**: `frame_bytes` = raw float32 per frame.
  - **Show3DVolume**: `volume_bytes` = raw float32 for entire volume.
  - **Show4D**: `frame_bytes` = raw float32 per signal frame, `nav_image_bytes` = raw float32 navigation image.
  - **Show4DSTEM**: `frame_bytes` = raw float32 per diffraction pattern.
  - **ShowComplex2D**: `real_bytes` + `imag_bytes` = raw float32 (JS computes amplitude/phase/HSV).
  - **Mark2D**: `frame_bytes` = raw float32.
  - **Edit2D**: `frame_bytes` = raw float32, `mask_bytes` = uint8 binary mask.
  - **Align2D**: `image_a_bytes` + `image_b_bytes` = raw float32.
- JS handles: log scale, auto-contrast, percentile clipping, histogram, colormap LUT application.
- Python handles: statistics computation, ROI mean, GIF/ZIP export (using `_normalize_frame`).

## Histogram & Contrast Pipeline

The histogram widget provides interactive contrast adjustment across all image widgets. Two draggable handles set `imageVminPct` (left, 0–100) and `imageVmaxPct` (right, 0–100), which clip the displayed intensity range.

### Histogram data source

| Widget | Histogram computed from | Depends on |
|--------|------------------------|------------|
| Show2D | current image (`frame_bytes`) | `selected_idx` |
| Show3D | current frame (`frame_bytes`) | `slice_idx` |
| Show3DVolume | **full volume** (`volume_bytes`) | — (stable across slices) |
| Show4D | signal frame (`frame_bytes`) | `pos_row`, `pos_col` |
| Show4DSTEM | diffraction pattern (`frame_bytes`) | `pos_row`, `pos_col` |
| ShowComplex2D | current display mode output | `display_mode` |
| Mark2D | current image (`frame_bytes`) | `selected_idx` |
| Edit2D | current image (`frame_bytes`) | — |

Show3DVolume uses the full volume (not per-slice) so the histogram range stays stable as users scrub through slices.

### Rendering math — 2D canvas path

All 2D canvas widgets follow the same pipeline:

$$x' = \begin{cases} \ln(1 + x) & \text{if log scale enabled} \\ x & \text{otherwise} \end{cases}$$

`sliderRange()` in `js/stats.ts` converts slider percentages to data-space clipping bounds:

$$v_{\min} = d_{\min} + \frac{p_{\min}}{100} \cdot (d_{\max} - d_{\min})$$

$$v_{\max} = d_{\min} + \frac{p_{\max}}{100} \cdot (d_{\max} - d_{\min})$$

where $d_{\min}, d_{\max}$ are the data range and $p_{\min}, p_{\max} \in [0, 100]$ are the slider handle positions.

`applyColormap()` then normalizes each pixel into $[0, 1]$ for LUT lookup:

$$I = \text{clamp}\!\left(\frac{x' - v_{\min}}{v_{\max} - v_{\min}},\; 0,\; 1\right) \xrightarrow{\text{LUT}} \text{RGBA}$$

> **Example — 2D path.** Data range $[100, 500]$, sliders at $p_{\min}=20\%$, $p_{\max}=80\%$:
>
> $$v_{\min} = 100 + \frac{20}{100} \cdot 400 = 180, \quad v_{\max} = 100 + \frac{80}{100} \cdot 400 = 420$$
>
> A pixel with value $x = 300$:
>
> $$I = \frac{300 - 180}{420 - 180} = \frac{120}{240} = 0.5 \quad \text{(mid-colormap)}$$
>
> A pixel with $x = 150 < 180$ clamps to $I = 0$ (black). A pixel with $x = 450 > 420$ clamps to $I = 1$ (white).

### Rendering math — 3D volume path (Show3DVolume)

The 3D ray-caster works in normalized texture space, not data space. `uploadVolume()` maps raw float32 to an `r8unorm` GPU texture:

$$t = \frac{x - d_{\min}}{d_{\max} - d_{\min}} \in [0, 1]$$

Since the texture is already normalized, slider percentages map directly without `sliderRange()`:

$$v_{\min} = \frac{p_{\min}}{100}, \quad v_{\max} = \frac{p_{\max}}{100}$$

The WGSL fragment shader remaps the sampled texel, then applies brightness:

$$I = \text{clamp}\!\left(\frac{t - v_{\min}}{v_{\max} - v_{\min}},\; 0,\; 1\right)$$

$$I' = \text{clamp}(I \cdot b,\; 0,\; 1) \xrightarrow{\text{colormap texture}} \text{RGB}$$

where $b$ is the brightness multiplier ($0.1$–$3.0$). This remapping is applied to both the volume ray-casting samples and the three slice plane intersections (XY, XZ, YZ).

> **Example — 3D path.** Same data $[100, 500]$, sliders at $p_{\min}=20\%$, $p_{\max}=80\%$, brightness $b = 1.0$:
>
> $$v_{\min} = 0.20, \quad v_{\max} = 0.80$$
>
> A voxel with raw value $x = 300$ was normalized by `uploadVolume()` to $t = \frac{300 - 100}{400} = 0.5$:
>
> $$I = \frac{0.5 - 0.2}{0.8 - 0.2} = \frac{0.3}{0.6} = 0.5 \quad \text{(same as 2D path ✓)}$$
>
> A voxel with $x = 150 \to t = 0.125 < 0.2$ clamps to $I = 0$. A voxel with $x = 450 \to t = 0.875 > 0.8$ clamps to $I = 1$.

### Equivalence proof

Both paths compute the same normalized intensity. Substituting the 2D formulas into the colormap normalization and simplifying with $\Delta = d_{\max} - d_{\min}$ and $t = (x - d_{\min})/\Delta$:

$$\frac{x - v_{\min}}{v_{\max} - v_{\min}} = \frac{x - d_{\min} - \frac{p_{\min}}{100}\Delta}{\frac{p_{\max} - p_{\min}}{100}\Delta} = \frac{\frac{x - d_{\min}}{\Delta} - \frac{p_{\min}}{100}}{\frac{p_{\max} - p_{\min}}{100}} = \frac{t - v_{\min}^{\text{3D}}}{v_{\max}^{\text{3D}} - v_{\min}^{\text{3D}}}$$

> **Numerical check.** With $x = 300$, $d_{\min} = 100$, $\Delta = 400$, $p_{\min} = 20$, $p_{\max} = 80$:
>
> $$\text{2D: } \frac{300 - 180}{420 - 180} = 0.5 \qquad \text{3D: } \frac{0.5 - 0.2}{0.8 - 0.2} = 0.5 \quad \checkmark$$

### Auto-contrast interaction

When `auto_contrast` is enabled, `percentileClip(data, p_{\text{low}}, p_{\text{high}})` computes $v_{\min}, v_{\max}$ from data percentiles (e.g., 1st–99th), overriding the slider values. The slider handles visually snap to the percentile positions. When auto-contrast is off, the slider handles control the range directly.

> **Example.** Volume data with outlier hot pixels. 1st percentile = $120$, 99th percentile = $480$ (full range $[50, 10000]$). Auto-contrast sets $v_{\min} = 120$, $v_{\max} = 480$, ignoring the outliers. The slider handles move to $p_{\min} \approx 0.7\%$, $p_{\max} \approx 4.3\%$ to reflect these positions within the full range.

### Precedence

1. `auto_contrast` (if enabled): $v_{\min}, v_{\max}$ from `percentileClip()`, overrides sliders
2. Slider handles: $v_{\min}, v_{\max}$ via `sliderRange()` (2D) or $p/100$ (3D volume)
3. Default: $p_{\min} = 0,\; p_{\max} = 100$ — full data range, no clipping

## Array Compatibility

- All widgets accept NumPy, PyTorch, CuPy, and any `np.asarray()`-compatible object via duck typing (`array_utils.to_numpy()`).
- All widgets auto-extract metadata from quantem Dataset objects via duck typing (`hasattr` checks, no hard import dependency):
  - **Show2D**: `Dataset2d` → title, pixel_size (Å/nm auto-converted).
  - **Show3D**: `Dataset3d` → title, pixel_size (Å, with nm→Å conversion).
  - **Show3DVolume**: `Dataset3d` → title, pixel_size (Å/nm auto-converted).
  - **Show4D**: `Dataset4dstem` → nav_pixel_size, sig_pixel_size, units.
  - **Show4DSTEM**: `Dataset4dstem` → pixel_size (Å), k_pixel_size (mrad), unit detection.
  - **ShowComplex2D**: Dataset (generic) → title, pixel_size.
  - **Mark2D**: `Dataset2d` → extracts array, pixel_size.
  - **Edit2D**: `Dataset2d` → extracts array, pixel_size.
  - **Align2D**: `Dataset2d` → extracts array, pixel_size (Å, with nm→Å conversion). Accepts Dataset for either or both images.
- Explicit parameters always override auto-extracted values.
- Tests: `tests/test_widget_dataset_integration.py`.

## IO Module

`src/quantem/widget/io.py` — unified file loading for all widgets.

### IOResult

Lightweight dataclass returned by all IO methods. Widgets accept it via duck typing.

- `data: np.ndarray` — float32, 2D/3D/4D/5D
- `pixel_size: float | None` — in Å
- `units: str | None` — e.g. "Å", "nm"
- `title: str` — filename stem
- `labels: list[str]` — one per frame for stacks
- `metadata: dict` — raw metadata from file
- `frame_metadata: list[dict]` — per-frame metadata for 5D stacks (one dict per source file)

### Arina GPU Pipeline

- `IO.arina_file(path_or_list, det_bin, scan_bin, ...)` — single master.h5 → 4D, or list of paths → 5D
- `IO.arina_folder(folder_or_list, det_bin, ..., max_files, recursive, pattern)` — folder(s) → 5D stack
- GPU-accelerated decompression (bitshuffle+LZ4): Metal/MPS on macOS, CUDA on Linux, double-buffered IO/GPU overlap
- Shape validation: `arina_file(list)` raises ValueError on mismatch; `arina_folder` skips mismatched files with warning

### `IO.arina_folder` API

```python
IO.arina_folder(
    folder,              # str/Path or list of folders
    det_bin=1,           # detector binning (1/2/4/8 or "auto")
    scan_bin=1,          # scan-space binning
    scan_shape=None,     # override scan shape, e.g. (256, 256)
    hot_pixel_filter=True,
    backend="auto",      # "auto", "mps", "cuda", "cpu"
    max_files=50,        # cap number of files (0 = no limit)
    recursive=False,     # search subdirectories
    pattern=None,        # filter by filename, e.g. "SnMoS2"
    shape=None,          # filter by scan dims, e.g. (256, 256)
    skip=0,              # skip first N files
    dry_run=False,       # print memory table, don't load
    virtual_only=False,  # compute BF/ADF/HAADF, discard 4D
)
```

**Survey workflow** — screen all scans before committing to ptychography/SSB:

```python
# 1. Preview what's in the folder
IO.arina_folder(folder, det_bin=8, dry_run=True)

# 2. Lightweight survey: BF/ADF/HAADF only (~0.75 MB/file)
survey = IO.arina_folder(folder, virtual_only=True)
Show2D(survey, title="BF / ADF / HAADF", log_scale=True)

# 3. Deep dive into interesting scan
result = IO.arina_folder(folder, det_bin=4, skip=3, max_files=2)
Show4DSTEM(result)
```

**`virtual_only=True`** loads each file at full detector resolution, auto-detects BF disk center and radius (same algorithm as `Show4DSTEM.auto_detect_center`), computes BF/ADF/HAADF virtual images, and discards the 4D data immediately. Returns an IOResult with shape `(n_files * 3, scan_r, scan_c)` and labels like `["file1 BF", "file1 ADF", "file1 HAADF", ...]`. Detection uses `detector.py` internally — not exported at the top level.

**`dry_run=True`** reads only HDF5 headers (no data decompression), prints a table with per-file frame count, detector shape, binned shape, and estimated memory. Returns None.

**`skip=N`** skips first N files after sorting and pattern filtering, before `max_files` cap. Useful for resuming a survey: `skip=10, max_files=10` loads files 11–20.

All parameters compose: find files → filter by `pattern` → `skip` → cap at `max_files` → load (`virtual_only` or full 4D).

### Schema-Agnostic Metadata Design

**Critical**: HDF5 metadata is stored as-is from the file, with no fixed-schema parsing. Different Arina detectors at different facilities (Stanford, NCEM, etc.) may write different metadata fields. The metadata extraction walks the full HDF5 tree with `visititems()` and stores every scalar dataset keyed by its full HDF5 path. This means:

- Never assume specific metadata keys exist (no `meta["entry/instrument/detector/count_time"]` without `.get()`)
- Never parse or transform metadata values into a canonical schema
- `describe()` uses short-name matching (`count_time` matches `entry/instrument/detector/count_time`) for convenience
- `describe(diff=True)` (default) shows only columns where values differ across frames — useful for identifying what changed in a defocus/tilt series
- `describe(diff=False)` shows all matched columns including constants

### Memory Management

- `Show4DSTEM.free()` — deletes GPU tensor, runs `gc.collect()`, flushes GPU allocator cache (`torch.mps.empty_cache()` on macOS, `torch.cuda.empty_cache()` on Linux)
- GPU allocators cache freed buffers in a free-list; `del` alone does not return memory to the system
- Always call `free()` + `del result` when switching between large 4D-STEM datasets in a notebook session

## Code Style

- Python: no docstrings on internal methods, type hints on public API only.
- Python: use `Self` (from `typing`, PEP 673) for return-type annotations on methods that return `self`, not `"ClassName"` forward references.
- Python: use human-readable variable names — `center_row` not `cy`, `bf_radius` not `br`, `detector_rows` not `det_r`. Single-letter variables are only acceptable in tight math expressions or loop indices.
- JS/TS: no `any` types. Use strict TypeScript.
- Keep it simple. No over-engineering. No unnecessary abstractions.

## Widget State Protocol

Every widget implements a consistent state persistence protocol. This allows users to save interactive state (display settings, ROI positions, annotations) across kernel restarts and share exact configurations between colleagues via JSON files.

### Why state persistence matters
- **Kernel restarts**: Interactive work (placed points, ROI positions, display tuning) is lost when the kernel restarts. `save()` + `state` param preserves it.
- **Collaboration**: Share a JSON file alongside a notebook so colleagues see the exact same view — no need to manually recreate settings.
- **Reproducibility**: `state_dict()` captures the full display configuration for a figure, making analysis reproducible.
- **Inspection**: `summary()` gives a quick human-readable overview of the widget state without diving into trait values.

### Protocol methods (all widgets)
1. **`state_dict()`** → `dict` — Returns all user-configurable traits (display settings, ROI, annotations). Excludes image data and computed values (stats, bytes).
2. **`save(path)`** — Writes a versioned envelope JSON with `metadata_version`, `widget_name`, `widget_version`, and `state`.
3. **`load_state_dict(state)`** — Restores traits from a dict via `setattr`. Unknown keys are silently skipped.
4. **`state` param in `__init__`** — Accepts a `dict` or a file path (`str` / `pathlib.Path`). File-based state uses the versioned envelope format.
5. **`summary()`** — Prints a human-readable overview: image dimensions, display settings, ROI/annotation state, calibration.
6. **`__repr__`** — Compact one-liner: `WidgetName(shape, key_settings)`.

State JSON files are canonical versioned envelopes. Do not add legacy flat-dict save formats or legacy flat-file load paths.

### Why no formal Protocol/ABC
The protocols (state_dict, save, set_image, summary, __repr__) are enforced by tests (3+ per widget), not by a shared base class or `typing.Protocol`. Each widget is self-contained (one `.py` + one `.tsx`), and the implementation is ~15 lines of copy-paste. An ABC/Protocol would add an import dependency without reducing code. If the codebase grew to 20+ widgets or had external plugin authors, a formal Protocol would make sense. At the current number of widgets maintained internally, docs + tests are sufficient.

### Implementation pattern
```python
import json
import pathlib

from quantem.widget.json_state import save_state_file, unwrap_state_payload

class MyWidget(anywidget.AnyWidget):
    def __init__(self, data, ..., state=None, **kwargs):
        super().__init__(**kwargs)
        # ... all trait assignments and setup ...
        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = json.loads(pathlib.Path(state).read_text())
            state = unwrap_state_payload(state)
            self.load_state_dict(state)

    def state_dict(self):
        return {"trait_a": self.trait_a, "trait_b": self.trait_b, ...}

    def save(self, path: str):
        save_state_file(path, "MyWidget", self.state_dict())

    def load_state_dict(self, state):
        for key, val in state.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def summary(self):
        print(...)  # Human-readable formatted output

    def __repr__(self):
        return f"MyWidget(shape, key=value)"
```

### What goes in state_dict (and what doesn't)
- **Include**: all user-configurable display traits (cmap, log_scale, auto_contrast, show_fft, etc.), ROI parameters, annotation data (selected_points, profile_line), calibration overrides, playback settings.
- **Exclude**: image/volume data (frame_bytes, volume_bytes), computed stats (stats_mean, stats_min), internal state (_data, _buffer), playing/paused state.

### Tests (3 per widget)
- `test_<widget>_state_dict_roundtrip` — Create with settings, get state_dict, restore via `state` param, verify all values match.
- `test_<widget>_save_load_file(tmp_path)` — Save to JSON file, verify envelope fields (`metadata_version`, `widget_name`, `widget_version`, `state`), load from file path string.
- `test_<widget>_summary(capsys)` — Call summary(), verify output contains key information.

## set_image Protocol

All widgets implement `set_image()` (or `set_images()` for Align2D) to replace data while preserving display settings. This avoids recreating the widget when data changes.

| Widget | Method | Preserves | Resets |
|--------|--------|-----------|--------|
| Show1D | `set_data(data, x=None, labels=None)` | log_scale, auto_contrast, percentile_low/high, line_width, grid_density | colors, peak_fwhm |
| Show2D | `set_image(data, labels=None)` | cmap, log_scale, roi_list | selected_idx |
| Show3D | `set_image(data, labels=None)` | fps, loop, cmap | slice_idx |
| Show3DVolume | `set_image(data)` | cmap, log_scale | slice positions to center |
| Show4D | `set_image(data, nav_image=None)` | roi settings | nav range |
| Show4DSTEM | `set_image(data, scan_shape=None)` | roi/vi_roi config | caches, frame_idx |
| ShowComplex2D | `set_image(data)` | cmap, display_mode | stats recomputed |
| Mark2D | `set_image(data, labels=None)` | marker/roi settings | selected_points |
| Edit2D | `set_image(data, **kwargs)` | mode | crop bounds, mask |
| Align2D | `set_images(image_a, image_b, auto_align=True)` | cmap, opacity | dx/dy/rotation |
| Bin2D | `set_image(data, labels=None)` | cmap, log_scale, bin_factor | selected_idx |
| Bin4D | `set_data(data)` | cmap, log_scale, bin factors | shape traits, previews |

ShowComplex2D also accepts `(real, imag)` tuple input in both constructor and `set_image()`.

## save_image Protocol

Programmatic export API for saving widget views as PNG/PDF/TIFF from Python. Replicates the JS rendering pipeline (log scale, auto-contrast percentile clipping, colormap) in Python via matplotlib + Pillow.

| Widget | Method | Key params |
|--------|--------|------------|
| Show2D | `save_image(path, *, idx=None, format=None, dpi=150)` | `idx` for gallery mode |
| Show3D | `save_image(path, *, frame_idx=None, format=None, dpi=150)` | `frame_idx` for specific frame |
| Show3DVolume | `save_image(path, *, plane=None, slice_idx=None, format=None, dpi=150)` | `plane` = 'xy'/'xz'/'yz' |
| Show4D | `save_image(path, *, view=None, position=None, format=None, dpi=150)` | `view` = 'signal'/'nav' |
| Show4DSTEM | `save_image(path, *, view=None, position=None, ...)` | Full metadata sidecar support |
| ShowComplex2D | `save_image(path, *, display_mode=None, format=None, dpi=150)` | HSV uses vectorized `hsv_to_rgb` |
| Mark2D | `save_image(path, *, idx=None, include_markers=True, format=None, dpi=150)` | Renders point markers on export |
| Bin2D | `save_image(path, *, view=None, format=None, dpi=150)` | `view` = 'original'/'binned'/'grid' |
| Bin4D | `save_image(path, *, view=None, include_metadata=True, metadata_path=None)` | `view` = 'bf'/'adf'/'grid'; also `save_zip()`, `save_gif()` |

All return `pathlib.Path` of the written file. Format is inferred from extension if not specified.

## Line Profile Protocol

Widgets with line profiles share a consistent API:

- **Traits**: `profile_line` = `[{"row": float, "col": float}, {"row": float, "col": float}]`, `profile_width` (int, for thick profiles)
- **Methods**: `set_profile((row0, col0), (row1, col1))`, `clear_profile()`
- **Properties**: `profile` → `[(row0, col0), (row1, col1)]`, `profile_values` → `Float32Array`, `profile_distance` → float (calibrated if pixel_size set)
- **Widgets**: Show2D, Show3D, Show4D, Show4DSTEM, Mark2D

## ROI Protocol

Two ROI patterns exist depending on widget needs. Both use `(row, col)` coordinates.

**ROI shape order**: dropdown menus list shapes as `square → rectangle → circle → annular`. Default new ROI shape is `"square"`. This matches the most common microscopy workflow (square selection first, then refine to rectangle or circle).

### Multi-ROI list pattern (Show2D, Show3D, Mark2D)

For widgets where users draw/manage multiple independent ROIs:

- **Traits**:
  - `roi_active: Bool` — ROI mode on/off (Show2D, Show3D only; Mark2D always allows ROIs)
  - `roi_list: List` — list of ROI dicts
  - `roi_selected_idx: Int` — currently selected index, -1 = none (Show2D, Show3D only)
  - `roi_stats: Dict` — `{"mean": float, "min": float, "max": float, "std": float}` for selected ROI (Show3D only)
  - `roi_plot_data: Bytes` — ROI mean per frame as float32 (Show3D only)
- **ROI dict keys**: `{"row", "col", "radius", "radius_inner", "width", "height", "shape"}` where `shape` is `"circle"` | `"square"` | `"rectangle"` | `"annular"`
  - Mark2D extends with: `{"id", "shape", "color", "opacity", "width", "height"}`
- **Methods** (Show2D, Show3D): `add_roi()`, `clear_rois()`, `set_roi()`, `roi_circle()`, `roi_square()`, `roi_rectangle()`, `roi_annular()` — all return `Self`
- **Methods** (Mark2D): `add_roi(row, col, shape, radius, ...)`, `clear_rois()`, `roi_center()`, `roi_radius()`, `roi_size()`
- **Tests**: ROI shape creation, stats computation, clear, state_dict roundtrip

### Single-ROI mode pattern (Show4D, Show4DSTEM, ShowComplex2D, Show3DVolume)

For widgets where one ROI defines a reduction mask (virtual imaging, signal integration) or region-specific FFT:

- **Traits**:
  - `roi_mode: Unicode` — `"off"` | `"circle"` | `"square"` | `"rect"` | `"annular"` (+ `"point"` for Show4DSTEM)
  - `roi_center_row, roi_center_col: Float` — center position
  - `roi_center: List[Float]` — compound `[row, col]` for batched JS drag updates
  - `roi_radius, roi_radius_inner, roi_width, roi_height: Float` — geometry
  - `roi_reduce: Unicode` — `"mean"` | `"max"` | `"min"` | `"sum"` (Show4D only)
- **Methods** (Show4DSTEM): `roi_circle()`, `roi_square()`, `roi_annular()`, `roi_rect()`, `roi_point()` — all return `Self`
- **Methods** (ShowComplex2D): `roi_circle()`, `roi_square()`, `roi_rect()` — all return `Self`
- **Methods** (Show3DVolume): `roi_circle()`, `roi_square()`, `roi_rect()` — all return `Self`. ROI applies to XY slice only.
- **Show4DSTEM dual ROI**: also has scan-space VI-ROI (`vi_roi_mode`, `vi_roi_center_row/col`, `vi_roi_radius`, `vi_roi_width`, `vi_roi_height`) for summed diffraction pattern computation

### Widgets without ROI

Edit2D, Align2D — no ROI traits. Edit2D uses crop bounds instead.

### ROI FFT (Show2D, Show3D, Mark2D, Show4D, Show4DSTEM, ShowComplex2D, Show3DVolume)

When both ROI and FFT are active, the FFT shows only the cropped ROI region instead of the full image. This lets microscopists inspect local crystal structure by viewing the FFT of a specific area.

- **Activation**: derived flag `roiFftActive = effectiveShowFft && <roi_is_active>` — no extra trait needed.
- **Crop helper**: `cropROIRegion()` (multi-ROI widgets) or `cropSingleROI()` (single-mode widgets) extracts bounding-box region from raw float32 data. Circle/annular shapes zero-mask outside the radius before FFT.
- **Power-of-2 padding**: crop data is pre-padded to `nextPow2(cropWidth) × nextPow2(cropHeight)` before calling `fft2d`. This avoids truncation artifacts from `fft2d`'s internal padding.
- **State**: `fftCropDims = { cropWidth, cropHeight, fftWidth, fftHeight }` tracks crop vs padded dimensions. `cropWidth/Height` for the UI label, `fftWidth/Height` for all FFT processing and coordinate mapping.
- **Label**: green `accentGreen` text showing `"ROI FFT (W×H)"` when active, falls back to default FFT label when inactive.
- **No debouncing**: FFT recomputes on every ROI change (position, size, shape) in real-time. Never add `setTimeout`-based debouncing or throttling. Instead, use the **rAF generation counter** pattern: increment a generation counter, yield via `requestAnimationFrame` (lets browser paint the ROI update), then check the counter — stale requests are discarded. This coalesces rapid drag events to ≤1 FFT per display frame with zero artificial delay.
- **D-spacing**: click-to-measure uses `fftCropDims.fftWidth` for frequency bin computation. Accuracy scales with crop size (larger ROI = better resolution).
- **Fallback**: when ROI is deselected or disabled, reverts to full-image FFT immediately.

## Playback Protocol

Stack/animation widgets share a playback API. Two variants exist.

### Frame playback (Show3D, Show3DVolume)

- **Traits**:
  - `playing: Bool`, `fps: Float`, `loop: Bool`, `reverse: Bool`, `boomerang: Bool`
  - `loop_start, loop_end: Int` — frame range (Show3D only)
  - `play_axis: Int` — 0=Z, 1=Y, 2=X, 3=All (Show3DVolume only)
  - `bookmarked_frames: List[Int]`, `playback_path: List[Int]` — custom order (Show3D only)
- **Methods**: `play() → Self`, `pause() → Self`, `stop() → Self`
  - `goto(index) → Self`, `set_playback_path(path)`, `clear_playback_path()` — Show3D only
- **Prefetch buffer** (Show3D internal, not in state_dict): `_buffer_bytes`, `_buffer_start`, `_buffer_count`, `_prefetch_request` — 64-frame sliding window, 64MB cap
- **state_dict includes**: `fps`, `loop`, `reverse`, `boomerang`, `loop_start`, `loop_end`
- **state_dict excludes**: `playing` (always starts paused)

### 5D frame animation (Show4DSTEM)

For time/tilt series with independent frame dimension:

- **Traits**: `frame_idx`, `n_frames`, `frame_dim_label` (e.g. `"Time"`, `"Tilt"`), `frame_playing`, `frame_fps`, `frame_loop`, `frame_reverse`, `frame_boomerang`
- **No methods** — frame playback is JS-controlled via trait sync
- **state_dict includes**: `frame_idx`, `frame_dim_label`, `frame_fps`, `frame_loop`, `frame_reverse`, `frame_boomerang`

## Path Animation Protocol

Scan-space traversal for 4D widgets. Shared by Show4D and Show4DSTEM with identical API.

- **Traits**: `path_playing: Bool`, `path_index: Int`, `path_length: Int`, `path_interval_ms: Int`, `path_loop: Bool`
- **Internal**: `_path_points: List[Tuple[int, int]]` (not synced to JS)
- **Methods** (identical in both widgets):
  - `set_path(points, interval_ms=100, loop=True, autoplay=True) → Self` — points as `[(row, col), ...]`
  - `play() → Self`, `pause() → Self`, `stop() → Self`, `goto(index) → Self`
  - `raster(step=1, bidirectional=False, interval_ms=100, loop=True) → Self` — generates raster scan path
- **Observer**: `@self.observe("path_index")` → updates `pos_row`/`pos_col` → triggers frame update
- **GIF export**: path animation frames are exported via `_generate_gif()`

## Statistics Protocol

All canvas-based widgets compute and display image statistics.

### Common traits

- **`show_stats: Bool`** — toggle the stats bar below the canvas. Present in all widgets except Align2D.

### Stats format variants

| Widget | Trait(s) | Type | Scope |
|--------|----------|------|-------|
| Show1D | `stats_mean/min/max/std` | `List[Float]` | per-trace |
| Show2D | `stats_mean/min/max/std` | `List[Float]` | per-image (gallery) |
| Show3D | `stats_mean/min/max/std` | `Float` | current frame |
| Show3DVolume | `stats_mean/min/max/std` | `List[Float]` (3) | per slice (XY, XZ, YZ) |
| Show4D | `nav_stats`, `sig_stats` | `List[Float]` (4) | `[mean, min, max, std]` |
| Show4DSTEM | `dp_stats`, `vi_stats` | `List[Float]` (4) | `[mean, min, max, std]` |
| ShowComplex2D | `stats_mean/min/max/std` | `Float` | recomputed per display_mode |
| Mark2D | `stats_mean/min/max/std` | `List[Float]` | per-image |
| Edit2D | `stats_mean/min/max/std` | `Float` | current image |

### Naming convention

- Single-panel widgets: `stats_mean`, `stats_min`, `stats_max`, `stats_std` (4 scalar or list traits)
- Dual-panel widgets (Show4D, Show4DSTEM): `nav_stats`/`sig_stats` or `dp_stats`/`vi_stats` (single list trait with 4 elements: `[mean, min, max, std]`)

Stats are **excluded** from `state_dict()` — they are computed values, not user-configurable.

## Display Traits Protocol

Common display traits shared across widgets. Show4DSTEM is an intentional outlier (multi-panel decomposition).

### Standard display traits (all widgets except Show4DSTEM)

| Trait | Type | Default | Widgets |
|-------|------|---------|---------|
| `title` | `Unicode` | varies | all |
| `cmap` | `Unicode` | `"inferno"` | Show2D, Show3D, Show3DVolume, Show4D, ShowComplex2D, Mark2D, Edit2D, Align2D, Bin2D, Bin4D |
| `log_scale` | `Bool` | `False` | Show1D, Show2D, Show3D, Show3DVolume, Show4D, ShowComplex2D, Mark2D, Edit2D, Bin2D, Bin4D |
| `auto_contrast` | `Bool` | `False` | Show1D, Show2D, Show3D, Show3DVolume, Show4D, ShowComplex2D, Mark2D, Edit2D |
| `percentile_low` | `Float` | varies | Show1D (2.0), Show3D (1.0), Show4D (0.5), ShowComplex2D (1.0), Mark2D (2.0) |
| `percentile_high` | `Float` | varies | Show1D (98.0), Show3D (99.0), Show4D (99.5), ShowComplex2D (99.0), Mark2D (98.0) |
| `show_fft` | `Bool` | `False` | Show2D, Show3D, Show3DVolume, Show4D, ShowComplex2D, Mark2D |
| `fft_window` | `Bool` | `True` | Show2D, Show3D, Show3DVolume, Show4D, Show4DSTEM, ShowComplex2D, Mark2D |
| `show_stats` | `Bool` | `True` | all except Align2D |
| `show_controls` | `Bool` | `True` | Show1D, Show2D, Show3D, Show3DVolume, Show4D, Show4DSTEM, ShowComplex2D, Mark2D, Edit2D, Bin2D, Bin4D |
| `scale_bar_visible` | `Bool` | `True` | Show2D, Show3DVolume, ShowComplex2D |

### Show4DSTEM multi-panel display traits

Show4DSTEM decomposes display traits per panel (diffraction, virtual image, FFT):

| Standard trait | Show4DSTEM equivalent |
|----------------|----------------------|
| `cmap` | `dp_colormap`, `vi_colormap`, `fft_colormap` |
| `log_scale` | `dp_scale_mode`, `vi_scale_mode`, `fft_scale_mode` (`"linear"` / `"log"` / `"power"`) |
| `percentile_*` | `dp_vmin_pct`/`dp_vmax_pct`, `vi_vmin_pct`/`vi_vmax_pct`, `fft_vmin_pct`/`fft_vmax_pct` (0-100) |
| — | `dp_power_exp`, `vi_power_exp`, `fft_power_exp` (power law exponent) |
| `show_fft` | `show_fft` (same) |
| — | `mask_dc`, `dp_show_colorbar`, `fft_auto` |

### Pixel size naming

All widgets use `pixel_size` in Å/px. Show4D and Show4DSTEM have additional calibration traits:

| Trait name | Unit | Widgets |
|------------|------|---------|
| `pixel_size` | Å/px | Show2D, Show3D, Show3DVolume, Show4DSTEM, ShowComplex2D, Mark2D, Edit2D, Align2D |
| `nav_pixel_size`, `sig_pixel_size` | user unit (with `nav_pixel_unit`, `sig_pixel_unit`) | Show4D |
| `k_pixel_size` | mrad/px | Show4DSTEM |

### What goes in state_dict

All display traits above are **included** in `state_dict()`. Image data (`frame_bytes`, `volume_bytes`) and computed stats are **excluded**.

## Protocol Overview

Summary of which widgets implement which protocols:

| Protocol | Show1D | Show2D | Show3D | Show3DVolume | Show4D | Show4DSTEM | ShowComplex2D | Mark2D | Edit2D | Align2D | Bin2D | Bin4D |
|----------|--------|--------|--------|-------------|--------|-----------|--------------|--------|--------|---------|------|------|
| State persistence | x | x | x | x | x | x | x | x | x | x | x | x |
| set_data/set_image | x*** | x | x | x | x | x | x | x | x | x* | x | x**** |
| Line profile | | x | x | | x | x | | x | | | | |
| ROI (multi-list) | | x | x | | | | | x | | | | |
| ROI (single-mode) | | | | x | x | x | x | | | | | |
| Playback | | | x | x | | x** | | | | | | |
| Path animation | | | | | x | x | | | | | | |
| Export figure | x | x | x | x | x | x | x | x | | | x | |
| GIF/ZIP export | | | x | x | x | x | | | | | | x |
| Tool lock/hide | | x | x | x | x | x | x | x | x | x | x | x |

\* Align2D uses `set_images(image_a, image_b)`
\*\* Show4DSTEM has 5D frame playback (separate from path animation)
\*\*\* Show1D uses `set_data()` instead of `set_image()`, plus `add_trace()`, `remove_trace()`, `clear()`
\*\*\*\* Bin4D uses `set_data()` instead of `set_image()`

## Export Figure Protocol

JS-side publication-quality figure export via `exportFigure()` in `js/scalebar.ts`:
- Renders image with title, colorbar (gradient + labels), scale bar, and custom annotations onto a white-background canvas
- Available in: Show2D, Show3D, Show3DVolume, Show4D, Show4DSTEM, ShowComplex2D, Mark2D

## Export Dropdown Pattern

All widgets use a single **Export dropdown button** (MUI Menu) instead of separate Export/Figure buttons:
- **Export** button opens a dropdown menu. Default figure export format is **PDF** (vector, Illustrator/Inkscape-ready). PNG is secondary.
- Menu items (in order): "PDF + colorbar", "PDF", "PNG", then stack-specific items. Show2D has "PDF + scalebar + colorbar", "PDF + scalebar", "PDF".
- **Copy** stays as a separate button (clipboard copy of the current/focused canvas as PNG).
- Stack widgets (Show3D, Show3DVolume, Show4D, Show4DSTEM) add extra menu items: "ZIP (all frames)", "GIF"
- **Show3DVolume Copy**: copies the currently focused slice panel (not all three). Uses `lastInteractedAxis` ref to track which slice the user last clicked/hovered.
- Implementation: `exportAnchor` state + `<Menu>` / `<MenuItem>`, handlers call `setExportAnchor(null)` first to close menu
- Reference pattern: Show3D `js/show3d/index.tsx`

## GIF/ZIP Export Protocol

Stack/animation widgets support GIF and ZIP export for frame sequences:
- Python-side rendering: `_normalize_frame()` applies colormap + contrast, `_generate_gif()` / `_generate_zip()` assembles frames
- Triggered via `_gif_export_requested` trait (bool), result returned via `_gif_export_data` / `_gif_data` trait (bytes)
- JS downloads the result via `downloadDataView()`
- Show3D/Show3DVolume: support frame range (loop_start/loop_end), axis selection
- Show4D/Show4DSTEM: export path animation frames

### Export metadata JSON schema (all metadata-bearing exports)

When an export writes `metadata.json` (ZIP) or a sidecar JSON, include:
- `metadata_version` (current `"1.0"`)
- `widget_name`
- `widget_version`
- `format` (`png`/`pdf`/`zip`)
- `export_kind` (explicit operation name, e.g. `single_view_image`, `multi_panel_bundle`)

Prefer canonical coordinate/shape keys: `position` (`row`,`col`), `scan_shape`, `detector_shape`, and structured `display`/`calibration` blocks.

For GIF trait-based exports, Python must also emit `_gif_metadata_json`, and frontend should download a same-prefix `.json` sidecar with the GIF.

## Show4DSTEM Programmatic Export Protocol

`Show4DSTEM.save_image()` is the Python-side export API for deterministic, scriptable output.

- **Canonical views**: `diffraction`, `virtual`, `fft`, `all`
- **Formats**: `png`, `pdf` (inferred from file extension when `format=None`)
- **State overrides**: `position=(row, col)` and `frame_idx` can be provided for one-off exports; with `restore_state=True`, widget state is restored after writing.
- **Display parity**: export uses synced Python traits (`dp_*`, `vi_*`, `fft_*`, `show_fft`) so saved output matches interactive colormap/scale/percentile settings.

### JSON sidecar schema

When `include_metadata=True`, write a sidecar JSON (`<image>.json` unless `metadata_path` is provided).

- Required versioning fields:
  - `metadata_version` (current: `"1.0"`)
  - `widget_name` (`"Show4DSTEM"`)
  - `widget_version` (`quantem.widget.__version__`)
- Required content:
  - `view`, `format`, `path`
  - `position`, `frame_idx`, `n_frames`
  - `scan_shape`, `detector_shape`
  - `roi`, `vi_roi`
  - `calibration` (including explicit units, e.g. `pixel_size_unit`, `k_pixel_size_unit`)
  - `display` (exact render parameters including colormap, scale mode, vmin/vmax %, and resolved `vmin`/`vmax`)

If the JSON schema changes, bump `metadata_version`.

## Versioning

- Follow [PEP 440](https://packaging.python.org/en/latest/discussions/versioning/) semantic versioning: `major.minor.patch` (e.g., `0.4.0`).
- Major = breaking API changes, minor = new features/widgets, patch = bug fixes.
- Pre-release suffixes when needed: `0.5.0a1` (alpha), `0.5.0b1` (beta), `0.5.0rc1` (release candidate).
- Version lives in `pyproject.toml` (`version = "X.Y.Z"`). Single source of truth — no other files to update.
- Always bump the version before tagging. TestPyPI rejects duplicate versions.

## Publishing

- Publish to TestPyPI via GitHub Actions: push a version tag (`git tag vX.Y.Z && git push origin vX.Y.Z`).
- CI workflow (`.github/workflows/publish.yml`) builds JS, builds the Python package, and uploads to TestPyPI.
- After CI finishes, verify locally: `./scripts/verify_testpypi.sh X.Y.Z` (creates fresh conda env, installs from TestPyPI, opens JupyterLab for visual inspection).
- In docs and install instructions, use `widget-env` as the conda environment name (not abbreviations like `qw`). Be explicit for users who are not familiar with conda.
- Default Python version in install instructions: `python=3.13` (latest stable). Update when a newer stable release is available.

## Changelog

- `docs/changelog.md` — user-facing, feature-focused.
- Write what users can **do** now, not internal implementation details.
- No internal cleanup, refactoring, test harness changes, or dead code removal — those are not features.
- Breaking changes get a `**breaking:**` prefix with migration instructions.
- Group by widget name; use "New widgets" heading for brand-new widgets.
- One bullet per feature, keep it short.

## GitHub Issues

Use this template for all issues:

```markdown
## Problem

[What's wrong, missing, or limiting — concrete, not vague]

## Proposed Solution

[How to fix it — API examples, trait names, which files to change]

## Files

- `path/to/file.py` — what changes here
```

- Lead with **Problem** (what the user can't do or what's broken), then **Proposed Solution** (concrete design with code examples).
- No "Options A/B" — pick one approach. If genuinely uncertain, ask before filing.
- Include `## Files` listing affected files so the scope is clear.
- Keep it short. One screen, not a design doc.

## Git

- Commit messages: one line, lowercase, no "Co-Authored-By" trailer.
