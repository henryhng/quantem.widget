# GPU Optimization Case Study: Show2D & Show3D

How we made Show2D and Show3D the fastest browser-based electron microscopy
viewers — going from 580ms to **7ms** colormap changes on 12 x 4096x4096
images using WebGPU compute shaders and zero-copy OffscreenCanvas rendering.

This guide is written for widget developers who want to apply the same
patterns to other widgets (Show4D, Show4DSTEM, etc.) or build new
GPU-accelerated Jupyter widgets.

## The Problem

Show2D displays a gallery of STEM HAADF images — typically 12-48 images
at 4096x4096 pixels each (768 MB - 3 GB of float32 data). Every time the
user changes a colormap, drags a histogram slider, toggles log scale, or
enables auto-contrast, the widget must:

1. Apply a colormap LUT to every pixel (201 million pixels for 12 images)
2. Render the result to canvas
3. Update the histogram display

Before optimization, this took **580ms per colormap change** on a MacBook
M5 — visible lag that breaks the interactive experience.

## Architecture Overview

```
Python (data)          JS (rendering)              GPU (compute)
─────────────         ─────────────────           ─────────────────
float32 data  ──────> extractFloat32()  ──────>  uploadData()
                      (DataView → F32)            (GPU storage buffer)

trait change  ──────> React effect      ──────>  renderSlots()
(cmap/log/...)        (computes vmin/vmax)        (WGSL shader)
                                                       │
                                        ◄──────  mapAsync + copy
                                        ImageData ← GPU mapped memory
                                        putImageData → offscreen canvas
                                              │
                                        drawImage → visible canvas
```

The GPU pipeline has three stages:
1. **Data upload** (once, on data change): float32 → GPU storage buffer
2. **Compute** (every interaction): WGSL shader applies colormap, log scale, contrast
3. **Readback** (every interaction): GPU → CPU → canvas

## The Six Optimizations

### 1. GPU Colormap Shader (580ms → 100ms)

**Before:** CPU loop iterating 201M pixels, doing clamp + divide + LUT lookup per pixel.

**After:** WGSL compute shader running on GPU with 16x16 workgroups.

```wgsl
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  var val = data[idx];
  if (params.log_scale == 1u) { val = log(1.0 + max(val, 0.0)); }
  let range = max(params.vmax - params.vmin, 1e-30);
  let clipped = clamp(val, params.vmin, params.vmax);
  let t = (clipped - params.vmin) / range;
  let lutIdx = min(u32(t * 255.0), 255u);
  let rgb = lut[lutIdx];
  rgba[idx] = rgb | 0xFF000000u;
}
```

**Key design decisions:**

- **2D dispatch** (`workgroup_size(16, 16)`): Stays within the 65535 workgroup
  limit. 1D dispatch with `workgroup_size(256)` would need 65536 workgroups
  for 4096x4096 — exceeds the limit by 1 and silently fails.

- **Log scale in shader**: The `log_scale` uniform flag means log toggle doesn't
  require re-uploading data. The shader computes `log(1 + max(val, 0))` per pixel.
  This eliminated a 1049ms CPU log1p transform.

- **LUT packed as u32**: The 256-entry RGB LUT is packed as `R | (G << 8) | (B << 16)`,
  and the shader adds alpha with `rgb | 0xFF000000u`. One memory read per LUT lookup.

### 2. Zero-Copy Readback (`renderSlots`)

**Before:** GPU compute → `mapAsync` → allocate 768MB `Uint8ClampedArray` → copy from
mapped memory → copy to `ImageData` → `putImageData`. Two copies of 768 MB.

**After:** GPU compute → `mapAsync` → write directly from mapped memory into `ImageData` →
`putImageData`. One copy of 768 MB.

```typescript
// Before: 2 copies (580ms)
const rgba = new Uint8ClampedArray(count * 4);    // 64 MB allocation
rgba.set(new Uint8ClampedArray(mapped));           // copy 1: GPU → temp
imgData.data.set(rgba);                            // copy 2: temp → ImageData

// After: 1 copy (100ms)
imgData.data.set(new Uint8ClampedArray(mapped));   // direct: GPU → ImageData
offscreen.getContext("2d")!.putImageData(imgData, 0, 0);
```

This eliminated 768 MB of intermediate allocation per render.

### 3. Persistent GPU Buffers

**Before:** Every `applySlots` call created and destroyed 24 GPU buffers (12 params +
12 read buffers). Buffer creation is expensive in WebGPU.

**After:** Each image slot has persistent buffers created once in `uploadData`:

```typescript
slots[idx] = {
  dataBuffer,      // float32 input (persistent)
  rgbaBuffer,      // RGBA output (persistent)
  readBuffer,      // MAP_READ for CPU readback (persistent)
  paramsBuffer,    // 24-byte uniform (persistent, just writeBuffer)
  histBinsBuffer,  // 256-bin histogram (persistent)
  histReadBuffer,  // histogram readback (persistent)
};
```

Only `device.queue.writeBuffer(paramsBuffer, ...)` on each render — no buffer
creation or destruction.

### 4. GPU Histogram for Auto-Contrast (768ms → 272ms first, 4ms cached)

**Before:** CPU `percentileClip` scanned 201M floats to build a histogram, then
computed percentiles. With log scale, also did 201M `log1p` transforms.

**After:** WGSL histogram shader using `atomicAdd`:

```wgsl
@group(0) @binding(2) var<storage, read_write> bins: array<atomic<u32>>;

@compute @workgroup_size(16, 16)
fn histogram(@builtin(global_invocation_id) gid: vec3u) {
  // ... normalize val to [0, 1], compute bin index ...
  let bin = min(u32(t * 256.0), 255u);
  atomicAdd(&bins[bin], 1u);
}
```

The JS derives percentiles from the 256-bin histogram (scanning 256 values,
not 201M):

```typescript
// Percentile from GPU histogram bins
let running = 0;
for (let b = 0; b < 256; b++) {
  running += bins[b];
  if (running >= targetLow) binLow = b;
  if (running >= targetHigh) { binHigh = b; break; }
}
vmin = dataMin + (binLow / 255) * range;
vmax = dataMin + (binHigh / 255) * range;
```

**Caching:** The auto-contrast effect depends on `[autoContrast, dataVersion, logScale]`.
Colormap changes do NOT recompute histograms — they use cached percentile ranges.

### 5. Auto-Bin for Large Galleries (OOM → works)

**Before:** 24+ images at 4K crashed the GPU (6+ GB of buffers).

**After:** Python auto-bins the display data to fit a GPU memory budget:

```python
_GPU_DISPLAY_BUDGET_MB = 2500  # fits 12×4K, auto-bins above

if display_bin == "auto":
    per_image_mb = (H * W * 4 * 3) / (1024 * 1024)  # 3 buffers per image
    total_mb = n_images * per_image_mb
    if total_mb > gpu_budget_mb:
        for bf in [2, 4, 8]:
            if total_mb / (bf * bf) <= gpu_budget_mb:
                self._display_bin = bf
                break
```

| Images | Source | Display | GPU Memory | Colormap |
|--------|--------|---------|------------|----------|
| 12 | 4096x4096 | 4096x4096 (full) | 2.3 GB | 100ms |
| 24 | 4096x4096 | 2048x2048 (bin 2x) | 1.2 GB | 77ms |
| 48 | 4096x4096 | 1024x1024 (bin 4x) | 0.6 GB | 37ms |
| 100 | 4096x4096 | 1024x1024 (bin 4x) | 1.2 GB | ~80ms |

Full-resolution data stays in Python for export, statistics, and analysis.
The `display_bin` parameter lets users override: `"auto"` (default), `1`
(force full-res), or `2`/`4`/`8` (force specific bin).

### 6. Zero-Copy Rendering via OffscreenCanvas (100ms → 7ms)

**Before:** GPU compute → `mapAsync` (10ms) → JS `Uint8ClampedArray.set` (35ms)
→ `putImageData` (5ms). The 35ms JS memcpy is at the hardware limit — 768 MB
at ~20 GB/s V8 engine throughput. Can't make it faster.

**After:** GPU compute → render pass → `OffscreenCanvas` texture →
`transferToImageBitmap()` → `drawImage()`. Zero CPU readback.

```
Before (mapAsync path):
  GPU compute        50ms
  mapAsync           10ms
  JS memcpy 768MB    35ms  ← hardware limit, can't reduce
  putImageData        5ms
  ─────────────────────
  Total             100ms

After (zero-copy path):
  GPU compute        5ms   ← same shader, but no readback fence
  render pass → OC   1ms   ← blit to OffscreenCanvas texture
  transferToImageBitmap 0ms ← browser internal, zero-copy
  drawImage           1ms  ← GPU texture → 2D canvas (composited)
  ─────────────────────
  Total               7ms  ← 14× faster
```

**How it works:**

1. The compute shader writes RGBA to a GPU storage buffer (same as before).
2. A fullscreen-triangle render pass blits the RGBA buffer to an
   `OffscreenCanvas` that has a `webgpu` context.
3. `transferToImageBitmap()` returns an `ImageBitmap` — this is a zero-copy
   operation that transfers ownership of the GPU texture to the bitmap.
4. `drawImage(bitmap)` on the visible 2D canvas composites the GPU texture.

```typescript
// The key method: renderSlotsToImageBitmap
const oc = new OffscreenCanvas(width, height);
const ctx = oc.getContext("webgpu") as GPUCanvasContext;
ctx.configure({ device, format, alphaMode: "opaque" });

// ... compute pass (same as renderSlots) ...
// ... render pass blits RGBA buffer → OffscreenCanvas texture ...

device.queue.submit([encoder.finish()]);
const bitmap = oc.transferToImageBitmap();  // zero-copy!

// Draw on visible 2D canvas
offscreen2d.getContext("2d").drawImage(bitmap, 0, 0);
```

**Why OffscreenCanvas, not in-DOM canvas?** A `<canvas>` element can only have
one context type (`"2d"` or `"webgpu"`, not both). The visible canvases need
`"2d"` for zoom/pan `drawImage` transforms. `OffscreenCanvas` is created
off-DOM, gets a `"webgpu"` context, and `transferToImageBitmap()` bridges the
two worlds. Supported in Chrome 94+ (all modern browsers).

**Fallback:** If `OffscreenCanvas` WebGPU is not available (older browsers),
falls back to the `renderSlots` path (mapAsync + memcpy, ~100ms).

## Optimization Strategy Summary

The six optimizations form a pipeline, each eliminating a different bottleneck:

| # | Optimization | What it eliminates | Speedup |
|---|-------------|-------------------|---------|
| 1 | GPU colormap shader | CPU per-pixel loop | 580ms → 100ms |
| 2 | Zero-copy readback (`renderSlots`) | Intermediate 768MB array | 580ms → 100ms |
| 3 | Persistent GPU buffers | Buffer create/destroy per frame | ~50ms overhead |
| 4 | GPU histogram batch | CPU percentile scan | 768ms → 8ms cached |
| 5 | Auto-bin for large galleries | GPU OOM on 24+ images | OOM → works |
| 6 | OffscreenCanvas zero-copy | mapAsync + JS memcpy | 100ms → 7ms |

**Total: 580ms → 7ms = 82× faster.** Log toggle: 1049ms → 4ms = 262× faster.

The strategy at each step was:
1. **Profile** — measure exactly where time goes (GPU compute, mapAsync, memcpy)
2. **Identify the bottleneck** — which step dominates?
3. **Eliminate the copy** — every optimization removes a data copy or transfer
4. **Test on real data** — Chrome CDP with real 4K EMD files, not synthetic benchmarks
5. **Measure again** — verify the improvement with numbers

## Timing Breakdown

Where the time goes for 12 x 4096x4096 colormap change:

**mapAsync path (optimization #2, before zero-copy):**

```
Step                    Time    Notes
──────────────────      ─────   ─────────────────────────────────
GPU compute             ~50ms   WGSL shader, 201M pixels (includes 15ms API overhead)
mapAsync                ~10ms   Apple unified memory cache flush
JS memcpy (768 MB)      ~35ms   Hardware limit: V8 typed array at ~20 GB/s
putImageData            ~5ms    Browser internal
──────────────────      ─────
Total                   ~100ms  (1.4× theoretical minimum of 70ms)
```

**Zero-copy path (optimization #6, current):**

```
Step                    Time    Notes
──────────────────      ─────   ─────────────────────────────────
GPU compute             ~5ms    Same shader, no readback fence needed
Render pass → OC        ~1ms    Blit to OffscreenCanvas texture
transferToImageBitmap   ~0ms    Zero-copy ownership transfer
drawImage               ~1ms    GPU texture → 2D canvas composite
──────────────────      ─────
Total                   ~7ms    (14× faster, eliminates all CPU copies)
```

The GPU compute dropped from 50ms to 5ms because without `mapAsync` there is
no readback fence — the GPU pipeline runs fully asynchronous. The 35ms JS
memcpy is eliminated entirely by `transferToImageBitmap()` which transfers GPU
texture ownership without copying.

## What NOT to Do

These patterns were discovered through bugs during development:

1. **Never use React state for GPU readiness flags.** Use refs instead.
   `gpuCmapReadyRef` (ref) doesn't trigger re-renders. `setGpuReady(true)`
   (state) causes effects to re-fire → double computation → images go black.

2. **Never use generation counters to discard GPU results.** Stale renders are
   always better than blank canvases. The `requestAnimationFrame` pattern
   naturally coalesces rapid events to 1 per frame.

3. **Never call `setState` from GPU init callbacks.** GPU init only sets refs and
   uploads data. The CPU-rendered first frame is visually identical. GPU kicks
   in on the next user interaction (or via the warm-up render).

4. **Never debounce FFT or colormap.** Use the rAF generation counter pattern:
   increment a counter, yield via `requestAnimationFrame`, check the counter
   after yield. Stale generations are discarded. Zero artificial delay.

5. **Watch 2D dispatch limits.** `ceil(4096 * 4096 / 256) = 65536` exceeds
   WebGPU's 65535 `maxComputeWorkgroupsPerDimension`. Use 2D dispatch
   `@workgroup_size(16, 16)` instead: `ceil(4096/16) = 256` per dimension.

## Testing GPU Code

Headless Playwright has no WebGPU. The CPU fallback path always passes smoke
tests — you'll miss GPU bugs.

**The real test is Chrome CDP** (Chrome DevTools Protocol):

```bash
# Per-widget GPU test:
python tests/gpu/run.py show2d --build --scale

# Structure:
tests/gpu/
  cdp.py              # Shared: CDPConnection, KernelConnection, benchmarks
  test_show2d.py      # Show2D: colormap, log, auto-contrast, scale, histogram cache
  test_show3d.py      # Show3D: (to be added)
  run.py              # Runner: python tests/gpu/run.py [widget] [--build] [--scale]
```

The test connects to real Chrome with real WebGPU, executes real notebook cells
with real EMD data, changes colormaps via kernel API, reads GPU timing from
console logs, and verifies pixel output. See `docs/dev/testing.md` for setup.

## Applying to Other Widgets

Every widget that uses `renderToOffscreenReuse` or `applyColormap` can benefit
from the same GPU pipeline. The pattern:

### Step 1: Import and initialize

```typescript
import { getGPUColormapEngine, GPUColormapEngine } from "../colormaps";

const gpuCmapRef = React.useRef<GPUColormapEngine | null>(null);
const gpuCmapReadyRef = React.useRef(false);

React.useEffect(() => {
  getGPUColormapEngine().then(engine => {
    if (engine) {
      gpuCmapRef.current = engine;
      gpuCmapReadyRef.current = true;
    }
  });
}, []);
```

### Step 2: Upload data when it changes

```typescript
const engine = gpuCmapRef.current;
if (engine && gpuCmapReadyRef.current) {
  engine.uploadData(0, floatData, width, height);
  engine.uploadLUT(cmap, COLORMAPS[cmap]);
}
```

### Step 3: Render with GPU, fall back to CPU

```typescript
if (engine && gpuCmapReadyRef.current) {
  requestAnimationFrame(async () => {
    const rendered = await engine.renderSlots(
      [0], [{ vmin, vmax }], [offscreen], [imgData], logScale
    );
    if (rendered === 0) {
      // CPU fallback
      renderToOffscreenReuse(data, lut, vmin, vmax, offscreen, imgData);
    }
  });
} else {
  renderToOffscreenReuse(data, lut, vmin, vmax, offscreen, imgData);
}
```

### Step 4: Add GPU histogram (optional, for auto-contrast)

```typescript
const bins = await engine.computeHistogramBatch(indices, ranges, logScale);
// Derive percentiles from 256 bins (instant, no data scan)
```

### Step 5: Add GPU test

Create `tests/gpu/test_mywidget.py` using the shared `tests/gpu/cdp.py`
infrastructure. Register in `tests/gpu/run.py`.

## Performance Reference

All numbers measured on **MacBook Pro M5, 24 GB RAM** with real 4096×4096
STEM HAADF gold nanoparticle EMD data via Chrome CDP (not synthetic benchmarks).
Data source: `20260409_gold_drift_v3/` — 12 Velox EMD files, 4096×4096 uint16.

### Show2D (gallery viewer)

```
Show2D: 12 × 4096×4096 images (768 MB float32)

Operation                     CPU (before)   GPU+mapAsync   Zero-copy    Total speedup
─────────────────────────     ────────────   ────────────   ─────────    ─────────────
Colormap change (warm)        580ms          100ms          7ms          82×
Log scale toggle              1049ms         132ms          4ms          262×
Auto-contrast ON              768ms          272ms          13ms         59×
Auto-contrast (cached)        768ms          4ms            4ms          192×
Histogram slider drag         580ms          100ms          7ms          82×
24 images                     OOM            77ms           —            ∞
48 images                     CRASH          37ms           —            ∞
```

**Timing breakdown (12×4K colormap change, zero-copy path):**

| Step | Time | Notes |
|------|------|-------|
| GPU compute (WGSL shader) | ~5ms | 201M pixels, no readback fence |
| Render pass → OffscreenCanvas | ~1ms | Blit fullscreen triangle |
| transferToImageBitmap | ~0ms | Zero-copy GPU texture transfer |
| drawImage to 2D canvas | ~1ms | Browser GPU composite |
| **Total** | **~7ms** | **82× faster than CPU** |

**Show2D scaling (auto-bin, zero-copy):**

| Images | Source | Display | GPU Memory | Colormap |
|--------|--------|---------|------------|----------|
| 12 | 4096×4096 | 4096×4096 (full) | 2.3 GB | 7ms |
| 24 | 4096×4096 | 2048×2048 (bin 2×) | 1.2 GB | 77ms |
| 48 | 4096×4096 | 1024×1024 (bin 4×) | 0.6 GB | 37ms |

### Show3D (stack viewer)

```
Show3D: 12 frames × 4096×4096 (768 MB float32)

                        Binned 2K         No-bin 4K
                        (16 MB/frame)     (64 MB/frame)
────────────────────    ──────────────    ──────────────
Frame scrub (avg)       22ms              ~60ms
Frame scrub (min)       5ms               ~30ms
Colormap change         6-12ms            8-12ms
Log toggle              5-8ms             7-8ms
GPU compute             19ms              19ms (same)
GPU → canvas copy       1ms               4ms
Python trait sync       15ms              33ms
```

**Show3D scaling (frame count does NOT affect per-frame latency):**

| Frames | Display | Frame Scrub | Colormap | Constructor |
|--------|---------|-------------|----------|-------------|
| 12 | 2048×2048 (bin 2×) | 22ms | 6ms | ~800ms |
| 48 | 2048×2048 (bin 2×) | 22ms | 11ms | ~4.8s |
| 96 | 2048×2048 (bin 2×) | 22ms | — | ~15s |

**Show3D multi-panel (side-by-side comparison):**

| Config | Display | Frame Scrub | Constructor |
|--------|---------|-------------|-------------|
| 2 panels × 12×4K | 1024×2050 (bin 4×) | 3ms | 698ms |
| 3 panels × 12×4K | 1024×3076 (bin 4×) | 3ms | 1716ms |

### Where the time goes

For a single Show3D frame at 2K (the interactive path):

```
Python trait sync:  15ms  ┐
JS parse Float32:    1ms  │  Framework overhead (can't reduce)
                          │
GPU compute:        19ms  ┤  WebGPU overhead: 15ms fence + 4ms shader
GPU → canvas:        1ms  ┘  Actual shader: ~4ms for 4M pixels

Total:              36ms → 28 fps
```

**Framework limits:**
- Python→JS trait sync: ~0.5 ms per MB (traitlets/anywidget comm protocol)
- WebGPU submission: ~15ms fixed overhead per `queue.submit()` (fence + validation)
- JS typed array copy: ~20 GB/s (V8 engine limit)

**To reach 60fps:** would require eliminating the per-frame trait sync by
pre-uploading all frames to GPU storage buffers at init. Frame changes would
then be a uniform update (~1ms) instead of a 15-33ms data transfer.

### Hardware Compatibility

| Hardware | GPU Memory | 12×4K Gallery | Show3D 4K | Notes |
|----------|-----------|---------------|-----------|-------|
| MacBook M5 24GB | Shared ~6GB | 100ms | 22ms/frame | All numbers above |
| MacBook M1/M2 8GB | Shared ~3GB | ~120ms | ~25ms/frame | Budget may need tuning |
| NVIDIA 4GB (microscope) | 3.5GB dedicated | ~100ms | ~22ms/frame | Auto-bin at 12+ images |
| NVIDIA 8GB+ | 7GB+ dedicated | ~80ms | ~20ms/frame | More headroom |
| No WebGPU (old browser) | N/A | CPU 580ms | CPU ~150ms | Still works, just slower |

The auto-bin budget (`_GPU_DISPLAY_BUDGET_MB = 2500`) is conservative enough
for 4GB NVIDIA GPUs. The `_gpu_max_buffer_mb` trait reports the GPU's actual
limit from JS, which can be used for finer tuning in the future.

### Auto-bin decision table

| Widget | Condition | Bin Factor | Display Size |
|--------|-----------|------------|--------------|
| Show2D | Gallery GPU budget > 2.5 GB | 2×, 4×, 8× | Per image |
| Show3D | Frame > 32 MB (> ~2800×2800) | 2×, 4×, 8× | Per frame |
| Show3D multi-panel | Combined width × height × 4 > 32 MB | 2×, 4×, 8× | Per panel |

Users can override with `display_bin=1` (force full-res) or `display_bin=N`
(force specific factor). Default is `display_bin="auto"`.

## Interaction Audit

Every user interaction categorized by latency. Measured on MacBook M5 24 GB
with 12 × 4096×4096 real STEM HAADF EMD data. Use this as a checklist when
optimizing — anything in SLOW needs work.

### Show2D (gallery viewer)

**FAST (<16ms) — feels instant, 60fps:**

| Interaction | Time | How |
|------------|------|-----|
| Colormap change | 7ms | Zero-copy GPU (OffscreenCanvas → ImageBitmap) |
| Log scale toggle | 4ms | GPU shader log1p, no JS data transform |
| Auto-contrast (cached) | 4ms | GPU histogram cached across cmap changes |
| Zoom / pan | <1ms | CSS transform only, no pixel recompute |
| Gallery image select | <1ms | Border highlight, no recompute |
| Scale bar update | <1ms | Overlay canvas redraw |

**OK (16–100ms) — noticeable but acceptable:**

| Interaction | Time | Bottleneck |
|------------|------|-----------|
| Histogram slider drag | ~17ms | 7ms GPU + ~10ms React effect overhead |
| Auto-contrast ON (first) | ~285ms | 272ms GPU histogram batch + 13ms render |
| FFT toggle (single 4K) | ~400ms | WebGPU FFT on 16M complex points |
| ROI drag | ~7ms | GPU re-render only (no FFT recompute unless ROI-FFT active) |
| Profile drag | ~1ms per image | CPU bilinear sampling along line (fast, <100 samples) |

**SLOW (>100ms) — user waits:**

| Interaction | Time | Root cause |
|------------|------|-----------|
| FFT toggle (12 images) | ~5.7s | 12 × 4K FFT, progressive batch-4 |
| `set_image` (24 imgs) | ~5.7s | 384 MB trait sync (auto-binned from 1.5 GB) |
| Constructor (12×4K) | ~3s | Stats + 768 MB trait sync |

**Known remaining bottlenecks:**

- FFT on 48 images: ~23s (batch-4 progressive, but total time is long)
- ROI-FFT recompute on ROI drag: one 4K FFT per drag event (~400ms)
- Export to PDF: Python-side matplotlib rendering (~500ms per image)

### Show3D (stack viewer)

**FAST (<16ms) — feels instant:**

| Interaction | Time | How |
|------------|------|-----|
| Colormap change | 4–5ms | Zero-copy GPU |
| Log scale toggle | 5ms | Zero-copy GPU |
| Zoom / pan | <1ms | CSS transform |

**OK (16–100ms):**

| Interaction | Time | Bottleneck |
|------------|------|-----------|
| Frame scrub (binned 2K) | ~20ms | 15ms trait sync (16 MB) + 5ms GPU |
| Frame scrub (no-bin 4K) | ~60ms | 33ms trait sync (64 MB) + 5ms GPU |
| Playback at 30fps | 20ms/frame | Fine with binning |
| 3-panel frame scrub | 3ms | Tiny frames (12 MB) |

**SLOW (>100ms):**

| Interaction | Time | Root cause |
|------------|------|-----------|
| FFT on 4K frame | ~400ms | WebGPU FFT, inherent O(N log N) |
| Constructor (12×4K) | ~800ms | bin2d + stats + trait sync |
| Constructor (3-panel) | ~1.7s | 3× bin + normalize + concat |
| `set_image` (48 frames) | ~4.8s | bin2d on 3 GB + trait sync |
| Playback at 60fps (no-bin) | 33ms/frame | Drops to 30fps (trait sync limit) |
| GIF export | seconds | Python-side CPU rendering per frame |

**Known remaining bottlenecks (code review):**

- **ROI plot** (`_compute_roi_plot`): scans ALL frames with a mask on every
  ROI move. For 48×4K: ~2s CPU. Should be GPU-accelerated or lazy.
- **Diff mode change**: recomputes min/max for ALL frames. For 48×4K: ~1s.
- **GIF/ZIP export**: Python-side per-frame normalization + colormap. No GPU.

### Audit methodology

To reproduce this audit on different hardware:

1. Start JupyterLab + Chrome with WebGPU (`--remote-debugging-port=9222`)
2. Load real 4K EMD data via `IO.folder`
3. Create widget (`Show2D(result, ncols=4)` or `Show3D(result.data)`)
4. Install console interceptor via CDP
5. Trigger each interaction via kernel API (`w.cmap='hot'`, `w.slice_idx=5`, etc.)
6. Read GPU timing from `window._gpuLogs`
7. For Python-side timing: use `time.perf_counter()` around trait changes

See `tests/gpu/run.py show2d --build` for the automated version.
