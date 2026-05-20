# Widget core dedup / consolidation — change log

Branch: `dedup/widget-consolidation` (off synced `main`). **No commits, no pushes.**

This file documents every code change made in this session, what was deliberately
*not* changed, and why. It is a reading guide for the diff; it is not a feature
changelog (per the project rule that `docs/changelog.md` is user-facing only).

---

## Goal

Reduce duplicated logic across 14 widgets — both intra-widget copy-paste and
overlap with non-widget `quantem` core — without changing behavior. Implemented
"What's next" items #2, #3, #5, #6, #7 from the prior analysis report; items #1
and #4 were deliberately skipped (see § Deliberate non-changes).

---

## 1 — Shared Python helpers in `array_utils.py`

`src/quantem/widget/array_utils.py` (+236 lines, additive).

Three new public functions consolidate logic that used to be copy-pasted across
many widgets. Behavior is preserved exactly per call site — see §3 for what is
intentionally NOT routed through these helpers.

### `extract_dataset_meta(data, *, sampling_axis=-1, ...) -> DatasetMeta`

Duck-types the three input kinds a widget accepts and unwraps each to a raw
array plus a `DatasetMeta` NamedTuple `(array, title, pixel_size, units, labels)`.

| Input kind | Detection | Source of `pixel_size` |
|---|---|---|
| `IOResult` (from `quantem.widget.io`) | `isinstance` | `data.pixel_size` (already Å, used verbatim) |
| quantem `Dataset` | `hasattr` `.array` + `.name` + `.sampling` | `sampling[sampling_axis]`, with nm→Å when `units[sampling_axis]` is in `nm_units` |
| plain numpy / torch / cupy | fallthrough | `None` (no metadata to extract) |

**Per-widget axis choices** (this is the load-bearing parameter — different
widgets read different sampling axes; see the rewiring spec in §2):

| Caller | `sampling_axis` | `nm_units` |
|---|---|---|
| Show2D, Show3DVolume, Edit2D, Mark2D, Align2D, ShowComplex2D | `-1` | `("nm",)` (default) |
| Show3D | `1` | `("nm", "nanometer")` |
| Show4D nav, Show4DSTEM real-space, ShowDiffraction | (excluded — see §3) | — |

**Returns `pixel_size = None`** for unknown unit strings (e.g. `"mrad"`,
`"pixels"`), matching every existing `if angstrom / elif nm / else pass`
branch — *no* widget previously assigned a raw sampling value to `pixel_size`
for an unrecognized unit.

The lazy `from quantem.widget.io import IOResult` inside the helper avoids a
circular import (`io.py` already imports from `array_utils.py`).

### `normalize_frame(frame, *, log_scale=False, vmin=None, vmax=None, auto_contrast=False, plo=2.0, phi=98.0) -> uint8 ndarray`

Verbatim port of the per-widget `_normalize_frame` body. Precedence:

1. **Manual** — both `vmin` and `vmax` not `None`: use them directly. Bounds
   are themselves `log1p`'d when `log_scale=True`.
2. **Auto-contrast** — `np.percentile(frame, plo)` / `np.percentile(frame, phi)`
   on the already-log-transformed frame.
3. **Min/max** — `frame.min()` / `frame.max()` of the already-log-transformed
   frame.

Output: `clip((frame - lo) / (hi - lo) * 255, 0, 255).astype(uint8)`. Returns an
all-zero `uint8` array when `hi <= lo`.

**The log formula is intentionally NOT switched to quantem's
`LogarithmicStretch`** (`log(a·x+1)/log(a+1)`, astropy convention). That would
change the displayed tone curve — `log1p` on raw data is what every widget
currently does. Behavior preservation > algorithmic unification here.

### `compute_stats(arr) -> {"mean": float, "min": float, "max": float, "std": float}`

Reduces a numpy / torch / generic array. Torch path uses `.mean().item()` etc.
(matching existing widget fast paths); numpy path uses `np.mean / min / max / std`.

For widgets whose stats traits are lists (Show4D `nav_stats`, `sig_stats`), the
canonical call is `list(compute_stats(x).values())` → `[mean, min, max, std]`.

> Note: torch `.std()` defaults to unbiased (N−1) while `np.std` defaults to
> biased (N). This split pre-existed in Show3D's torch-vs-numpy branches; the
> helper preserves it. Not a regression.

---

## 2 — Widget rewiring (Python)

13 widgets now route their duck-typing / normalize / stats blocks through the
helpers. Each rewire is **behavior-preserving** — the precedence guards
(`if not title`, `if pixel_size == 0.0`, `if labels is None`) are kept around
the helper call so an explicit constructor argument still overrides extracted
metadata.

Files touched and roughly what changed in each (see `git diff --stat` for exact
counts):

| File | Δ | Changes |
|---|---|---|
| `show2d.py` | +19 −42 | IOResult+Dataset duck-typing → `extract_dataset_meta(data, sampling_axis=-1)`; `_normalize_frame` body → `normalize_frame(...)`; removed unused `IO`/`IOResult` import |
| `show3d.py` | +19 −22 | duck-typing → `extract_dataset_meta(data, sampling_axis=1, nm_units=("nm","nanometer"))` with the original `len(sampling) >= 3` guard preserved; `set_image` unwrap → `extract_dataset_meta(data).array`; torch/numpy stats branches → `compute_stats(...)`. `_normalize_frame` left inline (excluded — see §3). |
| `show3dvolume.py` | +11 −37 | IOResult+Dataset blocks → `extract_dataset_meta(data, sampling_axis=-1)`; `data_b` unwrap → `extract_dataset_meta(data_b)`; `set_image` × 2 → `extract_dataset_meta(...).array`; removed unused `IOResult` import |
| `show4d.py` | +15 −30 | **Only the IOResult half** of the metadata block (still gated by `isinstance(data, IOResult)` so the multi-axis Dataset block below it still fires); `_normalize_frame` → `normalize_frame(...)`; `nav_stats` and `sig_stats` → `list(compute_stats(...).values())`. Dataset block + `set_image` unwrap left inline (see §3). |
| `show4dstem.py` | +6 −7 | IOResult half only, gated by `isinstance`. Dataset block + `set_image` left inline (real + k-space two-axis extraction). |
| `showcomplex.py` | +22 −42 | IOResult + first Dataset block → `extract_dataset_meta(...)`; `set_image` → `extract_dataset_meta(data).array`; `_normalize_frame` → `normalize_frame(...)`; `_update_stats` → `compute_stats(data)`. |
| `showdiffraction.py` | +16 −8 | IOResult-only routing in `__init__` and `set_image`; Dataset blocks left inline (real + k-space). |
| `show1d.py` | +5 −7 | `_compute_stats` per-trace loop → `compute_stats(self._data[i])` then append to the four lists. |
| `mark2d.py` | +18 −15 | IOResult block + `_set_data` Dataset block both routed through the helper. |
| `edit2d.py` | +12 −30 | IOResult + Dataset blocks unified through one helper call; `set_image` routed through `extract_dataset_meta(data)`; removed unused `IOResult` import. |
| `align2d.py` | +11 −6 | IOResult halves routed for both `image_a` and `image_b`, guarded by `isinstance(..., IOResult)` so the Dataset block below it still fires. Dataset block left inline (see §3). |
| `bin2d.py` | +8 −9 | IOResult half only; Dataset block left inline (different `.data`/`.unit` protocol). |

The agents' first cut had two cases where unconditionally calling
`extract_dataset_meta` would have *also* unwrapped a `Dataset` and silently
disabled the deliberately-excluded multi-axis Dataset blocks below it
(silently dropping `nav_pixel_size`/`sig_pixel_size`, `k_pixel_size`,
`k_calibrated`). Those call sites now keep an explicit
`if isinstance(data, IOResult):` guard around the helper call — see
`show4d.py`, `show4dstem.py`, `align2d.py`. This is intentional.

---

## 3 — Deliberately NOT routed through the helpers

These divergences came up during rewiring; in each case routing through the
helper would have silently changed behavior. They were excluded as instructed,
not by oversight.

1. **Show3D `_normalize_frame` / `_get_color_range`** — Show3D's min/max
   fallback uses precomputed `self._vmin` / `self._vmax`, not `frame.min/max`.
   The helper would log-transform those bounds a second time. Kept inline.
2. **Show4D Dataset metadata block** — extracts *two* pixel sizes
   (`nav_pixel_size` from `sampling[0]`, `sig_pixel_size` from `sampling[2]`)
   and has a `"mrad"` unit branch. Single-axis helper cannot model this.
3. **Show4DSTEM Dataset block** + **ShowDiffraction Dataset blocks** — both
   extract a real-space `pixel_size` (`sampling[0]`) *and* a separate k-space
   `k_pixel_size` (`sampling[2]`, units `"mrad"` / `"1/Å"`) and set a
   `k_calibrated` flag.
4. **ShowComplex2D's second duck-typing block** and **Bin2D's Dataset block** —
   use `.data`/`.pixel_size`/`.unit` (substring `"nm" in unit`) protocol, not
   `.array`/`.name`/`.sampling`. Different shape; helper does not apply.
5. **Align2D Dataset block** — assigns `pixel_size` for *every* unit
   (nm converted, all others verbatim). The helper returns `None` for unknown
   units, so routing through it would drop pixel size for e.g. `"pixels"`-unit
   Datasets. Excluded.
6. **`set_image` unwrap one-liners on Show4DSTEM and Show4D** — guard on
   `hasattr(.sampling) and hasattr(.array)` only (no `.name`). The helper
   requires `.name` too. Consolidation gain is near-zero; left inline.

---

## 4 — Dead-code removal in `bin.py`

`src/quantem/widget/bin.py` (+3 −47).

Deleted the 47-line `Bin._bin_axis_torch` method — it was a near-verbatim
duplicate of `_bin_axis_torch` in `bin_batch.py`, which `bin.py` line 26
**already imported** as `_bin_axis_standalone`. The only in-file caller of
`self._bin_axis_torch` (inside `Bin._bin_4d_torch`) was redirected to the
imported alias.

**`Bin._bin_4d_torch` was NOT deleted.** It genuinely diverges from
`bin_batch._bin_4d_torch`:

| | `Bin._bin_4d_torch` | `bin_batch._bin_4d_torch` |
|---|---|---|
| Signature | `(self, data4d, factors, mode, edge)` | `(data4d, preset, device_str)` |
| Input | torch tensor | numpy array + `BinPreset` |
| Output | torch tensor | numpy array |

Different I/O types, different call sites. Left as-is.

---

## 5 — JS: exact colormap LUTs + JCh phase wheel

Before this change, the widget had **two inconsistent colormap pipelines**:

- live canvas: 8–11 hand-typed control points interpolated to 256 entries
  (`js/colormaps.ts COLORMAP_POINTS`) → approximate colors
- PNG export (`save_image`): true matplotlib LUTs via `cm.get_cmap(...)` → exact

Same widget, two different appearances. Also, the phase colorwheel used naive
HSV instead of quantem's perceptually-uniform JCh "domain coloring"
(`array_to_rgba` in `quantem.core.visualization.visualization_utils`).

### Files

- **`scripts/gen_colormaps.py`** (new, 140 lines) — standalone generator. For
  each of `inferno, viridis, plasma, magma, hot, gray, hsv, turbo, RdBu`,
  samples matplotlib at 256 evenly-spaced points → exact RGB256. Also computes
  a 256-entry cyclic JCh phase wheel replicating quantem's `array_to_rgba`
  math (`J = amp·61.5`, `C = min(chroma_boost·98·J/123, 110)`, `h = deg(angle)+180`,
  then `cspace_convert(JCh, "JCh", "sRGB1")`) at full lightness with
  `chroma_boost=1.0`.
- **`js/colormap-data.ts`** (new, 194 lines, generated) —
  `COLORMAP_LUTS: Record<string, number[]>` (9 × 768 ints) plus
  `JCH_PHASE_WHEEL: number[]` (768 ints). Header comment marks it generated
  and points back at the script.
- **`js/colormaps.ts`** (+13 −50) — removed `COLORMAP_POINTS` and the linear
  interpolation in `createColormapLUT`; now imports `COLORMAP_LUTS` and uses
  the exact tables directly. **All public exports unchanged in signature**
  (`COLORMAPS`, `COLORMAP_NAMES`, `applyColormap`, `renderToOffscreen`,
  `renderToOffscreenReuse`, `GPUColormapEngine`, `getGPUColormapEngine`,
  `getGPUMaxBufferSize`).
- **`js/showcomplex/index.tsx`** (+30 −37) — `renderHSV` is now `rgb =
  JCH_PHASE_WHEEL[hueIndex]` scaled by amplitude-as-lightness;
  `drawPhaseColorwheel` draws each angular segment from the same LUT. The
  colorwheel matches quantem's `array_to_rgba` exactly; the field uses the
  same hue ramp scaled by amplitude (a deliberate, recommended approximation
  of the full 2-D JCh transform).

### Verification

| Check | Result |
|---|---|
| All 9 colormap LUTs vs `matplotlib.colormaps[name]` | **0 error** (byte-exact) |
| JCh phase wheel — 256 RGB entries | range [28, 252], max step between adjacent entries = 3, wrap gap (last → first) = 2 — smooth cyclic |
| `npm run typecheck` | pass |
| `npm run build` | pass (17 bundles) |

> **Visual screenshot verification:** could not be done here — the headless E2E
> harness is broken in this devcontainer (confirmed via the untouched `browse`
> widget, which fails identically). The CLAUDE.md-mandated screenshot review
> must be done by the user in JupyterLab.

---

## 6 — Item #4 (`dft_upsample` reuse) — kept the widget's version

The plan called for replacing `align2d._dft_upsample` (Guizar-Sicairos
matrix-DFT upsampling) with `quantem.core.utils.imaging_utils.dft_upsample`.
On inspection the two are the **same algorithm family** but have
**incompatible interfaces**:

| | widget `_dft_upsample` | quantem `dft_upsample` |
|---|---|---|
| Returns | the *absolute peak coordinate* `(ups_y[up_y], ups_x[up_x])` (caller stops) | the *upsampled patch array* (caller does its own argmax + offset arithmetic) |
| Shift param | integer `(peak_y, peak_x)`; builds `peak + (arange(size) - size//2)/up` internally | `shift` tuple; builds `arange(-du, du+1)` with `r_shift = shift - M//2` |

Swapping would require rewriting `_cross_correlate_fft`'s peak-recovery logic
(argmax + `(peak - up) / up` conversion) and reconciling the centering — a
behavior-affecting rewrite, not a clean argument reorder. Per the prompt's
explicit guidance ("if signatures/semantics differ enough that a clean swap
risks behavior change, DO NOT force it"), the widget's `_dft_upsample` and
`_cross_correlate_fft` are unchanged. Phase-correlation normalization and
Tukey window are also untouched.

A proper consolidation would require quantem's
`cross_correlation_shift` to gain `phase_correlation=True` and `window=`
options — followup, not in scope here.

---

## 7 — New parity tests

CLAUDE.md mandates that JS math be ported line-by-line to Python and
validated against NumPy/SciPy ground truth. Two such tests were missing.

### `tests/test_fft_parity.py` (new, 286 lines)

Ports the CPU-fallback FFT primitives from `js/webgpu-fft.ts` (`fft1d`, `fft2d`,
`fftshift`, `computeMagnitude`) line-by-line to Python (`_js_*` helpers), then
asserts they match `numpy.fft` within tolerance.

**Real bug surfaced:** `fft2d` zero-pads to next-pow2 then **crops back** to
the original size. For non-power-of-two dimensions the cropped output is
**not** equal to `np.fft.fft2` of the original input — only the DC term
survives correctly. CLAUDE.md's "ROI FFT" section requires callers to pre-pad
to nextPow2, so this is documented usage, but any caller passing raw non-pow2
dims gets a spectrum that does not correspond to its input. The test asserts
the actual (divergent) behavior with a prominent `PARITY FLAG` comment so the
suite stays honest. **Worth either an explicit guard in the JS or a docstring
warning on `fft2d`.**

### `tests/test_line_profile_parity.py` (new, 289 lines)

Ports `sampleSingleLine` / `sampleLineProfile` (identical across show2d /
show3d / show4d / show4dstem / mark2d) including the thick-profile
`profile_width` averaging. Validates:

- sample-count contract `n = max(2, ceil(length))`
- exact bilinear values (hand-computed `12.0` / `17.5`)
- edge-clamping
- endpoint parity with `skimage.measure.profile_line`
- full-profile parity on a linear field (resampled to JS grid, atol = 1e-9)
- axis-aligned integer-line parity (no interpolation, exact)
- thick-profile mean-of-offset-lines identity

> **Note on the linear-field test:** the agent's first draft of
> `test_js_profile_matches_skimage_resampled` used a random field and
> `atol = 0.05`. A random field interpolated bilinearly is only
> piecewise-bilinear, and resampling the two off-by-one sample grids onto each
> other via `np.interp` introduces error larger than 0.05 — so the test's
> premise was inherently noisy. Switched to a linear field
> `f(row, col) = 0.7·row − 0.4·col + 3.0` where bilinear interpolation is
> exact, both samplers give exact analytic values, and `np.interp` resampling
> is also exact → machine-precision agreement at `atol = 1e-9`.

**Documented algorithmic difference:** the JS sampler emits
`n = max(2, ceil(length))` samples; skimage emits `ceil(length) + 1`. The
profiles are NOT element-comparable in general. The tests handle this
explicitly (compare on a shared parameterization, or use endpoints only).

---

## 8 — Verification matrix

| Check | Result |
|---|---|
| `pytest tests/ --ignore-glob='tests/test_e2e_*.py'` | **1466 passed** |
| New parity tests (`test_fft_parity.py`, `test_line_profile_parity.py`) | **31 passed** |
| `npm run typecheck` | pass |
| `npm run build` | pass — all 17 bundles |
| Generated colormap LUTs vs matplotlib | byte-exact (0 error on all 9) |
| JCh phase wheel | 256 entries, smooth cyclic (max step 3, wrap gap 2) |
| E2E smoke screenshots (CLAUDE.md mandated) | **could not run** — pre-existing devcontainer infrastructure failure, confirmed via the untouched `browse` widget which fails identically |

---

## 9 — Deliberate non-changes

1. **`detector_calibration.py` (755 dead lines).** Untracked git-ignored
   *personal* file with no callers anywhere in the widget repo. Decided
   against deleting (unrecoverable; the file is the user's). Decision needed:
   wire into Show4DSTEM (detector presets / ellipse correction / k-calibration
   are genuinely useful) or delete.
2. **`align2d._dft_upsample` → quantem swap (item #4).** Kept widget's version
   — incompatible return semantics, see §6.
3. **No commits, no pushes, no PRs** — all changes live on branch
   `dedup/widget-consolidation`, working tree only.
4. **`.gitignore` modification** — pre-existing in the working tree before
   this session, not authored by these changes.

---

## What's next

1. **Visually verify the colormap + JCh phase-wheel changes in a real
   JupyterLab session** (`npm run dev`, open Show2D and ShowComplex2D, change
   colormaps, drag histogram, view the phase colorwheel) — confirms the
   exact-LUT swap and the JCh wheel render correctly so microscopists reading
   ptychography/exit-wave phase get accurate domain coloring instead of
   contrast-distorting naive HSV.
2. **Decide the fate of `detector_calibration.py`** — wire it into Show4DSTEM
   or delete the 755 dead lines. Dead code with a live-looking test invites a
   future contributor to ship half-tested ellipse/k-calibration into an
   acquisition workflow.
3. **Fix or guard the `fft2d` non-power-of-two divergence** flagged by the new
   parity test — either pre-pad inside `fft2d` itself or add a docstring
   warning + caller-side assertion. An undetected FFT mismatch corrupts the
   lattice/d-spacing readout an operator uses to confirm focus and zone-axis.
4. **Repair the E2E smoke harness in this devcontainer** — every widget root
   times out at 30 s (proven via untouched `browse`). Restores the screenshot
   safety net that catches black-panel / broken-overlay regressions before
   they reach a live session.
5. **Add `phase_correlation=True` and `window=` options to
   `quantem.core.utils.imaging_utils.cross_correlation_shift`** so the widget's
   `_cross_correlate_fft` can later retire onto shared code. Unifies the
   auto-align kernel with quantem's drift correction → consistent sub-pixel
   registration across the toolkit.

---

## Files modified

- `src/quantem/widget/array_utils.py` (+236 −0)
- `src/quantem/widget/bin.py` (+3 −47)
- `src/quantem/widget/show2d.py` (+19 −42)
- `src/quantem/widget/show3d.py` (+19 −22)
- `src/quantem/widget/show3dvolume.py` (+11 −37)
- `src/quantem/widget/show4d.py` (+15 −30)
- `src/quantem/widget/show4dstem.py` (+6 −7)
- `src/quantem/widget/showcomplex.py` (+22 −42)
- `src/quantem/widget/showdiffraction.py` (+16 −8)
- `src/quantem/widget/show1d.py` (+5 −7)
- `src/quantem/widget/mark2d.py` (+18 −15)
- `src/quantem/widget/edit2d.py` (+12 −30)
- `src/quantem/widget/align2d.py` (+11 −6)
- `src/quantem/widget/bin2d.py` (+8 −9)
- `js/colormaps.ts` (+13 −50)
- `js/showcomplex/index.tsx` (+30 −37)
- `js/colormap-data.ts` (new, 194 lines, generated)
- `scripts/gen_colormaps.py` (new, 140 lines)
- `tests/test_fft_parity.py` (new, 286 lines)
- `tests/test_line_profile_parity.py` (new, 289 lines)
- `CHANGES_dedup-consolidation.md` (new, this file)
