"""
TransferFunctionEditor: Interactive transfer function (LUT) designer for volume rendering.

Standalone widget — takes an optional numpy array (or no data), lets the user
place opacity/color control handles over a histogram of the data, and emits a
256-entry RGBA LUT (``tf_lut_bytes``) computed by linear interpolation between
sorted handles. Designed to plug into Show3DVolume's WebGPU ray-caster in a
future PR; this PR is the editor only.

Research grounding
------------------
- Tomviz's "Integrated Histogram, Opacity and Color Transfer Function Editor"
  (Kitware, 2016): histogram-backed editor with draggable handles, presets,
  rescale, invert, and live preview.
- VTK ``vtkPiecewiseFunction`` + ``vtkColorTransferFunction``: opacity and
  color are two independent piecewise-linear functions over scalar value.
  Standard practice combines color (RGB) with opacity into a single RGBA LUT.
- 3D Slicer Volume Rendering presets: opacity / color / gradient transfer
  functions are pre-tuned per modality (CT bone, soft tissue, fat, air, MR).
- napari rendering modes (translucent / MIP / iso / additive / attenuated_mip)
  motivate the need for a per-volume opacity ramp distinct from intensity LUT.
- 2D transfer functions add gradient magnitude as a second axis; we stay 1D
  for now — opacity = f(intensity), color = g(intensity).

Design choices
--------------
- **Standalone**: this PR ships the editor only, not the Show3DVolume hookup.
  That keeps the diff small and the editor reviewable in isolation. The
  ``tf_lut_bytes`` trait is the integration point; Show3DVolume will read it
  in a follow-up.
- **Three independent 1D functions** ≈ VTK pattern, simplified to one struct
  per handle ``{x, opacity, color}``. The LUT is computed in Python (256×4
  uint8 RGBA) on every handle change — this matches the data shape Show3DVolume
  expects.
- **Quantem normalizations** drive the optional intensity stretch
  (``stretch_preset``): ``linear``/``log``/``power``/``asinh`` map to
  ``LinearStretch``/``LogarithmicStretch``/``PowerLawStretch``/
  ``InverseHyperbolicSineStretch`` from
  ``quantem.core.visualization.custom_normalizations``. The stretch is applied
  before the histogram is computed (and conceptually before LUT lookup at
  render time).
- **Histogram log toggle** for EM tomograms with long tails (Tomviz convention).
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Optional, Self

import anywidget
import numpy as np
import traitlets

from quantem.core.visualization.custom_normalizations import (
    InverseHyperbolicSineStretch,
    LinearStretch,
    LogarithmicStretch,
    PowerLawStretch,
)

from quantem.widget.array_utils import to_numpy
from quantem.widget.json_state import (
    resolve_widget_version,
    save_state_file,
    unwrap_state_payload,
)


_STATIC = pathlib.Path(__file__).parent / "static"
_TFE_ESM = _STATIC / "transfer_function_editor.js"
_TFE_CSS = _STATIC / "transfer_function_editor.css"


# Matplotlib-style sampled colormap anchor points (RGB uint8). Kept tiny: just
# the ones already in js/colormaps.ts so JS + Python agree on the ramp colors.
_COLORMAP_ANCHORS: dict[str, list[tuple[int, int, int]]] = {
    "inferno": [
        (0, 0, 4), (40, 11, 84), (101, 21, 110), (159, 42, 99),
        (212, 72, 66), (245, 125, 21), (252, 193, 57), (252, 255, 164),
    ],
    "viridis": [
        (68, 1, 84), (72, 36, 117), (65, 68, 135), (53, 95, 141),
        (42, 120, 142), (33, 145, 140), (34, 168, 132), (68, 191, 112),
        (122, 209, 81), (189, 223, 38), (253, 231, 37),
    ],
    "plasma": [
        (13, 8, 135), (75, 3, 161), (126, 3, 168), (168, 34, 150),
        (203, 70, 121), (229, 107, 93), (248, 148, 65), (253, 195, 40), (240, 249, 33),
    ],
    "magma": [
        (0, 0, 4), (28, 16, 68), (79, 18, 123), (129, 37, 129),
        (181, 54, 122), (229, 80, 100), (251, 135, 97), (254, 194, 135), (252, 253, 191),
    ],
    "gray": [(0, 0, 0), (255, 255, 255)],
    "hot": [
        (0, 0, 0), (87, 0, 0), (173, 0, 0), (255, 0, 0),
        (255, 87, 0), (255, 173, 0), (255, 255, 0), (255, 255, 128), (255, 255, 255),
    ],
    "turbo": [
        (48, 18, 59), (69, 55, 161), (66, 107, 230), (30, 162, 230),
        (29, 212, 169), (79, 241, 89), (175, 240, 32), (244, 195, 12),
        (248, 118, 11), (207, 46, 3), (122, 4, 2),
    ],
}


def _sample_cmap_color(cmap: str, x: float) -> list[int]:
    """Sample the named colormap ramp at ``x`` in ``[0, 1]`` -> ``[r, g, b]`` uint8."""
    pts = _COLORMAP_ANCHORS.get(cmap) or _COLORMAP_ANCHORS["viridis"]
    if len(pts) == 1:
        return [int(pts[0][0]), int(pts[0][1]), int(pts[0][2])]
    t = float(max(0.0, min(1.0, x))) * (len(pts) - 1)
    idx = int(t)
    frac = t - idx
    if idx >= len(pts) - 1:
        return [int(pts[-1][0]), int(pts[-1][1]), int(pts[-1][2])]
    p0 = pts[idx]
    p1 = pts[idx + 1]
    return [
        int(round(p0[0] + frac * (p1[0] - p0[0]))),
        int(round(p0[1] + frac * (p1[1] - p0[1]))),
        int(round(p0[2] + frac * (p1[2] - p0[2]))),
    ]


_STRETCH_PRESETS = ("linear", "log", "power", "asinh")


def _stretch_to_unit(values: np.ndarray, preset: str) -> np.ndarray:
    """Map ``values`` from data-space to ``[0, 1]`` using a quantem stretch.

    The interval is min/max of finite values; the stretch is one of
    ``linear``/``log``/``power``/``asinh``.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.zeros(0, dtype=np.float64)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax <= vmin:
        return np.zeros_like(arr)
    normed = (arr - vmin) / (vmax - vmin)
    np.clip(normed, 0.0, 1.0, out=normed)

    if preset == "linear":
        return LinearStretch()(normed)
    if preset == "log":
        return LogarithmicStretch(a=1000.0)(normed, copy=True)
    if preset == "power":
        return PowerLawStretch(power=0.5)(normed, copy=True)
    if preset == "asinh":
        return InverseHyperbolicSineStretch(a=0.1)(normed, copy=True)
    return normed


def _compute_histogram(
    data: Optional[np.ndarray], n_bins: int, stretch_preset: str
) -> tuple[np.ndarray, float, float]:
    """Return ``(histogram_float32, data_min, data_max)``.

    The histogram is computed after the stretch maps the data into ``[0, 1]``,
    so the x-axis of the editor is the stretched intensity. ``data_min``/
    ``data_max`` keep the original data-space domain for tick labels.
    """
    if data is None or data.size == 0:
        return np.zeros(n_bins, dtype=np.float32), 0.0, 1.0
    arr = np.asarray(data).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.zeros(n_bins, dtype=np.float32), 0.0, 1.0
    data_min = float(arr.min())
    data_max = float(arr.max())
    if data_max <= data_min:
        return np.zeros(n_bins, dtype=np.float32), data_min, data_max
    stretched = _stretch_to_unit(arr, stretch_preset)
    counts, _ = np.histogram(stretched, bins=n_bins, range=(0.0, 1.0))
    counts_f = counts.astype(np.float32)
    peak = float(counts_f.max())
    if peak > 0:
        counts_f /= peak
    return counts_f, data_min, data_max


def _compute_lut(handles: list[dict[str, Any]], n_bins: int) -> bytes:
    """Compute an ``n_bins × 4`` uint8 RGBA LUT from sorted handles via linear interp."""
    if not handles:
        return np.zeros((n_bins, 4), dtype=np.uint8).tobytes()
    xs = np.array([float(h["x"]) for h in handles], dtype=np.float32)
    opacities = np.array([float(h["opacity"]) for h in handles], dtype=np.float32)
    colors = np.array(
        [[float(c) for c in h["color"]] for h in handles], dtype=np.float32
    )
    t = np.linspace(0.0, 1.0, n_bins, dtype=np.float32)
    opacity_lut = np.interp(t, xs, opacities)
    r = np.interp(t, xs, colors[:, 0])
    g = np.interp(t, xs, colors[:, 1])
    b = np.interp(t, xs, colors[:, 2])
    lut = np.stack([r, g, b, opacity_lut * 255.0], axis=-1)
    lut = np.clip(lut, 0.0, 255.0).astype(np.uint8)
    return lut.tobytes()


def _normalize_handle(h: dict[str, Any]) -> dict[str, Any]:
    """Coerce one handle dict into the canonical shape."""
    x = float(h.get("x", 0.0))
    opacity = float(h.get("opacity", 0.0))
    raw_color = h.get("color", [255, 255, 255])
    if isinstance(raw_color, (tuple, list, np.ndarray)):
        color_list = [int(round(float(c))) for c in list(raw_color)[:3]]
    else:
        color_list = [255, 255, 255]
    while len(color_list) < 3:
        color_list.append(0)
    return {
        "x": float(max(0.0, min(1.0, x))),
        "opacity": float(max(0.0, min(1.0, opacity))),
        "color": [max(0, min(255, c)) for c in color_list[:3]],
    }


def _normalize_handles(raw: Any) -> list[dict[str, Any]]:
    """Coerce + sort by x. Endpoints are clamped to [0, 1] but not auto-added."""
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise traitlets.TraitError("tf_handles must be a list of dicts")
    out: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise traitlets.TraitError("Each handle must be a dict")
        out.append(_normalize_handle(entry))
    out.sort(key=lambda h: h["x"])
    return out


def _default_handles(cmap: str) -> list[dict[str, Any]]:
    return [
        {"x": 0.0, "opacity": 0.0, "color": _sample_cmap_color(cmap, 0.0)},
        {"x": 1.0, "opacity": 1.0, "color": _sample_cmap_color(cmap, 1.0)},
    ]


class TransferFunctionEditor(anywidget.AnyWidget):
    """Interactive transfer function (LUT) editor.

    Parameters
    ----------
    data : array_like, optional
        Numpy/torch/cupy array used to compute the displayed histogram. If
        ``None``, the histogram is empty but a TF can still be designed.
    title : str, default "Transfer Function Editor"
        Title shown in the widget header.
    cmap : str, default "viridis"
        Initial colormap ramp used to color new handles.
    n_bins : int, default 256
        Resolution of the histogram and the output LUT.
    stretch_preset : {"linear", "log", "power", "asinh"}, default "linear"
        Quantem custom-normalization stretch applied before the histogram is
        computed. Maps data-space intensity to [0, 1].
    log_histogram : bool, default False
        Display log-scaled histogram bars (data is unchanged; UI only).
    show_stats : bool, default True
        Show the stats bar (data range, n handles).
    show_controls : bool, default True
        Show the control row (reset, stretch dropdown, histogram log toggle).
    state : dict or str or Path, optional
        Restore state from a dict or JSON envelope file.

    Notes
    -----
    The widget emits a 256 × 4 uint8 RGBA buffer in ``tf_lut_bytes`` whenever
    handles change. ``tf_lut_bytes`` is the integration point for downstream
    consumers such as Show3DVolume.
    """

    _esm = _TFE_ESM if _TFE_ESM.exists() else "export function render() {}"
    _css = _TFE_CSS if _TFE_CSS.exists() else ""

    widget_version = traitlets.Unicode("unknown").tag(sync=True)

    title = traitlets.Unicode("Transfer Function Editor").tag(sync=True)
    cmap = traitlets.Unicode("viridis").tag(sync=True)

    tf_handles = traitlets.List(trait=traitlets.Dict()).tag(sync=True)
    tf_lut_bytes = traitlets.Bytes(b"").tag(sync=True)

    data_min = traitlets.Float(0.0).tag(sync=True)
    data_max = traitlets.Float(1.0).tag(sync=True)
    histogram_bytes = traitlets.Bytes(b"").tag(sync=True)
    n_bins = traitlets.Int(256).tag(sync=True)
    log_histogram = traitlets.Bool(False).tag(sync=True)
    stretch_preset = traitlets.Unicode("linear").tag(sync=True)

    show_stats = traitlets.Bool(True).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)

    # Suppression flags so that programmatic batched edits during __init__
    # don't recompute the histogram / LUT N times. We always recompute at
    # the end of __init__.
    _suppress_lut = False
    _suppress_histogram = False

    @traitlets.validate("tf_handles")
    def _validate_tf_handles(self, proposal: dict[str, Any]) -> list[dict[str, Any]]:
        return _normalize_handles(proposal["value"])

    @traitlets.validate("stretch_preset")
    def _validate_stretch_preset(self, proposal: dict[str, Any]) -> str:
        v = str(proposal["value"]).lower()
        if v not in _STRETCH_PRESETS:
            raise traitlets.TraitError(
                f"stretch_preset must be one of {_STRETCH_PRESETS}, got {v!r}"
            )
        return v

    @traitlets.validate("n_bins")
    def _validate_n_bins(self, proposal: dict[str, Any]) -> int:
        v = int(proposal["value"])
        if v < 2:
            raise traitlets.TraitError("n_bins must be >= 2")
        return v

    def __init__(
        self,
        data: Any = None,
        *,
        title: str = "Transfer Function Editor",
        cmap: str = "viridis",
        n_bins: int = 256,
        stretch_preset: str = "linear",
        log_histogram: bool = False,
        show_stats: bool = True,
        show_controls: bool = True,
        tf_handles: Optional[list[dict[str, Any]]] = None,
        state: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()

        # Initialize internal state before any trait assignment so observers
        # that fire during __init__ have a consistent view of the widget.
        self._data: Optional[np.ndarray] = None
        if data is not None:
            self._data = to_numpy(data, dtype=np.float32)

        # Don't recompute LUT/histogram on every trait set during init; we do
        # both explicitly at the end.
        self._suppress_lut = True
        self._suppress_histogram = True
        try:
            self.title = title
            self.cmap = cmap
            self.n_bins = int(n_bins)
            self.stretch_preset = stretch_preset
            self.log_histogram = bool(log_histogram)
            self.show_stats = bool(show_stats)
            self.show_controls = bool(show_controls)

            self.tf_handles = (
                _normalize_handles(tf_handles)
                if tf_handles is not None
                else _default_handles(self.cmap)
            )
        finally:
            self._suppress_lut = False
            self._suppress_histogram = False

        self._refresh_histogram()
        self._refresh_lut()

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = json.loads(pathlib.Path(state).read_text())
            state = unwrap_state_payload(state)
            self.load_state_dict(state)

    # ------------------------------------------------------------------
    # Observers
    # ------------------------------------------------------------------

    @traitlets.observe("tf_handles")
    def _on_handles_changed(self, change: dict[str, Any]) -> None:  # noqa: ARG002
        self._refresh_lut()

    @traitlets.observe("n_bins")
    def _on_n_bins_changed(self, change: dict[str, Any]) -> None:  # noqa: ARG002
        self._refresh_histogram()
        self._refresh_lut()

    @traitlets.observe("stretch_preset")
    def _on_stretch_changed(self, change: dict[str, Any]) -> None:  # noqa: ARG002
        self._refresh_histogram()

    @traitlets.observe("cmap")
    def _on_cmap_changed(self, change: dict[str, Any]) -> None:  # noqa: ARG002
        # Only auto-recolor existing endpoint handles if they appear to be
        # untouched (default endpoints). We do not silently overwrite custom
        # palettes the user designed.
        pass

    # ------------------------------------------------------------------
    # Recompute helpers
    # ------------------------------------------------------------------

    def _refresh_histogram(self) -> None:
        if self._suppress_histogram:
            return
        counts, dmin, dmax = _compute_histogram(
            self._data, int(self.n_bins), str(self.stretch_preset)
        )
        self.histogram_bytes = counts.astype(np.float32).tobytes()
        self.data_min = float(dmin)
        self.data_max = float(dmax)

    def _refresh_lut(self) -> None:
        if self._suppress_lut:
            return
        self.tf_lut_bytes = _compute_lut(list(self.tf_handles), int(self.n_bins))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_image(self, data: Any) -> Self:
        """Replace the data used for the histogram. Preserves ``tf_handles``."""
        if data is None:
            self._data = None
        else:
            self._data = to_numpy(data, dtype=np.float32)
        self._refresh_histogram()
        return self

    def reset(self) -> Self:
        """Reset to the default two-handle ramp using the current ``cmap``."""
        self.tf_handles = _default_handles(self.cmap)
        return self

    def get_lut(self) -> np.ndarray:
        """Return the current LUT as a ``(n_bins, 4)`` ``uint8`` array."""
        if not self.tf_lut_bytes:
            self._refresh_lut()
        return np.frombuffer(self.tf_lut_bytes, dtype=np.uint8).reshape(-1, 4).copy()

    # ------------------------------------------------------------------
    # State protocol
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "cmap": self.cmap,
            "tf_handles": [dict(h) for h in self.tf_handles],
            "n_bins": int(self.n_bins),
            "stretch_preset": self.stretch_preset,
            "log_histogram": bool(self.log_histogram),
            "show_stats": bool(self.show_stats),
            "show_controls": bool(self.show_controls),
        }

    def save(self, path: str) -> None:
        save_state_file(path, "TransferFunctionEditor", self.state_dict())

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key, val in state.items():
            if hasattr(self, key):
                try:
                    setattr(self, key, val)
                except traitlets.TraitError:
                    # Skip values that fail validation rather than crash on load.
                    continue

    def summary(self) -> None:
        n_handles = len(self.tf_handles)
        print("TransferFunctionEditor Summary")
        print(f"  Title:    {self.title}")
        print(f"  Cmap:     {self.cmap}")
        print(f"  Handles:  {n_handles}")
        print(f"  Bins:     {self.n_bins}")
        print(f"  Stretch:  {self.stretch_preset}")
        print(f"  Log hist: {self.log_histogram}")
        print(f"  Domain:   [{self.data_min:.4g}, {self.data_max:.4g}]")
        if self._data is not None:
            print(f"  Data:     shape={self._data.shape}, dtype={self._data.dtype}")
        else:
            print("  Data:     (none)")

    def __repr__(self) -> str:
        if self._data is not None:
            shape = "×".join(str(s) for s in self._data.shape)
            data_info = f", data={shape}"
        else:
            data_info = ""
        return (
            f"TransferFunctionEditor(n_handles={len(self.tf_handles)}, "
            f"cmap={self.cmap}, stretch={self.stretch_preset}{data_info})"
        )
