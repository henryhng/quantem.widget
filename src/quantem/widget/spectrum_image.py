"""
spectrum_image: Interactive 2-panel viewer for hyperspectral data.

Displays a spatial map (left, integrated over an energy window) and a spectrum
panel (right, at the navigation cursor). Supports power-law background
subtraction and sum/max/argmax/mean map modes. Intended for EELS / EDS-style
spectrum images of shape ``(ny, nx, n_energy)``.

Scope: visualization + integration + optional power-law background fit.
EELS ionization edge databases, EDS line databases, and model fitting are
out of scope and belong in a dedicated model-fitting widget.
"""

import json
import pathlib
from typing import Self

import anywidget
import numpy as np
import traitlets

from quantem.widget.array_utils import to_numpy
from quantem.widget.json_state import (
    resolve_widget_version,
    save_state_file,
    unwrap_state_payload,
)
from quantem.widget.tool_parity import (
    bind_tool_runtime_api,
    build_tool_groups,
    normalize_tool_groups,
)


_BG_EPS = 1e-9


def _extract_energy_axis_from_dataset(data) -> tuple[np.ndarray, str]:
    """Pull an energy axis from a Dataset3d's sampling/origin metadata.

    The third axis is assumed to be the spectral axis. Returns
    (energy_axis, unit). Falls back to a unit-less integer axis when
    metadata is missing.
    """
    n_e = int(data.array.shape[2])
    sampling = getattr(data, "sampling", None)
    origin = getattr(data, "origin", None)
    units = getattr(data, "units", None)
    step = 1.0
    o0 = 0.0
    if sampling is not None and len(sampling) >= 3:
        step = float(sampling[2])
    if origin is not None and len(origin) >= 3:
        o0 = float(origin[2])
    unit = ""
    if units is not None and len(units) >= 3:
        unit = str(units[2])
    return (o0 + step * np.arange(n_e, dtype=np.float64), unit)


class SpectrumImage(anywidget.AnyWidget):
    """
    Interactive 2-panel viewer for hyperspectral data.

    Parameters
    ----------
    data : array_like or Dataset3d
        3D array of shape ``(ny, nx, n_energy)``. Accepts NumPy, PyTorch,
        CuPy, or a quantem ``Dataset3d``.
    energy_axis : array_like, optional
        Energy values per spectral bin. If omitted and ``data`` is a
        ``Dataset3d``, the axis is derived from
        ``origin[2] + sampling[2] * arange(n_energy)``. Otherwise defaults
        to ``arange(n_energy)``.
    energy_unit : str, default "eV"
        Display unit for the energy axis.
    title : str, optional
        Title displayed in the widget header.
    """

    _esm = pathlib.Path(__file__).parent / "static" / "spectrum_image.js"

    widget_version = traitlets.Unicode("unknown").tag(sync=True)

    # ── Shape ───────────────────────────────────────────────────────────────
    ny = traitlets.Int(1).tag(sync=True)
    nx = traitlets.Int(1).tag(sync=True)
    n_energy = traitlets.Int(1).tag(sync=True)

    # ── Data transfer ───────────────────────────────────────────────────────
    map_bytes = traitlets.Bytes(b"").tag(sync=True)
    spectrum_bytes = traitlets.Bytes(b"").tag(sync=True)
    bg_curve_bytes = traitlets.Bytes(b"").tag(sync=True)
    energy_axis_bytes = traitlets.Bytes(b"").tag(sync=True)
    bg_params = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0]).tag(sync=True)

    # ── Cursor / navigation ─────────────────────────────────────────────────
    nav_index = traitlets.List(traitlets.Int(), default_value=[0, 0]).tag(sync=True)
    cursor_sync = traitlets.Bool(True).tag(sync=True)

    # ── Integration window (in energy_unit) ─────────────────────────────────
    window_e_min = traitlets.Float(0.0).tag(sync=True)
    window_e_max = traitlets.Float(1.0).tag(sync=True)

    # ── Map mode ────────────────────────────────────────────────────────────
    map_mode = traitlets.Unicode("sum").tag(sync=True)  # sum | max | argmax | mean

    # ── Background fit ──────────────────────────────────────────────────────
    bg_subtract = traitlets.Bool(False).tag(sync=True)
    bg_e_min = traitlets.Float(0.0).tag(sync=True)
    bg_e_max = traitlets.Float(1.0).tag(sync=True)

    # ── Display traits ──────────────────────────────────────────────────────
    title = traitlets.Unicode("").tag(sync=True)
    energy_unit = traitlets.Unicode("eV").tag(sync=True)
    cmap = traitlets.Unicode("viridis").tag(sync=True)
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(False).tag(sync=True)
    percentile_low = traitlets.Float(1.0).tag(sync=True)
    percentile_high = traitlets.Float(99.0).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)

    # ── Map statistics [mean, min, max, std] ─────────────────────────────────
    map_stats_mean = traitlets.Float(0.0).tag(sync=True)
    map_stats_min = traitlets.Float(0.0).tag(sync=True)
    map_stats_max = traitlets.Float(0.0).tag(sync=True)
    map_stats_std = traitlets.Float(0.0).tag(sync=True)

    # ── Tool lock/hide ──────────────────────────────────────────────────────
    disabled_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    hidden_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)

    _VALID_MAP_MODES = ("sum", "max", "argmax", "mean")

    @classmethod
    def _normalize_tool_groups(cls, tool_groups):
        return normalize_tool_groups("SpectrumImage", tool_groups)

    @classmethod
    def _build_disabled_tools(
        cls,
        disabled_tools=None,
        disable_display: bool = False,
        disable_histogram: bool = False,
        disable_stats: bool = False,
        disable_navigation: bool = False,
        disable_view: bool = False,
        disable_export: bool = False,
        disable_background: bool = False,
        disable_window: bool = False,
        disable_all: bool = False,
    ):
        return build_tool_groups(
            "SpectrumImage",
            tool_groups=disabled_tools,
            all_flag=disable_all,
            flag_map={
                "display": disable_display,
                "histogram": disable_histogram,
                "stats": disable_stats,
                "navigation": disable_navigation,
                "view": disable_view,
                "export": disable_export,
                "background": disable_background,
                "window": disable_window,
            },
        )

    @classmethod
    def _build_hidden_tools(
        cls,
        hidden_tools=None,
        hide_display: bool = False,
        hide_histogram: bool = False,
        hide_stats: bool = False,
        hide_navigation: bool = False,
        hide_view: bool = False,
        hide_export: bool = False,
        hide_background: bool = False,
        hide_window: bool = False,
        hide_all: bool = False,
    ):
        return build_tool_groups(
            "SpectrumImage",
            tool_groups=hidden_tools,
            all_flag=hide_all,
            flag_map={
                "display": hide_display,
                "histogram": hide_histogram,
                "stats": hide_stats,
                "navigation": hide_navigation,
                "view": hide_view,
                "export": hide_export,
                "background": hide_background,
                "window": hide_window,
            },
        )

    @traitlets.validate("disabled_tools")
    def _validate_disabled_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    @traitlets.validate("hidden_tools")
    def _validate_hidden_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    @traitlets.validate("map_mode")
    def _validate_map_mode(self, proposal):
        value = str(proposal["value"]).strip().lower()
        if value not in self._VALID_MAP_MODES:
            supported = ", ".join(self._VALID_MAP_MODES)
            raise ValueError(
                f"Unknown map_mode {proposal['value']!r}. Supported: {supported}."
            )
        return value

    @traitlets.validate("nav_index")
    def _validate_nav_index(self, proposal):
        value = list(proposal["value"])
        if len(value) != 2:
            raise ValueError(f"nav_index must have length 2, got {len(value)}.")
        return [int(v) for v in value]

    def __init__(
        self,
        data,
        energy_axis=None,
        energy_unit: str = "eV",
        title: str | None = None,
        cmap: str = "viridis",
        log_scale: bool = False,
        auto_contrast: bool = False,
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
        show_stats: bool = True,
        show_controls: bool = True,
        scale_bar_visible: bool = True,
        cursor_sync: bool = True,
        map_mode: str = "sum",
        nav_index=None,
        window_e_min: float | None = None,
        window_e_max: float | None = None,
        bg_subtract: bool = False,
        bg_e_min: float | None = None,
        bg_e_max: float | None = None,
        disabled_tools=None,
        disable_display: bool = False,
        disable_histogram: bool = False,
        disable_stats: bool = False,
        disable_navigation: bool = False,
        disable_view: bool = False,
        disable_export: bool = False,
        disable_background: bool = False,
        disable_window: bool = False,
        disable_all: bool = False,
        hidden_tools=None,
        hide_display: bool = False,
        hide_histogram: bool = False,
        hide_stats: bool = False,
        hide_navigation: bool = False,
        hide_view: bool = False,
        hide_export: bool = False,
        hide_background: bool = False,
        hide_window: bool = False,
        hide_all: bool = False,
        state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()

        # Dataset3d duck typing
        self._dataset = None
        extracted_axis: np.ndarray | None = None
        extracted_unit = ""
        if hasattr(data, "array") and hasattr(data, "sampling") and getattr(data, "ndim", None) == 3:
            self._dataset = data
            if not title and getattr(data, "name", None):
                title = data.name
            if energy_axis is None:
                extracted_axis, extracted_unit = _extract_energy_axis_from_dataset(data)
            data = data.array

        # Normalize to (ny, nx, n_energy) float32
        data_np = to_numpy(data).astype(np.float32, copy=False)
        if data_np.ndim != 3:
            raise ValueError(
                f"Expected 3D array (ny, nx, n_energy), got {data_np.ndim}D."
            )
        self._data: np.ndarray = np.ascontiguousarray(data_np)

        self.ny = int(self._data.shape[0])
        self.nx = int(self._data.shape[1])
        self.n_energy = int(self._data.shape[2])

        # Energy axis
        if energy_axis is not None:
            ax = np.asarray(energy_axis, dtype=np.float64).ravel()
            if ax.size != self.n_energy:
                raise ValueError(
                    f"energy_axis has {ax.size} values but data has "
                    f"{self.n_energy} energy bins."
                )
            self._energy_axis = ax
        elif extracted_axis is not None:
            self._energy_axis = extracted_axis
        else:
            self._energy_axis = np.arange(self.n_energy, dtype=np.float64)

        # Use unit from Dataset if not overridden by caller. The kwarg
        # default is "eV", so only override when the Dataset3d unit is set
        # and the user did not pass a custom value.
        if extracted_unit and energy_unit == "eV":
            energy_unit = extracted_unit

        # ── Apply traits ────────────────────────────────────────────────────
        self.title = title or ""
        self.energy_unit = energy_unit
        self.cmap = str(cmap)
        self.log_scale = bool(log_scale)
        self.auto_contrast = bool(auto_contrast)
        self.percentile_low = float(percentile_low)
        self.percentile_high = float(percentile_high)
        self.show_stats = bool(show_stats)
        self.show_controls = bool(show_controls)
        self.scale_bar_visible = bool(scale_bar_visible)
        self.cursor_sync = bool(cursor_sync)
        self.map_mode = map_mode

        e_lo = float(self._energy_axis[0])
        e_hi = float(self._energy_axis[-1])
        if e_hi <= e_lo:
            e_hi = e_lo + 1.0

        # Integration window: full range by default
        self.window_e_min = float(window_e_min) if window_e_min is not None else e_lo
        self.window_e_max = float(window_e_max) if window_e_max is not None else e_hi

        # Background window: pre-edge first quarter by default
        bg_default_min = e_lo
        bg_default_max = e_lo + 0.25 * (e_hi - e_lo)
        self.bg_e_min = float(bg_e_min) if bg_e_min is not None else bg_default_min
        self.bg_e_max = float(bg_e_max) if bg_e_max is not None else bg_default_max
        self.bg_subtract = bool(bg_subtract)

        # Cursor at array center
        if nav_index is not None:
            if len(nav_index) != 2:
                raise ValueError("nav_index must have length 2.")
            r = max(0, min(self.ny - 1, int(nav_index[0])))
            c = max(0, min(self.nx - 1, int(nav_index[1])))
        else:
            r = self.ny // 2
            c = self.nx // 2
        self.nav_index = [r, c]

        # Tool parity
        self.disabled_tools = self._build_disabled_tools(
            disabled_tools=disabled_tools,
            disable_display=disable_display,
            disable_histogram=disable_histogram,
            disable_stats=disable_stats,
            disable_navigation=disable_navigation,
            disable_view=disable_view,
            disable_export=disable_export,
            disable_background=disable_background,
            disable_window=disable_window,
            disable_all=disable_all,
        )
        self.hidden_tools = self._build_hidden_tools(
            hidden_tools=hidden_tools,
            hide_display=hide_display,
            hide_histogram=hide_histogram,
            hide_stats=hide_stats,
            hide_navigation=hide_navigation,
            hide_view=hide_view,
            hide_export=hide_export,
            hide_background=hide_background,
            hide_window=hide_window,
            hide_all=hide_all,
        )

        # Cached fit results (recomputed on demand)
        self._bg_fit_A = 0.0
        self._bg_fit_r = 0.0
        self._bg_fit_window: tuple[float, float] | None = None
        self._bg_curve = np.zeros(self.n_energy, dtype=np.float32)

        # Send energy axis once
        self.energy_axis_bytes = self._energy_axis.astype(np.float32).tobytes()

        # Initial computations
        self._refresh_bg_if_needed()
        self._compute_spectrum()
        self._compute_map()

        # Observers
        self.observe(self._on_nav_change, names=["nav_index"])
        self.observe(
            self._on_window_change,
            names=["window_e_min", "window_e_max", "map_mode"],
        )
        self.observe(
            self._on_bg_change,
            names=["bg_subtract", "bg_e_min", "bg_e_max"],
        )

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                )
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)

    # ── Public API ─────────────────────────────────────────────────────────

    def set_image(self, data, energy_axis=None, **kw) -> Self:
        """Replace the spectrum image. Preserves display settings.

        Parameters
        ----------
        data : array_like or Dataset3d
            New 3D array of shape (ny, nx, n_energy).
        energy_axis : array_like, optional
            New energy axis. If omitted, pulled from a Dataset3d's
            sampling/origin or defaults to ``arange(n_energy)``.
        **kw : dict
            Optional ``energy_unit`` and ``title`` overrides.
        """
        extracted_axis: np.ndarray | None = None
        extracted_unit = ""
        new_title = kw.pop("title", None)
        new_unit = kw.pop("energy_unit", None)
        if hasattr(data, "array") and hasattr(data, "sampling") and getattr(data, "ndim", None) == 3:
            self._dataset = data
            if not new_title and getattr(data, "name", None):
                new_title = data.name
            if energy_axis is None:
                extracted_axis, extracted_unit = _extract_energy_axis_from_dataset(data)
            data = data.array
        else:
            self._dataset = None

        data_np = to_numpy(data).astype(np.float32, copy=False)
        if data_np.ndim != 3:
            raise ValueError(
                f"Expected 3D array (ny, nx, n_energy), got {data_np.ndim}D."
            )
        self._data = np.ascontiguousarray(data_np)
        self.ny = int(self._data.shape[0])
        self.nx = int(self._data.shape[1])
        self.n_energy = int(self._data.shape[2])

        if energy_axis is not None:
            ax = np.asarray(energy_axis, dtype=np.float64).ravel()
            if ax.size != self.n_energy:
                raise ValueError(
                    f"energy_axis has {ax.size} values but data has "
                    f"{self.n_energy} energy bins."
                )
            self._energy_axis = ax
        elif extracted_axis is not None:
            self._energy_axis = extracted_axis
        else:
            self._energy_axis = np.arange(self.n_energy, dtype=np.float64)

        if extracted_unit and not new_unit:
            new_unit = extracted_unit
        if new_unit:
            self.energy_unit = new_unit
        if new_title is not None:
            self.title = str(new_title)

        # Clamp current cursor and windows to new range
        r = max(0, min(self.ny - 1, int(self.nav_index[0])))
        c = max(0, min(self.nx - 1, int(self.nav_index[1])))
        self.nav_index = [r, c]

        e_lo = float(self._energy_axis[0])
        e_hi = float(self._energy_axis[-1])
        if e_hi <= e_lo:
            e_hi = e_lo + 1.0
        self.window_e_min = float(np.clip(self.window_e_min, e_lo, e_hi))
        self.window_e_max = float(np.clip(self.window_e_max, e_lo, e_hi))
        self.bg_e_min = float(np.clip(self.bg_e_min, e_lo, e_hi))
        self.bg_e_max = float(np.clip(self.bg_e_max, e_lo, e_hi))

        self.energy_axis_bytes = self._energy_axis.astype(np.float32).tobytes()
        self._bg_fit_window = None
        self._refresh_bg_if_needed()
        self._compute_spectrum()
        self._compute_map()
        return self

    def save_image(
        self,
        path: str | pathlib.Path,
        *,
        view: str = "map",
        format: str | None = None,
        dpi: int = 150,
    ) -> pathlib.Path:
        """Export the current view to PNG/PDF/TIFF via matplotlib.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path.
        view : str, default "map"
            ``"map"`` for the spatial map only; ``"all"`` for the
            map + spectrum side-by-side figure.
        format : str, optional
            ``"png"``, ``"pdf"``, or ``"tiff"``. Inferred from extension
            if omitted.
        dpi : int, default 150
            Output resolution.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path = pathlib.Path(path)
        fmt = (format or path.suffix.lstrip(".").lower() or "png").lower()
        if fmt not in ("png", "pdf", "tiff", "tif"):
            raise ValueError(
                f"Unsupported format: {fmt!r}. Use 'png', 'pdf', or 'tiff'."
            )
        view_key = (view or "map").lower()
        if view_key not in ("map", "all"):
            raise ValueError(f"Unknown view: {view_key!r}. Use 'map' or 'all'.")

        map_img = self._current_map().reshape(self.ny, self.nx)
        spectrum = self._current_spectrum()

        if view_key == "map":
            fig, ax = plt.subplots(figsize=(5, 4), dpi=dpi)
            im_kwargs = self._imshow_kwargs(map_img)
            ax.imshow(map_img, **im_kwargs)
            if self.title:
                ax.set_title(self.title)
            ax.set_xlabel("col")
            ax.set_ylabel("row")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=dpi)
            ax_map, ax_spec = axes
            im_kwargs = self._imshow_kwargs(map_img)
            ax_map.imshow(map_img, **im_kwargs)
            ax_map.plot(
                self.nav_index[1], self.nav_index[0], "+", color="red",
                markersize=10, markeredgewidth=1.5,
            )
            if self.title:
                ax_map.set_title(self.title)
            ax_map.set_xlabel("col")
            ax_map.set_ylabel("row")
            ax_spec.plot(self._energy_axis, spectrum, color="#4fc3f7", lw=1.2)
            ax_spec.axvspan(
                self.window_e_min, self.window_e_max, color="#4fc3f7", alpha=0.18,
            )
            if self.bg_subtract:
                ax_spec.plot(
                    self._energy_axis, self._bg_curve,
                    color="#aaaaaa", lw=1.0, linestyle="--",
                )
                ax_spec.axvspan(
                    self.bg_e_min, self.bg_e_max,
                    color="#aaaaaa", alpha=0.12,
                )
            ax_spec.set_xlabel(f"Energy ({self.energy_unit})")
            ax_spec.set_ylabel("Intensity")

        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), format=fmt if fmt != "tif" else "tiff", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def _imshow_kwargs(self, img: np.ndarray) -> dict:
        kwargs = {"cmap": self.cmap, "origin": "upper", "interpolation": "nearest"}
        if self.auto_contrast:
            kwargs["vmin"] = float(np.percentile(img, self.percentile_low))
            kwargs["vmax"] = float(np.percentile(img, self.percentile_high))
        return kwargs

    # ── State persistence ──────────────────────────────────────────────────

    def state_dict(self):
        return {
            "title": self.title,
            "energy_unit": self.energy_unit,
            "cmap": self.cmap,
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            "percentile_low": self.percentile_low,
            "percentile_high": self.percentile_high,
            "show_stats": self.show_stats,
            "show_controls": self.show_controls,
            "scale_bar_visible": self.scale_bar_visible,
            "cursor_sync": self.cursor_sync,
            "nav_index": list(self.nav_index),
            "map_mode": self.map_mode,
            "window_e_min": self.window_e_min,
            "window_e_max": self.window_e_max,
            "bg_subtract": self.bg_subtract,
            "bg_e_min": self.bg_e_min,
            "bg_e_max": self.bg_e_max,
            "disabled_tools": list(self.disabled_tools),
            "hidden_tools": list(self.hidden_tools),
        }

    def save(self, path: str):
        save_state_file(path, "SpectrumImage", self.state_dict())

    def load_state_dict(self, state):
        for key, val in state.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def summary(self):
        lines = [self.title or "SpectrumImage", "=" * 32]
        lines.append(f"Shape:    {self.ny}×{self.nx} x {self.n_energy} bins")
        lines.append(
            f"Energy:   {float(self._energy_axis[0]):.4g} – "
            f"{float(self._energy_axis[-1]):.4g} {self.energy_unit}"
        )
        lines.append(f"Cursor:   row={self.nav_index[0]}, col={self.nav_index[1]}")
        lines.append(
            f"Window:   [{self.window_e_min:.4g}, {self.window_e_max:.4g}] "
            f"{self.energy_unit} (mode={self.map_mode})"
        )
        if self.bg_subtract:
            lines.append(
                f"BG fit:   [{self.bg_e_min:.4g}, {self.bg_e_max:.4g}] "
                f"{self.energy_unit}  A={self._bg_fit_A:.4g}  r={self._bg_fit_r:.4g}"
            )
        scale = "log" if self.log_scale else "linear"
        contrast = "auto" if self.auto_contrast else "manual"
        lines.append(f"Display:  {self.cmap} | {contrast} | {scale}")
        if self.disabled_tools:
            lines.append(f"Locked:   {', '.join(self.disabled_tools)}")
        if self.hidden_tools:
            lines.append(f"Hidden:   {', '.join(self.hidden_tools)}")
        print("\n".join(lines))

    def __repr__(self) -> str:
        return (
            f"SpectrumImage(shape=({self.ny}, {self.nx}, {self.n_energy}), "
            f"mode={self.map_mode!r}, "
            f"window=[{self.window_e_min:.4g}, {self.window_e_max:.4g}] {self.energy_unit})"
        )

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def energy_axis(self) -> np.ndarray:
        return np.asarray(self._energy_axis)

    @property
    def map_image(self) -> np.ndarray:
        return self._current_map().reshape(self.ny, self.nx)

    @property
    def spectrum(self) -> np.ndarray:
        return self._current_spectrum()

    # ── Core compute ───────────────────────────────────────────────────────

    def _window_indices(self, e_min: float, e_max: float) -> tuple[int, int]:
        lo, hi = (float(e_min), float(e_max))
        if lo > hi:
            lo, hi = hi, lo
        axis = self._energy_axis
        # searchsorted gives [start, stop) such that axis[start:stop] is in [lo, hi]
        start = int(np.searchsorted(axis, lo, side="left"))
        stop = int(np.searchsorted(axis, hi, side="right"))
        start = max(0, min(self.n_energy, start))
        stop = max(start + 1, min(self.n_energy, stop))
        if stop <= start:
            stop = min(self.n_energy, start + 1)
        return (start, stop)

    def _fit_background(
        self, e_min: float | None = None, e_max: float | None = None,
    ) -> tuple[float, float]:
        """Least-squares power-law fit ``I = A * E^(-r)`` in log-log space.

        Performed per-cell once across all spatial positions by working on
        ``(ny*nx, n_e_in_window)``. Returns mean ``(A, r)`` across pixels.
        Cached values are stored for the bg curve at each pixel separately.
        """
        e_min = self.bg_e_min if e_min is None else float(e_min)
        e_max = self.bg_e_max if e_max is None else float(e_max)
        start, stop = self._window_indices(e_min, e_max)
        if stop - start < 2:
            self._bg_fit_A = 0.0
            self._bg_fit_r = 0.0
            self.bg_params = [0.0, 0.0]
            self._bg_curve = np.zeros(self.n_energy, dtype=np.float32)
            self._bg_fit_window = (e_min, e_max)
            return (0.0, 0.0)

        e_fit = self._energy_axis[start:stop]
        if not np.all(e_fit > 0):
            # Shift to keep E > 0; otherwise log undefined. Use small offset.
            shift = max(0.0, -float(e_fit.min())) + 1.0
            e_fit_shifted = e_fit + shift
            full_axis = self._energy_axis + shift
        else:
            shift = 0.0
            e_fit_shifted = e_fit
            full_axis = self._energy_axis

        log_e = np.log(e_fit_shifted).astype(np.float64)
        # Flat (n_pixels, n_e_fit) of intensities; clamp to >0 for log
        intens = self._data[..., start:stop].reshape(-1, e_fit.size).astype(np.float64)
        intens_safe = np.maximum(intens, _BG_EPS)
        log_I = np.log(intens_safe)  # (n_pixels, n_e_fit)

        # Linear LS per row in log-log space: log_I = m * log_e + b
        # → slope m = -r, intercept b = log(A)
        x = log_e
        x_mean = x.mean()
        x_centered = x - x_mean
        denom = float(np.sum(x_centered * x_centered))
        if denom <= 0:
            self._bg_fit_A = 0.0
            self._bg_fit_r = 0.0
            self.bg_params = [0.0, 0.0]
            self._bg_curve = np.zeros(self.n_energy, dtype=np.float32)
            self._bg_fit_window = (e_min, e_max)
            return (0.0, 0.0)
        # slopes per pixel
        y = log_I  # (n_pixels, n_e_fit)
        y_mean = y.mean(axis=1, keepdims=True)
        y_centered = y - y_mean
        slopes = (y_centered * x_centered[None, :]).sum(axis=1) / denom  # (n_pixels,)
        intercepts = y_mean.ravel() - slopes * x_mean
        r_per_pixel = -slopes  # since log_I = log_A - r * log_E
        A_per_pixel = np.exp(intercepts)

        # Build per-pixel BG curve evaluated on full energy axis
        full_log_e = np.log(np.maximum(full_axis, _BG_EPS))
        # bg_log = log_A - r * log_E
        # bg = exp(log_A - r * log_E) = A_per_pixel * E^(-r) with shifted axis
        bg_log = intercepts[:, None] - r_per_pixel[:, None] * full_log_e[None, :]
        bg_curve_pix = np.exp(bg_log)  # (n_pixels, n_energy)
        self._bg_curve_pixels = bg_curve_pix.astype(np.float32)

        # Cell-at-cursor curve sent to JS
        idx = self.nav_index[0] * self.nx + self.nav_index[1]
        self._bg_curve = self._bg_curve_pixels[idx]

        # Aggregate fit params (mean over valid pixels)
        valid = np.isfinite(A_per_pixel) & np.isfinite(r_per_pixel)
        if valid.any():
            A_mean = float(np.nanmean(A_per_pixel[valid]))
            r_mean = float(np.nanmean(r_per_pixel[valid]))
        else:
            A_mean, r_mean = 0.0, 0.0
        self._bg_fit_A = A_mean
        self._bg_fit_r = r_mean
        self.bg_params = [A_mean, r_mean]
        self._bg_fit_window = (e_min, e_max)
        return (A_mean, r_mean)

    def _refresh_bg_if_needed(self):
        if not self.bg_subtract:
            self._bg_fit_A = 0.0
            self._bg_fit_r = 0.0
            self.bg_params = [0.0, 0.0]
            self._bg_curve = np.zeros(self.n_energy, dtype=np.float32)
            self._bg_curve_pixels = None
            self._bg_fit_window = None
            return
        win = (float(self.bg_e_min), float(self.bg_e_max))
        if self._bg_fit_window != win:
            self._fit_background(*win)

    def _current_corrected_cube(self) -> np.ndarray:
        """Return (ny, nx, n_energy) with background subtracted if active."""
        if self.bg_subtract and getattr(self, "_bg_curve_pixels", None) is not None:
            bg_pix = self._bg_curve_pixels.reshape(self.ny, self.nx, self.n_energy)
            return self._data - bg_pix
        return self._data

    def _current_map(self) -> np.ndarray:
        start, stop = self._window_indices(self.window_e_min, self.window_e_max)
        cube = self._current_corrected_cube()
        window = cube[..., start:stop]
        mode = self.map_mode
        if mode == "sum":
            out = window.sum(axis=2)
        elif mode == "mean":
            out = window.mean(axis=2)
        elif mode == "max":
            out = window.max(axis=2)
        elif mode == "argmax":
            # Index of max within window, mapped back to energy value
            idx_in_win = window.argmax(axis=2)
            energies = self._energy_axis[start:stop]
            out = energies[idx_in_win]
        else:
            out = window.sum(axis=2)
        return out.astype(np.float32, copy=False)

    def _current_spectrum(self) -> np.ndarray:
        r, c = self.nav_index
        r = max(0, min(self.ny - 1, int(r)))
        c = max(0, min(self.nx - 1, int(c)))
        return np.ascontiguousarray(self._data[r, c, :], dtype=np.float32)

    def _compute_map(self):
        map_img = self._current_map()
        with self.hold_sync():
            self.map_stats_mean = float(map_img.mean())
            self.map_stats_min = float(map_img.min())
            self.map_stats_max = float(map_img.max())
            self.map_stats_std = float(map_img.std())
            self.map_bytes = np.ascontiguousarray(map_img).tobytes()

    def _compute_spectrum(self):
        spec = self._current_spectrum()
        # Update per-cell bg curve if we have one
        if self.bg_subtract and getattr(self, "_bg_curve_pixels", None) is not None:
            idx = self.nav_index[0] * self.nx + self.nav_index[1]
            self._bg_curve = self._bg_curve_pixels[idx]
        with self.hold_sync():
            self.spectrum_bytes = spec.tobytes()
            self.bg_curve_bytes = np.ascontiguousarray(
                self._bg_curve, dtype=np.float32,
            ).tobytes()

    # ── Observers ──────────────────────────────────────────────────────────

    def _on_nav_change(self, change=None):
        self._compute_spectrum()

    def _on_window_change(self, change=None):
        self._compute_map()

    def _on_bg_change(self, change=None):
        self._refresh_bg_if_needed()
        self._compute_spectrum()
        self._compute_map()


bind_tool_runtime_api(SpectrumImage, "SpectrumImage")
