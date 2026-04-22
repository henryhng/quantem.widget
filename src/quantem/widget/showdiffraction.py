"""
showdiffraction: Interactive d-spacing measurement for 4D-STEM diffraction patterns.

Dual-panel widget (diffraction pattern + virtual image) with click-to-measure
d-spacing, snap-to-peak, and a spots table showing position, d-spacing, |g|,
and intensity.
"""

import json
import math
import pathlib
import time
from typing import List, Optional, Self

import anywidget
import numpy as np
import torch
import traitlets

from quantem.widget.array_utils import to_numpy
from quantem.widget.io import IOResult
from quantem.widget.json_state import resolve_widget_version, save_state_file, unwrap_state_payload
from quantem.widget.tool_parity import (
    bind_tool_runtime_api,
    build_tool_groups,
    normalize_tool_groups,
)

DEFAULT_BF_RATIO = 0.125


class ShowDiffraction(anywidget.AnyWidget):
    """
    Interactive d-spacing measurement for 4D-STEM diffraction patterns.

    Displays a diffraction pattern (left panel) and virtual image (right panel).
    Click on the diffraction pattern to measure d-spacing. Spots are shown in
    a table with position, d-spacing, |g|, and intensity.

    Parameters
    ----------
    data : array_like
        4D array (scan_rows, scan_cols, det_rows, det_cols) or 3D array
        (N, det_rows, det_cols) with scan_shape.
    scan_shape : tuple of int, optional
        Reshape 3D input into (scan_rows, scan_cols).
    k_pixel_size : float, optional
        Reciprocal-space pixel size in 1/angstrom per pixel.
    pixel_size : float, optional
        Real-space pixel size in angstrom per pixel.
    center : tuple of float, optional
        BF disk center as (row, col). Auto-detected if not provided.
    bf_radius : float, optional
        BF disk radius in pixels. Auto-detected if not provided.
    title : str, default ""
        Title displayed in the widget header.
    snap_enabled : bool, default False
        Enable snap-to-peak when clicking to add spots.
    snap_radius : int, default 5
        Radius in pixels for snap-to-peak search.

    Examples
    --------
    >>> from quantem.widget import ShowDiffraction
    >>> widget = ShowDiffraction(data_4d, k_pixel_size=0.025)
    >>> widget.add_spot(row=89, col=64)
    >>> for spot in widget.spots:
    ...     print(f"d = {spot['d_spacing']:.2f} Å")
    """

    _esm = pathlib.Path(__file__).parent / "static" / "showdiffraction.js"
    _css = pathlib.Path(__file__).parent / "static" / "showdiffraction.css"

    # ── Core state ───────────────────────────────────────────────────────
    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    pos_row = traitlets.Int(0).tag(sync=True)
    pos_col = traitlets.Int(0).tag(sync=True)
    shape_rows = traitlets.Int(1).tag(sync=True)
    shape_cols = traitlets.Int(1).tag(sync=True)
    det_rows = traitlets.Int(1).tag(sync=True)
    det_cols = traitlets.Int(1).tag(sync=True)

    # ── Data bytes (raw float32, JS handles colormap) ────────────────────
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    virtual_image_bytes = traitlets.Bytes(b"").tag(sync=True)

    # ── Calibration ──────────────────────────────────────────────────────
    center_row = traitlets.Float(0.0).tag(sync=True)
    center_col = traitlets.Float(0.0).tag(sync=True)
    bf_radius = traitlets.Float(0.0).tag(sync=True)
    pixel_size = traitlets.Float(1.0).tag(sync=True)
    k_pixel_size = traitlets.Float(0.0).tag(sync=True)
    k_calibrated = traitlets.Bool(False).tag(sync=True)

    # ── Global min/max for DP normalization ──────────────────────────────
    dp_global_min = traitlets.Float(0.0).tag(sync=True)
    dp_global_max = traitlets.Float(1.0).tag(sync=True)

    # ── Spots ────────────────────────────────────────────────────────────
    spots = traitlets.List(traitlets.Dict()).tag(sync=True)
    snap_enabled = traitlets.Bool(False).tag(sync=True)
    snap_radius = traitlets.Int(5).tag(sync=True)

    # ── Spot triggers (JS → Python) ──────────────────────────────────────
    _spot_add_request = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    _spot_undo_request = traitlets.Bool(False).tag(sync=True)
    _spot_clear_request = traitlets.Bool(False).tag(sync=True)

    # ── Display ──────────────────────────────────────────────────────────
    dp_colormap = traitlets.Unicode("inferno").tag(sync=True)
    dp_scale_mode = traitlets.Unicode("log").tag(sync=True)
    dp_vmin_pct = traitlets.Float(0.0).tag(sync=True)
    dp_vmax_pct = traitlets.Float(100.0).tag(sync=True)
    vi_colormap = traitlets.Unicode("inferno").tag(sync=True)
    vi_vmin_pct = traitlets.Float(0.0).tag(sync=True)
    vi_vmax_pct = traitlets.Float(100.0).tag(sync=True)

    # ── Statistics ───────────────────────────────────────────────────────
    dp_stats = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 0.0, 0.0]).tag(
        sync=True
    )
    vi_stats = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 0.0, 0.0]).tag(
        sync=True
    )

    # ── UI ───────────────────────────────────────────────────────────────
    show_stats = traitlets.Bool(True).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)

    # ── Tool visibility ──────────────────────────────────────────────────
    disabled_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    hidden_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)

    @classmethod
    def _normalize_tool_groups(cls, tool_groups) -> List[str]:
        return normalize_tool_groups("ShowDiffraction", tool_groups)

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
        disable_spots: bool = False,
        disable_all: bool = False,
    ) -> List[str]:
        return build_tool_groups(
            "ShowDiffraction",
            tool_groups=disabled_tools,
            all_flag=disable_all,
            flag_map={
                "display": disable_display,
                "histogram": disable_histogram,
                "stats": disable_stats,
                "navigation": disable_navigation,
                "view": disable_view,
                "export": disable_export,
                "spots": disable_spots,
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
        hide_spots: bool = False,
        hide_all: bool = False,
    ) -> List[str]:
        return build_tool_groups(
            "ShowDiffraction",
            tool_groups=hidden_tools,
            all_flag=hide_all,
            flag_map={
                "display": hide_display,
                "histogram": hide_histogram,
                "stats": hide_stats,
                "navigation": hide_navigation,
                "view": hide_view,
                "export": hide_export,
                "spots": hide_spots,
            },
        )

    @traitlets.validate("disabled_tools")
    def _validate_disabled_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    @traitlets.validate("hidden_tools")
    def _validate_hidden_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    def __init__(
        self,
        data,
        scan_shape: tuple[int, int] | None = None,
        k_pixel_size: float | None = None,
        pixel_size: float | None = None,
        center: tuple[float, float] | None = None,
        bf_radius: float | None = None,
        title: str = "",
        snap_enabled: bool = False,
        snap_radius: int = 5,
        dp_scale_mode: str = "log",
        show_stats: bool = True,
        show_controls: bool = True,
        disabled_tools: Optional[List[str]] = None,
        disable_display: bool = False,
        disable_histogram: bool = False,
        disable_stats: bool = False,
        disable_navigation: bool = False,
        disable_view: bool = False,
        disable_export: bool = False,
        disable_spots: bool = False,
        disable_all: bool = False,
        hidden_tools: Optional[List[str]] = None,
        hide_display: bool = False,
        hide_histogram: bool = False,
        hide_stats: bool = False,
        hide_navigation: bool = False,
        hide_view: bool = False,
        hide_export: bool = False,
        hide_spots: bool = False,
        hide_all: bool = False,
        verbose: bool = True,
        state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        _t0 = time.perf_counter()
        _verbose = verbose
        self.widget_version = resolve_widget_version()

        # ── Extract metadata from IOResult ───────────────────────────────
        if isinstance(data, IOResult):
            if not title and data.title:
                title = data.title
            if pixel_size is None and data.pixel_size is not None:
                pixel_size = data.pixel_size
            data = data.data

        # ── Dataset duck typing ──────────────────────────────────────────
        k_calibrated = False
        if hasattr(data, "sampling") and hasattr(data, "array"):
            if not title and hasattr(data, "name") and data.name:
                title = str(data.name)
            units = getattr(data, "units", ["pixels"] * 4)
            if pixel_size is None and units[0] in ("Å", "angstrom", "A", "nm"):
                pixel_size = float(data.sampling[0])
                if units[0] == "nm":
                    pixel_size *= 10
            if k_pixel_size is None and units[2] in ("1/Å", "1/A"):
                k_pixel_size = float(data.sampling[2])
                k_calibrated = True
            data = data.array

        # ── Parse and store data ─────────────────────────────────────────
        self._device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self._ingest_data(data, scan_shape)

        # ── Calibration ──────────────────────────────────────────────────
        if pixel_size is not None:
            self.pixel_size = float(pixel_size)
        if k_pixel_size is not None and k_pixel_size > 0:
            self.k_pixel_size = float(k_pixel_size)
            self.k_calibrated = True
        elif k_calibrated:
            self.k_calibrated = True

        self.title = title
        self.dp_scale_mode = dp_scale_mode
        self.snap_enabled = snap_enabled
        self.snap_radius = snap_radius
        self.show_stats = show_stats
        self.show_controls = show_controls

        self.disabled_tools = self._build_disabled_tools(
            disabled_tools=disabled_tools,
            disable_display=disable_display,
            disable_histogram=disable_histogram,
            disable_stats=disable_stats,
            disable_navigation=disable_navigation,
            disable_view=disable_view,
            disable_export=disable_export,
            disable_spots=disable_spots,
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
            hide_spots=hide_spots,
            hide_all=hide_all,
        )

        # ── Center & BF radius ───────────────────────────────────────────
        det_size = min(self.det_rows, self.det_cols)
        if center is not None:
            self.center_row = float(center[0])
            self.center_col = float(center[1])
        else:
            self.center_row = float(self.det_rows / 2)
            self.center_col = float(self.det_cols / 2)

        if bf_radius is not None:
            self.bf_radius = float(bf_radius)
        else:
            self.bf_radius = det_size * DEFAULT_BF_RATIO

        if center is None and bf_radius is None:
            self.auto_detect_center()

        # ── Initial position ─────────────────────────────────────────────
        self.pos_row = self._scan_shape[0] // 2
        self.pos_col = self._scan_shape[1] // 2

        # ── Compute virtual image (BF) ───────────────────────────────────
        self._compute_virtual_image()

        # ── Initial frame ────────────────────────────────────────────────
        self._update_frame()

        # ── Observers ────────────────────────────────────────────────────
        self.observe(self._on_position_change, names=["pos_row", "pos_col"])
        self.observe(self._on_spot_add_request, names=["_spot_add_request"])
        self.observe(self._on_spot_undo_request, names=["_spot_undo_request"])
        self.observe(self._on_spot_clear_request, names=["_spot_clear_request"])

        if _verbose:
            mem = self._data.nelement() * 4 / 1e6
            print(f"  to {self._device}: {time.perf_counter() - _t0:.2f}s ({mem:.1f} MB)")

        # ── State restoration ────────────────────────────────────────────
        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                )
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)

    # =====================================================================
    # Data ingestion (shared by __init__ and set_image)
    # =====================================================================

    def _ingest_data(self, data, scan_shape=None):
        data_np = to_numpy(data)
        is_integer = np.issubdtype(data_np.dtype, np.integer)
        data_np = data_np.astype(np.float32)
        if data_np.size > 2**31 - 1 and self._device.type == "mps":
            self._device = torch.device("cpu")
        if is_integer:
            global_max = float(data_np.max())
            p999 = float(np.percentile(data_np, 99.9))
            if global_max > p999 * 5:
                data_np[data_np > p999 * 3] = 0
        ndim = data_np.ndim
        if ndim == 3:
            if scan_shape is not None:
                self._scan_shape = scan_shape
            else:
                n = data_np.shape[0]
                side = int(n**0.5)
                if side * side != n:
                    raise ValueError(
                        f"Cannot infer square scan_shape from N={n}. "
                        f"Provide scan_shape explicitly."
                    )
                self._scan_shape = (side, side)
            self._det_shape = (data_np.shape[1], data_np.shape[2])
        elif ndim == 4:
            self._scan_shape = (data_np.shape[0], data_np.shape[1])
            self._det_shape = (data_np.shape[2], data_np.shape[3])
        else:
            raise ValueError(f"Expected 3D or 4D array, got {ndim}D")
        reshaped = data_np.reshape(
            self._scan_shape[0], self._scan_shape[1], self._det_shape[0], self._det_shape[1]
        )
        self._data = torch.from_numpy(reshaped).to(self._device)
        self.shape_rows = self._scan_shape[0]
        self.shape_cols = self._scan_shape[1]
        self.det_rows = self._det_shape[0]
        self.det_cols = self._det_shape[1]
        self.dp_global_min = float(self._data.min().item())
        self.dp_global_max = float(self._data.max().item())

    # =====================================================================
    # Position
    # =====================================================================

    @property
    def position(self) -> tuple[int, int]:
        return (self.pos_row, self.pos_col)

    @position.setter
    def position(self, value: tuple[int, int]):
        self.pos_row = int(max(0, min(value[0], self.shape_rows - 1)))
        self.pos_col = int(max(0, min(value[1], self.shape_cols - 1)))

    @property
    def scan_shape(self) -> tuple[int, int]:
        return self._scan_shape

    @property
    def detector_shape(self) -> tuple[int, int]:
        return self._det_shape

    # =====================================================================
    # Auto-detect center
    # =====================================================================

    def auto_detect_center(self) -> Self:
        """Auto-detect BF disk center and radius from summed diffraction pattern."""
        summed_dp = self._data.sum(dim=(0, 1))

        threshold = summed_dp.mean() + summed_dp.std()
        mask = summed_dp > threshold

        total = mask.sum()
        if total == 0:
            return self

        row_coords = torch.arange(self.det_rows, device=self._device, dtype=torch.float32)[
            :, None
        ]
        col_coords = torch.arange(self.det_cols, device=self._device, dtype=torch.float32)[
            None, :
        ]

        cy = float((row_coords * mask).sum() / total)
        cx = float((col_coords * mask).sum() / total)

        radius = float(torch.sqrt(total / torch.pi))

        self.center_row = cy
        self.center_col = cx
        self.bf_radius = radius
        return self

    # =====================================================================
    # Frame update
    # =====================================================================

    def _get_frame(self, row: int, col: int) -> np.ndarray:
        row = max(0, min(row, self._scan_shape[0] - 1))
        col = max(0, min(col, self._scan_shape[1] - 1))
        return self._data[row, col].cpu().numpy().astype(np.float32)

    def _update_frame(self, change=None):
        frame = self._get_frame(self.pos_row, self.pos_col)
        self.dp_stats = [
            float(frame.mean()),
            float(frame.min()),
            float(frame.max()),
            float(frame.std()),
        ]
        self.frame_bytes = frame.tobytes()

    def _on_position_change(self, change=None):
        self._update_frame()

    # =====================================================================
    # Virtual image (BF)
    # =====================================================================

    def _compute_virtual_image(self):
        row_coords = torch.arange(self.det_rows, device=self._device, dtype=torch.float32)[
            :, None
        ]
        col_coords = torch.arange(self.det_cols, device=self._device, dtype=torch.float32)[
            None, :
        ]
        r2 = (row_coords - self.center_row) ** 2 + (col_coords - self.center_col) ** 2
        mask = (r2 <= self.bf_radius**2).float()

        data_4d = self._data
        vi = torch.tensordot(data_4d, mask, dims=([2, 3], [0, 1]))

        vi_np = vi.cpu().numpy().astype(np.float32)
        self.vi_stats = [
            float(vi_np.mean()),
            float(vi_np.min()),
            float(vi_np.max()),
            float(vi_np.std()),
        ]
        self.virtual_image_bytes = vi_np.tobytes()

    # =====================================================================
    # Spots
    # =====================================================================

    def _compute_spot_info(self, row: float, col: float) -> dict:
        r_pixels = math.sqrt((row - self.center_row) ** 2 + (col - self.center_col) ** 2)

        frame = self._get_frame(self.pos_row, self.pos_col)
        r_int = max(0, min(self.det_rows - 1, int(round(row))))
        c_int = max(0, min(self.det_cols - 1, int(round(col))))
        intensity = float(frame[r_int, c_int])

        if self.k_calibrated and self.k_pixel_size > 0:
            g_magnitude = r_pixels * self.k_pixel_size
            d_spacing = 1.0 / g_magnitude if g_magnitude > 0 else None
        else:
            g_magnitude = None
            d_spacing = None

        return {
            "d_spacing": d_spacing,
            "g_magnitude": g_magnitude,
            "r_pixels": r_pixels,
            "intensity": intensity,
        }

    def _snap_to_peak(self, row: float, col: float) -> tuple[float, float]:
        frame = self._get_frame(self.pos_row, self.pos_col)
        r, c = int(round(row)), int(round(col))
        radius = self.snap_radius
        r0 = max(0, r - radius)
        r1 = min(self.det_rows, r + radius + 1)
        c0 = max(0, c - radius)
        c1 = min(self.det_cols, c + radius + 1)
        region = frame[r0:r1, c0:c1]
        if region.size == 0:
            return float(row), float(col)
        idx = np.unravel_index(region.argmax(), region.shape)
        return float(r0 + idx[0]), float(c0 + idx[1])

    def add_spot(self, row: float, col: float) -> Self:
        """Add a spot at (row, col). Snaps to local peak if snap_enabled."""
        if self.snap_enabled:
            row, col = self._snap_to_peak(row, col)
        info = self._compute_spot_info(row, col)
        spot = {
            "id": len(self.spots) + 1,
            "row": float(row),
            "col": float(col),
            **info,
        }
        self.spots = list(self.spots) + [spot]
        return self

    def clear_spots(self) -> Self:
        """Remove all spots."""
        self.spots = []
        return self

    def undo_spot(self) -> Self:
        """Remove the last added spot."""
        if self.spots:
            self.spots = list(self.spots[:-1])
        return self

    def _on_spot_add_request(self, change=None):
        val = self._spot_add_request
        if val and len(val) == 2:
            self.add_spot(val[0], val[1])
            self._spot_add_request = []

    def _on_spot_undo_request(self, change=None):
        if self._spot_undo_request:
            self.undo_spot()
            self._spot_undo_request = False

    def _on_spot_clear_request(self, change=None):
        if self._spot_clear_request:
            self.clear_spots()
            self._spot_clear_request = False

    # =====================================================================
    # set_image
    # =====================================================================

    def set_image(self, data, scan_shape: tuple[int, int] | None = None) -> Self:
        """Replace data. Preserves display settings, clears spots."""
        if isinstance(data, IOResult):
            if data.pixel_size is not None:
                self.pixel_size = float(data.pixel_size)
            if data.title:
                self.title = data.title
            data = data.data
        if hasattr(data, "sampling") and hasattr(data, "array"):
            units = getattr(data, "units", ["pixels"] * 4)
            if units[0] in ("Å", "angstrom", "A", "nm"):
                px = float(data.sampling[0])
                if units[0] == "nm":
                    px *= 10
                self.pixel_size = px
            if len(units) > 2 and units[2] in ("1/Å", "1/A"):
                self.k_pixel_size = float(data.sampling[2])
                self.k_calibrated = True
            if hasattr(data, "name") and data.name:
                self.title = str(data.name)
            data = data.array
        self._ingest_data(data, scan_shape)
        self.pos_row = min(self.pos_row, self.shape_rows - 1)
        self.pos_col = min(self.pos_col, self.shape_cols - 1)
        self.spots = []
        self.auto_detect_center()
        self._compute_virtual_image()
        self._update_frame()
        return self

    # =====================================================================
    # Export
    # =====================================================================

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalize frame to uint8 with current display settings."""
        if self.dp_scale_mode == "log":
            frame = np.log1p(np.maximum(frame, 0))
        fmin, fmax = float(frame.min()), float(frame.max())
        vmin = fmin + (self.dp_vmin_pct / 100) * (fmax - fmin)
        vmax = fmin + (self.dp_vmax_pct / 100) * (fmax - fmin)
        if vmax > vmin:
            return np.clip((frame - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        return np.zeros(frame.shape, dtype=np.uint8)

    def _normalize_vi(self, vi: np.ndarray) -> np.ndarray:
        """Normalize virtual image to uint8 with current VI display settings."""
        fmin, fmax = float(vi.min()), float(vi.max())
        vmin = fmin + (self.vi_vmin_pct / 100) * (fmax - fmin)
        vmax = fmin + (self.vi_vmax_pct / 100) * (fmax - fmin)
        if vmax > vmin:
            return np.clip((vi - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        return np.zeros(vi.shape, dtype=np.uint8)

    def save_image(
        self,
        path: str | pathlib.Path,
        *,
        view: str | None = None,
        position: tuple[int, int] | None = None,
        format: str | None = None,
        dpi: int = 150,
    ) -> pathlib.Path:
        """Save the current visualization as PNG or PDF.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path.
        view : str, optional
            One of: "diffraction", "virtual", "all". Defaults to "diffraction".
        position : tuple[int, int], optional
            Temporary scan position override as (row, col).
        format : str, optional
            "png" or "pdf". If omitted, inferred from file extension.
        dpi : int, default 150
            Output DPI metadata.

        Returns
        -------
        pathlib.Path
            The written file path.
        """
        from matplotlib import colormaps
        from PIL import Image

        export_path = pathlib.Path(path)
        view_key = view or "diffraction"
        if view_key not in ("diffraction", "virtual", "all"):
            raise ValueError(f"view must be 'diffraction', 'virtual', or 'all', got {view_key!r}")
        fmt = (format or export_path.suffix.lstrip(".").lower() or "png").lower()
        if fmt not in ("png", "pdf"):
            raise ValueError(f"Unsupported format: {fmt!r}. Use 'png' or 'pdf'.")
        if dpi <= 0:
            raise ValueError(f"dpi must be > 0, got {dpi}")

        export_path.parent.mkdir(parents=True, exist_ok=True)

        prev_row, prev_col = self.pos_row, self.pos_col
        try:
            if position is not None:
                self.pos_row = int(max(0, min(position[0], self.shape_rows - 1)))
                self.pos_col = int(max(0, min(position[1], self.shape_cols - 1)))

            images = []
            if view_key in ("diffraction", "all"):
                frame = self._get_frame(self.pos_row, self.pos_col)
                normalized = self._normalize_frame(frame)
                cmap_fn = colormaps.get_cmap(self.dp_colormap)
                rgba = (cmap_fn(normalized / 255.0) * 255).astype(np.uint8)
                images.append(Image.fromarray(rgba))

            if view_key in ("virtual", "all"):
                vi = np.frombuffer(self.virtual_image_bytes, dtype=np.float32).reshape(
                    self.shape_rows, self.shape_cols
                )
                normalized = self._normalize_vi(vi)
                cmap_fn = colormaps.get_cmap(self.vi_colormap)
                rgba = (cmap_fn(normalized / 255.0) * 255).astype(np.uint8)
                images.append(Image.fromarray(rgba))

            if len(images) == 1:
                image = images[0]
            else:
                total_w = sum(im.width for im in images)
                max_h = max(im.height for im in images)
                image = Image.new("RGBA", (total_w, max_h))
                x = 0
                for im in images:
                    image.paste(im, (x, 0))
                    x += im.width

            if fmt == "pdf":
                Image.init()
                image = image.convert("RGB")
                image.save(str(export_path), format="PDF", resolution=dpi)
            else:
                image.save(str(export_path), format="PNG", dpi=(dpi, dpi))
        finally:
            self.pos_row = prev_row
            self.pos_col = prev_col

        return export_path

    # =====================================================================
    # State protocol
    # =====================================================================

    def state_dict(self):
        return {
            "title": self.title,
            "pos_row": self.pos_row,
            "pos_col": self.pos_col,
            "pixel_size": self.pixel_size,
            "k_pixel_size": self.k_pixel_size,
            "k_calibrated": self.k_calibrated,
            "center_row": self.center_row,
            "center_col": self.center_col,
            "bf_radius": self.bf_radius,
            "spots": list(self.spots),
            "snap_enabled": self.snap_enabled,
            "snap_radius": self.snap_radius,
            "dp_colormap": self.dp_colormap,
            "dp_scale_mode": self.dp_scale_mode,
            "dp_vmin_pct": self.dp_vmin_pct,
            "dp_vmax_pct": self.dp_vmax_pct,
            "vi_colormap": self.vi_colormap,
            "vi_vmin_pct": self.vi_vmin_pct,
            "vi_vmax_pct": self.vi_vmax_pct,
            "show_stats": self.show_stats,
            "show_controls": self.show_controls,
            "disabled_tools": self.disabled_tools,
            "hidden_tools": self.hidden_tools,
        }

    def save(self, path: str):
        """Save widget state to a JSON file."""
        save_state_file(path, "ShowDiffraction", self.state_dict())

    def load_state_dict(self, state):
        """Restore widget state from a dict."""
        allowed_keys = set(self.state_dict().keys())
        pending_pos_row = state.get("pos_row", None)
        pending_pos_col = state.get("pos_col", None)
        for key, val in state.items():
            if key in {"pos_row", "pos_col"}:
                continue
            if key in allowed_keys:
                setattr(self, key, val)
        if pending_pos_row is not None or pending_pos_col is not None:
            row = int(self.pos_row if pending_pos_row is None else pending_pos_row)
            col = int(self.pos_col if pending_pos_col is None else pending_pos_col)
            self.pos_row = int(max(0, min(row, self.shape_rows - 1)))
            self.pos_col = int(max(0, min(col, self.shape_cols - 1)))

    def summary(self):
        """Print a human-readable summary of the widget state."""
        name = self.title if self.title else "ShowDiffraction"
        lines = [name, "═" * 32]
        lines.append(f"Scan:     {self.shape_rows}×{self.shape_cols} ({self.pixel_size:.2f} Å/px)")
        k_unit = "1/Å" if self.k_calibrated else "px"
        k_val = f"{self.k_pixel_size:.4f}" if self.k_calibrated else "uncalibrated"
        lines.append(f"Detector: {self.det_rows}×{self.det_cols} ({k_val} {k_unit}/px)")
        lines.append(f"Position: ({self.pos_row}, {self.pos_col})")
        lines.append(
            f"Center:   ({self.center_row:.1f}, {self.center_col:.1f})  BF r={self.bf_radius:.1f} px"
        )
        lines.append(f"Spots:    {len(self.spots)}")
        if self.spots:
            for s in self.spots[:5]:
                d = f"{s['d_spacing']:.3f} Å" if s.get("d_spacing") else f"{s['r_pixels']:.1f} px"
                lines.append(f"  #{s['id']} ({s['row']:.1f}, {s['col']:.1f}) d={d}")
            if len(self.spots) > 5:
                lines.append(f"  ... +{len(self.spots) - 5} more")
        lines.append(f"Display:  {self.dp_colormap} | {self.dp_scale_mode}")
        if self.snap_enabled:
            lines.append(f"Snap:     radius={self.snap_radius}")
        print("\n".join(lines))

    def __repr__(self) -> str:
        k_unit = "1/Å" if self.k_calibrated else "px"
        shape = f"({self.shape_rows}, {self.shape_cols}, {self.det_rows}, {self.det_cols})"
        title_info = f", title='{self.title}'" if self.title else ""
        spots_info = f", spots={len(self.spots)}" if self.spots else ""
        return (
            f"ShowDiffraction(shape={shape}, "
            f"sampling=({self.pixel_size} Å, {self.k_pixel_size} {k_unit}), "
            f"pos=({self.pos_row}, {self.pos_col}){spots_info}{title_info})"
        )

    def free(self):
        """Free GPU memory."""
        if hasattr(self, "_data"):
            del self._data
        import gc

        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()


bind_tool_runtime_api(ShowDiffraction, "ShowDiffraction")
