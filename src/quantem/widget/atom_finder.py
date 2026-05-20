"""
atom_finder: Interactive atom column localization widget.

Detects atom column positions from a 2D image using blob_log + 2D Gaussian
sub-pixel refinement. Optionally partitions atoms into two sublattices by
intensity (HAADF Z-contrast workflow) and computes polarization vectors
between sublattices.

Algorithm (mirrors atomap defaults, single-image v1):
  1. Optional Gaussian preprocess (scipy.ndimage.gaussian_filter)
  2. Coarse detection (skimage.feature.blob_log)
  3. Sub-pixel refinement (per-blob 2D Gaussian fit via scipy.optimize.curve_fit)
  4. Sublattice partitioning (intensity-based, top fraction → A, rest → B)
  5. Polarization vectors (B-site displacement from 4-nearest-A centroid)
"""

import json
import pathlib
from typing import List, Optional, Self

import anywidget
import numpy as np
import traitlets

from quantem.widget.array_utils import to_numpy
from quantem.widget.io import IOResult
from quantem.widget.json_state import resolve_widget_version, save_state_file, unwrap_state_payload
from quantem.widget.tool_parity import (
    bind_tool_runtime_api,
    build_tool_groups,
    normalize_tool_groups,
)


# ─── Algorithm helpers (module-level so they can be unit-tested directly) ───


def _gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    """2D Gaussian with optional rotation. coords is a (2, N) array of (y, x)."""
    y, x = coords
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    a = (cos_t**2) / (2 * sigma_x**2) + (sin_t**2) / (2 * sigma_y**2)
    b = -np.sin(2 * theta) / (4 * sigma_x**2) + np.sin(2 * theta) / (4 * sigma_y**2)
    c = (sin_t**2) / (2 * sigma_x**2) + (cos_t**2) / (2 * sigma_y**2)
    dx = x - x0
    dy = y - y0
    return offset + amplitude * np.exp(-(a * dx * dx + 2 * b * dx * dy + c * dy * dy))


def _fit_gaussian_window(
    image: np.ndarray,
    row: float,
    col: float,
    radius: float,
    initial_sigma: float,
    rotation_enabled: bool,
) -> Optional[tuple]:
    """Fit a 2D Gaussian to a square window around (row, col).

    Returns (refined_row, refined_col, sigma_x, sigma_y, amplitude, theta) or None.
    """
    from scipy.optimize import curve_fit

    H, W = image.shape
    r = int(max(1, round(radius)))
    r0 = max(0, int(round(row)) - r)
    r1 = min(H, int(round(row)) + r + 1)
    c0 = max(0, int(round(col)) - r)
    c1 = min(W, int(round(col)) + r + 1)
    if r1 - r0 < 3 or c1 - c0 < 3:
        return None
    window = image[r0:r1, c0:c1].astype(np.float64)
    ys, xs = np.mgrid[r0:r1, c0:c1].astype(np.float64)
    offset_guess = float(window.min())
    amp_guess = float(window.max() - offset_guess)
    if amp_guess <= 0:
        return None
    sigma_guess = max(0.5, float(initial_sigma))
    p0 = [amp_guess, float(col), float(row), sigma_guess, sigma_guess, 0.0, offset_guess]
    # Bounds keep the fit centered within the window. When rotation is NOT
    # enabled we clamp theta to ~0 IN the optimizer (rather than zeroing
    # post-fit) so the fit cannot drift to a non-zero theta and then leave
    # spurious (sx ≠ sy) anisotropy behind on a rotationally-symmetric column.
    if rotation_enabled:
        theta_lo, theta_hi = -np.pi, np.pi
    else:
        theta_lo, theta_hi = -1e-6, 1e-6
    lower = [0.0, c0 - 0.5, r0 - 0.5, 0.1, 0.1, theta_lo, -np.inf]
    upper = [np.inf, c1 + 0.5, r1 + 0.5, max(radius * 4, 1.0), max(radius * 4, 1.0), theta_hi, np.inf]
    # `maxfev=1000` matches py4DSTEM / atomap defaults and scipy's auto value
    # (100·(N+1)=800 for 7 params); the previous 200 silently dropped crowded
    # fits via the bare `except Exception` below.
    try:
        popt, _ = curve_fit(
            _gaussian_2d,
            np.vstack([ys.ravel(), xs.ravel()]),
            window.ravel(),
            p0=p0,
            bounds=(lower, upper),
            maxfev=1000,
        )
    except Exception:
        return None
    amp, x0_f, y0_f, sx, sy, theta, _offset = popt
    if not rotation_enabled:
        # Bounded to ~0 above; force exactly 0 for the returned record.
        theta = 0.0
    # Reject runaway fits (center wandered outside window)
    if not (c0 - 1 <= x0_f <= c1 + 1 and r0 - 1 <= y0_f <= r1 + 1):
        return None
    return (float(y0_f), float(x0_f), float(sx), float(sy), float(amp), float(theta))


def _partition_by_intensity(
    positions: np.ndarray, intensities: np.ndarray, fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Split atoms into (A, B) index arrays by brightness."""
    n = positions.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    fraction = max(0.0, min(1.0, float(fraction)))
    n_a = max(0, min(n, int(round(n * fraction))))
    order = np.argsort(intensities)[::-1]  # brightest first
    a_idx = np.sort(order[:n_a]).astype(np.int32)
    b_idx = np.sort(order[n_a:]).astype(np.int32)
    return a_idx, b_idx


def _polarization_vectors(
    positions: np.ndarray, a_idx: np.ndarray, b_idx: np.ndarray
) -> np.ndarray:
    """For each B-site, compute displacement from the centroid of its 4 nearest A-sites.

    Returns (M, 4) float32 array of [row, col, drow, dcol].
    """
    if a_idx.size < 1 or b_idx.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    a_pos = positions[a_idx]
    b_pos = positions[b_idx]
    k = min(4, a_pos.shape[0])
    out = np.empty((b_pos.shape[0], 4), dtype=np.float32)
    for i, b in enumerate(b_pos):
        d2 = np.sum((a_pos - b) ** 2, axis=1)
        nn = np.argpartition(d2, k - 1)[:k]
        centroid = a_pos[nn].mean(axis=0)
        out[i, 0] = b[0]
        out[i, 1] = b[1]
        out[i, 2] = b[0] - centroid[0]
        out[i, 3] = b[1] - centroid[1]
    return out


class AtomFinder(anywidget.AnyWidget):
    """
    Interactive atom column localization for 2D images (HAADF, iDPC, ABF).

    Detects atom columns via blob_log coarse detection followed by per-blob
    2D Gaussian sub-pixel refinement. Optionally partitions atoms into two
    sublattices by intensity (HAADF Z-contrast assumption: A-cation brighter
    than B-cation / O) and computes polarization vectors as the displacement
    of each B-site from the centroid of its 4 nearest A-sites.

    The algorithm mirrors atomap's defaults (Nord et al. 2017) and uses
    quantem's ``Lattice`` class for the optional lattice-vector refinement
    step.

    Parameters
    ----------
    data : array_like
        2D image array ``(H, W)`` (NumPy, PyTorch, CuPy, or a ``Dataset2d`` /
        ``IOResult`` from ``IO.file()``).
    title : str, default "Atom Finder"
        Header title.
    cmap : str, default "gray"
        Colormap used for display. HAADF/iDPC typically use ``"gray"`` or
        ``"viridis"``.
    pixel_size : float, default 0.0
        Pixel size in angstroms. ``0`` means uncalibrated.
    units : str, default "Å"
        Display units for the scale bar.
    preprocess_sigma : float, default 0.0
        Gaussian blur applied before detection (pixels). ``0`` disables.
    min_sigma, max_sigma : float, defaults 2.0, 6.0
        Scale-space range for ``skimage.feature.blob_log``.
    blob_threshold : float, default 0.05
        ``blob_log`` detection threshold.
    fit_gaussian_subpixel : bool, default True
        Refine each blob centre with a 2D Gaussian fit.
    mask_radius_px : float, default 8.0
        Fit-window radius in pixels for the sub-pixel Gaussian fit.
    percent_to_nn : float, default 0.4
        Fraction of the nearest-neighbour distance used as the fit window
        when ``> 0``. Set to ``0`` to always use ``mask_radius_px``.
    rotation_enabled : bool, default False
        Allow the 2D Gaussian fit to rotate (``theta`` free parameter).
    n_sublattices : int, default 1
        ``1`` for a single sublattice, ``2`` to partition into A/B.
    sublattice_mode : str, default "intensity"
        Currently only ``"intensity"`` is supported.
        ``"kmeans_2_distances"`` is reserved for a future iteration.
    sublattice_fraction : float, default 0.5
        Top-intensity fraction assigned to sublattice A.
    polarization_active : bool, default False
        Compute B → A-centroid displacement vectors.
    auto_detect : bool, default True
        Run detection on construction. When ``False`` you must call
        :meth:`detect_atoms` manually.
    state : dict or path, optional
        Restore widget state from a state dict or JSON file.

    Notes
    -----
    The widget exposes a thin wrapper over ``quantem.imaging.lattice.Lattice``
    via :meth:`refine_lattice_vectors`: once atom positions are found, you can
    pick three positions (origin, u, v) and have the lattice class refine
    them by maximizing bilinear-interpolated intensity.

    Examples
    --------
    >>> from quantem.widget import AtomFinder
    >>> w = AtomFinder(haadf_image, pixel_size=0.18, n_sublattices=2,
    ...                polarization_active=True)
    >>> w.atom_positions.shape           # (N, 4)  row, col, intensity, sigma
    (84, 4)
    >>> w.sublattice_a_positions.shape   # bright atoms
    (42, 4)
    >>> w.polarization.shape             # (M, 4)  row, col, drow, dcol
    (42, 4)
    """

    _esm = pathlib.Path(__file__).parent / "static" / "atom_finder.js"

    # ── Versioning / image data ─────────────────────────────────────────
    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    title = traitlets.Unicode("Atom Finder").tag(sync=True)
    width = traitlets.Int(0).tag(sync=True)
    height = traitlets.Int(0).tag(sync=True)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    img_min = traitlets.Float(0.0).tag(sync=True)
    img_max = traitlets.Float(1.0).tag(sync=True)

    # ── Display ─────────────────────────────────────────────────────────
    cmap = traitlets.Unicode("gray").tag(sync=True)
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(True).tag(sync=True)
    percentile_low = traitlets.Float(2.0).tag(sync=True)
    percentile_high = traitlets.Float(98.0).tag(sync=True)
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)
    pixel_size = traitlets.Float(0.0).tag(sync=True)
    units = traitlets.Unicode("Å").tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)

    # ── Algorithm parameters ────────────────────────────────────────────
    preprocess_sigma = traitlets.Float(0.0).tag(sync=True)
    min_sigma = traitlets.Float(2.0).tag(sync=True)
    max_sigma = traitlets.Float(6.0).tag(sync=True)
    blob_threshold = traitlets.Float(0.05).tag(sync=True)
    fit_gaussian_subpixel = traitlets.Bool(True).tag(sync=True)
    mask_radius_px = traitlets.Float(8.0).tag(sync=True)
    percent_to_nn = traitlets.Float(0.4).tag(sync=True)
    rotation_enabled = traitlets.Bool(False).tag(sync=True)
    n_sublattices = traitlets.Int(1).tag(sync=True)
    sublattice_mode = traitlets.Unicode("intensity").tag(sync=True)
    sublattice_fraction = traitlets.Float(0.5).tag(sync=True)
    polarization_active = traitlets.Bool(False).tag(sync=True)
    polarization_scale = traitlets.Float(5.0).tag(sync=True)

    # ── Outputs (bytes, raw float32 / int32) ────────────────────────────
    atom_positions_bytes = traitlets.Bytes(b"").tag(sync=True)
    sublattice_a_indices_bytes = traitlets.Bytes(b"").tag(sync=True)
    sublattice_b_indices_bytes = traitlets.Bytes(b"").tag(sync=True)
    polarization_bytes = traitlets.Bytes(b"").tag(sync=True)
    n_atoms = traitlets.Int(0).tag(sync=True)

    # ── Statistics ──────────────────────────────────────────────────────
    stats_mean = traitlets.Float(0.0).tag(sync=True)
    stats_min = traitlets.Float(0.0).tag(sync=True)
    stats_max = traitlets.Float(0.0).tag(sync=True)
    stats_std = traitlets.Float(0.0).tag(sync=True)

    # ── Tool visibility ─────────────────────────────────────────────────
    disabled_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    hidden_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)

    @classmethod
    def _normalize_tool_groups(cls, tool_groups) -> List[str]:
        return normalize_tool_groups("AtomFinder", tool_groups)

    @classmethod
    def _build_disabled_tools(
        cls,
        disabled_tools=None,
        disable_display: bool = False,
        disable_histogram: bool = False,
        disable_stats: bool = False,
        disable_detection: bool = False,
        disable_sublattice: bool = False,
        disable_polarization: bool = False,
        disable_view: bool = False,
        disable_export: bool = False,
        disable_all: bool = False,
    ) -> List[str]:
        return build_tool_groups(
            "AtomFinder",
            tool_groups=disabled_tools,
            all_flag=disable_all,
            flag_map={
                "display": disable_display,
                "histogram": disable_histogram,
                "stats": disable_stats,
                "detection": disable_detection,
                "sublattice": disable_sublattice,
                "polarization": disable_polarization,
                "view": disable_view,
                "export": disable_export,
            },
        )

    @classmethod
    def _build_hidden_tools(
        cls,
        hidden_tools=None,
        hide_display: bool = False,
        hide_histogram: bool = False,
        hide_stats: bool = False,
        hide_detection: bool = False,
        hide_sublattice: bool = False,
        hide_polarization: bool = False,
        hide_view: bool = False,
        hide_export: bool = False,
        hide_all: bool = False,
    ) -> List[str]:
        return build_tool_groups(
            "AtomFinder",
            tool_groups=hidden_tools,
            all_flag=hide_all,
            flag_map={
                "display": hide_display,
                "histogram": hide_histogram,
                "stats": hide_stats,
                "detection": hide_detection,
                "sublattice": hide_sublattice,
                "polarization": hide_polarization,
                "view": hide_view,
                "export": hide_export,
            },
        )

    @traitlets.validate("disabled_tools")
    def _validate_disabled_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    @traitlets.validate("hidden_tools")
    def _validate_hidden_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    @traitlets.validate("n_sublattices")
    def _validate_n_sublattices(self, proposal):
        v = int(proposal["value"])
        if v not in (1, 2):
            raise ValueError(f"n_sublattices must be 1 or 2, got {v}.")
        return v

    @traitlets.validate("sublattice_mode")
    def _validate_sublattice_mode(self, proposal):
        v = str(proposal["value"])
        if v not in ("intensity",):
            # NOTE: "kmeans_2_distances" reserved for a follow-up iteration.
            raise ValueError(
                f"sublattice_mode must be 'intensity' (got {v!r}). "
                "'kmeans_2_distances' is planned but not yet implemented."
            )
        return v

    def __init__(
        self,
        data,
        title: str = "Atom Finder",
        cmap: str = "gray",
        pixel_size: float = 0.0,
        units: str = "Å",
        preprocess_sigma: float = 0.0,
        min_sigma: float = 2.0,
        max_sigma: float = 6.0,
        blob_threshold: float = 0.05,
        fit_gaussian_subpixel: bool = True,
        mask_radius_px: float = 8.0,
        percent_to_nn: float = 0.4,
        rotation_enabled: bool = False,
        n_sublattices: int = 1,
        sublattice_mode: str = "intensity",
        sublattice_fraction: float = 0.5,
        polarization_active: bool = False,
        polarization_scale: float = 5.0,
        log_scale: bool = False,
        auto_contrast: bool = True,
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
        scale_bar_visible: bool = True,
        show_stats: bool = True,
        show_controls: bool = True,
        auto_detect: bool = True,
        disabled_tools: Optional[List[str]] = None,
        disable_display: bool = False,
        disable_histogram: bool = False,
        disable_stats: bool = False,
        disable_detection: bool = False,
        disable_sublattice: bool = False,
        disable_polarization: bool = False,
        disable_view: bool = False,
        disable_export: bool = False,
        disable_all: bool = False,
        hidden_tools: Optional[List[str]] = None,
        hide_display: bool = False,
        hide_histogram: bool = False,
        hide_stats: bool = False,
        hide_detection: bool = False,
        hide_sublattice: bool = False,
        hide_polarization: bool = False,
        hide_view: bool = False,
        hide_export: bool = False,
        hide_all: bool = False,
        state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()

        # Auto-extract metadata from IOResult / Dataset2d before assignment
        if isinstance(data, IOResult):
            if data.title and title == "Atom Finder":
                title = data.title
            if data.pixel_size is not None and pixel_size == 0.0:
                pixel_size = data.pixel_size
            data = data.data
        if hasattr(data, "array") and hasattr(data, "sampling"):
            if hasattr(data, "name") and data.name and title == "Atom Finder":
                title = str(data.name)
            if hasattr(data, "units") and pixel_size == 0.0:
                ds_units = list(data.units)
                sampling_val = float(data.sampling[-1])
                if ds_units and ds_units[-1] == "nm":
                    pixel_size = sampling_val * 10
                    units = "Å"
                elif ds_units and ds_units[-1] in ("Å", "angstrom", "A"):
                    pixel_size = sampling_val
                    units = "Å"
            data = data.array

        self.title = title
        self.cmap = cmap
        self.pixel_size = pixel_size
        self.units = units
        self.preprocess_sigma = preprocess_sigma
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.blob_threshold = blob_threshold
        self.fit_gaussian_subpixel = fit_gaussian_subpixel
        self.mask_radius_px = mask_radius_px
        self.percent_to_nn = percent_to_nn
        self.rotation_enabled = rotation_enabled
        self.n_sublattices = n_sublattices
        self.sublattice_mode = sublattice_mode
        self.sublattice_fraction = sublattice_fraction
        self.polarization_active = polarization_active
        self.polarization_scale = polarization_scale
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.scale_bar_visible = scale_bar_visible
        self.show_stats = show_stats
        self.show_controls = show_controls
        self.disabled_tools = self._build_disabled_tools(
            disabled_tools=disabled_tools,
            disable_display=disable_display,
            disable_histogram=disable_histogram,
            disable_stats=disable_stats,
            disable_detection=disable_detection,
            disable_sublattice=disable_sublattice,
            disable_polarization=disable_polarization,
            disable_view=disable_view,
            disable_export=disable_export,
            disable_all=disable_all,
        )
        self.hidden_tools = self._build_hidden_tools(
            hidden_tools=hidden_tools,
            hide_display=hide_display,
            hide_histogram=hide_histogram,
            hide_stats=hide_stats,
            hide_detection=hide_detection,
            hide_sublattice=hide_sublattice,
            hide_polarization=hide_polarization,
            hide_view=hide_view,
            hide_export=hide_export,
            hide_all=hide_all,
        )

        # Suppress observer side-effects during initial trait assignment, then
        # run the full pipeline once at the end.
        self._recompute_blocked = True
        self._set_data(data)
        # Bind observers AFTER initial setup so we don't recompute eagerly.
        self.observe(
            self._on_input_change,
            names=[
                "preprocess_sigma",
                "min_sigma",
                "max_sigma",
                "blob_threshold",
                "fit_gaussian_subpixel",
                "mask_radius_px",
                "percent_to_nn",
                "rotation_enabled",
                "n_sublattices",
                "sublattice_mode",
                "sublattice_fraction",
                "polarization_active",
            ],
        )
        self._recompute_blocked = False

        if auto_detect:
            self._recompute_all()

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                )
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)

    # ── Data ingestion ─────────────────────────────────────────────────

    def _set_data(self, data) -> None:
        arr = to_numpy(data)
        if arr.ndim != 2:
            raise ValueError(f"AtomFinder expects a 2D image, got shape {arr.shape}.")
        arr = arr.astype(np.float32)
        if not np.isfinite(arr).all():
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        self._data = arr
        h, w = arr.shape
        self.height = h
        self.width = w
        self.img_min = float(arr.min())
        self.img_max = float(arr.max())
        self.stats_mean = float(arr.mean())
        self.stats_min = float(arr.min())
        self.stats_max = float(arr.max())
        self.stats_std = float(arr.std())
        self.frame_bytes = arr.tobytes()

    # ── Pipeline ───────────────────────────────────────────────────────

    def _preprocess(self) -> np.ndarray:
        img = self._data
        if self.preprocess_sigma and self.preprocess_sigma > 0:
            from scipy.ndimage import gaussian_filter

            img = gaussian_filter(img, sigma=float(self.preprocess_sigma))
        return img

    def detect_atoms(self) -> Self:
        """Run blob_log + optional sub-pixel Gaussian refinement.

        Idempotent — repeated calls re-detect with the current trait values.
        """
        from skimage.feature import blob_log

        img = self._preprocess()
        # Normalize to [0, 1] for blob_log's threshold to behave consistently
        img_min = float(img.min())
        img_max = float(img.max())
        if img_max > img_min:
            normalized = (img - img_min) / (img_max - img_min)
        else:
            normalized = np.zeros_like(img)
        try:
            blobs = blob_log(
                normalized,
                min_sigma=float(self.min_sigma),
                max_sigma=float(self.max_sigma),
                num_sigma=10,
                threshold=float(self.blob_threshold),
                exclude_border=False,
            )
        except Exception:
            blobs = np.empty((0, 3))

        if blobs.size == 0:
            self._atom_positions = np.empty((0, 4), dtype=np.float32)
            self._publish_positions()
            self._sublattice_a = np.empty(0, dtype=np.int32)
            self._sublattice_b = np.empty(0, dtype=np.int32)
            self._polarization = np.empty((0, 4), dtype=np.float32)
            self._publish_sublattices()
            self._publish_polarization()
            return self

        rows = blobs[:, 0].astype(np.float64)
        cols = blobs[:, 1].astype(np.float64)
        sigmas = blobs[:, 2].astype(np.float64)

        # Nearest-neighbour distance, used by percent_to_nn fit window
        nn_distance = self._estimate_nn_distance(np.column_stack([rows, cols]))

        refined = np.empty((blobs.shape[0], 4), dtype=np.float32)
        keep_mask = np.ones(blobs.shape[0], dtype=bool)
        for i in range(blobs.shape[0]):
            row, col, sigma = rows[i], cols[i], sigmas[i]
            if self.percent_to_nn > 0 and nn_distance > 0:
                radius = float(self.percent_to_nn) * nn_distance
                radius = min(radius, float(self.mask_radius_px))
            else:
                radius = float(self.mask_radius_px)
            if self.fit_gaussian_subpixel:
                fit = _fit_gaussian_window(
                    img,
                    row,
                    col,
                    radius=radius,
                    initial_sigma=sigma,
                    rotation_enabled=bool(self.rotation_enabled),
                )
                if fit is None:
                    keep_mask[i] = False
                    continue
                refined_row, refined_col, sx, sy, _amp, _theta = fit
            else:
                refined_row, refined_col = float(row), float(col)
                sx, sy = float(sigma), float(sigma)
            # Sample intensity at refined position (bilinear)
            intensity = float(_bilinear_sample(img, refined_row, refined_col))
            refined[i, 0] = refined_row
            refined[i, 1] = refined_col
            refined[i, 2] = intensity
            refined[i, 3] = 0.5 * (sx + sy)
        refined = refined[keep_mask]
        # Drop fits that landed outside the image
        H, W = img.shape
        in_bounds = (
            (refined[:, 0] >= 0)
            & (refined[:, 0] < H)
            & (refined[:, 1] >= 0)
            & (refined[:, 1] < W)
        )
        refined = refined[in_bounds]
        self._atom_positions = refined
        self._publish_positions()
        # Reset downstream
        self._sublattice_a = np.empty(0, dtype=np.int32)
        self._sublattice_b = np.empty(0, dtype=np.int32)
        self._polarization = np.empty((0, 4), dtype=np.float32)
        self._publish_sublattices()
        self._publish_polarization()
        return self

    @staticmethod
    def _estimate_nn_distance(positions: np.ndarray) -> float:
        if positions.shape[0] < 2:
            return 0.0
        # cKDTree is part of scipy.spatial (transitive dep) — fallback to brute force
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(positions)
            d, _ = tree.query(positions, k=2)
            return float(np.median(d[:, 1]))
        except Exception:
            # Brute force O(n^2) — fine for small N
            diff = positions[:, None, :] - positions[None, :, :]
            d2 = np.sum(diff * diff, axis=-1)
            np.fill_diagonal(d2, np.inf)
            return float(np.median(np.sqrt(d2.min(axis=1))))

    def partition_sublattices(self) -> Self:
        """Partition detected atoms into two sublattices when ``n_sublattices == 2``."""
        positions = getattr(self, "_atom_positions", np.empty((0, 4), dtype=np.float32))
        if self.n_sublattices != 2 or positions.shape[0] == 0:
            self._sublattice_a = np.empty(0, dtype=np.int32)
            self._sublattice_b = np.empty(0, dtype=np.int32)
            self._publish_sublattices()
            return self
        if self.sublattice_mode == "intensity":
            a_idx, b_idx = _partition_by_intensity(
                positions[:, :2], positions[:, 2], self.sublattice_fraction
            )
        else:
            # Validator already restricts this to known modes, but be defensive.
            a_idx = np.empty(0, dtype=np.int32)
            b_idx = np.empty(0, dtype=np.int32)
        self._sublattice_a = a_idx
        self._sublattice_b = b_idx
        self._publish_sublattices()
        return self

    def compute_polarization(self) -> Self:
        """Compute B-site displacement from the 4-nearest-A centroid."""
        positions = getattr(self, "_atom_positions", np.empty((0, 4), dtype=np.float32))
        if not self.polarization_active or self.n_sublattices != 2:
            self._polarization = np.empty((0, 4), dtype=np.float32)
            self._publish_polarization()
            return self
        a_idx = getattr(self, "_sublattice_a", np.empty(0, dtype=np.int32))
        b_idx = getattr(self, "_sublattice_b", np.empty(0, dtype=np.int32))
        self._polarization = _polarization_vectors(positions[:, :2], a_idx, b_idx).astype(
            np.float32
        )
        self._publish_polarization()
        return self

    def _recompute_all(self) -> None:
        if self._recompute_blocked:
            return
        self.detect_atoms()
        self.partition_sublattices()
        self.compute_polarization()

    def _on_input_change(self, change=None):
        if change is None:
            return
        name = change.get("name")
        if name in ("polarization_active",):
            # Only re-run polarization (cheap)
            self.compute_polarization()
            return
        if name in ("n_sublattices", "sublattice_mode", "sublattice_fraction"):
            self.partition_sublattices()
            self.compute_polarization()
            return
        # Detection-affecting traits → full pipeline
        self._recompute_all()

    # ── Publish helpers ────────────────────────────────────────────────

    def _publish_positions(self) -> None:
        arr = np.asarray(self._atom_positions, dtype=np.float32)
        self.atom_positions_bytes = arr.tobytes()
        self.n_atoms = int(arr.shape[0])

    def _publish_sublattices(self) -> None:
        self.sublattice_a_indices_bytes = np.asarray(
            self._sublattice_a, dtype=np.int32
        ).tobytes()
        self.sublattice_b_indices_bytes = np.asarray(
            self._sublattice_b, dtype=np.int32
        ).tobytes()

    def _publish_polarization(self) -> None:
        self.polarization_bytes = np.asarray(self._polarization, dtype=np.float32).tobytes()

    # ── Read-only views ────────────────────────────────────────────────

    @property
    def atom_positions(self) -> np.ndarray:
        """Detected atoms as ``(N, 4)`` float32 ``[row, col, intensity, sigma]``."""
        return getattr(self, "_atom_positions", np.empty((0, 4), dtype=np.float32)).copy()

    @property
    def sublattice_a_indices(self) -> np.ndarray:
        return getattr(self, "_sublattice_a", np.empty(0, dtype=np.int32)).copy()

    @property
    def sublattice_b_indices(self) -> np.ndarray:
        return getattr(self, "_sublattice_b", np.empty(0, dtype=np.int32)).copy()

    @property
    def sublattice_a_positions(self) -> np.ndarray:
        pos = self.atom_positions
        idx = self.sublattice_a_indices
        if idx.size == 0:
            return np.empty((0, 4), dtype=np.float32)
        return pos[idx]

    @property
    def sublattice_b_positions(self) -> np.ndarray:
        pos = self.atom_positions
        idx = self.sublattice_b_indices
        if idx.size == 0:
            return np.empty((0, 4), dtype=np.float32)
        return pos[idx]

    @property
    def polarization(self) -> np.ndarray:
        """Polarization vectors as ``(M, 4)`` float32 ``[row, col, drow, dcol]``."""
        return getattr(self, "_polarization", np.empty((0, 4), dtype=np.float32)).copy()

    # ── set_image / lattice refinement / export ────────────────────────

    def set_image(self, data, **kwargs) -> Self:
        """Replace the image and re-run detection. Preserves display settings."""
        if isinstance(data, IOResult):
            if data.title:
                self.title = data.title
            if data.pixel_size is not None:
                self.pixel_size = float(data.pixel_size)
            data = data.data
        if hasattr(data, "array") and hasattr(data, "sampling"):
            if hasattr(data, "name") and data.name:
                self.title = str(data.name)
            if hasattr(data, "units"):
                ds_units = list(data.units)
                sampling_val = float(data.sampling[-1])
                if ds_units and ds_units[-1] == "nm":
                    self.pixel_size = sampling_val * 10
                elif ds_units and ds_units[-1] in ("Å", "angstrom", "A"):
                    self.pixel_size = sampling_val
            data = data.array
        # Apply any caller overrides (e.g. pixel_size=...)
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self._set_data(data)
        self._recompute_all()
        return self

    def refine_lattice_vectors(
        self,
        origin,
        u,
        v,
        refine_lattice: bool = True,
    ):
        """Refine ``(origin, u, v)`` against the image via ``quantem.imaging.lattice.Lattice``.

        Returns the resulting ``Lattice`` instance. Atom positions found by
        the widget can be used to seed (origin, u, v) — typically pick a
        bright atom for the origin and two nearest-neighbour vectors.
        """
        from quantem.imaging.lattice import Lattice

        lattice = Lattice.from_data(self._data)
        lattice.define_lattice_vectors(
            origin=origin, u=u, v=v, refine_lattice=refine_lattice
        )
        return lattice

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.log_scale:
            frame = np.log1p(np.maximum(frame, 0))
        if self.auto_contrast:
            vmin = float(np.percentile(frame, self.percentile_low))
            vmax = float(np.percentile(frame, self.percentile_high))
        else:
            vmin = float(frame.min())
            vmax = float(frame.max())
        if vmax > vmin:
            normalized = np.clip((frame - vmin) / (vmax - vmin) * 255, 0, 255)
            return normalized.astype(np.uint8)
        return np.zeros(frame.shape, dtype=np.uint8)

    def save_image(
        self,
        path,
        *,
        format: str | None = None,
        dpi: int = 150,
        include_markers: bool = True,
        include_polarization: bool = True,
    ):
        """Save image with optional atom markers + polarization arrows."""
        import matplotlib.pyplot as plt
        from matplotlib import colormaps

        path = pathlib.Path(path)
        fmt = (format or path.suffix.lstrip(".").lower() or "png").lower()
        if fmt not in ("png", "pdf", "tiff", "tif"):
            raise ValueError(f"Unsupported format: {fmt!r}. Use 'png', 'pdf', or 'tiff'.")

        normalized = self._normalize_frame(self._data)
        cmap_fn = colormaps.get_cmap(self.cmap)

        fig, ax = plt.subplots(figsize=(self.width / 100, self.height / 100), dpi=dpi)
        ax.imshow(normalized, cmap=cmap_fn, vmin=0, vmax=255)
        ax.set_axis_off()
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)

        if include_markers:
            pos = self.atom_positions
            if pos.shape[0] > 0:
                if self.n_sublattices == 2:
                    a_idx = self.sublattice_a_indices
                    b_idx = self.sublattice_b_indices
                    if a_idx.size > 0:
                        ax.scatter(pos[a_idx, 1], pos[a_idx, 0], s=10, c="cyan",
                                   edgecolors="none")
                    if b_idx.size > 0:
                        ax.scatter(pos[b_idx, 1], pos[b_idx, 0], s=10, c="magenta",
                                   edgecolors="none")
                else:
                    ax.scatter(pos[:, 1], pos[:, 0], s=10, c="#00ff88", edgecolors="none")

        if include_polarization and self.polarization_active:
            pol = self.polarization
            if pol.shape[0] > 0:
                scale = float(self.polarization_scale)
                ax.quiver(
                    pol[:, 1], pol[:, 0],
                    pol[:, 3] * scale, pol[:, 2] * scale,
                    color="yellow",
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    width=0.003,
                )

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return path

    # ── State protocol ─────────────────────────────────────────────────

    def state_dict(self):
        return {
            "title": self.title,
            "cmap": self.cmap,
            "pixel_size": self.pixel_size,
            "units": self.units,
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            "percentile_low": self.percentile_low,
            "percentile_high": self.percentile_high,
            "scale_bar_visible": self.scale_bar_visible,
            "show_stats": self.show_stats,
            "show_controls": self.show_controls,
            "preprocess_sigma": self.preprocess_sigma,
            "min_sigma": self.min_sigma,
            "max_sigma": self.max_sigma,
            "blob_threshold": self.blob_threshold,
            "fit_gaussian_subpixel": self.fit_gaussian_subpixel,
            "mask_radius_px": self.mask_radius_px,
            "percent_to_nn": self.percent_to_nn,
            "rotation_enabled": self.rotation_enabled,
            "n_sublattices": self.n_sublattices,
            "sublattice_mode": self.sublattice_mode,
            "sublattice_fraction": self.sublattice_fraction,
            "polarization_active": self.polarization_active,
            "polarization_scale": self.polarization_scale,
            "disabled_tools": self.disabled_tools,
            "hidden_tools": self.hidden_tools,
        }

    def save(self, path: str):
        save_state_file(path, "AtomFinder", self.state_dict())

    def load_state_dict(self, state):
        allowed = set(self.state_dict().keys())
        # Apply non-pipeline traits first to avoid triggering multiple recomputes.
        self._recompute_blocked = True
        try:
            for key, val in state.items():
                if key in allowed and hasattr(self, key):
                    setattr(self, key, val)
        finally:
            self._recompute_blocked = False
        # Run pipeline once after restoring all traits
        self._recompute_all()

    def summary(self):
        """Print a human-readable widget summary."""
        name = self.title if self.title else "AtomFinder"
        lines = [name, "═" * 32]
        lines.append(f"Image:    {self.height}×{self.width}")
        if self.pixel_size > 0:
            lines[-1] += f"  ({self.pixel_size:.3f} {self.units}/px)"
        lines.append(
            f"Data:     min={self.stats_min:.4g}  max={self.stats_max:.4g}  mean={self.stats_mean:.4g}  std={self.stats_std:.4g}"
        )
        lines.append(
            f"Display:  cmap={self.cmap}  log={self.log_scale}  auto_contrast={self.auto_contrast}"
        )
        lines.append(
            f"Detect:   sigma=[{self.min_sigma}, {self.max_sigma}]  thr={self.blob_threshold}  preprocess_sigma={self.preprocess_sigma}"
        )
        lines.append(
            f"Refine:   subpixel={self.fit_gaussian_subpixel}  r={self.mask_radius_px}  pct_nn={self.percent_to_nn}  rotation={self.rotation_enabled}"
        )
        lines.append(
            f"Atoms:    {self.n_atoms} found  (sublattices={self.n_sublattices})"
        )
        if self.n_sublattices == 2:
            lines.append(
                f"Split:    A={self.sublattice_a_indices.size}  B={self.sublattice_b_indices.size}  fraction={self.sublattice_fraction}"
            )
            if self.polarization_active:
                lines.append(f"Polariz:  {self.polarization.shape[0]} vectors (scale={self.polarization_scale}×)")
        if self.disabled_tools:
            lines.append(f"Locked:   {', '.join(self.disabled_tools)}")
        if self.hidden_tools:
            lines.append(f"Hidden:   {', '.join(self.hidden_tools)}")
        print("\n".join(lines))

    def __repr__(self) -> str:
        parts = [f"AtomFinder({self.height}×{self.width}"]
        if self.title and self.title != "Atom Finder":
            parts.append(f"title={self.title!r}")
        parts.append(f"atoms={self.n_atoms}")
        if self.n_sublattices == 2:
            parts.append(f"sublattices=A:{self.sublattice_a_indices.size}/B:{self.sublattice_b_indices.size}")
            if self.polarization_active:
                parts.append(f"polariz={self.polarization.shape[0]}")
        if self.pixel_size > 0:
            parts.append(f"px={self.pixel_size:.3g}{self.units}")
        if self.cmap != "gray":
            parts.append(f"cmap={self.cmap}")
        return ", ".join(parts) + ")"


def _bilinear_sample(image: np.ndarray, row: float, col: float) -> float:
    """Bilinear-interpolate `image` at floating-point (row, col); clamp to bounds."""
    H, W = image.shape
    if H == 0 or W == 0:
        return 0.0
    r = float(np.clip(row, 0.0, H - 1 - 1e-6))
    c = float(np.clip(col, 0.0, W - 1 - 1e-6))
    r0 = int(np.floor(r))
    c0 = int(np.floor(c))
    r1 = min(r0 + 1, H - 1)
    c1 = min(c0 + 1, W - 1)
    dr = r - r0
    dc = c - c0
    return float(
        image[r0, c0] * (1 - dr) * (1 - dc)
        + image[r0, c1] * (1 - dr) * dc
        + image[r1, c0] * dr * (1 - dc)
        + image[r1, c1] * dr * dc
    )


bind_tool_runtime_api(AtomFinder, "AtomFinder")
