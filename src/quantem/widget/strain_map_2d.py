"""
strain_map_2d: Interactive 2D strain map widget for 4D-STEM Bragg peak data.

Given per-scan-pixel Bragg peak positions (qy, qx) in detector pixels, fits
reference lattice vectors (g1, g2) from a user-selected strain-free ROI and
computes a per-pixel strain tensor (e_xx, e_yy, e_xy) and rigid rotation
(theta) following the Hytch-style convention used by py4DSTEM:

    F = G_ref @ inv(G_local)
    eps = 0.5 * (F + F.T) - I
    theta = 0.5 * (F[1,0] - F[0,1])

Conventions:
- Input peak coordinates are (qy_px, qx_px) — (row, col) in detector pixel space.
- g1 / g2 are stored as length-2 lists [qy, qx] in detector pixels.
- Extensive strain is positive; rotation is CCW positive in radians.
- This widget does NOT detect Bragg peaks. Provide the peak array yourself
  (e.g., from py4DSTEM or another disk-detection pipeline).
"""

import json
import pathlib
from typing import Dict, List, Optional

import anywidget
import numpy as np
import traitlets

# Hard dependencies on quantem core
from quantem.diffractive_imaging.ptycho_utils import AffineTransform  # noqa: F401  (cited)
from quantem.imaging.lattice import Lattice  # noqa: F401  (cited)

from quantem.widget.array_utils import to_numpy
from quantem.widget.json_state import resolve_widget_version, save_state_file, unwrap_state_payload
from quantem.widget.tool_parity import (
    bind_tool_runtime_api,
    build_tool_groups,
    normalize_tool_groups,
)


def _peaks_to_dense(data) -> np.ndarray:
    """Accept either a dense ndarray (R_Nx, R_Ny, N_peaks, 2) or a dict
    {(rx, ry): ndarray (Nk, 2)} and return the dense float32 representation
    with np.nan padding for missing peaks."""
    if isinstance(data, dict):
        if not data:
            raise ValueError("Empty peak dict.")
        rxs = [int(k[0]) for k in data.keys()]
        rys = [int(k[1]) for k in data.keys()]
        R_Nx = max(rxs) + 1
        R_Ny = max(rys) + 1
        N_peaks = max(int(np.asarray(v).shape[0]) for v in data.values())
        out = np.full((R_Nx, R_Ny, N_peaks, 2), np.nan, dtype=np.float32)
        for (rx, ry), arr in data.items():
            a = np.asarray(arr, dtype=np.float32)
            if a.ndim != 2 or a.shape[1] != 2:
                raise ValueError(
                    f"Each peak list must have shape (Nk, 2); got {a.shape} at ({rx},{ry})."
                )
            n = a.shape[0]
            out[int(rx), int(ry), :n] = a
        return out

    arr = to_numpy(data).astype(np.float32, copy=False)
    if arr.ndim != 4 or arr.shape[-1] != 2:
        raise ValueError(
            f"Expected peaks of shape (R_Nx, R_Ny, N_peaks, 2); got {arr.shape}."
        )
    return arr


def _safe_lstsq(A: np.ndarray, b: np.ndarray):
    """Least-squares solver returning None on degenerate input."""
    try:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        if not np.all(np.isfinite(x)):
            return None
        return x
    except np.linalg.LinAlgError:
        return None


def _initial_g_from_peaks(peaks_xy: np.ndarray):
    """Given (N, 2) peak coordinates with NaN-padding removed, return an
    initial estimate of (g1, g2) as a pair of length-2 arrays in the same
    coordinate frame.

    Strategy: take the two shortest non-colinear vectors from the centroid
    out to the surrounding peaks. This is a coarse seed that the
    least-squares refit corrects."""
    if peaks_xy.shape[0] < 3:
        raise ValueError("Need at least 3 valid peaks to seed initial g1/g2.")
    center = np.nanmean(peaks_xy, axis=0)
    v = peaks_xy - center[None, :]
    r = np.linalg.norm(v, axis=1)
    order = np.argsort(r)
    # Skip vectors at r==0 (peak coincident with centroid)
    candidates = [v[i] for i in order if r[i] > 1e-6]
    if len(candidates) < 2:
        raise ValueError("Could not find two non-zero candidate vectors.")
    g1 = candidates[0]
    # find smallest vector that is not (anti-)parallel to g1
    g2 = None
    g1n = g1 / np.linalg.norm(g1)
    for cand in candidates[1:]:
        cn = cand / max(np.linalg.norm(cand), 1e-12)
        cross = abs(g1n[0] * cn[1] - g1n[1] * cn[0])  # |sin angle|
        if cross > 0.2:  # > ~11.5 degrees
            g2 = cand
            break
    if g2 is None:
        raise ValueError("Could not find a non-collinear second lattice vector.")
    return g1.astype(np.float32), g2.astype(np.float32)


def _assign_indices(peaks_xy: np.ndarray, g1: np.ndarray, g2: np.ndarray):
    """For each peak vector q, solve [g1 g2] · [h k]^T ≈ q. Round to integers."""
    if peaks_xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int32)
    M = np.column_stack([g1, g2])  # shape (2,2) columns = g1, g2
    hk_float = np.linalg.lstsq(M, peaks_xy.T, rcond=None)[0].T  # (N, 2)
    return np.round(hk_float).astype(np.int32)


def _fit_g_from_indexed(peaks_xy: np.ndarray, hk: np.ndarray):
    """Given (N, 2) peak vectors and (N, 2) integer indices, solve for
    G = [g1 g2] (2x2 with g1, g2 as columns) such that q ≈ G @ [h, k]^T."""
    if peaks_xy.shape[0] < 2:
        return None
    A = hk.astype(np.float64)  # (N, 2)
    b = peaks_xy.astype(np.float64)  # (N, 2)
    # We have A @ G^T ≈ b (each row: hk @ G^T = q). Solve for G^T.
    Gt = _safe_lstsq(A, b)
    if Gt is None:
        return None
    return Gt.T  # 2x2: columns are g1, g2


class StrainMap2D(anywidget.AnyWidget):
    """
    Interactive 2D strain map viewer for 4D-STEM Bragg peak data.

    Computes per-scan-pixel strain tensor components (eps_xx, eps_yy,
    eps_xy) and rigid rotation (theta) given an array of indexed Bragg
    peaks and a user-defined strain-free reference region. The widget
    displays the four channels in a 2×2 grid with a shared diverging
    colormap centered at zero.

    The peak finding step is NOT performed here — supply the peak array
    from py4DSTEM, the feat/show4dstem-bragg branch, or any other
    detector.

    Parameters
    ----------
    data : array_like or dict
        Per-pixel peak coordinates. Two accepted forms:

        - Dense ndarray of shape ``(R_Nx, R_Ny, N_peaks, 2)`` storing
          ``(qy_px, qx_px)`` in detector pixels with ``np.nan`` padding
          for missing peaks.
        - Dict ``{(rx, ry): ndarray (Nk, 2)}`` (sparse-friendly).

        The dense form is canonical; the dict form is converted internally.
    intensities : array_like, optional
        Optional per-peak intensities of shape ``(R_Nx, R_Ny, N_peaks)``,
        used as weights in least-squares fits. Defaults to uniform.
    title : str, default "Strain Map"
    cmap_strain : str, default "RdBu"
        Diverging colormap for the 4 strain channels.
    cmap_theta : str, default "RdBu"
        Colormap for the rotation channel.
    vmin_pct, vmax_pct : float
        Percentile clip for auto-contrast (defaults 2, 98).
    ref_roi : dict, optional
        Rectangle in scan space ``{"top","left","bottom","right"}`` defining
        the strain-free reference region. Defaults to a small top-left
        square.
    max_peak_spacing_px : float, default 6.0
        Assignment radius (detector pixels) for matching peaks to ideal
        (h,k)·G_ref positions during the per-pixel fit.
    unit : str, default "strain"
        Display unit for strain channels: "strain" (raw) or "%".

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget import StrainMap2D
    >>> # peaks shape: (R_Nx, R_Ny, N_peaks, 2) with NaN padding
    >>> w = StrainMap2D(peaks, ref_roi={"top": 0, "left": 0, "bottom": 4, "right": 4})
    """

    _esm = pathlib.Path(__file__).parent / "static" / "strain_map_2d.js"
    _css = pathlib.Path(__file__).parent / "static" / "strain_map_2d.css"

    # Core shape
    R_Nx = traitlets.Int(1).tag(sync=True)
    R_Ny = traitlets.Int(1).tag(sync=True)
    N_peaks = traitlets.Int(0).tag(sync=True)
    title = traitlets.Unicode("Strain Map").tag(sync=True)

    # Display
    cmap_strain = traitlets.Unicode("RdBu").tag(sync=True)
    cmap_theta = traitlets.Unicode("RdBu").tag(sync=True)
    vmin_pct = traitlets.Float(2.0).tag(sync=True)
    vmax_pct = traitlets.Float(98.0).tag(sync=True)
    unit = traitlets.Unicode("strain").tag(sync=True)

    # Reference region (scan-space rectangle)
    ref_roi = traitlets.Dict(
        default_value={"top": 0, "left": 0, "bottom": 8, "right": 8}
    ).tag(sync=True)
    max_peak_spacing_px = traitlets.Float(6.0).tag(sync=True)

    # Refined reference lattice vectors (qy, qx) in detector pixels
    g1 = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0]).tag(sync=True)
    g2 = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0]).tag(sync=True)

    # Strain channel bytes (float32, R_Nx × R_Ny)
    e_xx_bytes = traitlets.Bytes(b"").tag(sync=True)
    e_yy_bytes = traitlets.Bytes(b"").tag(sync=True)
    e_xy_bytes = traitlets.Bytes(b"").tag(sync=True)
    theta_bytes = traitlets.Bytes(b"").tag(sync=True)
    mask_bytes = traitlets.Bytes(b"").tag(sync=True)

    # Stats (per channel)
    stats_e_xx = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 0.0]).tag(sync=True)
    stats_e_yy = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 0.0]).tag(sync=True)
    stats_e_xy = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 0.0]).tag(sync=True)
    stats_theta = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 0.0]).tag(sync=True)

    # UI
    show_stats = traitlets.Bool(True).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    canvas_size = traitlets.Int(0).tag(sync=True)
    disabled_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    hidden_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)

    @classmethod
    def _normalize_tool_groups(cls, tool_groups) -> List[str]:
        return normalize_tool_groups("StrainMap2D", tool_groups)

    @classmethod
    def _build_disabled_tools(
        cls,
        disabled_tools=None,
        disable_display: bool = False,
        disable_stats: bool = False,
        disable_reference: bool = False,
        disable_strain: bool = False,
        disable_export: bool = False,
        disable_view: bool = False,
        disable_all: bool = False,
    ) -> List[str]:
        return build_tool_groups(
            "StrainMap2D",
            tool_groups=disabled_tools,
            all_flag=disable_all,
            flag_map={
                "display": disable_display,
                "stats": disable_stats,
                "reference": disable_reference,
                "strain": disable_strain,
                "export": disable_export,
                "view": disable_view,
            },
        )

    @classmethod
    def _build_hidden_tools(
        cls,
        hidden_tools=None,
        hide_display: bool = False,
        hide_stats: bool = False,
        hide_reference: bool = False,
        hide_strain: bool = False,
        hide_export: bool = False,
        hide_view: bool = False,
        hide_all: bool = False,
    ) -> List[str]:
        return build_tool_groups(
            "StrainMap2D",
            tool_groups=hidden_tools,
            all_flag=hide_all,
            flag_map={
                "display": hide_display,
                "stats": hide_stats,
                "reference": hide_reference,
                "strain": hide_strain,
                "export": hide_export,
                "view": hide_view,
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
        intensities=None,
        title: str = "Strain Map",
        cmap_strain: str = "RdBu",
        cmap_theta: str = "RdBu",
        vmin_pct: float = 2.0,
        vmax_pct: float = 98.0,
        ref_roi: Optional[Dict[str, int]] = None,
        max_peak_spacing_px: float = 6.0,
        unit: str = "strain",
        show_stats: bool = True,
        show_controls: bool = True,
        canvas_size: int = 0,
        auto_compute: bool = True,
        disabled_tools: Optional[List[str]] = None,
        disable_display: bool = False,
        disable_stats: bool = False,
        disable_reference: bool = False,
        disable_strain: bool = False,
        disable_export: bool = False,
        disable_view: bool = False,
        disable_all: bool = False,
        hidden_tools: Optional[List[str]] = None,
        hide_display: bool = False,
        hide_stats: bool = False,
        hide_reference: bool = False,
        hide_strain: bool = False,
        hide_export: bool = False,
        hide_view: bool = False,
        hide_all: bool = False,
        state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()

        # Ingest peaks
        self._peaks = _peaks_to_dense(data)
        self._intensities = self._validate_intensities(intensities)

        R_Nx, R_Ny, N_peaks, _ = self._peaks.shape
        self.R_Nx = int(R_Nx)
        self.R_Ny = int(R_Ny)
        self.N_peaks = int(N_peaks)

        # Display traits
        self.title = title
        self.cmap_strain = cmap_strain
        self.cmap_theta = cmap_theta
        self.vmin_pct = float(vmin_pct)
        self.vmax_pct = float(vmax_pct)
        self.unit = unit
        self.max_peak_spacing_px = float(max_peak_spacing_px)
        if ref_roi is None:
            self.ref_roi = {
                "top": 0,
                "left": 0,
                "bottom": min(8, R_Nx),
                "right": min(8, R_Ny),
            }
        else:
            self.ref_roi = self._validate_ref_roi(ref_roi)

        self.show_stats = show_stats
        self.show_controls = show_controls
        self.canvas_size = canvas_size
        self.disabled_tools = self._build_disabled_tools(
            disabled_tools=disabled_tools,
            disable_display=disable_display,
            disable_stats=disable_stats,
            disable_reference=disable_reference,
            disable_strain=disable_strain,
            disable_export=disable_export,
            disable_view=disable_view,
            disable_all=disable_all,
        )
        self.hidden_tools = self._build_hidden_tools(
            hidden_tools=hidden_tools,
            hide_display=hide_display,
            hide_stats=hide_stats,
            hide_reference=hide_reference,
            hide_strain=hide_strain,
            hide_export=hide_export,
            hide_view=hide_view,
            hide_all=hide_all,
        )

        # Output arrays (filled by compute_strain)
        self._e_xx = np.full((R_Nx, R_Ny), np.nan, dtype=np.float32)
        self._e_yy = np.full((R_Nx, R_Ny), np.nan, dtype=np.float32)
        self._e_xy = np.full((R_Nx, R_Ny), np.nan, dtype=np.float32)
        self._theta = np.full((R_Nx, R_Ny), np.nan, dtype=np.float32)
        self._mask = np.zeros((R_Nx, R_Ny), dtype=np.uint8)

        if auto_compute:
            try:
                self.fit_reference()
                self.compute_strain()
            except Exception:
                # Compute failures (e.g. no peaks in ROI) leave outputs at NaN
                self._push_bytes()
        else:
            self._push_bytes()

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                )
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)

    # ---------------------------------------------------------------------
    # Validation helpers
    # ---------------------------------------------------------------------

    def _validate_intensities(self, intensities) -> Optional[np.ndarray]:
        if intensities is None:
            return None
        arr = to_numpy(intensities).astype(np.float32, copy=False)
        expected = self._peaks.shape[:3]
        if arr.shape != expected:
            raise ValueError(
                f"intensities must have shape {expected}; got {arr.shape}."
            )
        return arr

    def _validate_ref_roi(self, roi: Dict[str, int]) -> Dict[str, int]:
        keys = {"top", "left", "bottom", "right"}
        if not isinstance(roi, dict) or set(roi.keys()) < keys:
            raise ValueError(
                "ref_roi must be a dict with keys top/left/bottom/right."
            )
        top = max(0, int(roi["top"]))
        left = max(0, int(roi["left"]))
        bottom = min(self.R_Nx, int(roi["bottom"]))
        right = min(self.R_Ny, int(roi["right"]))
        if bottom <= top or right <= left:
            raise ValueError(
                f"ref_roi must have bottom>top and right>left; got "
                f"top={top}, left={left}, bottom={bottom}, right={right}."
            )
        return {"top": top, "left": left, "bottom": bottom, "right": right}

    @traitlets.validate("ref_roi")
    def _validate_ref_roi_trait(self, proposal):
        return self._validate_ref_roi(proposal["value"])

    # ---------------------------------------------------------------------
    # Reference & strain computation
    # ---------------------------------------------------------------------

    def fit_reference(self) -> "StrainMap2D":
        """Fit reference lattice vectors (g1, g2) from peaks inside ref_roi.

        Strategy:
        1. Collect all valid peak coordinates inside the reference rectangle.
        2. Compute a seed (g1, g2) from the two shortest non-colinear
           displacement vectors from the centroid.
        3. Assign integer (h,k) indices to each peak using the seed.
        4. Least-squares refit ``G = [g1 g2]`` (columns) such that
           ``q ≈ G @ [h,k]^T``.
        """
        top = self.ref_roi["top"]
        left = self.ref_roi["left"]
        bottom = self.ref_roi["bottom"]
        right = self.ref_roi["right"]
        block = self._peaks[top:bottom, left:right]  # (rN, rM, N_peaks, 2)
        flat = block.reshape(-1, 2)
        valid = flat[np.isfinite(flat).all(axis=1)]
        if valid.shape[0] < 3:
            raise ValueError(
                "Not enough valid peaks in ref_roi to fit reference lattice."
            )

        g1_seed, g2_seed = _initial_g_from_peaks(valid)

        # Refinement pass: assign integer indices, then least-squares
        for _ in range(3):
            hk = _assign_indices(valid, g1_seed, g2_seed)
            # filter rows where (h, k) == (0, 0) — those carry no information
            keep = ~((hk[:, 0] == 0) & (hk[:, 1] == 0))
            if int(keep.sum()) < 2:
                break
            G = _fit_g_from_indexed(valid[keep], hk[keep])
            if G is None:
                break
            new_g1 = G[:, 0].astype(np.float32)
            new_g2 = G[:, 1].astype(np.float32)
            delta = float(
                np.linalg.norm(new_g1 - g1_seed) + np.linalg.norm(new_g2 - g2_seed)
            )
            g1_seed, g2_seed = new_g1, new_g2
            if delta < 1e-5:
                break

        self.g1 = [float(g1_seed[0]), float(g1_seed[1])]
        self.g2 = [float(g2_seed[0]), float(g2_seed[1])]
        return self

    def compute_strain(self) -> "StrainMap2D":
        """Compute per-scan-pixel strain map.

        Coordinate convention
        ---------------------
        Input peaks are stored as ``(qy_px, qx_px)``. Internally this
        method swaps to ``(qx, qy)`` order before applying the linear
        algebra so the sign / index convention exactly matches py4DSTEM's
        :func:`py4DSTEM.process.strain.latticevectors.get_strain_from_reference_g1g2`.

        Algorithm (per scan pixel)
        --------------------------
        1. Drop NaN-padded peaks.
        2. Predict ideal ``(h,k)·G_ref`` positions and assign each peak to
           its nearest predicted index within ``max_peak_spacing_px``.
        3. Least-squares fit ``G_local`` (2×2 columns) such that
           ``q ≈ G_local @ [h, k]^T``.
        4. Solve ``M_rows @ beta = alpha_rows`` with rows being the ref
           and local g-vectors in ``(qx, qy)`` order. Take ``beta = .T``
           of the lstsq result, exactly like py4DSTEM.
        5. Decompose:

               e_xx   = 1 - beta[0, 0]
               e_yy   = 1 - beta[1, 1]
               e_xy   = -0.5 * (beta[0, 1] + beta[1, 0])
               theta  = 0.5 * (beta[0, 1] - beta[1, 0])

        Here ``e_xx`` is the infinitesimal strain along the first
        reference lattice vector (``g1``) and ``e_yy`` along ``g2``.
        Extensive strain is positive and CCW rotation is positive
        (radians). Pixels with fewer than 2 matched non-zero peaks get
        ``mask=0`` and NaN output.
        """
        # Swap (qy, qx) -> (qx, qy) for the strain math
        g1_xy = np.array([self.g1[1], self.g1[0]], dtype=np.float64)
        g2_xy = np.array([self.g2[1], self.g2[0]], dtype=np.float64)
        g_ref_xy = np.column_stack([g1_xy, g2_xy])  # columns: g1, g2 (in xy)
        if not np.all(np.isfinite(g_ref_xy)) or abs(np.linalg.det(g_ref_xy)) < 1e-12:
            raise ValueError("Reference g1/g2 are unset or degenerate. Call fit_reference() first.")

        # Reference matrix in py4DSTEM row form: rows are g1, g2 in (x, y)
        M_rows = np.array([g1_xy, g2_xy], dtype=np.float64)

        R_Nx, R_Ny, _, _ = self._peaks.shape
        self._e_xx[:] = np.nan
        self._e_yy[:] = np.nan
        self._e_xy[:] = np.nan
        self._theta[:] = np.nan
        self._mask[:] = 0

        max_r2 = float(self.max_peak_spacing_px) ** 2

        for rx in range(R_Nx):
            for ry in range(R_Ny):
                peaks = self._peaks[rx, ry]
                valid_mask = np.isfinite(peaks).all(axis=1)
                if not np.any(valid_mask):
                    continue
                # Swap each peak (qy, qx) -> (qx, qy)
                q_yx = peaks[valid_mask].astype(np.float64)
                q = np.column_stack([q_yx[:, 1], q_yx[:, 0]])  # (N, 2) in (x, y)
                weights = None
                if self._intensities is not None:
                    weights = self._intensities[rx, ry][valid_mask].astype(np.float64)

                # Assign each peak to its closest predicted index using g_ref_xy
                hk_float = np.linalg.lstsq(g_ref_xy, q.T, rcond=None)[0].T  # (N, 2)
                hk = np.round(hk_float).astype(np.int32)
                pred = (g_ref_xy @ hk.T).T  # (N, 2)
                r2 = np.sum((q - pred) ** 2, axis=1)
                ok = r2 <= max_r2
                ok &= ~((hk[:, 0] == 0) & (hk[:, 1] == 0))
                if int(ok.sum()) < 2:
                    continue

                hk_ok = hk[ok].astype(np.float64)
                q_ok = q[ok]
                if weights is not None:
                    w = weights[ok]
                    w = np.maximum(w, 0.0)
                    if w.sum() <= 0:
                        w = np.ones_like(w)
                    sw = np.sqrt(w)[:, None]
                    A = hk_ok * sw
                    b = q_ok * sw
                else:
                    A = hk_ok
                    b = q_ok

                # Solve A @ G_local_cols^T ≈ b => G_local_cols^T = lstsq(A, b)
                Gt = _safe_lstsq(A, b)
                if Gt is None:
                    continue
                # Gt is (2,2) where Gt[h,axis] gives the projection of the
                # h-th lattice direction onto the axis. Row-wise:
                #   row 0 = g1_local in (x,y) — exactly the format py4DSTEM
                #   uses for alpha.
                alpha_rows = Gt  # rows: g1_local, g2_local in (x, y)
                if abs(np.linalg.det(alpha_rows)) < 1e-12:
                    continue

                beta = _safe_lstsq(M_rows, alpha_rows)
                if beta is None:
                    continue
                beta = beta.T  # match py4DSTEM's `.T` after lstsq

                self._e_xx[rx, ry] = 1.0 - beta[0, 0]
                self._e_yy[rx, ry] = 1.0 - beta[1, 1]
                self._e_xy[rx, ry] = -0.5 * (beta[0, 1] + beta[1, 0])
                self._theta[rx, ry] = 0.5 * (beta[0, 1] - beta[1, 0])
                self._mask[rx, ry] = 1

        self._update_stats()
        self._push_bytes()
        return self

    # ---------------------------------------------------------------------
    # set_image / refresh
    # ---------------------------------------------------------------------

    def set_image(self, data, intensities=None) -> "StrainMap2D":
        """Replace input peaks (and optional intensities) and recompute."""
        self._peaks = _peaks_to_dense(data)
        self._intensities = self._validate_intensities(intensities)
        R_Nx, R_Ny, N_peaks, _ = self._peaks.shape
        self.R_Nx = int(R_Nx)
        self.R_Ny = int(R_Ny)
        self.N_peaks = int(N_peaks)
        # Clip ref_roi to new shape
        self.ref_roi = self._validate_ref_roi({
            "top": min(self.ref_roi["top"], max(0, R_Nx - 1)),
            "left": min(self.ref_roi["left"], max(0, R_Ny - 1)),
            "bottom": min(self.ref_roi["bottom"], R_Nx),
            "right": min(self.ref_roi["right"], R_Ny),
        })
        self._e_xx = np.full((R_Nx, R_Ny), np.nan, dtype=np.float32)
        self._e_yy = np.full((R_Nx, R_Ny), np.nan, dtype=np.float32)
        self._e_xy = np.full((R_Nx, R_Ny), np.nan, dtype=np.float32)
        self._theta = np.full((R_Nx, R_Ny), np.nan, dtype=np.float32)
        self._mask = np.zeros((R_Nx, R_Ny), dtype=np.uint8)
        try:
            self.fit_reference()
            self.compute_strain()
        except Exception:
            self._push_bytes()
        return self

    # ---------------------------------------------------------------------
    # Stats and serialization helpers
    # ---------------------------------------------------------------------

    def _channel_stats(self, arr: np.ndarray) -> List[float]:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return [0.0, 0.0, 0.0]
        return [float(finite.min()), float(finite.max()), float(finite.std())]

    def _update_stats(self):
        self.stats_e_xx = self._channel_stats(self._e_xx)
        self.stats_e_yy = self._channel_stats(self._e_yy)
        self.stats_e_xy = self._channel_stats(self._e_xy)
        self.stats_theta = self._channel_stats(self._theta)

    def _push_bytes(self):
        self.e_xx_bytes = np.ascontiguousarray(self._e_xx, dtype=np.float32).tobytes()
        self.e_yy_bytes = np.ascontiguousarray(self._e_yy, dtype=np.float32).tobytes()
        self.e_xy_bytes = np.ascontiguousarray(self._e_xy, dtype=np.float32).tobytes()
        self.theta_bytes = np.ascontiguousarray(self._theta, dtype=np.float32).tobytes()
        self.mask_bytes = np.ascontiguousarray(self._mask, dtype=np.uint8).tobytes()
        self._update_stats()

    # Public accessors
    @property
    def e_xx(self) -> np.ndarray:
        return self._e_xx.copy()

    @property
    def e_yy(self) -> np.ndarray:
        return self._e_yy.copy()

    @property
    def e_xy(self) -> np.ndarray:
        return self._e_xy.copy()

    @property
    def theta(self) -> np.ndarray:
        return self._theta.copy()

    @property
    def mask(self) -> np.ndarray:
        return self._mask.copy()

    # ---------------------------------------------------------------------
    # Export
    # ---------------------------------------------------------------------

    def save_image(
        self,
        path: str | pathlib.Path,
        *,
        view: str = "all",
        format: str | None = None,
        dpi: int = 150,
    ) -> pathlib.Path:
        """Save the strain map as PNG or PDF via matplotlib.

        Parameters
        ----------
        path : str or pathlib.Path
            Output path.
        view : str, default "all"
            "all" → 2×2 panel; "e_xx", "e_yy", "e_xy", or "theta" → single panel.
        format : str, optional
            "png" or "pdf"; inferred from extension if omitted.
        dpi : int, default 150
        """
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        path = pathlib.Path(path)
        fmt = (format or path.suffix.lstrip(".").lower() or "png").lower()
        if fmt not in ("png", "pdf"):
            raise ValueError(f"Unsupported format: {fmt!r}. Use 'png' or 'pdf'.")

        scale = 100.0 if self.unit == "%" else 1.0
        channels = {
            "e_xx": (self._e_xx * scale, self.cmap_strain, r"$\epsilon_{xx}$"),
            "e_yy": (self._e_yy * scale, self.cmap_strain, r"$\epsilon_{yy}$"),
            "e_xy": (self._e_xy * scale, self.cmap_strain, r"$\epsilon_{xy}$"),
            "theta": (self._theta, self.cmap_theta, r"$\theta$ (rad)"),
        }

        if view == "all":
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))
            order = [("e_xx", axes[0, 0]), ("e_yy", axes[0, 1]),
                     ("e_xy", axes[1, 0]), ("theta", axes[1, 1])]
            for key, ax in order:
                arr, cmap, label = channels[key]
                self._plot_channel(ax, arr, cmap, label)
            fig.suptitle(self.title)
        elif view in channels:
            fig, ax = plt.subplots(figsize=(5, 5))
            arr, cmap, label = channels[view]
            self._plot_channel(ax, arr, cmap, label)
            fig.suptitle(self.title)
        else:
            raise ValueError(
                f"Unknown view {view!r}. Use 'all', 'e_xx', 'e_yy', 'e_xy', or 'theta'."
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=dpi, format=fmt, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_channel(self, ax, arr: np.ndarray, cmap: str, label: str):
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            lo = float(np.percentile(finite, self.vmin_pct))
            hi = float(np.percentile(finite, self.vmax_pct))
            amax = max(abs(lo), abs(hi), 1e-12)
            vmin, vmax = -amax, amax
        else:
            vmin, vmax = -1.0, 1.0
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])

    # ---------------------------------------------------------------------
    # State protocol
    # ---------------------------------------------------------------------

    def state_dict(self):
        return {
            "title": self.title,
            "cmap_strain": self.cmap_strain,
            "cmap_theta": self.cmap_theta,
            "vmin_pct": self.vmin_pct,
            "vmax_pct": self.vmax_pct,
            "ref_roi": dict(self.ref_roi),
            "max_peak_spacing_px": self.max_peak_spacing_px,
            "unit": self.unit,
            "g1": list(self.g1),
            "g2": list(self.g2),
            "show_stats": self.show_stats,
            "show_controls": self.show_controls,
            "canvas_size": self.canvas_size,
            "disabled_tools": list(self.disabled_tools),
            "hidden_tools": list(self.hidden_tools),
        }

    def save(self, path: str):
        save_state_file(path, "StrainMap2D", self.state_dict())

    def load_state_dict(self, state):
        for key, val in state.items():
            if hasattr(self, key):
                try:
                    setattr(self, key, val)
                except traitlets.TraitError:
                    # Silently skip values that fail trait validation
                    pass

    def summary(self):
        name = self.title if self.title else "StrainMap2D"
        lines = [name, "═" * 32]
        lines.append(f"Scan:     {self.R_Nx}×{self.R_Ny}  (N_peaks={self.N_peaks})")
        roi = self.ref_roi
        lines.append(
            f"Ref ROI:  top={roi['top']} left={roi['left']} "
            f"bottom={roi['bottom']} right={roi['right']}"
        )
        lines.append(
            f"g1:       ({self.g1[0]:.3f}, {self.g1[1]:.3f}) px"
        )
        lines.append(
            f"g2:       ({self.g2[0]:.3f}, {self.g2[1]:.3f}) px"
        )
        n_ok = int(self._mask.sum())
        n_total = int(self._mask.size)
        lines.append(f"Fit:      {n_ok}/{n_total} pixels solved")
        if n_ok > 0:
            lines.append(
                f"e_xx:     min={self.stats_e_xx[0]:.4g}  max={self.stats_e_xx[1]:.4g}  std={self.stats_e_xx[2]:.4g}"
            )
            lines.append(
                f"e_yy:     min={self.stats_e_yy[0]:.4g}  max={self.stats_e_yy[1]:.4g}  std={self.stats_e_yy[2]:.4g}"
            )
            lines.append(
                f"e_xy:     min={self.stats_e_xy[0]:.4g}  max={self.stats_e_xy[1]:.4g}  std={self.stats_e_xy[2]:.4g}"
            )
            lines.append(
                f"theta:    min={self.stats_theta[0]:.4g}  max={self.stats_theta[1]:.4g}  std={self.stats_theta[2]:.4g}"
            )
        lines.append(
            f"Display:  cmap_strain={self.cmap_strain}  cmap_theta={self.cmap_theta}  unit={self.unit}"
        )
        print("\n".join(lines))

    def __repr__(self) -> str:
        name = self.title if self.title else "StrainMap2D"
        n_ok = int(self._mask.sum())
        return (
            f"{name}({self.R_Nx}×{self.R_Ny}, "
            f"N_peaks={self.N_peaks}, fit={n_ok}/{self._mask.size}, "
            f"unit={self.unit})"
        )


bind_tool_runtime_api(StrainMap2D, "StrainMap2D")
