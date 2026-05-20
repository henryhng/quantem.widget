"""
orientation_map: Minimal template-matching orientation viewer for 4D-STEM.

Given a precomputed library of polar-transformed diffraction templates and
their rotation labels, computes the per-scan-pixel best-match correlation
score and best-match rotation by polar-resampling each DP and taking the
cyclic cross-correlation of the polar DP against every template along the
angular (theta) axis. The result is rendered as an HSV map (hue =
rotation, value = score).

Scope (intentionally narrow):
- Inputs are precomputed: this widget does NOT build templates from a
  Crystal object; the user supplies a polar template library.
- No crystal symmetry handling, no Euler-angle/zone-axis decomposition,
  no IPF coloring with TSL/orix-style symmetry — just plain HSV by
  rotation angle.
- No n-best matches, no sub-pixel refinement, no Friedel/inversion
  symmetry. (Those belong to downstream PRs.)

The scoring follows the py4DSTEM ACOM template-matching kernel
(Ophus, MAM 2022; arXiv 2111.00171):

    score[t, gamma] = sum_q( real(ifft_theta( T_t_fft * conj(P_fft) )) )[gamma]

with templates L2-normalized once and stored as ``conj(fft(T, axis=theta))``.

References
----------
- py4DSTEM ``crystal_ACOM.match_single_pattern`` / ``match_orientations``
  for the correlation kernel and FFT-domain matching.
- For true crystallographic IPF coloring, see ``orix.plot.IPFColorKeyTSL``.
  We use plain HSV here as an interim visualization.
"""

import json
import math
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


class OrientationMap(anywidget.AnyWidget):
    """
    Minimal template-matching orientation map for 4D-STEM.

    The widget polar-transforms every diffraction pattern, then for each scan
    pixel finds the (template, in-plane rotation) pair that maximizes the
    cyclic cross-correlation of the polar DP against the user-supplied polar
    template library. The best-match rotation is rendered as hue (HSV) and
    the best-match score modulates the brightness.

    Parameters
    ----------
    data : array_like
        4D-STEM dataset, shape ``(R_Nx, R_Ny, det_y, det_x)``.
    templates : array_like
        Polar-transformed templates, shape ``(n_templates, n_q, n_theta)``.
        The user is responsible for generating these from a Crystal /
        template-library — this widget does not build them.
    template_rotations : array_like
        Per-template rotation labels in radians, shape ``(n_templates,)``.
        Added to the best-match in-plane offset to produce the final
        rotation angle stored in ``orientation_rad_bytes``.
    center : (float, float), optional
        Diffraction pattern center as ``(row, col)``. Defaults to the
        detector center.
    q_min_mrad, q_max_mrad : float, optional
        Polar resampler q range. ``q_max_mrad=0`` (default) uses the
        inscribed-disk radius from the detector geometry.
    n_q, n_theta : int, optional
        Polar grid size. **Must match the templates** (the constructor
        validates and overrides from the template shape).
    k_pixel_size_mrad : float, optional
        Detector calibration in mrad/px. When 0, polar coords fall back
        to pixel units.
    title : str, optional
        Widget title.

    Notes
    -----
    What is intentionally NOT in scope here:
      - Building templates from a Crystal class.
      - Crystal-symmetry-aware IPF coloring (orix.IPFColorKeyTSL).
      - Multiple best matches per pixel / Friedel symmetry.
      - Sub-pixel angular refinement.
    """

    _esm = pathlib.Path(__file__).parent / "static" / "orientation_map.js"
    _css = pathlib.Path(__file__).parent / "static" / "orientation_map.css"

    # ── Core state ───────────────────────────────────────────────────────
    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    shape_rows = traitlets.Int(1).tag(sync=True)
    shape_cols = traitlets.Int(1).tag(sync=True)
    det_rows = traitlets.Int(1).tag(sync=True)
    det_cols = traitlets.Int(1).tag(sync=True)
    n_templates = traitlets.Int(0).tag(sync=True)

    # ── Polar resampler params ──────────────────────────────────────────
    q_min_mrad = traitlets.Float(0.0).tag(sync=True)
    q_max_mrad = traitlets.Float(0.0).tag(sync=True)
    n_q = traitlets.Int(64).tag(sync=True)
    n_theta = traitlets.Int(180).tag(sync=True)
    k_pixel_size_mrad = traitlets.Float(0.0).tag(sync=True)
    center_row = traitlets.Float(0.0).tag(sync=True)
    center_col = traitlets.Float(0.0).tag(sync=True)

    # ── Display ──────────────────────────────────────────────────────────
    cmap = traitlets.Unicode("hsv").tag(sync=True)
    show_score = traitlets.Bool(True).tag(sync=True)
    score_threshold = traitlets.Float(0.0).tag(sync=True)

    # ── Outputs (raw bytes for JS rendering) ─────────────────────────────
    orientation_rad_bytes = traitlets.Bytes(b"").tag(sync=True)
    score_bytes = traitlets.Bytes(b"").tag(sync=True)
    rgb_bytes = traitlets.Bytes(b"").tag(sync=True)
    score_min = traitlets.Float(0.0).tag(sync=True)
    score_max = traitlets.Float(1.0).tag(sync=True)

    # ── Scale bar ───────────────────────────────────────────────────────
    pixel_size = traitlets.Float(0.0).tag(sync=True)
    units = traitlets.Unicode("Å").tag(sync=True)
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)

    # ── UI ───────────────────────────────────────────────────────────────
    show_stats = traitlets.Bool(True).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)

    # ── Tool visibility ──────────────────────────────────────────────────
    disabled_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    hidden_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)

    @classmethod
    def _normalize_tool_groups(cls, tool_groups) -> List[str]:
        return normalize_tool_groups("OrientationMap", tool_groups)

    @classmethod
    def _build_disabled_tools(
        cls,
        disabled_tools=None,
        disable_display: bool = False,
        disable_threshold: bool = False,
        disable_view: bool = False,
        disable_export: bool = False,
        disable_stats: bool = False,
        disable_all: bool = False,
    ) -> List[str]:
        return build_tool_groups(
            "OrientationMap",
            tool_groups=disabled_tools,
            all_flag=disable_all,
            flag_map={
                "display": disable_display,
                "threshold": disable_threshold,
                "view": disable_view,
                "export": disable_export,
                "stats": disable_stats,
            },
        )

    @classmethod
    def _build_hidden_tools(
        cls,
        hidden_tools=None,
        hide_display: bool = False,
        hide_threshold: bool = False,
        hide_view: bool = False,
        hide_export: bool = False,
        hide_stats: bool = False,
        hide_all: bool = False,
    ) -> List[str]:
        return build_tool_groups(
            "OrientationMap",
            tool_groups=hidden_tools,
            all_flag=hide_all,
            flag_map={
                "display": hide_display,
                "threshold": hide_threshold,
                "view": hide_view,
                "export": hide_export,
                "stats": hide_stats,
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
        templates,
        template_rotations,
        center: tuple[float, float] | None = None,
        q_min_mrad: float = 0.0,
        q_max_mrad: float = 0.0,
        n_q: int | None = None,
        n_theta: int | None = None,
        k_pixel_size_mrad: float = 0.0,
        title: str = "",
        cmap: str = "hsv",
        show_score: bool = True,
        score_threshold: float = 0.0,
        pixel_size: float = 0.0,
        units: str = "Å",
        show_stats: bool = True,
        show_controls: bool = True,
        disabled_tools: Optional[List[str]] = None,
        disable_display: bool = False,
        disable_threshold: bool = False,
        disable_view: bool = False,
        disable_export: bool = False,
        disable_stats: bool = False,
        disable_all: bool = False,
        hidden_tools: Optional[List[str]] = None,
        hide_display: bool = False,
        hide_threshold: bool = False,
        hide_view: bool = False,
        hide_export: bool = False,
        hide_stats: bool = False,
        hide_all: bool = False,
        state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()

        # ── IOResult duck typing for data ────────────────────────────────
        if isinstance(data, IOResult):
            if not title and data.title:
                title = data.title
            if pixel_size <= 0 and data.pixel_size is not None:
                pixel_size = data.pixel_size
            data = data.data

        self._ingest_data(data)
        self._ingest_templates(templates, template_rotations)

        # Override n_q / n_theta from templates (templates are ground truth).
        if n_q is not None and n_q != self._templates.shape[1]:
            raise ValueError(
                f"n_q={n_q} does not match templates.shape[1]={self._templates.shape[1]}"
            )
        if n_theta is not None and n_theta != self._templates.shape[2]:
            raise ValueError(
                f"n_theta={n_theta} does not match templates.shape[2]={self._templates.shape[2]}"
            )
        self.n_q = int(self._templates.shape[1])
        self.n_theta = int(self._templates.shape[2])

        self.q_min_mrad = float(q_min_mrad)
        self.q_max_mrad = float(q_max_mrad)
        self.k_pixel_size_mrad = float(k_pixel_size_mrad)
        self.title = title
        self.cmap = cmap
        self.show_score = bool(show_score)
        self.score_threshold = float(score_threshold)
        self.pixel_size = float(pixel_size)
        self.units = units
        self.show_stats = show_stats
        self.show_controls = show_controls

        # Center defaults to detector center.
        if center is not None:
            self.center_row = float(center[0])
            self.center_col = float(center[1])
        else:
            self.center_row = (self.det_rows - 1) / 2.0
            self.center_col = (self.det_cols - 1) / 2.0

        self.disabled_tools = self._build_disabled_tools(
            disabled_tools=disabled_tools,
            disable_display=disable_display,
            disable_threshold=disable_threshold,
            disable_view=disable_view,
            disable_export=disable_export,
            disable_stats=disable_stats,
            disable_all=disable_all,
        )
        self.hidden_tools = self._build_hidden_tools(
            hidden_tools=hidden_tools,
            hide_display=hide_display,
            hide_threshold=hide_threshold,
            hide_view=hide_view,
            hide_export=hide_export,
            hide_stats=hide_stats,
            hide_all=hide_all,
        )

        # Compute initial match + render
        self._compute_match()
        self._render_rgb()

        # Observers that retrigger rendering only (no re-match).
        self.observe(
            self._on_render_param_change,
            names=["cmap", "show_score", "score_threshold"],
        )
        # Observers that retrigger match + render.
        self.observe(
            self._on_polar_param_change,
            names=[
                "q_min_mrad",
                "q_max_mrad",
                "k_pixel_size_mrad",
                "center_row",
                "center_col",
            ],
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

    # =========================================================================
    # Data / template ingestion
    # =========================================================================

    def _ingest_data(self, data):
        arr = np.asarray(to_numpy(data), dtype=np.float32)
        if arr.ndim != 4:
            raise ValueError(f"Expected 4D data (R_Nx, R_Ny, det_y, det_x), got {arr.ndim}D")
        self._data = np.ascontiguousarray(arr)
        self.shape_rows = int(arr.shape[0])
        self.shape_cols = int(arr.shape[1])
        self.det_rows = int(arr.shape[2])
        self.det_cols = int(arr.shape[3])

    def _ingest_templates(self, templates, template_rotations):
        t = np.asarray(to_numpy(templates), dtype=np.float32)
        if t.ndim != 3:
            raise ValueError(
                f"Expected 3D templates (n_templates, n_q, n_theta), got {t.ndim}D"
            )
        rot = np.asarray(to_numpy(template_rotations), dtype=np.float32).reshape(-1)
        if rot.shape[0] != t.shape[0]:
            raise ValueError(
                f"template_rotations length {rot.shape[0]} does not match "
                f"templates.shape[0]={t.shape[0]}"
            )
        # L2-normalize each template independently (matches py4DSTEM).
        # Store the FFT-domain conjugate along theta for fast cyclic correlation.
        norms = np.sqrt(np.sum(t * t, axis=(1, 2), keepdims=True))
        norms = np.where(norms > 0, norms, 1.0)
        t_norm = t / norms
        self._templates = t_norm.astype(np.float32)
        self._template_rotations = rot.astype(np.float32)
        # FFT along theta axis; store conj for kernel.
        self._templates_fft_conj = np.conj(np.fft.fft(t_norm, axis=2)).astype(np.complex64)
        self.n_templates = int(t.shape[0])

    # =========================================================================
    # Polar transform — inline bilinear gather (~50 lines)
    # Re-implements the show4dstem polar resampler so this widget is
    # self-contained and not coupled to that PR's branch.
    # =========================================================================

    def _polar_grid_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (row_pix, col_pix) arrays of shape (n_q, n_theta) for sampling."""
        n_q = int(self.n_q)
        n_theta = int(self.n_theta)

        det_diag = float(min(self.det_rows, self.det_cols)) / 2.0
        k_pixel = float(self.k_pixel_size_mrad)
        if k_pixel <= 0:
            # Pixel-unit fallback: q axis is in detector pixels directly.
            q_min = max(0.0, float(self.q_min_mrad))
            q_max = float(self.q_max_mrad)
            if q_max <= q_min:
                q_max = det_diag
            q_axis_px = np.linspace(q_min, q_max, n_q, dtype=np.float32)
        else:
            q_min = max(0.0, float(self.q_min_mrad))
            q_max = float(self.q_max_mrad)
            if q_max <= q_min:
                q_max = det_diag * k_pixel
            q_axis = np.linspace(q_min, q_max, n_q, dtype=np.float32)
            q_axis_px = q_axis / k_pixel

        # Exclude duplicate endpoint at 2*pi.
        theta_axis = np.linspace(0.0, 2.0 * math.pi, n_theta + 1, dtype=np.float32)[:-1]
        cos_t = np.cos(theta_axis)[None, :]
        sin_t = np.sin(theta_axis)[None, :]
        q_grid = q_axis_px[:, None]
        # No ellipse correction in this minimal widget.
        col_pix = self.center_col + q_grid * cos_t
        row_pix = self.center_row + q_grid * sin_t
        return row_pix, col_pix

    def _polar_transform(self, dp: np.ndarray) -> np.ndarray:
        """Bilinear-gather polar resample of a single DP to (n_q, n_theta)."""
        row_pix, col_pix = self._polar_grid_coords()
        n_rows = int(self.det_rows)
        n_cols = int(self.det_cols)

        r0 = np.floor(row_pix).astype(np.int64)
        c0 = np.floor(col_pix).astype(np.int64)
        dr = (row_pix - r0).astype(np.float32)
        dc = (col_pix - c0).astype(np.float32)

        valid = (r0 >= 0) & (r0 < n_rows - 1) & (c0 >= 0) & (c0 < n_cols - 1)
        r0c = np.clip(r0, 0, n_rows - 2)
        c0c = np.clip(c0, 0, n_cols - 2)
        r1c = r0c + 1
        c1c = c0c + 1

        v00 = dp[r0c, c0c]
        v01 = dp[r0c, c1c]
        v10 = dp[r1c, c0c]
        v11 = dp[r1c, c1c]
        w00 = (1.0 - dr) * (1.0 - dc)
        w01 = (1.0 - dr) * dc
        w10 = dr * (1.0 - dc)
        w11 = dr * dc
        out = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11
        out = np.where(valid, out, 0.0).astype(np.float32)
        return out

    # =========================================================================
    # Matching — vectorized over scan grid, FFT-based over rotation offsets
    # =========================================================================

    def _compute_match(self):
        """Compute per-scan-pixel best-match score + rotation.

        Algorithm (py4DSTEM ACOM kernel):
            P = polar_transform(DP)                       # (n_q, n_theta)
            P /= ||P||_2                                  # L2 normalize
            P_fft = fft(P, axis=theta)
            For each template t (stored as T_fft_conj):
                corr[t, gamma] = sum_q( real( ifft( T_fft_conj[t] * P_fft ) ) )
                ind_phi = argmax_gamma corr[t, gamma]
            Best (t*, gamma*) = argmax over (t, ind_phi[t]) of corr.
        """
        n_rows = int(self.shape_rows)
        n_cols = int(self.shape_cols)
        n_theta = int(self.n_theta)
        n_templates = int(self.n_templates)

        # Precompute the polar grid coords once (shared across all DPs).
        # This keeps the inner loop cheap: only the bilinear gather changes.
        row_pix, col_pix = self._polar_grid_coords()
        r0 = np.floor(row_pix).astype(np.int64)
        c0 = np.floor(col_pix).astype(np.int64)
        dr = (row_pix - r0).astype(np.float32)
        dc = (col_pix - c0).astype(np.float32)
        valid = (r0 >= 0) & (r0 < self.det_rows - 1) & (c0 >= 0) & (c0 < self.det_cols - 1)
        r0c = np.clip(r0, 0, self.det_rows - 2)
        c0c = np.clip(c0, 0, self.det_cols - 2)
        r1c = r0c + 1
        c1c = c0c + 1
        w00 = ((1.0 - dr) * (1.0 - dc)).astype(np.float32)
        w01 = ((1.0 - dr) * dc).astype(np.float32)
        w10 = (dr * (1.0 - dc)).astype(np.float32)
        w11 = (dr * dc).astype(np.float32)
        valid_mask = valid.astype(np.float32)

        orientation_rad = np.zeros((n_rows, n_cols), dtype=np.float32)
        score_map = np.zeros((n_rows, n_cols), dtype=np.float32)

        # Templates FFT conj: (n_templates, n_q, n_theta) complex64
        tfft = self._templates_fft_conj
        theta_step = (2.0 * math.pi) / float(n_theta)
        template_rot = self._template_rotations

        # Per-scan-pixel loop (vectorized over templates via einsum).
        # n_rows * n_cols polar transforms + FFTs is unavoidable; the
        # template-axis is the parallel one.
        for r in range(n_rows):
            for c in range(n_cols):
                dp = self._data[r, c]
                # Bilinear gather
                polar = (
                    w00 * dp[r0c, c0c]
                    + w01 * dp[r0c, c1c]
                    + w10 * dp[r1c, c0c]
                    + w11 * dp[r1c, c1c]
                ) * valid_mask
                # L2 normalize (match py4DSTEM scaling regime)
                norm = float(np.sqrt(np.sum(polar * polar)))
                if norm > 0:
                    polar = polar / norm
                # FFT along theta axis
                p_fft = np.fft.fft(polar, axis=1)  # (n_q, n_theta), complex
                # Multiply templates by P_fft along (q, theta), then sum over q
                # after ifft -> sum-then-ifft is equivalent because ifft is linear:
                #   ifft(sum_q(T*P)) == sum_q(ifft(T*P))
                # which lets us collapse q before the (cheaper) ifft.
                merged = np.einsum("tqs,qs->ts", tfft, p_fft)  # (n_templates, n_theta)
                corr_full = np.real(np.fft.ifft(merged, axis=1))
                np.maximum(corr_full, 0, out=corr_full)
                # Best (template, gamma)
                flat = corr_full.argmax()
                best_t, best_g = np.unravel_index(flat, corr_full.shape)
                best_score = float(corr_full[best_t, best_g])
                # gamma index -> in-plane offset angle (radians).
                # ifft shift index k corresponds to rotation by +k * 2pi/n_theta.
                offset = float(best_g) * theta_step
                rot = float(template_rot[best_t]) + offset
                # Wrap to [0, 2π)
                rot = float(np.mod(rot, 2.0 * math.pi))
                orientation_rad[r, c] = rot
                score_map[r, c] = best_score

        self._orientation_rad = orientation_rad
        self._score = score_map
        self.score_min = float(score_map.min())
        self.score_max = float(score_map.max())
        with self.hold_sync():
            self.orientation_rad_bytes = orientation_rad.astype(np.float32).tobytes()
            self.score_bytes = score_map.astype(np.float32).tobytes()

    # =========================================================================
    # HSV render
    # =========================================================================

    @staticmethod
    def _hsv_to_rgb_vec(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Vectorized HSV -> RGB (float [0,1] -> uint8 [0,255]).
        h, s, v all in [0, 1]. Returns (..., 3) uint8.
        """
        h6 = (h * 6.0) % 6.0
        i = np.floor(h6).astype(np.int32)
        f = (h6 - i).astype(np.float32)
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        r = np.where(i == 0, v, np.where(i == 1, q, np.where(i == 2, p,
            np.where(i == 3, p, np.where(i == 4, t, v)))))
        g = np.where(i == 0, t, np.where(i == 1, v, np.where(i == 2, v,
            np.where(i == 3, q, np.where(i == 4, p, p)))))
        b = np.where(i == 0, p, np.where(i == 1, p, np.where(i == 2, t,
            np.where(i == 3, v, np.where(i == 4, v, q)))))
        rgb = np.stack([r, g, b], axis=-1)
        return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    def _render_rgb(self):
        """Build the per-scan-pixel HSV-by-rotation RGB image."""
        rot = self._orientation_rad
        score = self._score

        # Hue = rotation / (2π)
        hue = (rot / (2.0 * math.pi)).astype(np.float32)
        hue = np.mod(hue, 1.0)

        # Value channel: optionally modulated by score (normalized to [0,1]).
        if self.show_score:
            smin = float(score.min())
            smax = float(score.max())
            if smax > smin:
                val = ((score - smin) / (smax - smin)).astype(np.float32)
            else:
                val = np.ones_like(score, dtype=np.float32)
            sat = np.ones_like(hue, dtype=np.float32)
        else:
            val = np.ones_like(hue, dtype=np.float32)
            sat = np.ones_like(hue, dtype=np.float32)

        # Threshold: pixels below score_threshold render black.
        # Threshold is interpreted in the SAME units as score (raw correlation).
        mask_below = score < float(self.score_threshold)

        rgb = self._hsv_to_rgb_vec(hue, sat, val)
        rgb[mask_below] = 0
        with self.hold_sync():
            self.rgb_bytes = np.ascontiguousarray(rgb).tobytes()

    # =========================================================================
    # Observers
    # =========================================================================

    def _on_render_param_change(self, change=None):
        # Only re-render — matching does not depend on display params.
        self._render_rgb()

    def _on_polar_param_change(self, change=None):
        # Polar params change → recompute match + re-render.
        self._compute_match()
        self._render_rgb()

    # =========================================================================
    # set_image
    # =========================================================================

    def set_image(self, data=None, templates=None, template_rotations=None, **kw) -> Self:
        """Replace data and/or templates and recompute.

        Parameters
        ----------
        data : array_like, optional
            New 4D-STEM dataset. If None, keep current data.
        templates : array_like, optional
            New polar template library. Requires ``template_rotations``.
        template_rotations : array_like, optional
            Per-template rotation labels.
        **kw : optional
            ``center``, ``k_pixel_size_mrad``, ``q_min_mrad``,
            ``q_max_mrad``, ``pixel_size``, ``title``, ``units``.
        """
        if (templates is None) != (template_rotations is None):
            raise ValueError(
                "Pass templates and template_rotations together, or neither."
            )
        if isinstance(data, IOResult):
            if data.pixel_size is not None:
                self.pixel_size = float(data.pixel_size)
            if data.title:
                self.title = data.title
            data = data.data
        if data is not None:
            self._ingest_data(data)
        if templates is not None:
            self._ingest_templates(templates, template_rotations)
        # Honor any kw overrides
        if "title" in kw:
            self.title = str(kw["title"])
        if "pixel_size" in kw:
            self.pixel_size = float(kw["pixel_size"])
        if "units" in kw:
            self.units = str(kw["units"])
        if "k_pixel_size_mrad" in kw:
            self.k_pixel_size_mrad = float(kw["k_pixel_size_mrad"])
        if "q_min_mrad" in kw:
            self.q_min_mrad = float(kw["q_min_mrad"])
        if "q_max_mrad" in kw:
            self.q_max_mrad = float(kw["q_max_mrad"])
        if "center" in kw and kw["center"] is not None:
            self.center_row = float(kw["center"][0])
            self.center_col = float(kw["center"][1])
        self._compute_match()
        self._render_rgb()
        return self

    # =========================================================================
    # Export
    # =========================================================================

    def save_image(
        self,
        path: str | pathlib.Path,
        *,
        view: str = "rgb",
        format: str | None = None,
        dpi: int = 150,
    ) -> pathlib.Path:
        """Save the orientation map as PNG or PDF.

        Parameters
        ----------
        path : str or pathlib.Path
            Output path.
        view : {"rgb", "score", "rotation"}, default "rgb"
            Which view to export. "score" is grayscale of the correlation
            score, "rotation" is grayscale of the rotation angle in [0, 2π).
        format : {"png", "pdf"}, optional
            Inferred from path extension if omitted.
        dpi : int, default 150
            Output DPI metadata.
        """
        from matplotlib import colormaps
        from PIL import Image

        export_path = pathlib.Path(path)
        view_key = view.lower()
        if view_key not in ("rgb", "score", "rotation"):
            raise ValueError(
                f"view must be one of 'rgb', 'score', 'rotation', got {view_key!r}"
            )
        fmt = (format or export_path.suffix.lstrip(".").lower() or "png").lower()
        if fmt not in ("png", "pdf"):
            raise ValueError(f"Unsupported format: {fmt!r}. Use 'png' or 'pdf'.")
        export_path.parent.mkdir(parents=True, exist_ok=True)

        if view_key == "rgb":
            rgb = np.frombuffer(self.rgb_bytes, dtype=np.uint8).reshape(
                self.shape_rows, self.shape_cols, 3
            )
            image = Image.fromarray(rgb, mode="RGB")
        elif view_key == "score":
            s = self._score
            smin, smax = float(s.min()), float(s.max())
            if smax > smin:
                norm = (s - smin) / (smax - smin)
            else:
                norm = np.zeros_like(s)
            cmap_fn = colormaps.get_cmap("viridis")
            rgba = (cmap_fn(norm) * 255).astype(np.uint8)
            image = Image.fromarray(rgba)
        else:  # rotation
            rot01 = (self._orientation_rad / (2.0 * math.pi)).astype(np.float32)
            cmap_fn = colormaps.get_cmap("hsv")
            rgba = (cmap_fn(np.mod(rot01, 1.0)) * 255).astype(np.uint8)
            image = Image.fromarray(rgba)

        if fmt == "pdf":
            Image.init()
            image = image.convert("RGB")
            image.save(str(export_path), format="PDF", resolution=dpi)
        else:
            image.save(str(export_path), format="PNG", dpi=(dpi, dpi))
        return export_path

    # =========================================================================
    # Public read-only views (NumPy arrays for downstream analysis)
    # =========================================================================

    @property
    def orientation_rad(self) -> np.ndarray:
        return np.frombuffer(self.orientation_rad_bytes, dtype=np.float32).reshape(
            self.shape_rows, self.shape_cols
        ).copy()

    @property
    def score(self) -> np.ndarray:
        return np.frombuffer(self.score_bytes, dtype=np.float32).reshape(
            self.shape_rows, self.shape_cols
        ).copy()

    @property
    def rgb(self) -> np.ndarray:
        return np.frombuffer(self.rgb_bytes, dtype=np.uint8).reshape(
            self.shape_rows, self.shape_cols, 3
        ).copy()

    # =========================================================================
    # State protocol
    # =========================================================================

    def state_dict(self):
        return {
            "title": self.title,
            "cmap": self.cmap,
            "show_score": self.show_score,
            "score_threshold": self.score_threshold,
            "q_min_mrad": self.q_min_mrad,
            "q_max_mrad": self.q_max_mrad,
            "n_q": self.n_q,
            "n_theta": self.n_theta,
            "k_pixel_size_mrad": self.k_pixel_size_mrad,
            "center_row": self.center_row,
            "center_col": self.center_col,
            "pixel_size": self.pixel_size,
            "units": self.units,
            "scale_bar_visible": self.scale_bar_visible,
            "show_stats": self.show_stats,
            "show_controls": self.show_controls,
            "disabled_tools": list(self.disabled_tools),
            "hidden_tools": list(self.hidden_tools),
        }

    def save(self, path: str):
        save_state_file(path, "OrientationMap", self.state_dict())

    def load_state_dict(self, state):
        allowed = set(self.state_dict().keys())
        for key, val in state.items():
            if key in allowed and hasattr(self, key):
                setattr(self, key, val)

    def summary(self):
        name = self.title if self.title else "OrientationMap"
        lines = [name, "═" * 32]
        lines.append(f"Scan:       {self.shape_rows}×{self.shape_cols}")
        lines.append(f"Detector:   {self.det_rows}×{self.det_cols}")
        lines.append(f"Templates:  {self.n_templates}")
        lines.append(f"Polar grid: {self.n_q} q × {self.n_theta} θ")
        k_unit = "mrad" if self.k_pixel_size_mrad > 0 else "px"
        if self.k_pixel_size_mrad > 0:
            lines.append(
                f"Q range:    [{self.q_min_mrad:.2f}, {self.q_max_mrad:.2f}] mrad "
                f"(k={self.k_pixel_size_mrad:.4f} mrad/px)"
            )
        else:
            lines.append(f"Q range:    [{self.q_min_mrad:.2f}, {self.q_max_mrad:.2f}] {k_unit}")
        lines.append(f"Center:     ({self.center_row:.1f}, {self.center_col:.1f})")
        lines.append(f"Score:      min={self.score_min:.4f}  max={self.score_max:.4f}")
        lines.append(f"Display:    cmap={self.cmap}  show_score={self.show_score}  threshold={self.score_threshold}")
        print("\n".join(lines))

    def __repr__(self) -> str:
        title_info = f", title='{self.title}'" if self.title else ""
        return (
            f"OrientationMap(scan=({self.shape_rows}, {self.shape_cols}), "
            f"detector=({self.det_rows}, {self.det_cols}), "
            f"n_templates={self.n_templates}, "
            f"polar=({self.n_q}, {self.n_theta}){title_info})"
        )


bind_tool_runtime_api(OrientationMap, "OrientationMap")
