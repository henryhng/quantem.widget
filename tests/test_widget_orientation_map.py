"""Tests for the OrientationMap minimal template-matching widget.

Scope: verifies that per-scan-pixel best-match rotation recovery is correct
on synthetic data where the ground truth is known, that score thresholding
zeroes RGB rows, that ``set_image`` recomputes, and the 3 required widget
protocol tests.

Out of scope (these are downstream PRs, see widget docstring):
  - Crystal class / template library generation
  - True IPF coloring with crystal symmetry
  - n-best matches / sub-pixel refinement
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from quantem.widget import OrientationMap
from quantem.widget.io import IOResult


# ─────────────────────────────────────────────────────────────────────────
# Helpers — build a synthetic 4D-STEM dataset with one Gaussian "spot"
# placed at a known angular position around the DP center. Templates are
# the same Gaussian spot but rotated through a discrete set of angles.
# ─────────────────────────────────────────────────────────────────────────


def _make_polar_spot_template(
    n_q: int,
    n_theta: int,
    q_center: int,
    theta_idx: int,
    sigma_q: float = 1.0,
    sigma_theta: float = 1.5,
) -> np.ndarray:
    """A Gaussian blob in (q, theta) space at fixed (q_center, theta_idx)."""
    q = np.arange(n_q, dtype=np.float32)[:, None]
    t = np.arange(n_theta, dtype=np.float32)[None, :]
    dq = q - q_center
    # Cyclic distance in theta
    dt = np.abs(t - theta_idx)
    dt = np.minimum(dt, n_theta - dt)
    return np.exp(-(dq**2) / (2 * sigma_q**2) - (dt**2) / (2 * sigma_theta**2)).astype(
        np.float32
    )


def _make_dp_with_spot(
    det_size: int,
    center: tuple[float, float],
    radius: float,
    angle_rad: float,
    sigma_px: float = 1.2,
) -> np.ndarray:
    """A 2D diffraction pattern with a Gaussian spot at (radius, angle)."""
    cy, cx = center
    spot_x = cx + radius * math.cos(angle_rad)
    spot_y = cy + radius * math.sin(angle_rad)
    yy, xx = np.mgrid[0:det_size, 0:det_size].astype(np.float32)
    return np.exp(-((xx - spot_x) ** 2 + (yy - spot_y) ** 2) / (2 * sigma_px**2)).astype(
        np.float32
    )


def _build_synthetic_dataset(
    scan_shape: tuple[int, int] = (3, 4),
    det_size: int = 32,
    radius_px: float = 10.0,
    n_q: int = 24,
    n_theta: int = 36,
):
    """Build (data, templates, template_rotations, true_rotations).

    Each scan pixel has a unique spot rotation. The template library is a
    single Gaussian spot at the same radius, rotated to ``n_templates``
    discrete angles. Ground-truth rotation per scan pixel is recoverable
    up to the angular resolution ``2π / n_theta``.
    """
    nr, nc = scan_shape
    center = ((det_size - 1) / 2, (det_size - 1) / 2)
    rng = np.random.default_rng(seed=0)
    # Random ground-truth rotations, one per scan pixel.
    true_rot = rng.uniform(0.0, 2 * math.pi, size=(nr, nc)).astype(np.float32)
    data = np.zeros((nr, nc, det_size, det_size), dtype=np.float32)
    for r in range(nr):
        for c in range(nc):
            data[r, c] = _make_dp_with_spot(det_size, center, radius_px, float(true_rot[r, c]))
    # Template library — same spot, rotated discretely.
    n_templates = 24
    template_rotations = np.linspace(
        0.0, 2 * math.pi, n_templates, endpoint=False, dtype=np.float32
    )
    # Place the template's reference spot at theta=0 (so the per-template
    # rotation label is the angular offset of the spot).
    # The polar grid q_max is `(det_size-1)/2` so the spot at radius_px lives
    # at fractional q-bin `radius_px / q_max * n_q`. Round to nearest integer.
    q_max_px = (det_size - 1) / 2.0
    q_center = int(round(radius_px / q_max_px * (n_q - 1)))
    # Build each template by setting its blob at theta_idx = rot * n_theta/(2π).
    templates = np.zeros((n_templates, n_q, n_theta), dtype=np.float32)
    for i, rot in enumerate(template_rotations):
        theta_idx = int(round(float(rot) * n_theta / (2 * math.pi))) % n_theta
        templates[i] = _make_polar_spot_template(n_q, n_theta, q_center, theta_idx)
    return data, templates, template_rotations, true_rot, center


def _circ_diff_rad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.mod(a - b, 2 * math.pi)
    d = np.minimum(d, 2 * math.pi - d)
    return d


# ─────────────────────────────────────────────────────────────────────────
# Construction & basic invariants
# ─────────────────────────────────────────────────────────────────────────


def test_orientation_map_construction_basic():
    data, tmpl, rot, _, center = _build_synthetic_dataset()
    w = OrientationMap(data, tmpl, rot, center=center)
    assert (w.shape_rows, w.shape_cols) == data.shape[:2]
    assert (w.det_rows, w.det_cols) == data.shape[2:]
    assert w.n_templates == tmpl.shape[0]
    assert w.n_q == tmpl.shape[1]
    assert w.n_theta == tmpl.shape[2]
    # Outputs populated
    assert w.orientation_rad_bytes != b""
    assert w.score_bytes != b""
    assert w.rgb_bytes != b""
    rgb = w.rgb
    assert rgb.shape == (w.shape_rows, w.shape_cols, 3)
    assert rgb.dtype == np.uint8


def test_orientation_map_wrong_ndim_raises():
    data = np.zeros((4, 4), dtype=np.float32)
    tmpl = np.zeros((1, 8, 16), dtype=np.float32)
    rot = np.zeros((1,), dtype=np.float32)
    with pytest.raises(ValueError, match="Expected 4D"):
        OrientationMap(data, tmpl, rot)


def test_orientation_map_template_shape_mismatch_raises():
    data = np.zeros((2, 2, 8, 8), dtype=np.float32)
    tmpl = np.zeros((2, 4, 8), dtype=np.float32)
    rot = np.zeros((3,), dtype=np.float32)  # wrong length
    with pytest.raises(ValueError, match="template_rotations length"):
        OrientationMap(data, tmpl, rot)


def test_orientation_map_n_q_mismatch_raises():
    data = np.zeros((2, 2, 8, 8), dtype=np.float32)
    tmpl = np.zeros((2, 4, 8), dtype=np.float32)
    rot = np.zeros((2,), dtype=np.float32)
    with pytest.raises(ValueError, match="n_q="):
        OrientationMap(data, tmpl, rot, n_q=8)


# ─────────────────────────────────────────────────────────────────────────
# Synthetic ground-truth recovery
# ─────────────────────────────────────────────────────────────────────────


def test_orientation_map_recovers_ground_truth_rotation():
    """Each scan pixel's recovered rotation should match the synthetic
    ground truth within one ``n_theta`` bin.

    The synthetic dataset places a single Gaussian spot at a known angle.
    The template library covers ``n_templates`` rotations of the same spot.
    The best (template, in-plane offset) combination should reproduce the
    ground-truth rotation up to ``2π/n_theta``.
    """
    data, tmpl, rot, true_rot, center = _build_synthetic_dataset()
    w = OrientationMap(data, tmpl, rot, center=center)
    recovered = w.orientation_rad
    assert recovered.shape == true_rot.shape
    diffs = _circ_diff_rad(recovered, true_rot)
    tol = 2 * math.pi / w.n_theta + 1e-3
    # Every pixel must match within one angular bin
    assert diffs.max() < tol, (
        f"max angular diff = {diffs.max():.4f} rad, "
        f"tolerance = 2π/n_theta = {tol:.4f}"
    )


def test_orientation_map_score_threshold_zeroes_rgb_rows():
    """Pixels with score below ``score_threshold`` must render to black."""
    data, tmpl, rot, _, center = _build_synthetic_dataset()
    w = OrientationMap(data, tmpl, rot, center=center)
    # Threshold above all scores → everything below threshold → all black.
    w.score_threshold = float(w.score_max + 1.0)
    rgb = w.rgb
    assert np.all(rgb == 0), "Above-max threshold should zero every pixel"
    # Threshold below all scores → nothing zeroed.
    w.score_threshold = float(w.score_min - 1.0)
    rgb2 = w.rgb
    assert rgb2.any(), "Below-min threshold should leave RGB nonzero somewhere"


def test_orientation_map_show_score_toggle():
    """Toggling ``show_score`` modulates the value channel."""
    data, tmpl, rot, _, center = _build_synthetic_dataset()
    w = OrientationMap(data, tmpl, rot, center=center)
    w.show_score = True
    rgb_modulated = w.rgb.copy()
    w.show_score = False
    rgb_flat = w.rgb.copy()
    # When show_score is False, value is uniform 1 → brightness is generally
    # brighter than the score-modulated version (which scales by score in [0,1]).
    assert rgb_flat.mean() >= rgb_modulated.mean()


# ─────────────────────────────────────────────────────────────────────────
# set_image: replace data and/or templates → recompute
# ─────────────────────────────────────────────────────────────────────────


def test_orientation_map_set_image_with_new_templates_recomputes():
    data, tmpl, rot, _, center = _build_synthetic_dataset()
    w = OrientationMap(data, tmpl, rot, center=center)
    orig_score_bytes = bytes(w.score_bytes)

    # New dataset with a different fixed rotation everywhere
    nr, nc, det = 2, 2, 32
    new_center = ((det - 1) / 2, (det - 1) / 2)
    fixed_angle = 1.2
    new_data = np.zeros((nr, nc, det, det), dtype=np.float32)
    for r in range(nr):
        for c in range(nc):
            new_data[r, c] = _make_dp_with_spot(det, new_center, 10.0, fixed_angle)
    # Templates aligned with same angle present
    n_templates = 12
    new_rot = np.linspace(0.0, 2 * math.pi, n_templates, endpoint=False, dtype=np.float32)
    q_max_px = (det - 1) / 2.0
    q_center = int(round(10.0 / q_max_px * (tmpl.shape[1] - 1)))
    new_tmpl = np.zeros((n_templates, tmpl.shape[1], tmpl.shape[2]), dtype=np.float32)
    for i, a in enumerate(new_rot):
        theta_idx = int(round(float(a) * tmpl.shape[2] / (2 * math.pi))) % tmpl.shape[2]
        new_tmpl[i] = _make_polar_spot_template(tmpl.shape[1], tmpl.shape[2], q_center, theta_idx)

    w.set_image(new_data, templates=new_tmpl, template_rotations=new_rot, center=new_center)
    assert w.shape_rows == nr
    assert w.shape_cols == nc
    assert w.n_templates == n_templates
    assert bytes(w.score_bytes) != orig_score_bytes  # recomputed
    # Recovered rotation should be near the fixed angle
    rec = w.orientation_rad
    diffs = _circ_diff_rad(rec, np.full_like(rec, fixed_angle))
    assert diffs.max() < 2 * math.pi / w.n_theta + 1e-3


def test_orientation_map_set_image_data_only():
    data, tmpl, rot, _, center = _build_synthetic_dataset()
    w = OrientationMap(data, tmpl, rot, center=center)
    new_data = np.random.RandomState(7).rand(*data.shape).astype(np.float32)
    w.set_image(new_data)
    assert bytes(w.score_bytes) != b""


def test_orientation_map_set_image_templates_required_together():
    data, tmpl, rot, _, _ = _build_synthetic_dataset()
    w = OrientationMap(data, tmpl, rot)
    with pytest.raises(ValueError, match="together"):
        w.set_image(templates=tmpl)


# ─────────────────────────────────────────────────────────────────────────
# Polar params change → recompute
# ─────────────────────────────────────────────────────────────────────────


def test_orientation_map_center_change_recomputes():
    data, tmpl, rot, _, center = _build_synthetic_dataset()
    w = OrientationMap(data, tmpl, rot, center=center)
    before = bytes(w.score_bytes)
    w.center_row = w.center_row + 2.0
    after = bytes(w.score_bytes)
    assert before != after


# ─────────────────────────────────────────────────────────────────────────
# 3 required protocol tests
# ─────────────────────────────────────────────────────────────────────────


def test_orientation_map_state_dict_roundtrip():
    data, tmpl, rot, _, center = _build_synthetic_dataset()
    w = OrientationMap(data, tmpl, rot, center=center)
    w.cmap = "viridis"
    w.show_score = False
    w.score_threshold = 0.05
    sd = w.state_dict()
    assert sd["cmap"] == "viridis"
    assert sd["show_score"] is False
    assert sd["score_threshold"] == pytest.approx(0.05)
    w2 = OrientationMap(data, tmpl, rot, center=center, state=sd)
    assert w2.cmap == "viridis"
    assert w2.show_score is False
    assert w2.score_threshold == pytest.approx(0.05)


def test_orientation_map_save_load_file(tmp_path):
    data, tmpl, rot, _, center = _build_synthetic_dataset()
    w = OrientationMap(data, tmpl, rot, center=center)
    w.cmap = "viridis"
    path = tmp_path / "orient_state.json"
    w.save(str(path))
    saved = json.loads(path.read_text())
    assert saved["metadata_version"] == "1.0"
    assert saved["widget_name"] == "OrientationMap"
    assert "widget_version" in saved
    assert saved["state"]["cmap"] == "viridis"
    w2 = OrientationMap(data, tmpl, rot, center=center, state=str(path))
    assert w2.cmap == "viridis"


def test_orientation_map_summary(capsys):
    data, tmpl, rot, _, center = _build_synthetic_dataset()
    w = OrientationMap(data, tmpl, rot, center=center, title="test")
    w.summary()
    out = capsys.readouterr().out
    assert "Scan:" in out
    assert "Templates:" in out
    assert "Polar grid:" in out


# ─────────────────────────────────────────────────────────────────────────
# __repr__
# ─────────────────────────────────────────────────────────────────────────


def test_orientation_map_repr():
    data, tmpl, rot, _, center = _build_synthetic_dataset()
    w = OrientationMap(data, tmpl, rot, center=center)
    r = repr(w)
    assert "OrientationMap" in r
    assert "n_templates" in r


# ─────────────────────────────────────────────────────────────────────────
# Tool visibility / IOResult
# ─────────────────────────────────────────────────────────────────────────


def test_orientation_map_tool_visibility():
    data, tmpl, rot, _, center = _build_synthetic_dataset()
    w = OrientationMap(data, tmpl, rot, center=center)
    w.disabled_tools = ["display", "threshold"]
    assert "display" in w.disabled_tools
    w.hidden_tools = ["stats"]
    assert "stats" in w.hidden_tools
    with pytest.raises(ValueError):
        w.disabled_tools = ["fake_tool"]


def test_orientation_map_accepts_ioresult():
    data, tmpl, rot, _, center = _build_synthetic_dataset()
    result = IOResult(
        data=data, title="ioresult_scan", pixel_size=1.5,
        units="Å", labels=[], metadata={}, frame_metadata=[],
    )
    w = OrientationMap(result, tmpl, rot, center=center)
    assert w.title == "ioresult_scan"
    assert w.pixel_size == 1.5


# ─────────────────────────────────────────────────────────────────────────
# save_image
# ─────────────────────────────────────────────────────────────────────────


def test_orientation_map_save_image(tmp_path):
    data, tmpl, rot, _, center = _build_synthetic_dataset()
    w = OrientationMap(data, tmpl, rot, center=center)
    assert w.save_image(str(tmp_path / "rgb.png")).exists()
    assert w.save_image(str(tmp_path / "score.png"), view="score").exists()
    assert w.save_image(str(tmp_path / "rot.png"), view="rotation").exists()
    with pytest.raises(ValueError):
        w.save_image(str(tmp_path / "bad.png"), view="garbage")
