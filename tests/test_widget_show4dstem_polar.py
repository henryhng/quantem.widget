"""Tests for the polar / azimuthal sub-view of Show4DSTEM."""

import json
import math
import pathlib

import numpy as np
import pytest

from quantem.widget import Show4DSTEM


def _make_widget(det_rows=32, det_cols=32, scan=4, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.random((scan, scan, det_rows, det_cols), dtype=np.float32) * 100.0
    return Show4DSTEM(data, k_pixel_size=0.5, verbose=False), data


def test_polar_trait_defaults():
    """All polar traits start at the documented defaults."""
    w, _ = _make_widget()
    assert w.show_polar is False
    assert w.polar_q_min_mrad == 0.0
    assert w.polar_q_max_mrad == 0.0
    assert w.polar_n_q == 128
    assert w.polar_n_theta == 180
    assert w.polar_ellipse_a == 1.0
    assert w.polar_ellipse_b == 1.0
    assert w.polar_ellipse_theta_rad == 0.0
    # polar_bytes / polar_radial_bytes are empty until enabled
    assert w.polar_bytes == b""
    assert w.polar_radial_bytes == b""
    assert w.polar_q_mrad_min == 0.0
    assert w.polar_q_mrad_max == 0.0


def test_polar_toggle_populates_bytes():
    """Turning show_polar on triggers computation; the bytes become non-empty."""
    w, _ = _make_widget()
    assert len(w.polar_bytes) == 0
    w.show_polar = True
    assert len(w.polar_bytes) > 0
    assert len(w.polar_radial_bytes) > 0
    # And turning it off again clears the outputs.
    w.show_polar = False
    assert w.polar_bytes == b""
    assert w.polar_radial_bytes == b""


def test_polar_bytes_shape():
    """polar_bytes shape is polar_n_q × polar_n_theta × 4 (float32)."""
    w, _ = _make_widget()
    w.show_polar = True
    expected = w.polar_n_q * w.polar_n_theta * 4
    assert len(w.polar_bytes) == expected

    # Changing the bin counts re-shapes the output.
    w.polar_n_q = 64
    w.polar_n_theta = 90
    assert len(w.polar_bytes) == 64 * 90 * 4


def test_polar_radial_bytes_shape_and_unbiased_against_corner_clipping():
    """polar_radial_bytes is `polar_n_q × 4` bytes and is unbiased even when
    the polar arc clips the detector at large q. Implemented as
    `polar.sum(theta) / valid_count(theta)` so out-of-detector wedges do not
    drag the average down (a plain `polar.mean(axis=1)` would).
    """
    # Detector 32×32, center (16,16). Inscribed-circle radius = 16 px; corner
    # reach = hypot(16,16) ≈ 22.6 px. Choose q_max so some thetas at the outer
    # q are off-detector (cardinal directions past 16 px) but others are still
    # inside (diagonal corners within 22.6 px). With k_pixel_size=0.5 mrad/px,
    # q_max ≈ 9 mrad → 18 px from center → cardinals (theta=0,π/2,...) off,
    # diagonals on. This is the regime where the bias matters.
    rng = np.random.default_rng(0)
    det = 32
    data = (rng.random((1, 1, det, det), dtype=np.float32) * 100.0 + 1.0)
    w = Show4DSTEM(data, k_pixel_size=0.5, verbose=False)
    w.polar_q_min_mrad = 0.0
    w.polar_q_max_mrad = 9.0  # 18 px from center -> mixed valid/invalid
    w.polar_n_q = 48
    w.polar_n_theta = 90
    w.show_polar = True

    assert len(w.polar_radial_bytes) == w.polar_n_q * 4
    polar = np.frombuffer(w.polar_bytes, dtype=np.float32).reshape(
        w.polar_n_q, w.polar_n_theta
    )
    radial = np.frombuffer(w.polar_radial_bytes, dtype=np.float32)
    naive_mean = polar.mean(axis=1)

    # Inner q (always-inside): unbiased equals naive mean.
    np.testing.assert_allclose(radial[:4], naive_mean[:4], rtol=1e-5, atol=1e-5)

    # Find the largest q where partial clipping happens (some zeros, not all).
    valid_per_q = (polar != 0).sum(axis=1)
    partial = np.where((valid_per_q > 0) & (valid_per_q < w.polar_n_theta))[0]
    assert partial.size > 0, "test scenario should produce partial clipping"
    q_idx = int(partial[-1])
    # At a partially-clipped q, the unbiased mean is strictly larger than the
    # naive (which divides by full n_theta instead of valid count).
    assert radial[q_idx] > naive_mean[q_idx] + 1.0, (
        f"radial profile is biased at partial-clip q={q_idx}: "
        f"unbiased={radial[q_idx]:.3f}, naive={naive_mean[q_idx]:.3f}"
    )


def test_polar_ellipse_correction_undistorts_ring_along_major_direction():
    """Direction check: build a DP with a ring distorted into an ellipse (major
    axis 1.2× along x). The widget's gather-direction correction with
    (a=1.2, b=1.0, theta_e=0) must straighten the ring to constant q across
    theta. The opposite scaling (1/a, 1/b) must make it worse. This pins the
    sign of the ellipse transform so a future refactor cannot silently flip it.
    """
    det = 128
    center = det / 2.0
    yy, xx = np.mgrid[:det, :det].astype(np.float32) - center
    # Ring at radius R=30, stretched by factor 1.2 along x:
    ring_R = 30.0
    ellipse_dist = np.sqrt((xx / (ring_R * 1.2)) ** 2 + (yy / ring_R) ** 2)
    dp = np.exp(-((ellipse_dist - 1.0) ** 2) / 0.005).astype(np.float32)
    data4d = dp[None, None, ...]

    w = Show4DSTEM(data4d, verbose=False, precompute_virtual_images=False)
    w.center_row = center
    w.center_col = center
    w.k_pixel_size = 1.0  # mrad/px
    w.polar_q_min_mrad = 0.0
    w.polar_q_max_mrad = 50.0
    w.polar_n_q = 100
    w.polar_n_theta = 90
    w.show_polar = True

    q_axis = np.linspace(w.polar_q_min_mrad, w.polar_q_max_mrad, w.polar_n_q)

    def ring_q_std(widget):
        polar = (
            np.frombuffer(widget.polar_bytes, dtype=np.float32)
            .copy()
            .reshape(widget.polar_n_q, widget.polar_n_theta)
        )
        return float(q_axis[polar.argmax(axis=0)].std())

    # Uncorrected: ring is at varying q across theta (the elliptical distortion).
    w.polar_ellipse_a = 1.0
    w.polar_ellipse_b = 1.0
    std_uncorrected = ring_q_std(w)

    # CORRECT direction: multiply by (1.2, 1.0) -> straightens the ring.
    w.polar_ellipse_a = 1.2
    w.polar_ellipse_b = 1.0
    std_correct = ring_q_std(w)

    # INVERTED direction: would make it worse, not better.
    w.polar_ellipse_a = 1.0 / 1.2
    w.polar_ellipse_b = 1.0
    std_inverted = ring_q_std(w)

    # The correct direction must reduce the variance an order of magnitude.
    assert std_correct < 0.3 * std_uncorrected, (
        f"ellipse correction did not straighten ring: "
        f"uncorrected std={std_uncorrected:.3f}, corrected std={std_correct:.3f}"
    )
    # And the inverted direction must clearly make it worse.
    assert std_inverted > std_uncorrected, (
        f"inverted ellipse direction should be worse than no correction: "
        f"uncorrected std={std_uncorrected:.3f}, inverted std={std_inverted:.3f}"
    )


def test_polar_deterministic_same_dp_same_bytes():
    """Computing twice for the same DP gives bit-identical output."""
    w, _ = _make_widget(seed=7)
    w.show_polar = True
    first = bytes(w.polar_bytes)
    # Trigger recomputation by flipping a polar trait back to itself
    w.polar_n_q = w.polar_n_q  # no-op assignment, but force compute via observer
    w._compute_polar()
    second = bytes(w.polar_bytes)
    assert first == second


def test_polar_state_dict_includes_polar_traits():
    """state_dict carries every polar trait, and load_state_dict restores them."""
    w, _ = _make_widget()
    w.show_polar = True
    w.polar_q_min_mrad = 1.5
    w.polar_q_max_mrad = 10.0
    w.polar_n_q = 64
    w.polar_n_theta = 90
    w.polar_ellipse_a = 0.95
    w.polar_ellipse_b = 1.05
    w.polar_ellipse_theta_rad = 0.3

    sd = w.state_dict()
    for key in (
        "show_polar",
        "polar_q_min_mrad",
        "polar_q_max_mrad",
        "polar_n_q",
        "polar_n_theta",
        "polar_ellipse_a",
        "polar_ellipse_b",
        "polar_ellipse_theta_rad",
    ):
        assert key in sd, f"missing {key} in state_dict"

    # Round-trip via constructor
    data = np.random.default_rng(0).random((4, 4, 32, 32), dtype=np.float32) * 100.0
    w2 = Show4DSTEM(data, k_pixel_size=0.5, state=sd, verbose=False)
    assert w2.show_polar is True
    assert w2.polar_q_min_mrad == pytest.approx(1.5)
    assert w2.polar_q_max_mrad == pytest.approx(10.0)
    assert w2.polar_n_q == 64
    assert w2.polar_n_theta == 90
    assert w2.polar_ellipse_a == pytest.approx(0.95)
    assert w2.polar_ellipse_b == pytest.approx(1.05)
    assert w2.polar_ellipse_theta_rad == pytest.approx(0.3)
    # After restore, polar bytes should be populated
    assert len(w2.polar_bytes) == 64 * 90 * 4


def test_polar_save_load_file_roundtrip(tmp_path):
    """save() / load via state= keeps polar settings intact."""
    w, _ = _make_widget()
    w.show_polar = True
    w.polar_n_q = 32
    w.polar_n_theta = 60
    path = tmp_path / "polar_state.json"
    w.save(str(path))
    saved = json.loads(pathlib.Path(path).read_text())
    assert saved["state"]["show_polar"] is True
    assert saved["state"]["polar_n_q"] == 32
    assert saved["state"]["polar_n_theta"] == 60

    data = np.random.default_rng(0).random((4, 4, 32, 32), dtype=np.float32) * 100.0
    w2 = Show4DSTEM(data, k_pixel_size=0.5, state=str(path), verbose=False)
    assert w2.show_polar is True
    assert w2.polar_n_q == 32
    assert w2.polar_n_theta == 60


def test_polar_summary_mentions_polar(capsys):
    """summary() mentions polar when enabled."""
    w, _ = _make_widget()
    w.show_polar = True
    w.summary()
    out = capsys.readouterr().out
    assert "Polar:" in out
    # And does NOT mention it when disabled
    w.show_polar = False
    w.summary()
    out = capsys.readouterr().out
    assert "Polar:" not in out


def test_polar_position_change_refreshes_bytes():
    """Moving the scan position recomputes polar (different DP → different bytes)."""
    rng = np.random.default_rng(1)
    # Inject very distinct DPs at two scan positions.
    data = rng.random((4, 4, 32, 32), dtype=np.float32) * 0.01
    data[0, 0] = rng.random((32, 32), dtype=np.float32) * 100 + 50
    data[3, 3] = rng.random((32, 32), dtype=np.float32) * 200 + 100
    w = Show4DSTEM(data, k_pixel_size=0.5, verbose=False)
    w.show_polar = True
    w.pos_row, w.pos_col = 0, 0
    bytes_00 = bytes(w.polar_bytes)
    w.pos_row, w.pos_col = 3, 3
    bytes_33 = bytes(w.polar_bytes)
    assert bytes_00 != bytes_33
