"""Tests for Show4DSTEM Bragg-disk detection (current-DP only).

The detection algorithm matches py4DSTEM's `_find_Bragg_disks_single`
(CPU path): probe-kernel matched filter -> Gaussian smoothed correlogram
-> local maxima -> filter -> subpixel refinement.

Scope notes
-----------
* These tests cover the per-DP detection path only.
* Full-4D detection (bragg_peaks_all) is intentionally deferred and not
  tested here.
"""

import json
import pathlib

import numpy as np
import pytest

from quantem.widget import Show4DSTEM
from quantem.widget.show4dstem import detect_bragg_disks_single


# ---------------------------------------------------------------------------
# Synthetic DP factory
# ---------------------------------------------------------------------------

def _make_dp_with_peaks(det: int, peak_positions, sigma: float = 2.0):
    """Return a (det, det) float32 DP with Gaussian disks at peak_positions."""
    yy, xx = np.meshgrid(np.arange(det), np.arange(det), indexing="ij")
    dp = np.zeros((det, det), dtype=np.float32)
    for r, c in peak_positions:
        dp += np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2.0 * sigma ** 2))
    return dp


def _make_dataset_with_peaks(scan: int, det: int, peak_positions, sigma: float = 2.0):
    """Return a (scan, scan, det, det) dataset where every DP has the given peaks."""
    dp = _make_dp_with_peaks(det, peak_positions, sigma=sigma)
    data = np.broadcast_to(dp, (scan, scan, det, det)).astype(np.float32).copy()
    return data


def _peaks_from_bytes(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32).reshape(-1, 3) if b else np.zeros((0, 3), np.float32)


def _match_peaks(found: np.ndarray, expected, tol: float):
    """For each expected (qy, qx), find a peak in `found` within `tol` pixels.

    Returns the matched peaks (one per expected position, in order). Raises
    AssertionError if any expected peak is missing.
    """
    matched = []
    for (er, ec) in expected:
        dy = found[:, 0] - er
        dx = found[:, 1] - ec
        d = np.sqrt(dy * dy + dx * dx)
        i = int(np.argmin(d))
        assert d[i] <= tol, (
            f"expected peak ({er}, {ec}) not found within {tol}px; "
            f"closest detected: ({found[i, 0]:.2f}, {found[i, 1]:.2f}) "
            f"distance={d[i]:.2f}"
        )
        matched.append(found[i])
    return np.stack(matched)


# ---------------------------------------------------------------------------
# Module-level helper tests (detect_bragg_disks_single)
# ---------------------------------------------------------------------------

def test_detect_bragg_disks_single_finds_three_peaks():
    det = 64
    peaks_xy = [(20, 25), (40, 30), (28, 50)]
    dp = _make_dp_with_peaks(det, peaks_xy)
    # Origin-aligned probe kernel (peak at (0,0) — matches py4DSTEM convention).
    yy, xx = np.meshgrid(np.arange(det), np.arange(det), indexing="ij")
    probe = (
        np.exp(-((yy) ** 2 + (xx) ** 2) / (2.0 * 2.0 ** 2))
        + np.exp(-((yy - det) ** 2 + (xx) ** 2) / (2.0 * 2.0 ** 2))
        + np.exp(-((yy) ** 2 + (xx - det) ** 2) / (2.0 * 2.0 ** 2))
        + np.exp(-((yy - det) ** 2 + (xx - det) ** 2) / (2.0 * 2.0 ** 2))
    ).astype(np.float32)

    found = detect_bragg_disks_single(dp, probe, subpixel="multicorr", upsample_factor=4)
    assert found.shape[0] >= 3
    _match_peaks(found, peaks_xy, tol=1.5)


def test_detect_bragg_disks_single_invalid_subpixel_raises():
    dp = np.zeros((16, 16), dtype=np.float32)
    probe = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(ValueError):
        detect_bragg_disks_single(dp, probe, subpixel="bogus")


def test_detect_bragg_disks_single_shape_mismatch_raises():
    dp = np.zeros((16, 16), dtype=np.float32)
    probe = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        detect_bragg_disks_single(dp, probe)


# ---------------------------------------------------------------------------
# Widget integration tests
# ---------------------------------------------------------------------------

def test_show4dstem_bragg_detects_three_peaks_within_1p5px():
    det = 64
    peaks_xy = [(20, 25), (40, 30), (28, 50)]
    data = _make_dataset_with_peaks(scan=4, det=det, peak_positions=peaks_xy)
    w = Show4DSTEM(data, center=(det // 2, det // 2), bf_radius=3.0,
                   precompute_virtual_images=False)
    w.bragg_detect_active = True
    found = _peaks_from_bytes(w.bragg_peaks_bytes)
    assert found.shape[0] >= 3
    _match_peaks(found, peaks_xy, tol=1.5)


@pytest.mark.parametrize("subpixel", ["pixel", "poly", "multicorr"])
def test_show4dstem_bragg_subpixel_modes_return_3_peaks(subpixel):
    det = 64
    peaks_xy = [(20, 25), (40, 30), (28, 50)]
    data = _make_dataset_with_peaks(scan=2, det=det, peak_positions=peaks_xy)
    w = Show4DSTEM(data, center=(det // 2, det // 2), bf_radius=3.0,
                   precompute_virtual_images=False)
    w.bragg_subpixel = subpixel
    w.bragg_detect_active = True
    found = _peaks_from_bytes(w.bragg_peaks_bytes)
    assert found.shape[0] >= 3, f"expected >=3 peaks for {subpixel}, got {found.shape[0]}"


def test_show4dstem_bragg_multicorr_subpixel_is_nonintegral():
    """multicorr should produce non-integer coordinates for off-grid peaks."""
    det = 64
    # Off-grid peaks (half-pixel positions)
    peaks_xy = [(20.5, 25.3), (40.2, 30.7)]
    dp = _make_dp_with_peaks(det, peaks_xy)
    data = np.broadcast_to(dp, (2, 2, det, det)).astype(np.float32).copy()
    w = Show4DSTEM(data, center=(det // 2, det // 2), bf_radius=3.0,
                   precompute_virtual_images=False)
    w.bragg_subpixel = "multicorr"
    w.bragg_upsample = 16
    w.bragg_detect_active = True
    found = _peaks_from_bytes(w.bragg_peaks_bytes)
    # At least one peak must be at a non-integer coordinate
    any_non_integer = np.any(
        (np.abs(found[:, 0] - np.round(found[:, 0])) > 1e-3)
        | (np.abs(found[:, 1] - np.round(found[:, 1])) > 1e-3)
    )
    assert any_non_integer, f"multicorr coords look integral: {found}"


def test_show4dstem_bragg_max_num_peaks_caps_output():
    det = 64
    # Many peaks (5×5 lattice — 25 peaks)
    peaks_xy = [(8 + 10 * i, 8 + 10 * j) for i in range(5) for j in range(5)]
    data = _make_dataset_with_peaks(scan=2, det=det, peak_positions=peaks_xy)
    w = Show4DSTEM(data, center=(det // 2, det // 2), bf_radius=3.0,
                   precompute_virtual_images=False)
    w.bragg_max_peaks = 7
    w.bragg_min_peak_spacing = 5.0
    w.bragg_detect_active = True
    found = _peaks_from_bytes(w.bragg_peaks_bytes)
    assert found.shape[0] <= 7, f"max_num_peaks=7 should cap at 7, got {found.shape[0]}"


def test_show4dstem_bragg_min_rel_threshold_removes_weak_peaks():
    det = 64
    # 3 peaks — 1 strong, 2 very weak
    yy, xx = np.meshgrid(np.arange(det), np.arange(det), indexing="ij")
    dp = (
        10.0 * np.exp(-((yy - 20) ** 2 + (xx - 25) ** 2) / 8.0)
        + 0.05 * np.exp(-((yy - 40) ** 2 + (xx - 30) ** 2) / 8.0)
        + 0.05 * np.exp(-((yy - 28) ** 2 + (xx - 50) ** 2) / 8.0)
    ).astype(np.float32)
    data = np.broadcast_to(dp, (2, 2, det, det)).astype(np.float32).copy()
    w = Show4DSTEM(data, center=(det // 2, det // 2), bf_radius=3.0,
                   precompute_virtual_images=False)

    # Low threshold: should find all 3 (or more)
    w.bragg_min_rel = 0.0
    w.bragg_detect_active = True
    found_low = _peaks_from_bytes(w.bragg_peaks_bytes)
    assert found_low.shape[0] >= 3

    # High threshold: should drop the weak ones
    w.bragg_min_rel = 0.5
    found_high = _peaks_from_bytes(w.bragg_peaks_bytes)
    assert found_high.shape[0] < found_low.shape[0]
    assert found_high.shape[0] >= 1


def test_show4dstem_bragg_inactive_bytes_empty():
    det = 32
    data = np.random.rand(2, 2, det, det).astype(np.float32)
    w = Show4DSTEM(data, center=(det // 2, det // 2), bf_radius=3.0,
                   precompute_virtual_images=False)
    assert w.bragg_detect_active is False
    assert w.bragg_peaks_bytes == b""

    # Activate then deactivate — bytes should be cleared
    w.bragg_detect_active = True
    assert w.bragg_peaks_bytes  # has data
    w.bragg_detect_active = False
    assert w.bragg_peaks_bytes == b""


def test_show4dstem_bragg_recomputes_on_pos_change():
    det = 64
    # Different peaks in different scan positions
    data = np.zeros((2, 2, det, det), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(det), np.arange(det), indexing="ij")
    data[0, 0] = np.exp(-((yy - 20) ** 2 + (xx - 25) ** 2) / 8.0)
    data[0, 1] = np.exp(-((yy - 40) ** 2 + (xx - 18) ** 2) / 8.0) + np.exp(
        -((yy - 15) ** 2 + (xx - 45) ** 2) / 8.0
    )
    data[1, 0] = np.exp(-((yy - 30) ** 2 + (xx - 35) ** 2) / 8.0)
    data[1, 1] = np.zeros((det, det), dtype=np.float32)

    w = Show4DSTEM(data, center=(det // 2, det // 2), bf_radius=3.0,
                   precompute_virtual_images=False)
    w.bragg_detect_active = True

    w.pos_row, w.pos_col = 0, 0
    p00 = _peaks_from_bytes(w.bragg_peaks_bytes)
    w.pos_row, w.pos_col = 0, 1
    p01 = _peaks_from_bytes(w.bragg_peaks_bytes)
    # Different DP -> different peak counts/positions
    if p00.shape[0] > 0 and p01.shape[0] > 0:
        # at least one of {count, top-peak-position} differs
        same_count = p00.shape[0] == p01.shape[0]
        same_top = (
            same_count
            and abs(p00[0, 0] - p01[0, 0]) < 0.5
            and abs(p00[0, 1] - p01[0, 1]) < 0.5
        )
        assert not same_top, "expected different peaks at different scan positions"


def test_show4dstem_set_vacuum_probe_accepts_custom_probe():
    det = 16
    data = np.random.rand(2, 2, det, det).astype(np.float32)
    w = Show4DSTEM(data, precompute_virtual_images=False)
    custom = np.ones((det, det), dtype=np.float32)
    w.set_vacuum_probe(custom)
    assert w.vacuum_probe_bytes  # non-empty
    arr = np.frombuffer(w.vacuum_probe_bytes, dtype=np.float32).reshape(det, det)
    assert np.allclose(arr, 1.0)


def test_show4dstem_set_vacuum_probe_none_resets_to_auto():
    det = 16
    data = np.random.rand(2, 2, det, det).astype(np.float32)
    w = Show4DSTEM(data, precompute_virtual_images=False)
    w.set_vacuum_probe(np.ones((det, det), dtype=np.float32))
    assert w.vacuum_probe_bytes
    w.set_vacuum_probe(None)
    assert w.vacuum_probe_bytes == b""
    # auto-built probe should be available via the helper
    probe = w._build_auto_vacuum_probe()
    assert probe.shape == (det, det)


def test_show4dstem_set_vacuum_probe_wrong_shape_raises():
    det = 16
    data = np.random.rand(2, 2, det, det).astype(np.float32)
    w = Show4DSTEM(data, precompute_virtual_images=False)
    with pytest.raises(ValueError):
        w.set_vacuum_probe(np.ones((8, 8), dtype=np.float32))


def test_show4dstem_bragg_state_dict_roundtrip():
    det = 16
    data = np.random.rand(2, 2, det, det).astype(np.float32)
    w = Show4DSTEM(data, precompute_virtual_images=False)
    w.bragg_detect_active = True
    w.bragg_sigma = 1.5
    w.bragg_corr_power = 0.5
    w.bragg_subpixel = "poly"
    w.bragg_max_peaks = 10
    w.bragg_min_rel = 0.01
    sd = w.state_dict()
    assert sd["bragg_detect_active"] is True
    assert sd["bragg_sigma"] == 1.5
    assert sd["bragg_corr_power"] == 0.5
    assert sd["bragg_subpixel"] == "poly"
    assert sd["bragg_max_peaks"] == 10
    assert sd["bragg_min_rel"] == 0.01

    w2 = Show4DSTEM(data, state=sd, precompute_virtual_images=False)
    assert w2.bragg_detect_active is True
    assert w2.bragg_sigma == 1.5
    assert w2.bragg_corr_power == 0.5
    assert w2.bragg_subpixel == "poly"
    assert w2.bragg_max_peaks == 10
    assert w2.bragg_min_rel == 0.01


def test_show4dstem_bragg_in_summary(capsys):
    det = 64
    peaks_xy = [(20, 25), (40, 30)]
    data = _make_dataset_with_peaks(scan=2, det=det, peak_positions=peaks_xy)
    w = Show4DSTEM(data, center=(det // 2, det // 2), bf_radius=3.0,
                   precompute_virtual_images=False)
    w.bragg_detect_active = True
    w.summary()
    out = capsys.readouterr().out
    assert "Bragg" in out
    # Subpixel mode is shown
    assert "multicorr" in out or "poly" in out or "pixel" in out


def test_show4dstem_bragg_invalid_subpixel_raises():
    det = 16
    data = np.random.rand(2, 2, det, det).astype(np.float32)
    w = Show4DSTEM(data, precompute_virtual_images=False)
    from traitlets import TraitError
    with pytest.raises(TraitError):
        w.bragg_subpixel = "bogus"


def test_show4dstem_detect_bragg_peaks_returns_array_when_inactive():
    """detect_bragg_peaks() should work whether or not bragg_detect_active is set."""
    det = 64
    peaks_xy = [(20, 25), (40, 30)]
    data = _make_dataset_with_peaks(scan=2, det=det, peak_positions=peaks_xy)
    w = Show4DSTEM(data, center=(det // 2, det // 2), bf_radius=3.0,
                   precompute_virtual_images=False)
    # active=False; method still works
    assert w.bragg_detect_active is False
    out = w.detect_bragg_peaks()
    assert out.shape[1] == 3
    assert out.shape[0] >= 2


# ─── Regression: sub-pixel center handled exactly (Fourier phase-ramp shift) ───


def test_bragg_subpixel_center_does_not_quantize_peaks():
    """Pin the Fourier phase-ramp center shift. With a vacuum probe whose
    true center is at (31.7, 32.4) (non-integer), peaks detected at known
    sub-pixel offsets must NOT snap to integer-rounded centers — which a
    plain ``np.roll`` would silently do.
    """
    import numpy as np
    from quantem.widget import Show4DSTEM

    det = 64
    cy, cx = 31.7, 32.4
    yy, xx = np.mgrid[:det, :det].astype(np.float32)
    # Three Bragg-like Gaussian spots at known sub-pixel offsets from (cy, cx).
    truth = [(cy + 12.3, cx - 8.6), (cy - 9.1, cx + 5.5), (cy + 4.4, cx + 14.2)]
    dp = np.zeros((det, det), dtype=np.float32)
    for ty, tx in truth:
        dp += np.exp(-((yy - ty) ** 2 + (xx - tx) ** 2) / (2.0 * 3.0 ** 2))
    data = dp[None, None, ...]

    w = Show4DSTEM(data, verbose=False, precompute_virtual_images=False)
    # Set a non-integer center; this is the key check.
    w.center_row = cy
    w.center_col = cx
    w.bf_radius = 5.0
    w.set_vacuum_probe(None)  # auto-build with center at (cy, cx)
    w.bragg_detect_active = True
    w.bragg_subpixel = "multicorr"
    w.bragg_min_peak_spacing = 4.0
    w.bragg_max_peaks = 10

    peaks = np.frombuffer(w.bragg_peaks_bytes, dtype=np.float32).reshape(-1, 3)
    assert peaks.shape[0] >= 3, f"expected >=3 peaks, got {peaks.shape[0]}"
    # For each truth peak, the nearest detected peak should be within 0.4 px.
    for ty, tx in truth:
        dists = np.hypot(peaks[:, 0] - ty, peaks[:, 1] - tx)
        nearest = float(dists.min())
        assert nearest < 0.4, (
            f"peak ({ty:.2f},{tx:.2f}) recovered to {nearest:.3f} px — "
            "indicates the sub-pixel center got quantized somewhere"
        )
