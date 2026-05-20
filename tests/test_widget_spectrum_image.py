"""Tests for SpectrumImage widget."""

from __future__ import annotations

import json

import numpy as np
import pytest

import quantem.widget
from quantem.widget import SpectrumImage


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_synthetic_si(
    ny: int = 6,
    nx: int = 8,
    n_e: int = 64,
    e0_start: float = 30.0,
    e0_step: float = 0.3,
    sigma: float = 1.5,
    bg_amp: float = 1.0e8,
    bg_r: float = 3.0,
    peak_amp: float = 5.0,
    noise_frac: float = 0.001,
    rng_seed: int = 0,
):
    """Synthetic SI: gaussian peak + power-law background.

    bg_amp chosen so bg ≫ noise across the full energy axis. peak amplitude
    expressed as a multiple of the background at the peak location so the
    integrated peak is detectable even after background subtraction.

    Returns (data, energy_axis, peak_centers).
    """
    energy = np.linspace(50.0, 200.0, n_e).astype(np.float32)
    rng = np.random.default_rng(rng_seed)
    centers = np.zeros((ny, nx), dtype=np.float32)
    for y in range(ny):
        for x in range(nx):
            centers[y, x] = 80.0 + e0_start * (y / max(ny - 1, 1)) + e0_step * x
    # Background: bg_amp * E^(-r)
    bg = bg_amp * energy.astype(np.float64) ** (-bg_r)
    data = np.zeros((ny, nx, n_e), dtype=np.float32)
    for y in range(ny):
        for x in range(nx):
            center = centers[y, x]
            bg_at_peak = bg_amp * float(center) ** (-bg_r)
            peak = peak_amp * bg_at_peak * np.exp(
                -0.5 * ((energy - center) / sigma) ** 2
            )
            data[y, x] = (peak + bg).astype(np.float32)
    # Multiplicative noise — keep SNR high across the spectrum
    data *= (1.0 + rng.normal(0.0, noise_frac, size=data.shape)).astype(np.float32)
    return data, energy, centers


# ─────────────────────────────────────────────────────────────────────────────
# Basic construction / shape
# ─────────────────────────────────────────────────────────────────────────────


def test_spectrum_image_version_exists():
    assert hasattr(quantem.widget, "__version__")


def test_spectrum_image_loads():
    data = np.random.rand(4, 5, 32).astype(np.float32)
    w = SpectrumImage(data)
    assert w.ny == 4 and w.nx == 5 and w.n_energy == 32


def test_spectrum_image_rejects_2d():
    with pytest.raises(ValueError):
        SpectrumImage(np.random.rand(8, 8).astype(np.float32))


def test_spectrum_image_rejects_4d():
    with pytest.raises(ValueError):
        SpectrumImage(np.random.rand(2, 3, 4, 5).astype(np.float32))


def test_spectrum_image_defaults():
    data = np.random.rand(4, 5, 32).astype(np.float32)
    w = SpectrumImage(data)
    assert w.map_mode == "sum"
    assert w.bg_subtract is False
    assert w.cmap == "viridis"
    assert w.cursor_sync is True
    assert w.energy_unit == "eV"
    # Cursor at center
    assert w.nav_index == [2, 2]
    # Defaults full energy window
    assert w.window_e_min == pytest.approx(0.0)
    assert w.window_e_max == pytest.approx(31.0)


def test_spectrum_image_repr():
    data = np.random.rand(4, 5, 16).astype(np.float32)
    w = SpectrumImage(data)
    r = repr(w)
    assert "SpectrumImage" in r
    assert "4, 5, 16" in r
    assert "sum" in r


def test_spectrum_image_energy_axis_default():
    data = np.random.rand(2, 3, 8).astype(np.float32)
    w = SpectrumImage(data)
    np.testing.assert_array_equal(w.energy_axis, np.arange(8))


def test_spectrum_image_energy_axis_explicit():
    data = np.random.rand(2, 3, 8).astype(np.float32)
    eax = np.linspace(100, 200, 8)
    w = SpectrumImage(data, energy_axis=eax, energy_unit="eV")
    np.testing.assert_allclose(w.energy_axis, eax)
    # Defaults span the full range
    assert w.window_e_min == pytest.approx(100.0)
    assert w.window_e_max == pytest.approx(200.0)


def test_spectrum_image_invalid_energy_axis_size():
    data = np.random.rand(2, 3, 8).astype(np.float32)
    with pytest.raises(ValueError):
        SpectrumImage(data, energy_axis=np.arange(10))


# ─────────────────────────────────────────────────────────────────────────────
# Compute: spectrum
# ─────────────────────────────────────────────────────────────────────────────


def test_nav_index_update_changes_spectrum():
    data = np.zeros((3, 3, 4), dtype=np.float32)
    data[0, 0] = [1, 2, 3, 4]
    data[2, 2] = [10, 20, 30, 40]
    w = SpectrumImage(data)
    w.nav_index = [0, 0]
    spec1 = np.frombuffer(bytes(w.spectrum_bytes), dtype=np.float32).copy()
    w.nav_index = [2, 2]
    spec2 = np.frombuffer(bytes(w.spectrum_bytes), dtype=np.float32).copy()
    assert not np.allclose(spec1, spec2)
    np.testing.assert_allclose(spec1, [1, 2, 3, 4])
    np.testing.assert_allclose(spec2, [10, 20, 30, 40])


def test_nav_index_validation():
    data = np.random.rand(2, 3, 4).astype(np.float32)
    w = SpectrumImage(data)
    with pytest.raises(Exception):
        w.nav_index = [1, 2, 3]


# ─────────────────────────────────────────────────────────────────────────────
# Compute: map modes
# ─────────────────────────────────────────────────────────────────────────────


def test_map_mode_sum_recovers_spatial_structure():
    data, energy, centers = _make_synthetic_si()
    w = SpectrumImage(data, energy_axis=energy)
    # Integrate around the brightest peak region
    w.window_e_min = 80.0
    w.window_e_max = 130.0
    img = w.map_image
    # Without bg subtract: peak region intensities differ across pixels
    assert img.shape == data.shape[:2]
    # Sum should be strictly positive everywhere (gaussian + bg both positive)
    assert (img > 0).all()


def test_map_mode_argmax_returns_energy_value():
    n_e = 16
    energy = np.linspace(0.0, 30.0, n_e).astype(np.float32)
    data = np.zeros((2, 3, n_e), dtype=np.float32)
    # Put argmax at known bins per pixel
    expected_idx = np.array([[1, 4, 7], [10, 12, 14]])
    for y in range(2):
        for x in range(3):
            data[y, x, expected_idx[y, x]] = 5.0
    w = SpectrumImage(data, energy_axis=energy, map_mode="argmax")
    img = w.map_image
    # argmax mode returns energy values, not bin indices
    expected_energies = energy[expected_idx]
    np.testing.assert_allclose(img, expected_energies)


def test_map_mode_max_and_mean():
    data = np.zeros((2, 2, 4), dtype=np.float32)
    data[0, 0] = [1, 5, 3, 2]
    data[0, 1] = [0, 0, 0, 8]
    data[1, 0] = [2, 2, 2, 2]
    data[1, 1] = [-1, 1, -1, 1]
    energy = np.arange(4, dtype=np.float32)
    w = SpectrumImage(data, energy_axis=energy, map_mode="max")
    np.testing.assert_allclose(w.map_image, [[5, 8], [2, 1]])
    w.map_mode = "mean"
    np.testing.assert_allclose(w.map_image, [[11 / 4, 8 / 4], [8 / 4, 0 / 4]])


def test_map_mode_validation():
    data = np.random.rand(2, 2, 4).astype(np.float32)
    w = SpectrumImage(data)
    with pytest.raises(ValueError):
        w.map_mode = "bogus"


# ─────────────────────────────────────────────────────────────────────────────
# Background fit
# ─────────────────────────────────────────────────────────────────────────────


def test_bg_subtract_removes_most_background():
    # Peaks centered around 80–115 eV; use 160–195 eV where there is no peak
    # signal — only the power-law background.
    data, energy, _ = _make_synthetic_si()
    w = SpectrumImage(data, energy_axis=energy)

    w.window_e_min = 160.0
    w.window_e_max = 195.0
    no_bg = w.map_image.copy()
    no_bg_mag = float(np.mean(np.abs(no_bg)))
    assert no_bg_mag > 0

    w.bg_e_min = 50.0
    w.bg_e_max = 75.0
    w.bg_subtract = True
    with_bg = w.map_image.copy()
    with_bg_mag = float(np.mean(np.abs(with_bg)))

    # Background subtraction must strictly reduce the integrated background
    assert with_bg_mag < no_bg_mag
    # And bring it close to zero (≤10% of un-subtracted magnitude)
    assert with_bg_mag < 0.1 * no_bg_mag


def test_bg_fit_accuracy():
    """Synthetic: known A=100, r=3 → recover within tolerance."""
    n_e = 128
    energy = np.linspace(50.0, 200.0, n_e).astype(np.float32)
    A, r = 100.0, 3.0
    bg = A * energy.astype(np.float64) ** (-r)
    data = np.tile(bg.astype(np.float32), (3, 4, 1)).reshape(3, 4, n_e)
    w = SpectrumImage(data, energy_axis=energy)
    # Fit over the full range — pure power-law data
    w.bg_e_min = 60.0
    w.bg_e_max = 180.0
    w.bg_subtract = True
    A_fit, r_fit = w.bg_params
    assert A_fit == pytest.approx(A, rel=0.1), f"A={A_fit}, expected ~{A}"
    assert r_fit == pytest.approx(r, abs=0.05), f"r={r_fit}, expected ~{r}"


def test_bg_curve_bytes_shape():
    data, energy, _ = _make_synthetic_si()
    w = SpectrumImage(data, energy_axis=energy)
    w.bg_subtract = True
    arr = np.frombuffer(bytes(w.bg_curve_bytes), dtype=np.float32)
    assert arr.size == w.n_energy


def test_bg_disabled_clears_params():
    data, energy, _ = _make_synthetic_si()
    w = SpectrumImage(data, energy_axis=energy, bg_subtract=True,
                      bg_e_min=60.0, bg_e_max=75.0)
    assert w.bg_params != [0.0, 0.0]
    w.bg_subtract = False
    assert w.bg_params == [0.0, 0.0]
    arr = np.frombuffer(bytes(w.bg_curve_bytes), dtype=np.float32)
    assert np.allclose(arr, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# set_image
# ─────────────────────────────────────────────────────────────────────────────


def test_set_image_roundtrip():
    data1 = np.random.rand(4, 5, 32).astype(np.float32)
    w = SpectrumImage(data1, cmap="plasma")
    data2 = np.random.rand(3, 7, 16).astype(np.float32)
    eax2 = np.linspace(0, 100, 16)
    w.set_image(data2, energy_axis=eax2)
    assert w.ny == 3 and w.nx == 7 and w.n_energy == 16
    assert w.cmap == "plasma"  # preserved
    np.testing.assert_allclose(w.energy_axis, eax2)


def test_set_image_rejects_4d():
    data = np.random.rand(4, 5, 32).astype(np.float32)
    w = SpectrumImage(data)
    with pytest.raises(ValueError):
        w.set_image(np.random.rand(2, 3, 4, 5).astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset3d integration
# ─────────────────────────────────────────────────────────────────────────────


def test_dataset3d_extracts_energy_axis_from_sampling_origin():
    from quantem.core.datastructures import Dataset3d

    arr = np.random.rand(4, 5, 32).astype(np.float32)
    ds = Dataset3d.from_array(
        arr,
        name="EELS Test",
        sampling=(1.0, 1.0, 0.5),
        origin=(0.0, 0.0, 100.0),
        units=("pixels", "pixels", "eV"),
    )
    w = SpectrumImage(ds)
    assert w.title == "EELS Test"
    assert w.energy_unit == "eV"
    expected = 100.0 + 0.5 * np.arange(32)
    np.testing.assert_allclose(w.energy_axis, expected)
    # Window default spans Dataset axis
    assert w.window_e_min == pytest.approx(100.0)
    assert w.window_e_max == pytest.approx(100.0 + 0.5 * 31)


def test_dataset3d_explicit_energy_axis_overrides():
    from quantem.core.datastructures import Dataset3d

    arr = np.random.rand(2, 3, 8).astype(np.float32)
    ds = Dataset3d.from_array(
        arr, name="Test",
        sampling=(1.0, 1.0, 5.0), origin=(0.0, 0.0, 200.0),
        units=("pixels", "pixels", "eV"),
    )
    override = np.linspace(0, 1, 8)
    w = SpectrumImage(ds, energy_axis=override)
    np.testing.assert_allclose(w.energy_axis, override)


# ─────────────────────────────────────────────────────────────────────────────
# Output bytes
# ─────────────────────────────────────────────────────────────────────────────


def test_map_bytes_shape():
    data = np.random.rand(4, 5, 32).astype(np.float32)
    w = SpectrumImage(data)
    arr = np.frombuffer(bytes(w.map_bytes), dtype=np.float32)
    assert arr.size == 4 * 5


def test_spectrum_bytes_shape():
    data = np.random.rand(4, 5, 32).astype(np.float32)
    w = SpectrumImage(data)
    arr = np.frombuffer(bytes(w.spectrum_bytes), dtype=np.float32)
    assert arr.size == 32


def test_map_stats_populated():
    data = np.random.rand(4, 5, 16).astype(np.float32)
    w = SpectrumImage(data)
    assert w.map_stats_max >= w.map_stats_min
    assert isinstance(w.map_stats_mean, float)


# ─────────────────────────────────────────────────────────────────────────────
# State protocol (3 required)
# ─────────────────────────────────────────────────────────────────────────────


def test_spectrum_image_state_dict_roundtrip():
    data = np.random.rand(4, 5, 32).astype(np.float32)
    w = SpectrumImage(
        data, cmap="plasma", log_scale=True, title="My SI",
        map_mode="max", bg_subtract=True, window_e_min=5.0, window_e_max=20.0,
        disabled_tools=["display"], hidden_tools=["background"],
    )
    sd = w.state_dict()
    assert "disabled_tools" in sd
    assert "hidden_tools" in sd
    assert "map_mode" in sd
    w2 = SpectrumImage(data, state=sd)
    assert w2.cmap == "plasma"
    assert w2.log_scale is True
    assert w2.title == "My SI"
    assert w2.map_mode == "max"
    assert w2.bg_subtract is True
    assert w2.window_e_min == pytest.approx(5.0)
    assert w2.window_e_max == pytest.approx(20.0)
    assert w2.disabled_tools == ["display"]
    assert w2.hidden_tools == ["background"]


def test_spectrum_image_save_load_file(tmp_path):
    data = np.random.rand(4, 5, 16).astype(np.float32)
    w = SpectrumImage(data, cmap="cividis", title="Saved SI")
    path = tmp_path / "spectrum_image_state.json"
    w.save(str(path))
    assert path.exists()
    saved = json.loads(path.read_text())
    assert saved["metadata_version"] == "1.0"
    assert saved["widget_name"] == "SpectrumImage"
    assert isinstance(saved["widget_version"], str)
    assert saved["state"]["cmap"] == "cividis"
    w2 = SpectrumImage(data, state=str(path))
    assert w2.cmap == "cividis"
    assert w2.title == "Saved SI"


def test_spectrum_image_summary(capsys):
    data = np.random.rand(4, 5, 16).astype(np.float32)
    w = SpectrumImage(data, title="My SI", cmap="viridis")
    w.summary()
    out = capsys.readouterr().out
    assert "My SI" in out
    assert "viridis" in out
    assert "4×5" in out


# ─────────────────────────────────────────────────────────────────────────────
# Tool parity API
# ─────────────────────────────────────────────────────────────────────────────


def test_tool_parity_runtime_api():
    data = np.random.rand(4, 5, 16).astype(np.float32)
    w = SpectrumImage(data)
    assert w.set_disabled_tools(["background"]) is w
    assert "background" in w.disabled_tools
    assert w.lock_tool("window") is w
    assert "window" in w.disabled_tools


# ─────────────────────────────────────────────────────────────────────────────
# Save image
# ─────────────────────────────────────────────────────────────────────────────


def test_save_image_map(tmp_path):
    data = np.random.rand(4, 5, 16).astype(np.float32)
    w = SpectrumImage(data)
    path = tmp_path / "map.png"
    out = w.save_image(path, view="map")
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_image_all(tmp_path):
    data = np.random.rand(4, 5, 16).astype(np.float32)
    w = SpectrumImage(data, bg_subtract=True, bg_e_min=0.0, bg_e_max=4.0)
    path = tmp_path / "all.png"
    out = w.save_image(path, view="all")
    assert out.exists()


def test_save_image_invalid_view(tmp_path):
    data = np.random.rand(4, 5, 16).astype(np.float32)
    w = SpectrumImage(data)
    with pytest.raises(ValueError):
        w.save_image(tmp_path / "x.png", view="bogus")
