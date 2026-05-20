"""Tests for TransferFunctionEditor.

Covers the required Widget State Protocol (state_dict_roundtrip,
save_load_file, summary), the LUT computation contract, handle
normalization, stretch-preset integration with
``quantem.core.visualization.custom_normalizations``, and the histogram
input-shape handling.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from quantem.widget import TransferFunctionEditor


# ---------------------------------------------------------------------------
# Construction / defaults
# ---------------------------------------------------------------------------


def test_construct_without_data():
    w = TransferFunctionEditor()
    # Default two-handle ramp
    assert len(w.tf_handles) == 2
    assert w.tf_handles[0]["x"] == 0.0
    assert w.tf_handles[-1]["x"] == 1.0
    # LUT was computed during __init__
    assert len(w.tf_lut_bytes) == w.n_bins * 4


def test_construct_with_numpy_2d():
    data = np.random.default_rng(0).standard_normal((32, 32)).astype(np.float32)
    w = TransferFunctionEditor(data)
    assert len(w.histogram_bytes) == w.n_bins * 4  # float32 per bin
    assert w.data_min < w.data_max


def test_construct_with_numpy_1d():
    data = np.linspace(0, 1, 100, dtype=np.float32)
    w = TransferFunctionEditor(data)
    assert len(w.histogram_bytes) == w.n_bins * 4
    assert w.data_min == pytest.approx(0.0)
    assert w.data_max == pytest.approx(1.0)


def test_construct_with_kwargs():
    w = TransferFunctionEditor(
        title="My TF",
        cmap="plasma",
        n_bins=128,
        stretch_preset="log",
        log_histogram=True,
        show_stats=False,
        show_controls=False,
    )
    assert w.title == "My TF"
    assert w.cmap == "plasma"
    assert w.n_bins == 128
    assert w.stretch_preset == "log"
    assert w.log_histogram is True
    assert w.show_stats is False
    assert w.show_controls is False
    assert len(w.tf_lut_bytes) == 128 * 4


# ---------------------------------------------------------------------------
# LUT contract
# ---------------------------------------------------------------------------


def test_default_lut_endpoint_alphas():
    """Default handles → first LUT row has alpha 0, last has alpha 255."""
    w = TransferFunctionEditor()
    lut = w.get_lut()
    assert lut.shape == (w.n_bins, 4)
    assert lut.dtype == np.uint8
    assert int(lut[0, 3]) == 0
    assert int(lut[-1, 3]) == 255


def test_add_handle_preserves_lut_length_and_opacity():
    """Add a handle at x=0.5 with opacity=0.7 — LUT length stays n_bins and
    the interpolated alpha at that x equals 255 * 0.7 (within rounding)."""
    w = TransferFunctionEditor()
    handles = list(w.tf_handles)
    handles.append({"x": 0.5, "opacity": 0.7, "color": [255, 0, 0]})
    w.tf_handles = handles
    lut = w.get_lut()
    assert lut.shape == (w.n_bins, 4)
    # Bin closest to x=0.5
    bin_idx = int(round(0.5 * (w.n_bins - 1)))
    assert int(lut[bin_idx, 3]) == pytest.approx(int(round(255 * 0.7)), abs=2)


def test_monotonic_x_ordering_corrected():
    """Passing unsorted handles is corrected by the validator."""
    w = TransferFunctionEditor()
    w.tf_handles = [
        {"x": 1.0, "opacity": 1.0, "color": [255, 255, 255]},
        {"x": 0.3, "opacity": 0.5, "color": [128, 128, 128]},
        {"x": 0.0, "opacity": 0.0, "color": [0, 0, 0]},
    ]
    xs = [h["x"] for h in w.tf_handles]
    assert xs == sorted(xs)
    assert xs == [0.0, 0.3, 1.0]


def test_handle_clamping():
    """Out-of-range x / opacity are clamped to [0, 1]."""
    w = TransferFunctionEditor()
    w.tf_handles = [
        {"x": -0.5, "opacity": 2.0, "color": [300, -10, 128]},
        {"x": 1.5, "opacity": -0.1, "color": [0, 0, 0]},
    ]
    h0, h1 = w.tf_handles
    assert h0["x"] == 0.0
    assert h0["opacity"] == 1.0
    assert h0["color"] == [255, 0, 128]
    assert h1["x"] == 1.0
    assert h1["opacity"] == 0.0


# ---------------------------------------------------------------------------
# Histogram + stretch
# ---------------------------------------------------------------------------


def test_histogram_shape_1d():
    # 50_000 samples into 256 bins → every bin is non-empty for a uniform ramp.
    data = np.linspace(0.0, 1.0, 50_000, dtype=np.float32)
    w = TransferFunctionEditor(data)
    counts = np.frombuffer(w.histogram_bytes, dtype=np.float32)
    assert counts.shape == (w.n_bins,)
    assert (counts > 0).sum() == w.n_bins
    assert counts.max() == pytest.approx(1.0)


def test_histogram_shape_2d():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 10, size=(50, 50)).astype(np.float32)
    w = TransferFunctionEditor(data)
    counts = np.frombuffer(w.histogram_bytes, dtype=np.float32)
    assert counts.shape == (w.n_bins,)
    assert counts.max() == pytest.approx(1.0)  # normalized to peak


def test_stretch_preset_log_changes_histogram():
    """Switching to log stretch with positive long-tail data should redistribute
    the histogram counts versus linear stretch."""
    rng = np.random.default_rng(0)
    # Long-tail-ish positive data: exponential
    data = rng.exponential(scale=1.0, size=10_000).astype(np.float32) + 1e-6
    w = TransferFunctionEditor(data, stretch_preset="linear")
    linear_counts = np.frombuffer(w.histogram_bytes, dtype=np.float32).copy()
    w.stretch_preset = "log"
    log_counts = np.frombuffer(w.histogram_bytes, dtype=np.float32).copy()
    # The two histograms must not be identical — log stretch spreads the
    # narrow low-end bulk out across the [0, 1] axis.
    assert not np.allclose(linear_counts, log_counts)


def test_invalid_stretch_preset_raises():
    w = TransferFunctionEditor()
    with pytest.raises(Exception):
        w.stretch_preset = "bogus"


# ---------------------------------------------------------------------------
# set_image protocol
# ---------------------------------------------------------------------------


def test_set_image_recomputes_histogram_keeps_handles():
    w = TransferFunctionEditor()
    # Customize handles, then replace the data
    custom_handles = [
        {"x": 0.0, "opacity": 0.0, "color": [10, 20, 30]},
        {"x": 0.5, "opacity": 0.9, "color": [200, 100, 50]},
        {"x": 1.0, "opacity": 1.0, "color": [255, 255, 255]},
    ]
    w.tf_handles = custom_handles
    saved = [dict(h) for h in w.tf_handles]

    new_data = np.random.default_rng(1).uniform(100, 500, size=(40, 40)).astype(np.float32)
    w.set_image(new_data)
    assert w.data_min >= 100.0
    assert w.data_max <= 500.0
    assert len(w.tf_handles) == 3
    assert [dict(h) for h in w.tf_handles] == saved


def test_set_image_none_clears_data():
    data = np.linspace(0, 1, 100, dtype=np.float32)
    w = TransferFunctionEditor(data)
    w.set_image(None)
    # Histogram is reset to zeros, domain falls back to default.
    counts = np.frombuffer(w.histogram_bytes, dtype=np.float32)
    assert counts.sum() == 0.0


# ---------------------------------------------------------------------------
# State protocol — REQUIRED by CLAUDE.md
# ---------------------------------------------------------------------------


def test_transfer_function_editor_state_dict_roundtrip():
    w = TransferFunctionEditor(
        title="Roundtrip",
        cmap="plasma",
        stretch_preset="power",
        log_histogram=True,
        n_bins=128,
    )
    w.tf_handles = [
        {"x": 0.0, "opacity": 0.0, "color": [0, 0, 0]},
        {"x": 0.4, "opacity": 0.6, "color": [200, 50, 50]},
        {"x": 1.0, "opacity": 1.0, "color": [255, 255, 255]},
    ]
    sd = w.state_dict()
    assert sd["title"] == "Roundtrip"
    assert sd["cmap"] == "plasma"
    assert sd["stretch_preset"] == "power"
    assert sd["log_histogram"] is True
    assert sd["n_bins"] == 128
    assert len(sd["tf_handles"]) == 3

    w2 = TransferFunctionEditor()
    w2.load_state_dict(sd)
    assert w2.title == "Roundtrip"
    assert w2.cmap == "plasma"
    assert w2.stretch_preset == "power"
    assert w2.log_histogram is True
    assert w2.n_bins == 128
    assert len(w2.tf_handles) == 3
    assert w2.tf_handles[1]["color"] == [200, 50, 50]


def test_transfer_function_editor_save_load_file(tmp_path):
    w = TransferFunctionEditor(title="Save Test", cmap="magma", stretch_preset="asinh")
    path = tmp_path / "tfe_state.json"
    w.save(str(path))
    assert path.exists()

    payload = json.loads(path.read_text())
    assert payload["metadata_version"] == "1.0"
    assert payload["widget_name"] == "TransferFunctionEditor"
    assert isinstance(payload["widget_version"], str)
    assert payload["state"]["title"] == "Save Test"
    assert payload["state"]["cmap"] == "magma"
    assert payload["state"]["stretch_preset"] == "asinh"

    # Restore from file path string
    w2 = TransferFunctionEditor(state=str(path))
    assert w2.title == "Save Test"
    assert w2.cmap == "magma"
    assert w2.stretch_preset == "asinh"


def test_transfer_function_editor_summary(capsys):
    w = TransferFunctionEditor(
        np.random.default_rng(0).standard_normal((32, 32)).astype(np.float32),
        title="My TFE",
        cmap="viridis",
        stretch_preset="linear",
    )
    w.summary()
    out = capsys.readouterr().out
    assert "TransferFunctionEditor" in out
    assert "My TFE" in out
    assert "viridis" in out
    assert "linear" in out
    assert "Handles" in out


def test_repr_contains_essentials():
    w = TransferFunctionEditor(cmap="plasma")
    r = repr(w)
    assert "TransferFunctionEditor" in r
    assert "plasma" in r
    assert "n_handles=2" in r


# ---------------------------------------------------------------------------
# Smoke test for n_bins change
# ---------------------------------------------------------------------------


def test_n_bins_change_resizes_lut():
    w = TransferFunctionEditor()
    w.n_bins = 64
    assert len(w.tf_lut_bytes) == 64 * 4
    w.n_bins = 512
    assert len(w.tf_lut_bytes) == 512 * 4


def test_n_bins_too_small_raises():
    w = TransferFunctionEditor()
    with pytest.raises(Exception):
        w.n_bins = 1


# ---------------------------------------------------------------------------
# Top-level import wiring
# ---------------------------------------------------------------------------


def test_top_level_import():
    from quantem.widget import TransferFunctionEditor as TFE_imported

    assert TFE_imported is TransferFunctionEditor
