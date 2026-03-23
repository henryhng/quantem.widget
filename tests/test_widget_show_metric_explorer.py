import json

import numpy as np
import pytest

from quantem.widget import MetricExplorer


def _make_points(n=11, h=32, w=32, n_groups=2):
    """Create synthetic points for testing."""
    points = []
    for g in range(n_groups):
        for i in range(n):
            c10 = -50.0 + i * 10.0
            points.append({
                "image": np.random.randn(h, w).astype(np.float32),
                "label": f"r={g * 10 + 20} C10={c10:+.0f}nm",
                "metrics": {
                    "variance_loss": 0.05 - 0.001 * abs(c10),
                    "tv": 5000.0 + 100 * abs(c10),
                    "fft_snr": 10.0 - 0.1 * abs(c10),
                },
                "params": {
                    "C10_nm": c10,
                    "bf_radius": g * 10 + 20,
                },
            })
    return points


# =========================================================================
# Basic construction
# =========================================================================
def test_basic_construction():
    pts = _make_points(n=5, n_groups=1)
    w = MetricExplorer(pts, x_key="C10_nm")
    assert w.n_points == 5
    assert w.height == 32
    assert w.width == 32
    assert len(w.metric_names) == 3
    assert "variance_loss" in w.metric_names


def test_construction_with_all_options():
    pts = _make_points(n=3, n_groups=2)
    w = MetricExplorer(
        pts,
        x_key="C10_nm",
        x_label="Defocus C10 (nm)",
        group_key="bf_radius",
        metric_labels={"tv": "Total Variation"},
        metric_directions={"fft_snr": "max"},
        cmap="viridis",
        pixel_size=0.185,
        pixel_unit="Å",
    )
    assert w.n_points == 6
    assert w.x_key == "C10_nm"
    assert w.x_label == "Defocus C10 (nm)"
    assert w.group_key == "bf_radius"
    assert w.cmap == "viridis"
    assert w.pixel_size == 0.185
    assert w.pixel_unit == "Å"
    assert w.metric_labels[w.metric_names.index("tv")] == "Total Variation"
    assert w.metric_directions[w.metric_names.index("fft_snr")] == "max"


def test_empty_points_raises():
    with pytest.raises(ValueError, match="non-empty"):
        MetricExplorer([])


def test_3d_image_raises():
    pts = [{"image": np.zeros((2, 3, 4)), "label": "x", "metrics": {"a": 1}, "params": {}}]
    with pytest.raises(ValueError, match="2D"):
        MetricExplorer(pts)


def test_mismatched_shapes_raises():
    pts = [
        {"image": np.zeros((32, 32)), "label": "a", "metrics": {"a": 1}, "params": {}},
        {"image": np.zeros((16, 16)), "label": "b", "metrics": {"a": 2}, "params": {}},
    ]
    with pytest.raises(ValueError, match="same shape"):
        MetricExplorer(pts)


# =========================================================================
# Properties
# =========================================================================
def test_selected_point():
    pts = _make_points(n=3, n_groups=1)
    w = MetricExplorer(pts, x_key="C10_nm")
    assert w.selected_index == 0
    assert w.selected_point == pts[0]


def test_selected_params():
    pts = _make_points(n=3, n_groups=1)
    w = MetricExplorer(pts, x_key="C10_nm")
    assert "C10_nm" in w.selected_params


def test_selected_image():
    pts = _make_points(n=3, n_groups=1)
    w = MetricExplorer(pts, x_key="C10_nm")
    img = w.selected_image
    assert img.shape == (32, 32)
    assert img.dtype == np.float32


def test_selected_index_change():
    pts = _make_points(n=5, n_groups=1)
    w = MetricExplorer(pts, x_key="C10_nm")
    w.selected_index = 3
    assert w.selected_point == pts[3]
    np.testing.assert_array_equal(w.selected_image, w._data[3])


# =========================================================================
# Data bytes
# =========================================================================
def test_points_bytes_size():
    pts = _make_points(n=4, n_groups=1)
    w = MetricExplorer(pts, x_key="C10_nm")
    expected = 4 * 32 * 32 * 4  # n_points * H * W * float32
    assert len(w.points_bytes) == expected


def test_pixel_size():
    pts = _make_points(n=2, n_groups=1)
    w = MetricExplorer(pts, pixel_size=0.185, pixel_unit="Å")
    assert w.pixel_size == 0.185
    assert w.pixel_unit == "Å"


# =========================================================================
# State persistence (3 required tests)
# =========================================================================
def test_state_dict_roundtrip():
    pts = _make_points(n=3, n_groups=1)
    w = MetricExplorer(pts, x_key="C10_nm", cmap="viridis")
    w.selected_index = 2

    state = w.state_dict()
    assert state["selected_index"] == 2
    assert state["cmap"] == "viridis"

    # Create new widget and restore
    w2 = MetricExplorer(pts, x_key="C10_nm", state=state)
    assert w2.selected_index == 2
    assert w2.cmap == "viridis"


def test_save_load_file(tmp_path):
    pts = _make_points(n=3, n_groups=1)
    w = MetricExplorer(pts, x_key="C10_nm", cmap="magma")
    w.selected_index = 1

    path = str(tmp_path / "explorer_state.json")
    w.save(path)

    # Verify envelope
    data = json.loads(open(path).read())
    assert "metadata_version" in data
    assert data["widget_name"] == "MetricExplorer"
    assert "widget_version" in data
    assert "state" in data
    assert data["state"]["cmap"] == "magma"
    assert data["state"]["selected_index"] == 1

    # Load from file
    w2 = MetricExplorer(pts, x_key="C10_nm", state=path)
    assert w2.cmap == "magma"
    assert w2.selected_index == 1


def test_summary(capsys):
    pts = _make_points(n=3, n_groups=1)
    w = MetricExplorer(
        pts,
        x_key="C10_nm",
        x_label="Defocus",
    )
    w.summary()
    captured = capsys.readouterr()
    assert "MetricExplorer" in captured.out
    assert "3" in captured.out
    assert "32 x 32" in captured.out
    assert "variance_loss" in captured.out


# =========================================================================
# Repr
# =========================================================================
def test_repr():
    pts = _make_points(n=5, n_groups=2)
    w = MetricExplorer(pts, x_key="C10_nm")
    r = repr(w)
    assert "10 points" in r
    assert "32x32" in r
    assert "3 metrics" in r


# =========================================================================
# Metric defaults
# =========================================================================
def test_default_metric_directions():
    pts = _make_points(n=2, n_groups=1)
    w = MetricExplorer(pts)
    # variance_loss should default to "min"
    idx = w.metric_names.index("variance_loss")
    assert w.metric_directions[idx] == "min"
    # fft_snr should default to "max"
    idx = w.metric_names.index("fft_snr")
    assert w.metric_directions[idx] == "max"


# =========================================================================
# Tool visibility
# =========================================================================
def test_tool_groups():
    pts = _make_points(n=2, n_groups=1)
    w = MetricExplorer(pts, disabled_tools=["display"])
    assert "display" in w.disabled_tools


def test_hidden_tools():
    pts = _make_points(n=2, n_groups=1)
    w = MetricExplorer(pts, hidden_tools=["export"])
    assert "export" in w.hidden_tools
