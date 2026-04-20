import json

import numpy as np
import pytest
import torch

from quantem.widget import ShowDiffraction
from quantem.widget.io import IOResult


# ── Basic Construction ─────────────────────────────────────────────────


def test_showdiffraction_loads():
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    w = ShowDiffraction(data, verbose=False)
    assert w.shape_rows == 8
    assert w.shape_cols == 8
    assert w.det_rows == 16
    assert w.det_cols == 16


def test_showdiffraction_3d_with_scan_shape():
    data = np.zeros((6, 4, 4), dtype=np.float32)
    w = ShowDiffraction(data, scan_shape=(2, 3), verbose=False)
    assert (w.shape_rows, w.shape_cols) == (2, 3)
    assert (w.det_rows, w.det_cols) == (4, 4)


def test_showdiffraction_3d_square_inferred():
    data = np.zeros((9, 4, 4), dtype=np.float32)
    w = ShowDiffraction(data, verbose=False)
    assert (w.shape_rows, w.shape_cols) == (3, 3)


def test_showdiffraction_3d_nonsquare_raises():
    data = np.zeros((7, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="Cannot infer"):
        ShowDiffraction(data, verbose=False)


def test_showdiffraction_wrong_ndim_raises():
    data = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="Expected 3D or 4D"):
        ShowDiffraction(data, verbose=False)


def test_showdiffraction_default_log_scale():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = ShowDiffraction(data, verbose=False)
    assert w.dp_scale_mode == "log"


def test_showdiffraction_custom_scale_mode():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = ShowDiffraction(data, dp_scale_mode="linear", verbose=False)
    assert w.dp_scale_mode == "linear"


# ── Auto-detect Center ─────────────────────────────────────────────────


def test_showdiffraction_auto_detect_center():
    data = np.zeros((2, 2, 7, 7), dtype=np.float32)
    for i in range(7):
        for j in range(7):
            if np.sqrt((i - 3) ** 2 + (j - 3) ** 2) <= 1.5:
                data[:, :, i, j] = 100.0
    w = ShowDiffraction(data, verbose=False)
    assert abs(w.center_row - 3.0) < 0.5
    assert abs(w.center_col - 3.0) < 0.5
    assert w.bf_radius > 0


def test_showdiffraction_manual_center():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = ShowDiffraction(data, center=(5.0, 6.0), bf_radius=3.0, verbose=False)
    assert w.center_row == 5.0
    assert w.center_col == 6.0
    assert w.bf_radius == 3.0


# ── d-spacing Computation ─────────────────────────────────────────────


def test_showdiffraction_add_spot_calibrated():
    data = np.random.rand(4, 4, 32, 32).astype(np.float32)
    w = ShowDiffraction(data, k_pixel_size=0.1, center=(16, 16), bf_radius=5, verbose=False)
    w.add_spot(16, 26)  # 10 pixels from center
    assert len(w.spots) == 1
    spot = w.spots[0]
    assert spot["id"] == 1
    assert abs(spot["r_pixels"] - 10.0) < 0.01
    assert abs(spot["g_magnitude"] - 1.0) < 0.01  # 10 * 0.1
    assert abs(spot["d_spacing"] - 1.0) < 0.01  # 1 / 1.0


def test_showdiffraction_add_spot_uncalibrated():
    data = np.random.rand(4, 4, 32, 32).astype(np.float32)
    w = ShowDiffraction(data, center=(16, 16), bf_radius=5, verbose=False)
    w.add_spot(16, 26)
    spot = w.spots[0]
    assert spot["d_spacing"] is None
    assert spot["g_magnitude"] is None
    assert abs(spot["r_pixels"] - 10.0) < 0.01


def test_showdiffraction_add_spot_diagonal():
    data = np.random.rand(4, 4, 32, 32).astype(np.float32)
    w = ShowDiffraction(data, k_pixel_size=0.05, center=(16, 16), bf_radius=5, verbose=False)
    w.add_spot(19, 20)  # sqrt(9+16) = 5 pixels from center
    spot = w.spots[0]
    expected_r = np.sqrt(9 + 16)
    assert abs(spot["r_pixels"] - expected_r) < 0.01
    assert abs(spot["g_magnitude"] - expected_r * 0.05) < 0.01


def test_showdiffraction_spot_at_center():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = ShowDiffraction(data, k_pixel_size=0.1, center=(8, 8), bf_radius=3, verbose=False)
    w.add_spot(8, 8)
    spot = w.spots[0]
    assert spot["r_pixels"] == pytest.approx(0.0)
    assert spot["d_spacing"] is None  # g=0, can't compute d


# ── Snap to Peak ──────────────────────────────────────────────────────


def test_showdiffraction_snap_to_peak():
    data = np.zeros((4, 4, 16, 16), dtype=np.float32)
    data[:, :, 5, 8] = 100.0  # bright spot at (5, 8)
    w = ShowDiffraction(data, snap_enabled=True, snap_radius=3, center=(8, 8), bf_radius=3, verbose=False)
    w.add_spot(6, 7)  # click near the peak
    spot = w.spots[0]
    assert spot["row"] == 5.0
    assert spot["col"] == 8.0


def test_showdiffraction_snap_disabled():
    data = np.zeros((4, 4, 16, 16), dtype=np.float32)
    data[:, :, 5, 8] = 100.0
    w = ShowDiffraction(data, snap_enabled=False, center=(8, 8), bf_radius=3, verbose=False)
    w.add_spot(6, 7)
    spot = w.spots[0]
    assert spot["row"] == 6.0
    assert spot["col"] == 7.0


# ── Spot Management ───────────────────────────────────────────────────


def test_showdiffraction_undo_spot():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = ShowDiffraction(data, center=(8, 8), bf_radius=3, verbose=False)
    w.add_spot(5, 5)
    w.add_spot(10, 10)
    assert len(w.spots) == 2
    w.undo_spot()
    assert len(w.spots) == 1
    assert w.spots[0]["id"] == 1


def test_showdiffraction_clear_spots():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = ShowDiffraction(data, center=(8, 8), bf_radius=3, verbose=False)
    w.add_spot(5, 5)
    w.add_spot(10, 10)
    w.clear_spots()
    assert len(w.spots) == 0


def test_showdiffraction_undo_empty_is_noop():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = ShowDiffraction(data, verbose=False)
    w.undo_spot()  # should not raise
    assert len(w.spots) == 0


def test_showdiffraction_multiple_spots():
    data = np.random.rand(4, 4, 32, 32).astype(np.float32)
    w = ShowDiffraction(data, k_pixel_size=0.1, center=(16, 16), bf_radius=5, verbose=False)
    w.add_spot(16, 26)
    w.add_spot(6, 16)
    w.add_spot(16, 6)
    assert len(w.spots) == 3
    assert w.spots[0]["id"] == 1
    assert w.spots[1]["id"] == 2
    assert w.spots[2]["id"] == 3


# ── Virtual Image ─────────────────────────────────────────────────────


def test_showdiffraction_virtual_image_computed():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = ShowDiffraction(data, verbose=False)
    assert len(w.virtual_image_bytes) > 0
    vi = np.frombuffer(w.virtual_image_bytes, dtype=np.float32)
    assert vi.size == w.shape_rows * w.shape_cols


# ── Position Property ─────────────────────────────────────────────────


def test_showdiffraction_position_property():
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    w = ShowDiffraction(data, verbose=False)
    w.position = (3, 5)
    assert w.position == (3, 5)
    assert w.pos_row == 3
    assert w.pos_col == 5


def test_showdiffraction_position_clamped():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = ShowDiffraction(data, verbose=False)
    w.position = (100, 100)
    assert w.pos_row == 3
    assert w.pos_col == 3


# ── State Persistence ─────────────────────────────────────────────────


def test_showdiffraction_state_dict_roundtrip():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = ShowDiffraction(data, center=(5.0, 6.0), bf_radius=3.0, k_pixel_size=0.1, verbose=False)
    w.dp_scale_mode = "linear"
    w.add_spot(8, 8)
    sd = w.state_dict()
    assert sd["dp_scale_mode"] == "linear"
    assert sd["center_row"] == 5.0
    assert len(sd["spots"]) == 1
    w2 = ShowDiffraction(data, state=sd, verbose=False)
    assert w2.dp_scale_mode == "linear"
    assert w2.bf_radius == 3.0
    assert len(w2.spots) == 1


def test_showdiffraction_save_load_file(tmp_path):
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = ShowDiffraction(data, verbose=False)
    w.dp_colormap = "viridis"
    path = tmp_path / "diff_state.json"
    w.save(str(path))
    assert path.exists()
    saved = json.loads(path.read_text())
    assert saved["metadata_version"] == "1.0"
    assert saved["widget_name"] == "ShowDiffraction"
    assert saved["state"]["dp_colormap"] == "viridis"
    w2 = ShowDiffraction(data, state=str(path), verbose=False)
    assert w2.dp_colormap == "viridis"


def test_showdiffraction_summary(capsys):
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = ShowDiffraction(data, pixel_size=2.39, k_pixel_size=0.1, verbose=False)
    w.add_spot(5, 5)
    w.summary()
    out = capsys.readouterr().out
    assert "ShowDiffraction" in out or "4×4" in out
    assert "Spots" in out


def test_showdiffraction_repr():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = ShowDiffraction(data, k_pixel_size=0.1, verbose=False)
    r = repr(w)
    assert "ShowDiffraction" in r
    assert "4, 4, 16, 16" in r


# ── set_image ─────────────────────────────────────────────────────────


def test_showdiffraction_set_image():
    data = np.random.rand(4, 4, 32, 32).astype(np.float32)
    w = ShowDiffraction(data, verbose=False)
    w.add_spot(10, 10)
    assert len(w.spots) == 1
    new_data = np.random.rand(8, 8, 64, 64).astype(np.float32)
    w.set_image(new_data)
    assert w.shape_rows == 8
    assert w.det_rows == 64
    assert len(w.spots) == 0  # spots cleared


# ── Tool Visibility ───────────────────────────────────────────────────


def test_showdiffraction_disabled_tools():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = ShowDiffraction(data, verbose=False)
    w.disabled_tools = ["display", "spots"]
    assert "display" in w.disabled_tools
    assert "spots" in w.disabled_tools


def test_showdiffraction_hidden_tools():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = ShowDiffraction(data, verbose=False)
    w.hidden_tools = ["histogram"]
    assert "histogram" in w.hidden_tools


def test_showdiffraction_invalid_tool_raises():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = ShowDiffraction(data, verbose=False)
    with pytest.raises(ValueError):
        w.disabled_tools = ["fake_tool"]


# ── Array Compatibility ───────────────────────────────────────────────


def test_showdiffraction_accepts_torch_tensor():
    data = torch.rand(4, 4, 16, 16)
    w = ShowDiffraction(data, verbose=False)
    assert w.shape_rows == 4


def test_showdiffraction_accepts_ioresult():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    result = IOResult(
        data=data, title="test_scan", pixel_size=2.0,
        units="Å", labels=[], metadata={}, frame_metadata=[],
    )
    w = ShowDiffraction(result, verbose=False)
    assert w.title == "test_scan"
    assert w.pixel_size == 2.0


# ── Hot Pixel Removal ─────────────────────────────────────────────────


def test_showdiffraction_hot_pixel_removal():
    data = np.ones((4, 4, 32, 32), dtype=np.uint16) * 100
    data[0, 0, 3, 5] = 65535  # single hot pixel (1 out of 16384)
    w = ShowDiffraction(data, verbose=False)
    frame = w._get_frame(0, 0)
    assert frame[3, 5] == 0  # hot pixel removed


# ── Free ──────────────────────────────────────────────────────────────


def test_showdiffraction_free():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = ShowDiffraction(data, verbose=False)
    w.free()
    assert not hasattr(w, "_data")
