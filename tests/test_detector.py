"""Tests for quantem.widget.detector — BF disk detection and virtual imaging."""

import numpy as np
import pytest
import torch

from quantem.widget.detector import detect_bf_disk, make_virtual_masks, virtual_images


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_4dstem_with_disk(
    scan_rows: int = 32,
    scan_cols: int = 32,
    det_rows: int = 48,
    det_cols: int = 48,
    center_row: float = 22.5,
    center_col: float = 25.5,
    bf_radius: float = 5.0,
    seed: int = 42,
) -> np.ndarray:
    """Create synthetic 4D-STEM data with a BF disk at a known position."""
    rng = np.random.default_rng(seed)
    data = rng.random((scan_rows, scan_cols, det_rows, det_cols), dtype=np.float32) * 0.1
    row_coords, col_coords = np.ogrid[:det_rows, :det_cols]
    disk = ((row_coords - center_row) ** 2 + (col_coords - center_col) ** 2) < bf_radius**2
    data[:, :, disk] += 5.0
    return data


# ---------------------------------------------------------------------------
# detect_bf_disk
# ---------------------------------------------------------------------------


class TestDetectBfDisk:
    def test_detects_centered_disk(self):
        data = _make_4dstem_with_disk(center_row=24.0, center_col=24.0, bf_radius=6.0)
        center_row, center_col, bf_radius = detect_bf_disk(data)
        assert abs(center_row - 24.0) < 0.5
        assert abs(center_col - 24.0) < 0.5
        assert abs(bf_radius - 6.0) < 1.0

    def test_detects_offcenter_disk(self):
        data = _make_4dstem_with_disk(center_row=15.0, center_col=30.0, bf_radius=4.0)
        center_row, center_col, bf_radius = detect_bf_disk(data)
        assert abs(center_row - 15.0) < 0.5
        assert abs(center_col - 30.0) < 0.5
        assert abs(bf_radius - 4.0) < 1.0

    def test_accepts_5d_input(self):
        data_4d = _make_4dstem_with_disk(center_row=20.0, center_col=20.0)
        data_5d = np.stack([data_4d, data_4d, data_4d])  # (3, 32, 32, 48, 48)
        center_row, center_col, bf_radius = detect_bf_disk(data_5d)
        assert abs(center_row - 20.0) < 0.5
        assert abs(center_col - 20.0) < 0.5

    def test_accepts_pytorch_tensor(self):
        data = _make_4dstem_with_disk(center_row=24.0, center_col=24.0)
        tensor = torch.from_numpy(data)
        center_row, center_col, bf_radius = detect_bf_disk(tensor)
        assert abs(center_row - 24.0) < 0.5
        assert abs(center_col - 24.0) < 0.5

    def test_fallback_on_uniform_data(self):
        data = np.ones((8, 8, 24, 24), dtype=np.float32)
        center_row, center_col, bf_radius = detect_bf_disk(data)
        # Uniform data → threshold filters everything → fallback to center
        assert center_row == pytest.approx(12.0)
        assert center_col == pytest.approx(12.0)

    def test_matches_show4dstem_algorithm(self):
        """Verify detect_bf_disk matches Show4DSTEM.auto_detect_center exactly."""
        data = _make_4dstem_with_disk(
            scan_rows=16, scan_cols=16, det_rows=48, det_cols=48,
            center_row=22.5, center_col=25.5, bf_radius=5.0,
        )
        # detector.py (numpy)
        center_row_np, center_col_np, bf_radius_np = detect_bf_disk(data)

        # Show4DSTEM algorithm (pytorch) — ported line-by-line
        tensor = torch.from_numpy(data)
        summed_dp = tensor.sum(dim=(0, 1))
        threshold = summed_dp.mean() + summed_dp.std()
        mask = summed_dp > threshold
        total = mask.sum()
        row_coords = torch.arange(48, dtype=torch.float64).reshape(-1, 1)
        col_coords = torch.arange(48, dtype=torch.float64).reshape(1, -1)
        center_row_pt = float((row_coords * mask).sum() / total)
        center_col_pt = float((col_coords * mask).sum() / total)
        bf_radius_pt = float(torch.sqrt(total / torch.pi))

        assert center_row_np == pytest.approx(center_row_pt, abs=0.01)
        assert center_col_np == pytest.approx(center_col_pt, abs=0.01)
        assert bf_radius_np == pytest.approx(bf_radius_pt, abs=0.01)


# ---------------------------------------------------------------------------
# make_virtual_masks
# ---------------------------------------------------------------------------


class TestMakeVirtualMasks:
    def test_masks_cover_detector(self):
        bf, adf, haadf = make_virtual_masks(48, 48, 24.0, 24.0, 6.0)
        # Every pixel should be in exactly one mask
        total = bf + adf + haadf
        assert np.all(total > 0.99)
        assert np.all(total < 1.01)

    def test_bf_mask_center_is_one(self):
        bf, _, _ = make_virtual_masks(48, 48, 24.0, 24.0, 6.0)
        assert bf[24, 24] == 1.0

    def test_haadf_mask_corner_is_one(self):
        _, _, haadf = make_virtual_masks(48, 48, 24.0, 24.0, 6.0)
        assert haadf[0, 0] == 1.0

    def test_masks_are_float32(self):
        bf, adf, haadf = make_virtual_masks(24, 24, 12.0, 12.0, 4.0)
        assert bf.dtype == np.float32
        assert adf.dtype == np.float32
        assert haadf.dtype == np.float32

    def test_offcenter_masks(self):
        bf, _, _ = make_virtual_masks(48, 48, 10.0, 10.0, 5.0)
        assert bf[10, 10] == 1.0
        assert bf[24, 24] == 0.0  # far from center


# ---------------------------------------------------------------------------
# virtual_images
# ---------------------------------------------------------------------------


class TestVirtualImages:
    def test_output_shapes_4d(self):
        data = _make_4dstem_with_disk(scan_rows=16, scan_cols=16)
        bf, adf, haadf = virtual_images(data)
        assert bf.shape == (16, 16)
        assert adf.shape == (16, 16)
        assert haadf.shape == (16, 16)

    def test_output_shapes_5d(self):
        data_4d = _make_4dstem_with_disk(scan_rows=8, scan_cols=8)
        data_5d = np.stack([data_4d, data_4d])
        bf, adf, haadf = virtual_images(data_5d)
        assert bf.shape == (2, 8, 8)

    def test_bf_brighter_than_haadf(self):
        """BF disk should produce higher mean in BF image than HAADF."""
        data = _make_4dstem_with_disk()
        bf, _, haadf = virtual_images(data)
        assert bf.mean() > haadf.mean()

    def test_custom_center(self):
        data = _make_4dstem_with_disk(center_row=24.0, center_col=24.0, bf_radius=5.0)
        bf1, _, _ = virtual_images(data)
        bf2, _, _ = virtual_images(data, center=(24.0, 24.0), bf_radius=5.0)
        # Explicitly passing the correct center should give similar results
        np.testing.assert_allclose(bf1, bf2, atol=0.1)

    def test_accepts_pytorch(self):
        data = _make_4dstem_with_disk(scan_rows=8, scan_cols=8)
        tensor = torch.from_numpy(data)
        bf, adf, haadf = virtual_images(tensor)
        assert bf.shape == (8, 8)

    def test_module_import(self):
        from quantem.widget.detector import virtual_images as vi
        assert callable(vi)

    def test_detect_bf_disk_module_import(self):
        from quantem.widget.detector import detect_bf_disk as dbf
        assert callable(dbf)


# ---------------------------------------------------------------------------
# Integration with real arina fixture
# ---------------------------------------------------------------------------


class TestVirtualImagesArina:
    FIXTURE = "tests/data/arina_fixture/test_arina_master.h5"

    def test_virtual_images_from_arina(self):
        """Compute virtual images from the real arina test fixture."""
        import os
        from quantem.widget import IO

        fixture_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            self.FIXTURE,
        )
        if not os.path.exists(fixture_path):
            pytest.skip("arina fixture not found")
        # Fixture is 16 frames, 24×24 detector → reshape to 4×4 scan
        result = IO.arina_file(fixture_path, det_bin=1, scan_shape=(4, 4))
        bf, adf, haadf = virtual_images(result.data)
        assert bf.ndim == 2
        assert bf.shape == (4, 4)
        assert adf.shape == (4, 4)
        assert haadf.shape == (4, 4)
