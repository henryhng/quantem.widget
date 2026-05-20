"""Tests for the AtomFinder widget."""

import json

import numpy as np
import pytest
import torch

from quantem.widget import AtomFinder
from quantem.widget.atom_finder import (
    _fit_gaussian_window,
    _gaussian_2d,
    _partition_by_intensity,
    _polarization_vectors,
)
from quantem.widget.io import IOResult


# ─── Synthetic image helpers ────────────────────────────────────────────


def _add_gaussian(img, row, col, amp=1.0, sigma=1.5):
    H, W = img.shape
    r = int(np.ceil(sigma * 4))
    r0, r1 = max(0, int(row) - r), min(H, int(row) + r + 1)
    c0, c1 = max(0, int(col) - r), min(W, int(col) + r + 1)
    ys, xs = np.mgrid[r0:r1, c0:c1]
    img[r0:r1, c0:c1] += amp * np.exp(-(((ys - row) ** 2 + (xs - col) ** 2) / (2 * sigma**2)))
    return img


def _make_synthetic_image(positions, amps, sigma=1.5, size=128, noise=0.0):
    img = np.zeros((size, size), dtype=np.float32)
    for (r, c), a in zip(positions, amps):
        _add_gaussian(img, r, c, amp=a, sigma=sigma)
    if noise > 0:
        rng = np.random.default_rng(0)
        img += rng.normal(0, noise, img.shape).astype(np.float32)
    return img


# ─── Construction ──────────────────────────────────────────────────────


def test_atomfinder_numpy_2d():
    data = np.zeros((64, 64), dtype=np.float32)
    _add_gaussian(data, 32, 32, amp=1.0, sigma=2.0)
    w = AtomFinder(data, auto_detect=False)
    assert w.height == 64
    assert w.width == 64
    assert len(w.frame_bytes) == 64 * 64 * 4
    assert w.n_atoms == 0  # auto_detect=False


def test_atomfinder_torch():
    img = torch.zeros(32, 32)
    img[15, 15] = 1.0
    w = AtomFinder(img, auto_detect=False)
    assert w.width == 32


def test_atomfinder_wrong_ndim_raises():
    with pytest.raises(ValueError):
        AtomFinder(np.zeros((4, 4, 4), dtype=np.float32))


def test_atomfinder_default_traits():
    img = _make_synthetic_image([(32.0, 32.0)], [1.0], size=64)
    w = AtomFinder(img, auto_detect=False)
    assert w.title == "Atom Finder"
    assert w.cmap == "gray"
    assert w.log_scale is False
    assert w.auto_contrast is True
    assert w.percentile_low == 2.0
    assert w.percentile_high == 98.0
    assert w.preprocess_sigma == 0.0
    assert w.min_sigma == 2.0
    assert w.max_sigma == 6.0
    assert w.blob_threshold == 0.05
    assert w.fit_gaussian_subpixel is True
    assert w.mask_radius_px == 8.0
    assert w.percent_to_nn == 0.4
    assert w.rotation_enabled is False
    assert w.n_sublattices == 1
    assert w.sublattice_mode == "intensity"
    assert w.sublattice_fraction == 0.5
    assert w.polarization_active is False


# ─── Detection accuracy ────────────────────────────────────────────────


def test_atomfinder_recovers_known_positions_subpixel():
    """Sub-pixel fit should locate Gaussians to better than 0.5 px."""
    positions = [(20.3, 30.7), (50.5, 60.2), (80.1, 90.8)]
    amps = [1.0, 1.0, 1.0]
    img = _make_synthetic_image(positions, amps, sigma=2.0, size=128)
    w = AtomFinder(
        img,
        min_sigma=1.0,
        max_sigma=3.5,
        blob_threshold=0.03,
        mask_radius_px=6.0,
        percent_to_nn=0.0,  # use mask_radius_px directly
        fit_gaussian_subpixel=True,
    )
    found = w.atom_positions
    assert found.shape[0] >= len(positions)
    # Each true position is within 0.5 px of some detection
    for r, c in positions:
        d = np.hypot(found[:, 0] - r, found[:, 1] - c)
        assert d.min() < 0.5, f"closest detection {d.min():.3f} px from ({r}, {c})"


def test_atomfinder_no_subpixel_gives_integer_like_positions():
    positions = [(20.3, 30.7), (50.5, 60.2)]
    img = _make_synthetic_image(positions, [1.0, 1.0], sigma=2.0, size=96)
    w = AtomFinder(
        img,
        min_sigma=1.0,
        max_sigma=3.5,
        blob_threshold=0.03,
        fit_gaussian_subpixel=False,
    )
    found = w.atom_positions
    assert found.shape[0] >= 2
    # Without subpixel, blob_log returns near-integer coords
    fractional = np.abs(found[:, :2] - np.round(found[:, :2]))
    assert fractional.max() < 1e-5

    # With subpixel, positions are refined (not integer)
    w.fit_gaussian_subpixel = True
    found_refined = w.atom_positions
    fractional_refined = np.abs(found_refined[:, :2] - np.round(found_refined[:, :2]))
    # At least one match should be off-integer
    assert fractional_refined.max() > 0.05


def test_atomfinder_mask_radius_constrains_fits():
    positions = [(20.0, 20.0), (40.0, 40.0), (60.0, 60.0)]
    img = _make_synthetic_image(positions, [1.0, 1.0, 1.0], sigma=2.0, size=96)
    # Sane fit window
    w_ok = AtomFinder(
        img,
        min_sigma=1.0,
        max_sigma=3.5,
        blob_threshold=0.03,
        mask_radius_px=8.0,
        percent_to_nn=0.0,
        fit_gaussian_subpixel=True,
    )
    n_ok = w_ok.n_atoms
    # Very tiny window: too few points for a robust Gaussian fit, some drop out
    w_tight = AtomFinder(
        img,
        min_sigma=1.0,
        max_sigma=3.5,
        blob_threshold=0.03,
        mask_radius_px=1.0,
        percent_to_nn=0.0,
        fit_gaussian_subpixel=True,
    )
    n_tight = w_tight.n_atoms
    assert n_tight <= n_ok


# ─── Sublattice partition ──────────────────────────────────────────────


def test_atomfinder_sublattice_intensity():
    # Two brightness groups: 3 bright + 3 dim
    bright = [(20.0, 20.0), (20.0, 60.0), (60.0, 40.0)]
    dim = [(40.0, 30.0), (40.0, 50.0), (50.0, 70.0)]
    img = _make_synthetic_image(
        bright + dim, [1.0, 1.0, 1.0, 0.3, 0.3, 0.3], sigma=2.0, size=96
    )
    w = AtomFinder(
        img,
        min_sigma=1.0,
        max_sigma=3.5,
        blob_threshold=0.02,
        n_sublattices=2,
        sublattice_fraction=0.5,
        mask_radius_px=6.0,
        percent_to_nn=0.0,
    )
    a_pos = w.sublattice_a_positions
    b_pos = w.sublattice_b_positions
    assert a_pos.shape[0] + b_pos.shape[0] == w.n_atoms
    # Mean intensity of A should be greater than B
    assert a_pos[:, 2].mean() > b_pos[:, 2].mean()


def test_atomfinder_single_sublattice_clears_partition():
    img = _make_synthetic_image([(32.0, 32.0), (50.0, 50.0)], [1.0, 1.0], size=80)
    w = AtomFinder(
        img,
        n_sublattices=2,
        sublattice_fraction=0.5,
        mask_radius_px=6.0,
        percent_to_nn=0.0,
        min_sigma=1.0,
        max_sigma=3.5,
        blob_threshold=0.02,
    )
    assert w.sublattice_a_indices.size + w.sublattice_b_indices.size == w.n_atoms
    w.n_sublattices = 1
    assert w.sublattice_a_indices.size == 0
    assert w.sublattice_b_indices.size == 0


# ─── Polarization ──────────────────────────────────────────────────────


def test_atomfinder_polarization_offcenter_b():
    """A regular square A lattice with B-site shifted off-centre yields non-zero polarization."""
    # 2x2 grid of bright A atoms (corners of a square), spacing 20 px.
    # B atom at the centre of the square, shifted (+3, +0).
    a_pos = [(20.0, 20.0), (20.0, 40.0), (40.0, 20.0), (40.0, 40.0)]
    b_pos = [(33.0, 30.0)]  # ideal centre = (30, 30), shifted by (+3, 0)
    img = _make_synthetic_image(
        a_pos + b_pos, [1.0] * 4 + [0.3], sigma=2.0, size=80
    )
    w = AtomFinder(
        img,
        min_sigma=1.0,
        max_sigma=3.5,
        blob_threshold=0.02,
        n_sublattices=2,
        sublattice_fraction=4.0 / 5.0,  # top 4/5 are A
        polarization_active=True,
        mask_radius_px=5.0,
        percent_to_nn=0.0,
    )
    pol = w.polarization
    assert pol.shape[0] >= 1
    # The single B is at (33, 30); its 4-NN A's are the 4 corners with
    # centroid (30, 30), so displacement ≈ (+3, 0).
    # Find the polarization entry near (33, 30).
    d = np.hypot(pol[:, 0] - 33.0, pol[:, 1] - 30.0)
    i = int(np.argmin(d))
    drow, dcol = float(pol[i, 2]), float(pol[i, 3])
    assert drow > 1.0, f"expected drow ≈ +3, got {drow}"
    assert abs(dcol) < 1.5, f"expected dcol ≈ 0, got {dcol}"


def test_atomfinder_polarization_disabled():
    img = _make_synthetic_image([(20.0, 20.0), (40.0, 40.0)], [1.0, 1.0], size=80)
    w = AtomFinder(
        img,
        min_sigma=1.0,
        max_sigma=3.5,
        blob_threshold=0.02,
        n_sublattices=2,
        polarization_active=False,
        mask_radius_px=6.0,
        percent_to_nn=0.0,
    )
    assert w.polarization.shape[0] == 0
    assert len(w.polarization_bytes) == 0
    w.polarization_active = True
    assert w.polarization.shape[0] >= 1


# ─── set_image round-trip ──────────────────────────────────────────────


def test_atomfinder_set_image_roundtrip():
    img1 = _make_synthetic_image([(20.0, 20.0)], [1.0], size=64)
    w = AtomFinder(img1, cmap="viridis", min_sigma=1.0, max_sigma=3.5, blob_threshold=0.02)
    assert w.n_atoms >= 1
    img2 = _make_synthetic_image([(50.0, 50.0), (60.0, 70.0)], [1.0, 1.0], size=128)
    w.set_image(img2)
    assert w.width == 128
    assert w.height == 128
    # cmap preserved
    assert w.cmap == "viridis"
    # New atoms detected
    assert w.n_atoms >= 2


def test_atomfinder_set_image_ioresult():
    img1 = _make_synthetic_image([(20.0, 20.0)], [1.0], size=64)
    w = AtomFinder(img1, auto_detect=False)
    new = IOResult(
        data=_make_synthetic_image([(30.0, 30.0)], [1.0], size=80),
        title="new", pixel_size=0.5, units="Å",
        labels=[], metadata={}, frame_metadata=[],
    )
    w.set_image(new)
    assert w.title == "new"
    assert w.pixel_size == 0.5
    assert w.width == 80


# ─── State persistence (3 required) ────────────────────────────────────


def test_atomfinder_state_dict_roundtrip():
    img = _make_synthetic_image([(20.0, 20.0)], [1.0], size=64)
    w = AtomFinder(img, cmap="viridis", min_sigma=1.0, max_sigma=3.5, blob_threshold=0.02)
    w.n_sublattices = 2
    w.polarization_active = True
    sd = w.state_dict()
    assert sd["cmap"] == "viridis"
    assert sd["n_sublattices"] == 2
    assert sd["polarization_active"] is True
    # Exclude bytes / stats from state_dict
    assert "frame_bytes" not in sd
    assert "atom_positions_bytes" not in sd
    assert "stats_mean" not in sd
    w2 = AtomFinder(img, state=sd)
    assert w2.cmap == "viridis"
    assert w2.n_sublattices == 2
    assert w2.polarization_active is True
    # Pipeline ran after restore
    assert w2.n_atoms >= 1


def test_atomfinder_save_load_file(tmp_path):
    img = _make_synthetic_image([(20.0, 20.0)], [1.0], size=64)
    w = AtomFinder(img, cmap="viridis", auto_detect=False)
    path = tmp_path / "atomfinder_state.json"
    w.save(str(path))
    saved = json.loads(path.read_text())
    assert saved["metadata_version"] == "1.0"
    assert saved["widget_name"] == "AtomFinder"
    assert "widget_version" in saved
    assert saved["state"]["cmap"] == "viridis"
    w2 = AtomFinder(img, state=str(path), auto_detect=False)
    assert w2.cmap == "viridis"


def test_atomfinder_summary(capsys):
    img = _make_synthetic_image([(20.0, 20.0)], [1.0], size=64)
    w = AtomFinder(
        img,
        pixel_size=0.18,
        min_sigma=1.0,
        max_sigma=3.5,
        blob_threshold=0.02,
        n_sublattices=2,
    )
    w.summary()
    out = capsys.readouterr().out
    assert "Image:" in out
    assert "Atoms:" in out
    assert "Detect:" in out
    assert "Refine:" in out
    assert "Split:" in out


# ─── repr ──────────────────────────────────────────────────────────────


def test_atomfinder_repr():
    img = _make_synthetic_image([(20.0, 20.0)], [1.0], size=64)
    w = AtomFinder(img, min_sigma=1.0, max_sigma=3.5, blob_threshold=0.02)
    r = repr(w)
    assert "AtomFinder" in r
    assert "atoms=" in r
    assert "64×64" in r


# ─── Edge cases ────────────────────────────────────────────────────────


def test_atomfinder_empty_image():
    img = np.zeros((64, 64), dtype=np.float32)
    w = AtomFinder(img, blob_threshold=0.05)
    assert w.n_atoms == 0
    assert w.atom_positions.shape == (0, 4)
    assert w.polarization.shape == (0, 4)


def test_atomfinder_single_peak_no_sublattices():
    img = _make_synthetic_image([(32.0, 32.0)], [1.0], size=64)
    w = AtomFinder(
        img,
        min_sigma=1.0,
        max_sigma=3.5,
        blob_threshold=0.02,
        n_sublattices=2,
        polarization_active=True,
    )
    # Polarization needs at least 1 A + 1 B
    # With one atom only and fraction=0.5, all goes to one bucket
    assert w.n_atoms == 1
    assert w.polarization.shape[0] == 0


def test_atomfinder_sublattice_mode_invalid():
    img = _make_synthetic_image([(20.0, 20.0)], [1.0], size=64)
    w = AtomFinder(img, auto_detect=False)
    with pytest.raises(ValueError):
        w.sublattice_mode = "kmeans_2_distances"


def test_atomfinder_n_sublattices_invalid():
    img = _make_synthetic_image([(20.0, 20.0)], [1.0], size=64)
    w = AtomFinder(img, auto_detect=False)
    with pytest.raises(ValueError):
        w.n_sublattices = 3


# ─── Module-level algorithm helpers ────────────────────────────────────


def test_helper_partition_by_intensity():
    pos = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float32)
    inten = np.array([0.1, 1.0, 0.5, 0.9])
    a, b = _partition_by_intensity(pos, inten, fraction=0.5)
    # Top 2: indices 1 and 3
    assert set(a.tolist()) == {1, 3}
    assert set(b.tolist()) == {0, 2}


def test_helper_polarization_vectors():
    pos = np.array(
        [[0, 0], [0, 10], [10, 0], [10, 10], [5, 5]], dtype=np.float32
    )
    a_idx = np.array([0, 1, 2, 3], dtype=np.int32)
    b_idx = np.array([4], dtype=np.int32)
    pol = _polarization_vectors(pos, a_idx, b_idx)
    assert pol.shape == (1, 4)
    # B at centroid → zero displacement
    assert abs(pol[0, 2]) < 1e-5
    assert abs(pol[0, 3]) < 1e-5


def test_helper_gaussian_2d_fit():
    """The Gaussian fit recovers known parameters within tight tolerance."""
    H, W = 21, 21
    ys, xs = np.mgrid[0:H, 0:W]
    img = _gaussian_2d(
        np.vstack([ys.ravel(), xs.ravel()]),
        amplitude=2.0,
        x0=10.4,
        y0=10.6,
        sigma_x=2.0,
        sigma_y=2.0,
        theta=0.0,
        offset=0.1,
    ).reshape(H, W)
    fit = _fit_gaussian_window(img, row=10.0, col=10.0, radius=8, initial_sigma=1.5, rotation_enabled=False)
    assert fit is not None
    refined_row, refined_col, sx, sy, amp, _theta = fit
    assert abs(refined_row - 10.6) < 0.1
    assert abs(refined_col - 10.4) < 0.1


# ─── Array compatibility ───────────────────────────────────────────────


def test_atomfinder_accepts_torch_tensor():
    img = torch.from_numpy(_make_synthetic_image([(20.0, 20.0)], [1.0], size=48))
    w = AtomFinder(img, min_sigma=1.0, max_sigma=3.5, blob_threshold=0.02)
    assert w.height == 48


def test_atomfinder_ioresult_metadata():
    arr = _make_synthetic_image([(20.0, 20.0)], [1.0], size=64)
    result = IOResult(
        data=arr, title="atom_image", pixel_size=0.18,
        units="Å", labels=[], metadata={}, frame_metadata=[],
    )
    w = AtomFinder(result, auto_detect=False)
    assert w.title == "atom_image"
    assert w.pixel_size == 0.18


# ─── Tool visibility ───────────────────────────────────────────────────


def test_atomfinder_tool_visibility():
    img = _make_synthetic_image([(20.0, 20.0)], [1.0], size=64)
    w = AtomFinder(img, auto_detect=False)
    w.disabled_tools = ["display", "detection"]
    assert "display" in w.disabled_tools
    assert "detection" in w.disabled_tools
    w.hidden_tools = ["histogram"]
    assert "histogram" in w.hidden_tools
    with pytest.raises(ValueError):
        w.disabled_tools = ["fake_tool"]


# ─── save_image ────────────────────────────────────────────────────────


def test_atomfinder_save_image_png(tmp_path):
    img = _make_synthetic_image([(20.0, 20.0), (50.0, 50.0)], [1.0, 0.5], size=80)
    w = AtomFinder(
        img,
        min_sigma=1.0,
        max_sigma=3.5,
        blob_threshold=0.02,
        n_sublattices=2,
        sublattice_fraction=0.5,
        polarization_active=True,
        mask_radius_px=6.0,
        percent_to_nn=0.0,
    )
    out = w.save_image(str(tmp_path / "out.png"))
    assert out.exists()
    assert out.stat().st_size > 0


def test_atomfinder_save_image_unknown_format_raises(tmp_path):
    img = _make_synthetic_image([(20.0, 20.0)], [1.0], size=48)
    w = AtomFinder(img, auto_detect=False)
    with pytest.raises(ValueError):
        w.save_image(str(tmp_path / "x.xyz"))


# ─── Observer triggers ─────────────────────────────────────────────────


def test_atomfinder_observer_reruns_on_threshold_change():
    img = _make_synthetic_image(
        [(20.0, 20.0), (40.0, 40.0), (60.0, 60.0)], [1.0, 0.5, 0.3], size=96
    )
    w = AtomFinder(
        img, min_sigma=1.0, max_sigma=3.5, blob_threshold=0.4, mask_radius_px=6.0,
        percent_to_nn=0.0,
    )
    n_high = w.n_atoms
    w.blob_threshold = 0.02
    n_low = w.n_atoms
    assert n_low >= n_high  # lower threshold should not lose detections


def test_atomfinder_n_atoms_consistent_with_bytes():
    img = _make_synthetic_image(
        [(20.0, 20.0), (40.0, 40.0)], [1.0, 1.0], size=80
    )
    w = AtomFinder(img, min_sigma=1.0, max_sigma=3.5, blob_threshold=0.02)
    decoded = np.frombuffer(w.atom_positions_bytes, dtype=np.float32).reshape(-1, 4)
    assert decoded.shape[0] == w.n_atoms
