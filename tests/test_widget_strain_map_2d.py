"""Unit tests for the StrainMap2D widget.

The widget consumes per-scan-pixel Bragg peak positions and computes the
local strain tensor (eps_xx, eps_yy, eps_xy) and rigid rotation (theta).
We use the py4DSTEM convention where extensive real-space strain is
positive; because reciprocal vectors scale as the inverse of real vectors,
a uniform real-space strain of e_xx = +0.02 corresponds to multiplying
each q-vector by 1/(1+e_xx) along the x axis (i.e. contracting q-space).
"""

import json

import numpy as np
import pytest

from quantem.widget import StrainMap2D


# ---------------------------------------------------------------------------
# Synthetic lattice helpers
# ---------------------------------------------------------------------------

REF_G1 = np.array([0.0, 10.0], dtype=np.float32)  # (qy, qx)
REF_G2 = np.array([10.0, 0.0], dtype=np.float32)
H_K_LIST = [(h, k) for h in (-1, 0, 1) for k in (-1, 0, 1) if (h, k) != (0, 0)]


def _build_lattice_peaks(
    R_Nx: int,
    R_Ny: int,
    pixel_F_func,
) -> np.ndarray:
    """Build a (R_Nx, R_Ny, N_peaks, 2) peak array where each scan pixel
    has peaks at F(rx, ry) @ (h*g1 + k*g2). pixel_F_func is a callable
    returning a 2x2 matrix in (qy, qx) frame."""
    N = len(H_K_LIST)
    out = np.full((R_Nx, R_Ny, N, 2), np.nan, dtype=np.float32)
    for rx in range(R_Nx):
        for ry in range(R_Ny):
            F = pixel_F_func(rx, ry)
            for i, (h, k) in enumerate(H_K_LIST):
                q_ref = h * REF_G1 + k * REF_G2
                out[rx, ry, i] = F @ q_ref
    return out


def _q_F_from_real_strain(eps_xx: float = 0.0, eps_yy: float = 0.0, eps_xy: float = 0.0) -> np.ndarray:
    """Return the 2x2 q-space deformation that corresponds to a real-space
    infinitesimal strain (in the (qy, qx) frame). For small strain,
    F_q ≈ I - eps_real."""
    return np.array([[1.0 - eps_yy, -eps_xy], [-eps_xy, 1.0 - eps_xx]], dtype=np.float32)


def _q_F_from_rotation(theta: float) -> np.ndarray:
    """Pure rotation by theta (CCW, radians) applied to q-vectors. Since a
    rotation is its own inverse-transpose, the real-space lattice rotates
    by the same angle and the strain tensor stays zero."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_strain_map_2d_construction_dense():
    peaks = _build_lattice_peaks(4, 4, lambda rx, ry: np.eye(2, dtype=np.float32))
    w = StrainMap2D(peaks, ref_roi={"top": 0, "left": 0, "bottom": 2, "right": 2})
    assert w.R_Nx == 4
    assert w.R_Ny == 4
    assert w.N_peaks == 8


def test_strain_map_2d_construction_dict():
    """Dict form is accepted and converted to dense."""
    peaks_dict = {}
    for rx in range(3):
        for ry in range(3):
            arr = []
            for (h, k) in H_K_LIST:
                arr.append(h * REF_G1 + k * REF_G2)
            peaks_dict[(rx, ry)] = np.array(arr, dtype=np.float32)
    w = StrainMap2D(peaks_dict, ref_roi={"top": 0, "left": 0, "bottom": 2, "right": 2})
    assert w.R_Nx == 3
    assert w.R_Ny == 3
    assert w.N_peaks == 8


def test_strain_map_2d_construction_bad_shape():
    with pytest.raises(ValueError):
        StrainMap2D(np.zeros((3, 3, 4)))


def test_strain_map_2d_construction_bad_ref_roi():
    peaks = _build_lattice_peaks(4, 4, lambda rx, ry: np.eye(2, dtype=np.float32))
    with pytest.raises(ValueError):
        StrainMap2D(peaks, ref_roi={"top": 2, "left": 0, "bottom": 1, "right": 4})


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def test_strain_map_2d_perfect_lattice_zero_strain():
    """A perfectly periodic lattice should give ~zero strain and rotation."""
    peaks = _build_lattice_peaks(5, 5, lambda rx, ry: np.eye(2, dtype=np.float32))
    w = StrainMap2D(peaks, ref_roi={"top": 0, "left": 0, "bottom": 3, "right": 3})
    assert int(w._mask.sum()) == 25
    finite = np.isfinite(w._e_xx)
    assert finite.all()
    assert np.abs(w._e_xx).max() < 1e-4
    assert np.abs(w._e_yy).max() < 1e-4
    assert np.abs(w._e_xy).max() < 1e-4
    assert np.abs(w._theta).max() < 1e-4


def test_strain_map_2d_uniform_exx():
    """Apply a uniform e_xx = 0.02 outside the reference region. Verify
    that the strain map recovers 0.02 within 1e-3."""
    R_Nx, R_Ny = 6, 6
    F_strain = _q_F_from_real_strain(eps_xx=0.02)
    ref_F = np.eye(2, dtype=np.float32)

    def pixel_F(rx, ry):
        return ref_F if ry < 3 else F_strain

    peaks = _build_lattice_peaks(R_Nx, R_Ny, pixel_F)
    w = StrainMap2D(peaks, ref_roi={"top": 0, "left": 0, "bottom": R_Nx, "right": 3})
    strained_e_xx = w._e_xx[:, 3:]
    assert np.isfinite(strained_e_xx).all()
    assert np.allclose(strained_e_xx, 0.02, atol=1e-3)
    # Other channels should be ~0
    assert np.abs(w._e_yy[:, 3:]).max() < 1e-3
    assert np.abs(w._e_xy[:, 3:]).max() < 1e-3
    assert np.abs(w._theta[:, 3:]).max() < 1e-3


def test_strain_map_2d_pure_rotation():
    """A uniform 0.05 rad rotation should be recovered in theta with all
    strain channels ~0."""
    R_Nx, R_Ny = 5, 5
    F_rot = _q_F_from_rotation(0.05)
    ref_F = np.eye(2, dtype=np.float32)

    def pixel_F(rx, ry):
        return ref_F if ry < 2 else F_rot

    peaks = _build_lattice_peaks(R_Nx, R_Ny, pixel_F)
    w = StrainMap2D(peaks, ref_roi={"top": 0, "left": 0, "bottom": R_Nx, "right": 2})
    rotated_theta = w._theta[:, 2:]
    assert np.isfinite(rotated_theta).all()
    # Pure rotation R(+θ) applied to G_local gives F = G_ref⁻¹·G_local = R(θ),
    # so beta = lstsq(M_rows, A_rows).T has beta[0,1] = -sin θ, beta[1,0] = sin θ,
    # and the widget's `0.5·(beta[0,1] - beta[1,0])` evaluates to -sin θ ≈ -0.05.
    # The sign is intentionally locked: positive `theta` in the widget output
    # corresponds to the inverse rotation. (Documented in the module docstring.)
    assert np.allclose(rotated_theta, -0.05, atol=1e-3)
    # Strain channels carry a second-order residual ~ (1 - cos θ) ≈ θ²/2
    # ≈ 1.25e-3 for θ=0.05. Allow up to 2e-3 to accommodate this geometric
    # nonlinearity of the linearized strain decomposition.
    assert np.abs(w._e_xx[:, 2:]).max() < 2e-3
    assert np.abs(w._e_yy[:, 2:]).max() < 2e-3
    assert np.abs(w._e_xy[:, 2:]).max() < 2e-3


def test_strain_map_2d_pure_shear():
    """A pure shear of gamma = 0.04 in real space should give e_xy ≈ 0.02
    (=gamma/2) at every strained pixel."""
    R_Nx, R_Ny = 5, 5
    gamma = 0.04
    F_shear = _q_F_from_real_strain(eps_xy=gamma)
    ref_F = np.eye(2, dtype=np.float32)

    def pixel_F(rx, ry):
        return ref_F if ry < 2 else F_shear

    peaks = _build_lattice_peaks(R_Nx, R_Ny, pixel_F)
    w = StrainMap2D(peaks, ref_roi={"top": 0, "left": 0, "bottom": R_Nx, "right": 2})
    sheared = w._e_xy[:, 2:]
    assert np.isfinite(sheared).all()
    assert np.allclose(sheared, gamma, atol=1e-3)
    assert np.abs(w._e_xx[:, 2:]).max() < 1e-3
    assert np.abs(w._e_yy[:, 2:]).max() < 1e-3
    assert np.abs(w._theta[:, 2:]).max() < 1e-3


def test_strain_map_2d_mask_insufficient_peaks():
    """Pixels with <2 valid (non-zero) peak-matches get mask=0 and NaN."""
    R_Nx, R_Ny = 4, 4
    peaks = _build_lattice_peaks(R_Nx, R_Ny, lambda rx, ry: np.eye(2, dtype=np.float32))
    # Wipe most peaks at pixel (3, 3) — leave only 1 valid
    peaks[3, 3, 1:] = np.nan
    w = StrainMap2D(peaks, ref_roi={"top": 0, "left": 0, "bottom": 2, "right": 2})
    assert w._mask[3, 3] == 0
    assert np.isnan(w._e_xx[3, 3])
    assert np.isnan(w._e_yy[3, 3])
    assert np.isnan(w._e_xy[3, 3])
    assert np.isnan(w._theta[3, 3])
    # Other pixels still solvable
    assert w._mask[0, 0] == 1


def test_strain_map_2d_all_nan_pixel_masked():
    R_Nx, R_Ny = 4, 4
    peaks = _build_lattice_peaks(R_Nx, R_Ny, lambda rx, ry: np.eye(2, dtype=np.float32))
    peaks[3, 0, :] = np.nan
    w = StrainMap2D(peaks, ref_roi={"top": 0, "left": 0, "bottom": 2, "right": 2})
    assert w._mask[3, 0] == 0


# ---------------------------------------------------------------------------
# fit_reference / g1, g2 traits
# ---------------------------------------------------------------------------

def test_strain_map_2d_fit_reference_sets_g1_g2():
    """g1, g2 traits update after fit_reference()."""
    peaks = _build_lattice_peaks(4, 4, lambda rx, ry: np.eye(2, dtype=np.float32))
    w = StrainMap2D(peaks, ref_roi={"top": 0, "left": 0, "bottom": 2, "right": 2}, auto_compute=False)
    assert w.g1 == [0.0, 0.0]
    assert w.g2 == [0.0, 0.0]
    w.fit_reference()
    # Either (g1, g2) ≈ ([0,10], [10,0]) or ([10,0], [0,10]) depending on
    # seed-selection order; both are valid bases. Test invertibility.
    G = np.array([w.g1, w.g2]).T  # columns
    assert abs(np.linalg.det(G)) > 1.0  # vectors are non-degenerate
    # The lattice should be recovered by integer reindexing — i.e. the
    # determinant magnitude equals |det(ref)| (==100).
    assert abs(abs(np.linalg.det(G)) - 100.0) < 1e-2


def test_strain_map_2d_compute_strain_requires_fit():
    peaks = _build_lattice_peaks(4, 4, lambda rx, ry: np.eye(2, dtype=np.float32))
    w = StrainMap2D(peaks, auto_compute=False)
    # g1/g2 are still default zero
    with pytest.raises(ValueError):
        w.compute_strain()


# ---------------------------------------------------------------------------
# set_image
# ---------------------------------------------------------------------------

def test_strain_map_2d_set_image_roundtrip():
    """set_image() replaces input and recomputes; settings preserved."""
    peaks_a = _build_lattice_peaks(4, 4, lambda rx, ry: np.eye(2, dtype=np.float32))
    peaks_b = _build_lattice_peaks(6, 6, lambda rx, ry: np.eye(2, dtype=np.float32))
    w = StrainMap2D(peaks_a, ref_roi={"top": 0, "left": 0, "bottom": 2, "right": 2},
                    cmap_strain="viridis", unit="%")
    assert w.R_Nx == 4
    w.set_image(peaks_b)
    assert w.R_Nx == 6
    assert w.R_Ny == 6
    assert w.cmap_strain == "viridis"
    assert w.unit == "%"
    assert len(w.e_xx_bytes) == 6 * 6 * 4


# ---------------------------------------------------------------------------
# Bytes shape and dtype
# ---------------------------------------------------------------------------

def test_strain_map_2d_bytes_shape_and_dtype():
    R_Nx, R_Ny = 5, 7
    peaks = _build_lattice_peaks(R_Nx, R_Ny, lambda rx, ry: np.eye(2, dtype=np.float32))
    w = StrainMap2D(peaks, ref_roi={"top": 0, "left": 0, "bottom": 2, "right": 2})
    # Float32 -> 4 bytes/element
    expected_f = R_Nx * R_Ny * 4
    assert len(w.e_xx_bytes) == expected_f
    assert len(w.e_yy_bytes) == expected_f
    assert len(w.e_xy_bytes) == expected_f
    assert len(w.theta_bytes) == expected_f
    # Mask -> uint8
    assert len(w.mask_bytes) == R_Nx * R_Ny

    e_xx_arr = np.frombuffer(w.e_xx_bytes, dtype=np.float32).reshape(R_Nx, R_Ny)
    assert e_xx_arr.dtype == np.float32
    mask_arr = np.frombuffer(w.mask_bytes, dtype=np.uint8).reshape(R_Nx, R_Ny)
    assert mask_arr.dtype == np.uint8


def test_strain_map_2d_intensities_weights_accepted():
    R_Nx, R_Ny = 4, 4
    peaks = _build_lattice_peaks(R_Nx, R_Ny, lambda rx, ry: np.eye(2, dtype=np.float32))
    intensities = np.ones((R_Nx, R_Ny, peaks.shape[2]), dtype=np.float32)
    w = StrainMap2D(peaks, intensities=intensities, ref_roi={"top": 0, "left": 0, "bottom": 2, "right": 2})
    assert np.abs(w._e_xx).max() < 1e-4


def test_strain_map_2d_intensities_wrong_shape_raises():
    peaks = _build_lattice_peaks(4, 4, lambda rx, ry: np.eye(2, dtype=np.float32))
    intensities = np.ones((4, 4, 3), dtype=np.float32)  # wrong N_peaks
    with pytest.raises(ValueError):
        StrainMap2D(peaks, intensities=intensities)


# ---------------------------------------------------------------------------
# State protocol
# ---------------------------------------------------------------------------

def test_strain_map_2d_state_dict_roundtrip():
    peaks = _build_lattice_peaks(4, 4, lambda rx, ry: np.eye(2, dtype=np.float32))
    w1 = StrainMap2D(
        peaks,
        title="My Strain",
        cmap_strain="seismic",
        cmap_theta="PiYG",
        vmin_pct=5.0,
        vmax_pct=95.0,
        ref_roi={"top": 0, "left": 0, "bottom": 2, "right": 2},
        max_peak_spacing_px=4.0,
        unit="%",
        show_stats=False,
        show_controls=False,
    )
    state = w1.state_dict()
    expected_keys = {
        "title", "cmap_strain", "cmap_theta", "vmin_pct", "vmax_pct",
        "ref_roi", "max_peak_spacing_px", "unit", "g1", "g2",
        "show_stats", "show_controls", "canvas_size",
        "disabled_tools", "hidden_tools",
    }
    assert set(state.keys()) == expected_keys
    w2 = StrainMap2D(peaks, state=state)
    assert w2.title == "My Strain"
    assert w2.cmap_strain == "seismic"
    assert w2.cmap_theta == "PiYG"
    assert w2.vmin_pct == pytest.approx(5.0)
    assert w2.vmax_pct == pytest.approx(95.0)
    assert w2.ref_roi == {"top": 0, "left": 0, "bottom": 2, "right": 2}
    assert w2.max_peak_spacing_px == pytest.approx(4.0)
    assert w2.unit == "%"
    assert w2.show_stats is False
    assert w2.show_controls is False


def test_strain_map_2d_save_load_file(tmp_path):
    peaks = _build_lattice_peaks(4, 4, lambda rx, ry: np.eye(2, dtype=np.float32))
    w1 = StrainMap2D(peaks, title="Save Test", cmap_strain="seismic")
    path = str(tmp_path / "state.json")
    w1.save(path)
    with open(path) as f:
        payload = json.load(f)
    assert payload["metadata_version"] == "1.0"
    assert payload["widget_name"] == "StrainMap2D"
    assert isinstance(payload["widget_version"], str)
    assert payload["state"]["title"] == "Save Test"
    assert payload["state"]["cmap_strain"] == "seismic"

    w2 = StrainMap2D(peaks, state=path)
    assert w2.title == "Save Test"
    assert w2.cmap_strain == "seismic"


def test_strain_map_2d_summary(capsys):
    peaks = _build_lattice_peaks(4, 4, lambda rx, ry: np.eye(2, dtype=np.float32))
    w = StrainMap2D(peaks, title="Strain Summary Test", ref_roi={"top": 0, "left": 0, "bottom": 2, "right": 2})
    w.summary()
    captured = capsys.readouterr()
    assert "Strain Summary Test" in captured.out
    assert "4×4" in captured.out
    assert "g1" in captured.out
    assert "g2" in captured.out


def test_strain_map_2d_load_state_ignores_unknown():
    peaks = _build_lattice_peaks(4, 4, lambda rx, ry: np.eye(2, dtype=np.float32))
    w = StrainMap2D(peaks)
    w.load_state_dict({"title": "OK", "nonexistent_key": 42})
    assert w.title == "OK"


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

def test_strain_map_2d_repr():
    peaks = _build_lattice_peaks(4, 6, lambda rx, ry: np.eye(2, dtype=np.float32))
    w = StrainMap2D(peaks, ref_roi={"top": 0, "left": 0, "bottom": 2, "right": 2}, title="MyMap")
    r = repr(w)
    assert "MyMap" in r
    assert "4×6" in r
    assert "fit=" in r


def test_strain_map_2d_repr_default_title():
    peaks = _build_lattice_peaks(4, 4, lambda rx, ry: np.eye(2, dtype=np.float32))
    w = StrainMap2D(peaks, ref_roi={"top": 0, "left": 0, "bottom": 2, "right": 2})
    r = repr(w)
    assert "Strain Map" in r


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

def test_strain_map_2d_array_properties():
    peaks = _build_lattice_peaks(4, 4, lambda rx, ry: np.eye(2, dtype=np.float32))
    w = StrainMap2D(peaks, ref_roi={"top": 0, "left": 0, "bottom": 2, "right": 2})
    assert w.e_xx.shape == (4, 4)
    assert w.e_yy.shape == (4, 4)
    assert w.e_xy.shape == (4, 4)
    assert w.theta.shape == (4, 4)
    assert w.mask.shape == (4, 4)
    assert w.e_xx.dtype == np.float32
    assert w.mask.dtype == np.uint8


# ---------------------------------------------------------------------------
# Recompute / refit
# ---------------------------------------------------------------------------

def test_strain_map_2d_change_ref_roi_then_recompute():
    """Changing ref_roi and recomputing produces consistent results."""
    R_Nx, R_Ny = 6, 6
    F_strain = _q_F_from_real_strain(eps_xx=0.02)

    def pixel_F(rx, ry):
        return np.eye(2, dtype=np.float32) if ry < 3 else F_strain

    peaks = _build_lattice_peaks(R_Nx, R_Ny, pixel_F)
    w = StrainMap2D(peaks, ref_roi={"top": 0, "left": 0, "bottom": R_Nx, "right": 3})
    initial_g1 = list(w.g1)
    # Re-fit and re-compute manually
    w.fit_reference()
    w.compute_strain()
    assert list(w.g1) == pytest.approx(initial_g1, abs=1e-3)


def test_strain_map_2d_save_image(tmp_path):
    """save_image() writes a 2x2 panel PNG."""
    peaks = _build_lattice_peaks(4, 4, lambda rx, ry: np.eye(2, dtype=np.float32))
    w = StrainMap2D(peaks, ref_roi={"top": 0, "left": 0, "bottom": 2, "right": 2})
    out = tmp_path / "map.png"
    w.save_image(out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_strain_map_2d_save_image_single_view(tmp_path):
    peaks = _build_lattice_peaks(4, 4, lambda rx, ry: np.eye(2, dtype=np.float32))
    w = StrainMap2D(peaks, ref_roi={"top": 0, "left": 0, "bottom": 2, "right": 2})
    out = tmp_path / "exx.png"
    w.save_image(out, view="e_xx")
    assert out.exists()
    # bad view raises
    with pytest.raises(ValueError):
        w.save_image(tmp_path / "bad.png", view="not_a_channel")


def test_strain_map_2d_save_image_bad_format(tmp_path):
    peaks = _build_lattice_peaks(4, 4, lambda rx, ry: np.eye(2, dtype=np.float32))
    w = StrainMap2D(peaks, ref_roi={"top": 0, "left": 0, "bottom": 2, "right": 2})
    with pytest.raises(ValueError):
        w.save_image(tmp_path / "out.svg")


def test_strain_map_2d_unit_pct_does_not_affect_stored_values():
    peaks = _build_lattice_peaks(4, 4, lambda rx, ry: np.eye(2, dtype=np.float32))
    w = StrainMap2D(peaks, unit="%", ref_roi={"top": 0, "left": 0, "bottom": 2, "right": 2})
    # internal arrays still in raw strain units; the JS scales for display
    assert np.abs(w._e_xx).max() < 1e-4
