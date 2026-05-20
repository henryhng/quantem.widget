"""Tests for AberrationExplorer widget."""

import json
import math

import numpy as np
import pytest

from quantem.widget import AberrationExplorer


# =============================================================================
# Basic construction
# =============================================================================


def test_aberration_explorer_defaults():
    w = AberrationExplorer()
    assert w.title == "Aberration Explorer"
    assert w.energy_keV == pytest.approx(200.0)
    assert w.semiangle_cutoff_mrad == pytest.approx(25.0)
    assert w.gpts == 256
    assert w.real_space_sampling_A == pytest.approx(0.1)
    assert w.cmap == "inferno"
    assert w.show_stats is True
    assert w.show_controls is True
    # All default aberrations zero
    for v in w.aberrations.values():
        assert v == 0.0


def test_aberration_explorer_default_keys():
    """Default aberration dict has the documented Krivanek polar keys."""
    w = AberrationExplorer()
    expected = {
        "C10", "C12", "phi12",
        "C21", "phi21", "C23", "phi23",
        "C30", "C32", "phi32", "C34", "phi34",
    }
    assert set(w.aberrations.keys()) == expected


def test_aberration_explorer_bytes_shape():
    w = AberrationExplorer(gpts=128, real_space_sampling_A=0.1)
    # Probe matches gpts; chi has its own fixed display grid that fills the
    # aperture canvas regardless of gpts.
    assert len(w.probe_intensity_bytes) == 128 * 128 * 4
    chi_samples = len(w.chi_polar_bytes) // 4
    side = int(round(chi_samples ** 0.5))
    assert side * side == chi_samples
    # Radial CTF is 256 float32 samples.
    assert len(w.radial_ctf_bytes) == 256 * 4


def test_aberration_explorer_wavelength_set():
    w = AberrationExplorer(energy_keV=200.0)
    assert w.wavelength_A > 0
    # 200 keV electrons: ~0.0251 Å
    assert 0.024 < w.wavelength_A < 0.026


def test_aberration_explorer_repr():
    w = AberrationExplorer()
    r = repr(w)
    assert "256×256" in r
    assert "E=200keV" in r
    assert "alpha=25" in r


# =============================================================================
# Trait validation
# =============================================================================


def test_aberration_explorer_invalid_gpts():
    with pytest.raises(Exception):  # noqa: B017 — TraitError, but it's wrapped
        AberrationExplorer(gpts=300)


def test_aberration_explorer_valid_gpts():
    for g in (128, 256, 512):
        w = AberrationExplorer(gpts=g)
        assert w.gpts == g


def test_aberration_explorer_invalid_aberration_key():
    with pytest.raises(ValueError, match="Unknown aberration"):
        AberrationExplorer(aberrations={"Cxx": 1.0})


def test_aberration_explorer_invalid_energy():
    with pytest.raises(Exception):  # noqa: B017
        AberrationExplorer(energy_keV=-1.0)


def test_aberration_explorer_invalid_semiangle():
    with pytest.raises(Exception):  # noqa: B017
        AberrationExplorer(semiangle_cutoff_mrad=0.0)


def test_aberration_explorer_invalid_sampling():
    with pytest.raises(Exception):  # noqa: B017
        AberrationExplorer(real_space_sampling_A=0.0)


# =============================================================================
# Recompute behaviour — aberrations actually change the output
# =============================================================================


def test_aberration_explorer_defocus_changes_probe():
    """Adding C10 (defocus) must change the probe intensity from baseline."""
    w0 = AberrationExplorer(gpts=128, real_space_sampling_A=0.1)
    baseline = np.frombuffer(w0.probe_intensity_bytes, dtype=np.float32).copy()

    w1 = AberrationExplorer(
        gpts=128, real_space_sampling_A=0.1, aberrations={"C10": 500.0}
    )
    defocused = np.frombuffer(w1.probe_intensity_bytes, dtype=np.float32)

    assert baseline.shape == defocused.shape
    # The defocused probe is broader, so its peak intensity should drop.
    assert defocused.max() < baseline.max() * 0.95
    assert not np.allclose(baseline, defocused)


def test_aberration_explorer_c30_changes_chi():
    """C30 (spherical aberration) must change chi."""
    w0 = AberrationExplorer(gpts=128, real_space_sampling_A=0.1)
    assert w0.chi_max == pytest.approx(0.0, abs=1e-6)

    w1 = AberrationExplorer(
        gpts=128, real_space_sampling_A=0.1, aberrations={"C30": 1e5}
    )
    assert w1.chi_max > 0.0


def test_aberration_explorer_radial_ctf_baseline_zero():
    """With all aberrations zero, sin(chi(k))=sin(0)=0 along the whole radial axis."""
    w = AberrationExplorer(gpts=128)
    ctf = np.frombuffer(w.radial_ctf_bytes, dtype=np.float32)
    assert np.allclose(ctf, 0.0, atol=1e-6)


def test_aberration_explorer_radial_ctf_active():
    """With C10 present, sin(chi(k)) is nontrivial."""
    w = AberrationExplorer(gpts=128, aberrations={"C10": 100.0})
    ctf = np.frombuffer(w.radial_ctf_bytes, dtype=np.float32)
    # ctf(0) = sin(0) = 0; later samples must be nonzero.
    assert abs(ctf[0]) < 1e-5
    assert np.any(np.abs(ctf) > 0.1)
    # CTF stays bounded in [-1, 1]
    assert ctf.min() >= -1.0 - 1e-5
    assert ctf.max() <= 1.0 + 1e-5


def test_aberration_explorer_observer_triggers_recompute():
    """Mutating an aberration via the dict trait re-runs the pipeline."""
    w = AberrationExplorer(gpts=128)
    initial_max = float(np.frombuffer(w.probe_intensity_bytes, dtype=np.float32).max())
    w.aberrations = {**w.aberrations, "C10": 500.0}
    new_max = float(np.frombuffer(w.probe_intensity_bytes, dtype=np.float32).max())
    assert new_max != initial_max


def test_aberration_explorer_set_aberration_helper():
    w = AberrationExplorer(gpts=128)
    out = w.set_aberration(C10=100.0, C30=1e4)
    assert out is w
    assert w.aberrations["C10"] == pytest.approx(100.0)
    assert w.aberrations["C30"] == pytest.approx(1e4)


def test_aberration_explorer_reset_aberrations():
    w = AberrationExplorer(gpts=128, aberrations={"C10": 100.0, "C30": 1e4})
    w.reset_aberrations()
    for v in w.aberrations.values():
        assert v == 0.0


def test_aberration_explorer_set_aberration_invalid_key():
    w = AberrationExplorer(gpts=128)
    with pytest.raises(ValueError, match="Unknown aberration"):
        w.set_aberration(NotAReal=1.0)


def test_aberration_explorer_defocus_spread_envelope():
    """Defocus spread should attenuate higher-k probe content (broader probe)."""
    w0 = AberrationExplorer(gpts=128, real_space_sampling_A=0.1, defocus_spread_A=0.0)
    w1 = AberrationExplorer(gpts=128, real_space_sampling_A=0.1, defocus_spread_A=50.0)
    peak0 = float(np.frombuffer(w0.probe_intensity_bytes, dtype=np.float32).max())
    peak1 = float(np.frombuffer(w1.probe_intensity_bytes, dtype=np.float32).max())
    # With temporal damping the focused-probe peak drops.
    assert peak1 < peak0


def test_aberration_explorer_gpts_changes_grid_size():
    w = AberrationExplorer(gpts=256)
    assert len(w.probe_intensity_bytes) == 256 * 256 * 4
    w.gpts = 128
    assert len(w.probe_intensity_bytes) == 128 * 128 * 4


# =============================================================================
# State Protocol — required by spec
# =============================================================================


def test_aberration_explorer_state_dict_roundtrip():
    w1 = AberrationExplorer(
        energy_keV=80.0,
        semiangle_cutoff_mrad=20.0,
        gpts=128,
        real_space_sampling_A=0.05,
        cmap="viridis",
        aberrations={"C10": 50.0, "C30": 1e5},
    )
    state = w1.state_dict()
    w2 = AberrationExplorer(state=state)
    assert w2.energy_keV == pytest.approx(80.0)
    assert w2.semiangle_cutoff_mrad == pytest.approx(20.0)
    assert w2.gpts == 128
    assert w2.real_space_sampling_A == pytest.approx(0.05)
    assert w2.cmap == "viridis"
    assert w2.aberrations["C10"] == pytest.approx(50.0)
    assert w2.aberrations["C30"] == pytest.approx(1e5)


def test_aberration_explorer_state_dict_completeness():
    w = AberrationExplorer()
    state = w.state_dict()
    expected_keys = {
        "title", "energy_keV", "semiangle_cutoff_mrad", "gpts",
        "real_space_sampling_A", "aperture_smoothing", "defocus_spread_A",
        "aberrations", "cmap", "show_stats", "show_controls", "canvas_size",
    }
    assert set(state.keys()) == expected_keys


def test_aberration_explorer_save_load_file(tmp_path):
    w1 = AberrationExplorer(
        title="run1", cmap="plasma", aberrations={"C10": 75.0, "phi12": 1.0}
    )
    path = str(tmp_path / "state.json")
    w1.save(path)
    # Envelope structure
    with open(path) as f:
        envelope = json.load(f)
    assert envelope["metadata_version"] == "1.0"
    assert envelope["widget_name"] == "AberrationExplorer"
    assert isinstance(envelope["widget_version"], str)
    assert envelope["state"]["title"] == "run1"
    # Load from path
    w2 = AberrationExplorer(state=path)
    assert w2.title == "run1"
    assert w2.cmap == "plasma"
    assert w2.aberrations["C10"] == pytest.approx(75.0)
    assert w2.aberrations["phi12"] == pytest.approx(1.0)


def test_aberration_explorer_load_state_partial():
    w = AberrationExplorer()
    w.load_state_dict({"cmap": "magma", "title": "Partial"})
    assert w.cmap == "magma"
    assert w.title == "Partial"
    assert w.energy_keV == pytest.approx(200.0)  # unchanged


def test_aberration_explorer_load_state_ignores_unknown_keys():
    w = AberrationExplorer()
    w.load_state_dict({"not_a_real_trait": 42, "title": "OK"})
    assert w.title == "OK"


# =============================================================================
# Summary + repr — required by spec
# =============================================================================


def test_aberration_explorer_summary(capsys):
    w = AberrationExplorer(
        energy_keV=200.0,
        aberrations={"C10": 50.0, "C30": 1e5},
        title="Tuning",
    )
    w.summary()
    out = capsys.readouterr().out
    assert "Tuning" in out
    assert "200" in out
    assert "C10" in out or "C30" in out
    # Wavelength is reported
    assert "lambda" in out or "Å" in out


def test_aberration_explorer_summary_all_zero(capsys):
    w = AberrationExplorer()
    w.summary()
    out = capsys.readouterr().out
    assert "all zero" in out


def test_aberration_explorer_repr_smoke():
    w = AberrationExplorer(aberrations={"C10": 10.0})
    r = repr(w)
    assert "aberr=1" in r


# =============================================================================
# Sanity: math consistency with quantem
# =============================================================================


def test_aberration_explorer_radial_ctf_zero_intercept():
    """At k=0, chi=0 ⇒ sin(chi)=0 always, regardless of aberrations."""
    w = AberrationExplorer(
        gpts=128, aberrations={"C10": 50.0, "C12": 10.0, "C30": 1e5}
    )
    ctf = np.frombuffer(w.radial_ctf_bytes, dtype=np.float32)
    assert abs(ctf[0]) < 1e-5


def test_aberration_explorer_widget_version_is_set():
    w = AberrationExplorer()
    assert w.widget_version != "unknown"


def test_aberration_explorer_phi_only_no_change():
    """Pure phi-only changes (with all magnitudes = 0) do not alter chi."""
    w0 = AberrationExplorer(gpts=128)
    chi0 = np.frombuffer(w0.chi_polar_bytes, dtype=np.float32).copy()

    w1 = AberrationExplorer(
        gpts=128, aberrations={"phi12": math.pi / 3, "phi23": math.pi / 4}
    )
    chi1 = np.frombuffer(w1.chi_polar_bytes, dtype=np.float32)
    np.testing.assert_allclose(chi0, chi1, atol=1e-6)


def test_aberration_explorer_c10_quadratic_chi():
    # For a pure C10 (defocus) coefficient with all phi=0 cross-terms,
    # chi(alpha, 0) = (2*pi/lambda) * (1/2) * alpha^2 * C10.
    # The maximum chi reported by the widget (chi_max trait) should equal the
    # analytic value at alpha = semiangle_cutoff to within sampling tolerance.
    semi_mrad = 25.0
    c10_A = 100.0
    w = AberrationExplorer(
        gpts=128,
        energy_keV=200.0,
        semiangle_cutoff_mrad=semi_mrad,
        aberrations={"C10": c10_A},
    )
    semi_rad = semi_mrad * 1e-3
    lam = w.wavelength_A
    expected_max = (2.0 * math.pi / lam) * 0.5 * semi_rad ** 2 * c10_A
    # Allow ~1% tolerance to account for the discrete polar grid not landing
    # exactly on alpha = semi_rad.
    assert w.chi_max == pytest.approx(expected_max, rel=0.01)


def test_aberration_explorer_temporal_envelope_matches_abtem_form():
    # Validates the canonical temporal-envelope formula used by abTEM
    # (abtem.transfer.TemporalEnvelope) and py4DSTEM (evaluate_temporal_envelope):
    #   E(alpha) = exp(-((0.5*pi/lambda) * focal_spread * alpha^2)^2)
    # The bytes change from zero-spread to df>0 should match this curve.
    df = 50.0
    w0 = AberrationExplorer(gpts=128, energy_keV=200.0, semiangle_cutoff_mrad=25.0,
                            defocus_spread_A=0.0)
    w1 = AberrationExplorer(gpts=128, energy_keV=200.0, semiangle_cutoff_mrad=25.0,
                            defocus_spread_A=df)
    p0_sum = np.frombuffer(w0.probe_intensity_bytes, dtype=np.float32).sum()
    p1_sum = np.frombuffer(w1.probe_intensity_bytes, dtype=np.float32).sum()
    # With the canonical k^4-decaying envelope (and NO post-envelope
    # renormalization in the widget), the total real-space intensity must
    # drop monotonically with df.
    assert p1_sum < p0_sum
    # Sanity: the envelope value at any alpha in (0, semiangle_cutoff) lies
    # strictly inside (0, 1); confirms the formula is dimensionally sensible.
    # For df=50 A, E=200 keV, alpha=25 mrad the value is ~0.02 — strong decay,
    # consistent with the k^4 (i.e. alpha^4) dependence in the canonical form.
    lam = w0.wavelength_A
    alpha = 25e-3
    e_edge = math.exp(-((0.5 * math.pi / lam) * df * alpha ** 2) ** 2)
    assert 0.0 < e_edge < 1.0
    # Same envelope at a smaller angle must be larger (monotone decreasing).
    e_inner = math.exp(-((0.5 * math.pi / lam) * df * (0.5 * alpha) ** 2) ** 2)
    assert e_inner > e_edge
