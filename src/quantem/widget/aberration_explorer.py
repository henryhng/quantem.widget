"""
aberration_explorer: Interactive aberration / CTF / probe diagnostics widget.

Live STEM column tuning. Drag Krivanek polar coefficient sliders; three
panels recompute in real time:
- Real-space probe intensity |psi(r)|^2.
- Aberration phase wheel chi(k, phi) rendered as a polar image in k-space.
- 1D radial CTF sin(chi(k, phi=0)) along radial k from 0 to semiangle_cutoff.

Backend math is provided by ``quantem.diffractive_imaging.complex_probe``.
"""

import json
import math
import pathlib
from typing import Any, Dict, List, Optional

import anywidget
import numpy as np
import torch
import traitlets

from quantem.core.utils.utils import electron_wavelength_angstrom
from quantem.diffractive_imaging.complex_probe import (
    POLAR_SYMBOLS,
    aberration_surface,
    fourier_space_probe,
    polar_spatial_frequencies,
    standardize_aberration_coefs,
)

from quantem.widget.json_state import (
    resolve_widget_version,
    save_state_file,
    unwrap_state_payload,
)


# Krivanek polar coefficients exposed by the widget (1st, 2nd, 3rd order).
# Magnitudes are in angstrom, angles in radians.
_DEFAULT_ABERRATIONS: Dict[str, float] = {
    "C10": 0.0, "C12": 0.0, "phi12": 0.0,
    "C21": 0.0, "phi21": 0.0, "C23": 0.0, "phi23": 0.0,
    "C30": 0.0, "C32": 0.0, "phi32": 0.0, "C34": 0.0, "phi34": 0.0,
}

_VALID_GPTS = (128, 256, 512)
_RADIAL_NK = 256  # number of samples for the 1D radial CTF
_CHI_DISPLAY_SIZE = 256  # px-per-side of the rendered chi phase wheel


class AberrationExplorer(anywidget.AnyWidget):
    """Interactive aberration / CTF / probe diagnostics."""

    _esm = pathlib.Path(__file__).parent / "static" / "aberration_explorer.js"
    _css = pathlib.Path(__file__).parent / "static" / "aberration_explorer.css"

    # --- Header / display ----------------------------------------------------
    title = traitlets.Unicode("Aberration Explorer").tag(sync=True)
    cmap = traitlets.Unicode("inferno").tag(sync=True)

    # --- Microscope / sampling ----------------------------------------------
    energy_keV = traitlets.Float(200.0).tag(sync=True)
    semiangle_cutoff_mrad = traitlets.Float(25.0).tag(sync=True)
    gpts = traitlets.Int(256).tag(sync=True)
    real_space_sampling_A = traitlets.Float(0.1).tag(sync=True)
    aperture_smoothing = traitlets.Float(0.0).tag(sync=True)  # mrad; 0 -> hard aperture
    defocus_spread_A = traitlets.Float(0.0).tag(sync=True)  # temporal envelope sigma (Å)

    # --- Aberrations (Krivanek polar; canonical keys only) ------------------
    aberrations = traitlets.Dict(default_value=dict(_DEFAULT_ABERRATIONS)).tag(sync=True)

    # --- Computed data shipped to JS ----------------------------------------
    probe_intensity_bytes = traitlets.Bytes(b"").tag(sync=True)
    chi_polar_bytes = traitlets.Bytes(b"").tag(sync=True)
    radial_ctf_bytes = traitlets.Bytes(b"").tag(sync=True)
    radial_k_max_mrad = traitlets.Float(25.0).tag(sync=True)
    chi_min = traitlets.Float(0.0).tag(sync=True)
    chi_max = traitlets.Float(0.0).tag(sync=True)
    real_space_extent_A = traitlets.Float(0.0).tag(sync=True)  # full FOV in Å
    wavelength_A = traitlets.Float(0.0).tag(sync=True)

    # --- Stats (probe intensity) --------------------------------------------
    stats_mean = traitlets.Float(0.0).tag(sync=True)
    stats_min = traitlets.Float(0.0).tag(sync=True)
    stats_max = traitlets.Float(0.0).tag(sync=True)
    stats_std = traitlets.Float(0.0).tag(sync=True)

    # --- UI toggles ---------------------------------------------------------
    show_stats = traitlets.Bool(True).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    canvas_size = traitlets.Int(0).tag(sync=True)

    # ------------------------------------------------------------------
    # validators
    # ------------------------------------------------------------------
    @traitlets.validate("gpts")
    def _validate_gpts(self, proposal):
        value = int(proposal["value"])
        if value not in _VALID_GPTS:
            raise traitlets.TraitError(
                f"gpts must be one of {_VALID_GPTS}, got {value}."
            )
        return value

    @traitlets.validate("aberrations")
    def _validate_aberrations(self, proposal):
        value = proposal["value"]
        if not isinstance(value, dict):
            raise traitlets.TraitError("aberrations must be a dict.")
        out: Dict[str, float] = {}
        for key, val in value.items():
            if key not in POLAR_SYMBOLS:
                raise traitlets.TraitError(
                    f"Unknown aberration key {key!r}. "
                    f"Expected one of {POLAR_SYMBOLS}."
                )
            out[key] = float(val)
        # Ensure all default keys exist so JS can rely on a stable shape.
        for key, default in _DEFAULT_ABERRATIONS.items():
            out.setdefault(key, default)
        return out

    @traitlets.validate("energy_keV")
    def _validate_energy(self, proposal):
        value = float(proposal["value"])
        if value <= 0:
            raise traitlets.TraitError("energy_keV must be > 0.")
        return value

    @traitlets.validate("semiangle_cutoff_mrad")
    def _validate_semiangle(self, proposal):
        value = float(proposal["value"])
        if value <= 0:
            raise traitlets.TraitError("semiangle_cutoff_mrad must be > 0.")
        return value

    @traitlets.validate("real_space_sampling_A")
    def _validate_sampling(self, proposal):
        value = float(proposal["value"])
        if value <= 0:
            raise traitlets.TraitError("real_space_sampling_A must be > 0.")
        return value

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        title: str = "Aberration Explorer",
        energy_keV: float = 200.0,
        semiangle_cutoff_mrad: float = 25.0,
        gpts: int = 256,
        real_space_sampling_A: float = 0.1,
        aperture_smoothing: float = 0.0,
        defocus_spread_A: float = 0.0,
        aberrations: Optional[Dict[str, float]] = None,
        cmap: str = "inferno",
        show_stats: bool = True,
        show_controls: bool = True,
        canvas_size: int = 0,
        state: Any = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()

        # Assign user-facing traits first.
        self.title = title
        self.energy_keV = float(energy_keV)
        self.semiangle_cutoff_mrad = float(semiangle_cutoff_mrad)
        self.gpts = int(gpts)
        self.real_space_sampling_A = float(real_space_sampling_A)
        self.aperture_smoothing = float(aperture_smoothing)
        self.defocus_spread_A = float(defocus_spread_A)
        self.cmap = cmap
        self.show_stats = show_stats
        self.show_controls = show_controls
        self.canvas_size = int(canvas_size)

        merged = dict(_DEFAULT_ABERRATIONS)
        if aberrations:
            for key, val in aberrations.items():
                if key not in POLAR_SYMBOLS:
                    raise ValueError(
                        f"Unknown aberration key {key!r}. "
                        f"Expected one of {POLAR_SYMBOLS}."
                    )
                merged[key] = float(val)
        self.aberrations = merged

        # Initial compute.
        self._recompute()

        # Observe recompute-driving traits.
        self.observe(
            self._on_recompute_trait_change,
            names=[
                "energy_keV",
                "semiangle_cutoff_mrad",
                "gpts",
                "real_space_sampling_A",
                "aperture_smoothing",
                "defocus_spread_A",
                "aberrations",
            ],
        )

        # State restoration last (so observers fire on assignment).
        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                )
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)

    # ------------------------------------------------------------------
    # observers
    # ------------------------------------------------------------------
    def _on_recompute_trait_change(self, change=None):  # noqa: ARG002
        self._recompute()

    # ------------------------------------------------------------------
    # recompute pipeline
    # ------------------------------------------------------------------
    def _recompute(self) -> None:
        gpts = int(self.gpts)
        sampling = float(self.real_space_sampling_A)
        energy_eV = float(self.energy_keV) * 1e3
        semiangle = float(self.semiangle_cutoff_mrad)
        soft_edges = bool(self.aperture_smoothing > 0)

        wavelength = float(electron_wavelength_angstrom(energy_eV))
        self.wavelength_A = wavelength

        coefs_input = dict(self.aberrations)
        coefs = standardize_aberration_coefs(coefs_input)

        # ---- Fourier-space probe & real-space probe via IFFT ----
        fourier_probe = fourier_space_probe(
            gpts=(gpts, gpts),
            sampling=(sampling, sampling),
            energy=energy_eV,
            semiangle_cutoff=semiangle,
            aberration_coefs=coefs,
            soft_edges=soft_edges,
            normalized=True,
            device="cpu",
        )

        # Apply optional temporal (defocus-spread) envelope on the Fourier probe.
        # Canonical form (abtem.transfer.TemporalEnvelope, py4DSTEM evaluate_temporal_envelope):
        #   E(alpha) = exp(-((0.5 * pi / lambda) * df * alpha^2)^2)
        # with alpha = k * lambda the scattering angle in radians and df the
        # defocus-spread standard deviation in Angstroms.
        if self.defocus_spread_A > 0:
            k_grid, _ = polar_spatial_frequencies(
                (gpts, gpts), (sampling, sampling), device="cpu"
            )
            df = float(self.defocus_spread_A)
            alpha = k_grid * wavelength
            envelope = torch.exp(
                -((0.5 * math.pi / wavelength) * df * alpha ** 2) ** 2
            )
            fourier_probe = fourier_probe * envelope

        real_probe = torch.fft.ifft2(fourier_probe)
        real_probe = torch.fft.fftshift(real_probe)
        intensity = real_probe.abs().square().cpu().numpy().astype(np.float32)

        # ---- chi(alpha, phi) on a dedicated polar-display grid ----
        # Sample on a Cartesian grid whose extent is exactly [-cutoff, +cutoff]
        # so the visualization fills the canvas. The pixel-radius in this grid
        # is _CHI_DISPLAY_SIZE / 2, matching the aperture used by the JS
        # renderer.
        nd = _CHI_DISPLAY_SIZE
        semiangle_rad = semiangle * 1e-3
        coords = torch.linspace(-semiangle_rad, semiangle_rad, nd, dtype=torch.float32)
        alpha_x, alpha_y = torch.meshgrid(coords, coords, indexing="ij")
        alpha_disp = torch.sqrt(alpha_x.square() + alpha_y.square())
        phi_disp = torch.arctan2(alpha_y, alpha_x)
        chi_disp = aberration_surface(alpha_disp, phi_disp, wavelength, coefs)
        chi_arr = chi_disp.cpu().numpy().astype(np.float32)

        # Mask outside the aperture (alpha > cutoff lies in the canvas corners).
        aperture_mask = (alpha_disp.cpu().numpy() <= semiangle_rad)
        chi_display = np.where(aperture_mask, chi_arr, 0.0).astype(np.float32)

        # ---- 1D radial CTF along phi = 0 ----
        alpha_r = torch.linspace(
            0.0, semiangle * 1e-3, _RADIAL_NK, dtype=torch.float32
        )
        phi_r = torch.zeros_like(alpha_r)
        chi_radial = aberration_surface(alpha_r, phi_r, wavelength, coefs)
        ctf_radial = torch.sin(chi_radial).cpu().numpy().astype(np.float32)

        # ---- Stats on probe intensity ----
        self.stats_mean = float(intensity.mean())
        self.stats_min = float(intensity.min())
        self.stats_max = float(intensity.max())
        self.stats_std = float(intensity.std())

        # ---- Phase / k extents ----
        self.real_space_extent_A = float(gpts * sampling)
        self.radial_k_max_mrad = float(semiangle)
        if aperture_mask.any():
            chi_in_aperture = chi_arr[aperture_mask]
            self.chi_min = float(chi_in_aperture.min())
            self.chi_max = float(chi_in_aperture.max())
        else:
            self.chi_min = float(chi_arr.min())
            self.chi_max = float(chi_arr.max())

        # ---- Ship bytes ----
        self.probe_intensity_bytes = np.ascontiguousarray(intensity).tobytes()
        self.chi_polar_bytes = np.ascontiguousarray(chi_display).tobytes()
        self.radial_ctf_bytes = np.ascontiguousarray(ctf_radial).tobytes()

    # ------------------------------------------------------------------
    # public helpers
    # ------------------------------------------------------------------
    def set_aberration(self, **coefs: float) -> "AberrationExplorer":
        """Update one or more aberration coefficients and trigger a recompute."""
        merged = dict(self.aberrations)
        for key, val in coefs.items():
            if key not in POLAR_SYMBOLS:
                raise ValueError(
                    f"Unknown aberration key {key!r}. Expected one of {POLAR_SYMBOLS}."
                )
            merged[key] = float(val)
        self.aberrations = merged
        return self

    def reset_aberrations(self) -> "AberrationExplorer":
        """Reset every aberration coefficient to zero."""
        self.aberrations = dict(_DEFAULT_ABERRATIONS)
        return self

    # ------------------------------------------------------------------
    # State protocol
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "energy_keV": self.energy_keV,
            "semiangle_cutoff_mrad": self.semiangle_cutoff_mrad,
            "gpts": self.gpts,
            "real_space_sampling_A": self.real_space_sampling_A,
            "aperture_smoothing": self.aperture_smoothing,
            "defocus_spread_A": self.defocus_spread_A,
            "aberrations": dict(self.aberrations),
            "cmap": self.cmap,
            "show_stats": self.show_stats,
            "show_controls": self.show_controls,
            "canvas_size": self.canvas_size,
        }

    def save(self, path: str) -> None:
        """Save widget state to a JSON file."""
        save_state_file(path, "AberrationExplorer", self.state_dict())

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore widget state from a dict."""
        for key, val in state.items():
            if hasattr(self, key):
                setattr(self, key, val)

    # ------------------------------------------------------------------
    # Repr / summary
    # ------------------------------------------------------------------
    def summary(self) -> None:
        """Print a human-readable summary of the widget state."""
        name = self.title if self.title else "AberrationExplorer"
        lines = [name, "=" * 32]
        lines.append(
            f"Energy:     {self.energy_keV:.2f} keV "
            f"(lambda = {self.wavelength_A:.4f} Å)"
        )
        lines.append(
            f"Aperture:   {self.semiangle_cutoff_mrad:.2f} mrad "
            f"({'soft' if self.aperture_smoothing > 0 else 'hard'})"
        )
        lines.append(
            f"Grid:       {self.gpts}×{self.gpts} at {self.real_space_sampling_A:.3f} Å/px "
            f"(FOV {self.real_space_extent_A:.2f} Å)"
        )
        if self.defocus_spread_A > 0:
            lines.append(f"Δf spread:  {self.defocus_spread_A:.3f} Å")
        active = {k: v for k, v in self.aberrations.items() if v != 0}
        if active:
            joined = ", ".join(
                f"{k}={v:.4g}" for k, v in active.items()
            )
            lines.append(f"Aberrations: {joined}")
        else:
            lines.append("Aberrations: all zero")
        lines.append(
            f"Probe I:    min={self.stats_min:.4g}  max={self.stats_max:.4g}  "
            f"mean={self.stats_mean:.4g}"
        )
        lines.append(
            f"chi:        min={self.chi_min:.4g}  max={self.chi_max:.4g} rad"
        )
        print("\n".join(lines))

    def __repr__(self) -> str:
        name = self.title if self.title else "AberrationExplorer"
        parts: List[str] = [
            f"{name}({self.gpts}×{self.gpts}",
            f"E={self.energy_keV:.0f}keV",
            f"alpha={self.semiangle_cutoff_mrad:.1f}mrad",
        ]
        active = sum(1 for v in self.aberrations.values() if v != 0)
        parts.append(f"aberr={active}")
        return ", ".join(parts) + ")"
