#!/usr/bin/env python3
"""
beam_conditioning_framework.py

Scalar-diffraction framework for a two-lens beam-conditioning / spatial-filter
system.

Goals:
  * uncertain input beams: Gaussian, elliptical Gaussian, dirty Gaussian, top-hat
  * input-defining apertures, optional pinhole, lens phases, arbitrary apertures
  * angular-spectrum propagation of a complex scalar field
  * optimize fraction of light outside a 3 mm diameter target at z = 1 m
  * keep track of throughput, with soft/hard goals like >90% and >75%

Units:
  * mm for all lengths
  * wavelength_mm = wavelength_nm * 1e-6

Caveats:
  * Scalar diffraction with thin-lens phases; not full CODE V aberration modeling.
  * Doublet prescriptions are used to compute EFL, FFL, BFL, and principal planes.
  * Good for scans/optimization; use CODE V/Zemax for final aberration checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scipy.optimize import differential_evolution, minimize


# ---------------------------------------------------------------------------
# Cemented doublet first-order model
# ---------------------------------------------------------------------------

@dataclass
class CementedDoublet:
    """
    Cemented doublet represented by an ABCD matrix in reduced coordinates.

    Sign convention:
        z increases to the right.
        R > 0 if the centre of curvature is to the right of the surface.
        R < 0 if the centre of curvature is to the left of the surface.

    Geometry:
        V1 at z = 0, V2 at z = t12, V3 at z = t12 + t23.

    Optical system:
        air -> n1 -> n2 -> air
    """

    R1: float
    R2: float
    R3: float
    n1: float
    n2: float
    t12: float
    t23: float
    aperture_radius: float
    name: str = "doublet"

    @property
    def L(self) -> float:
        return self.t12 + self.t23

    def reversed(self) -> "CementedDoublet":
        """Return the same physical doublet flipped left-right."""
        return CementedDoublet(
            R1=-self.R3,
            R2=-self.R2,
            R3=-self.R1,
            n1=self.n2,
            n2=self.n1,
            t12=self.t23,
            t23=self.t12,
            aperture_radius=self.aperture_radius,
            name=f"{self.name} reversed",
        )

    @staticmethod
    def surface_power(n_left: float, n_right: float, R: float) -> float:
        if np.isinf(R):
            return 0.0
        return (n_right - n_left) / R

    @staticmethod
    def S(phi: float) -> np.ndarray:
        return np.array([[1.0, 0.0], [-phi, 1.0]])

    @staticmethod
    def T(t: float, n: float) -> np.ndarray:
        return np.array([[1.0, t / n], [0.0, 1.0]])

    def matrix(self) -> np.ndarray:
        phi1 = self.surface_power(1.0, self.n1, self.R1)
        phi2 = self.surface_power(self.n1, self.n2, self.R2)
        phi3 = self.surface_power(self.n2, 1.0, self.R3)

        return (
            self.S(phi3)
            @ self.T(self.t23, self.n2)
            @ self.S(phi2)
            @ self.T(self.t12, self.n1)
            @ self.S(phi1)
        )

    def properties(self) -> dict:
        M = self.matrix()
        A, B = M[0, 0], M[0, 1]
        C, D = M[1, 0], M[1, 1]

        if abs(C) < 1e-15:
            raise ValueError("C is approximately zero; lens has infinite focal length.")

        efl = -1.0 / C
        ffl = D / C          # signed, from V1 to front focal point
        bfl = -A / C         # signed, from V3 to rear focal point

        z_H1 = (D - 1.0) / C
        z_H2_from_V3 = (1.0 - A) / C
        z_H2 = self.L + z_H2_from_V3

        return {
            "A": A,
            "B": B,
            "C": C,
            "D": D,
            "M": M,
            "EFL": efl,
            "FFL_from_V1": ffl,
            "BFL_from_V3": bfl,
            "H1_from_V1": z_H1,
            "H2_from_V3": z_H2_from_V3,
            "H2_from_V1": z_H2,
            "front_focus_from_V1": ffl,
            "rear_focus_from_V1": self.L + bfl,
        }


def thorlabs_AC254_030_A() -> CementedDoublet:
    """Approximate Thorlabs AC254-030-A prescription at the d-line."""
    return CementedDoublet(
        R1=20.9, R2=-16.7, R3=-79.8,
        n1=1.67003, n2=1.80518,
        t12=12.0, t23=2.0,
        aperture_radius=12.7,
        name="AC254-030-A f=30 mm",
    )


def thorlabs_AC254_050_A() -> CementedDoublet:
    """
    Thorlabs AC254-050-A-ML prescription from the Zemax .zmx file.

    Zemax surfaces:
        R1 = 1 / 0.02999400119976  = 33.3400 mm
        R2 = 1 / -0.04488330341113 = -22.2800 mm
        R3 = 1 / -0.00343559968392 = -291.0700 mm
        t12 = 9.0 mm, glass N-BAF10
        t23 = 2.5 mm, glass SF10

    The SF10 index below is the approximate Schott d-line value. This gives
    EFL ~= 50.17 mm and BFL ~= 43.44 mm with the ABCD model, matching the
    Zemax/catalog scale much better than the earlier placeholder SF6HT value.
    """
    return CementedDoublet(
        R1=1.0 / 0.02999400119976,
        R2=1.0 / -0.044883303411131101,
        R3=1.0 / -0.0034355996839248002,
        n1=1.67003,   # N-BAF10, d-line
        n2=1.72825,   # SF10, d-line approximate
        t12=9.0,
        t23=2.5,
        aperture_radius=12.7,
        name="AC254-050-A-ML f=50 mm",
    )


# ---------------------------------------------------------------------------
# Complex-field grid and propagation
# ---------------------------------------------------------------------------

@dataclass
class FieldGrid:
    N: int = 1024
    size_mm: float = 12.0
    wavelength_mm: float = 405e-6  # 405 nm by default

    def __post_init__(self):
        self.dx = self.size_mm / self.N
        x = (np.arange(self.N) - self.N // 2) * self.dx
        self.x = x
        self.y = x.copy()
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="xy")
        self.R = np.sqrt(self.X**2 + self.Y**2)
        fx = np.fft.fftfreq(self.N, d=self.dx)  # cycles/mm
        self.FX, self.FY = np.meshgrid(fx, fx, indexing="xy")

    @property
    def k(self) -> float:
        return 2.0 * np.pi / self.wavelength_mm


def angular_spectrum_propagate(U: np.ndarray, grid: FieldGrid, dz_mm: float) -> np.ndarray:
    """Propagate scalar field U by dz_mm using angular-spectrum propagation."""
    if abs(dz_mm) < 1e-15:
        return U

    wavelength = grid.wavelength_mm
    k = 2.0 * np.pi / wavelength
    kx = 2.0 * np.pi * grid.FX
    ky = 2.0 * np.pi * grid.FY
    kz2 = k**2 - kx**2 - ky**2
    kz = np.sqrt(np.maximum(kz2, 0.0))
    H = np.exp(1j * dz_mm * kz)
    H[kz2 < 0] = 0.0
    return np.fft.ifft2(np.fft.fft2(U) * H)


def thin_lens_phase(U: np.ndarray, grid: FieldGrid, f_mm: float) -> np.ndarray:
    """Apply a thin-lens phase U -> U exp[-i k r^2/(2f)]."""
    phase = np.exp(-1j * grid.k * (grid.X**2 + grid.Y**2) / (2.0 * f_mm))
    return U * phase


def circular_aperture(U: np.ndarray, grid: FieldGrid, radius_mm: float) -> np.ndarray:
    return U * (grid.R <= radius_mm)


def power(U: np.ndarray) -> float:
    return float(np.sum(np.abs(U)**2))


def fraction_outside_radius(U: np.ndarray, grid: FieldGrid, radius_mm: float) -> float:
    I = np.abs(U)**2
    total = I.sum()
    if total <= 0:
        return np.nan
    return float(I[grid.R > radius_mm].sum() / total)


def rms_radius(U: np.ndarray, grid: FieldGrid) -> float:
    I = np.abs(U)**2
    total = I.sum()
    if total <= 0:
        return np.nan
    return float(np.sqrt(np.sum(grid.R**2 * I) / total))


def encircled_energy(U: np.ndarray, grid: FieldGrid, radii_mm: np.ndarray) -> np.ndarray:
    I = np.abs(U)**2
    total = I.sum()
    if total <= 0:
        return np.full_like(radii_mm, np.nan, dtype=float)
    return np.array([I[grid.R <= r].sum() / total for r in radii_mm])


# ---------------------------------------------------------------------------
# Input fields
# ---------------------------------------------------------------------------

def gaussian_field(
    grid: FieldGrid,
    waist_radius_mm: float = 0.35,
    x0_mm: float = 0.0,
    y0_mm: float = 0.0,
    tilt_x_mrad: float = 0.0,
    tilt_y_mrad: float = 0.0,
) -> np.ndarray:
    """Circular Gaussian amplitude. waist_radius_mm is the 1/e amplitude radius."""
    X = grid.X - x0_mm
    Y = grid.Y - y0_mm
    amp = np.exp(-(X**2 + Y**2) / waist_radius_mm**2)
    theta_x = tilt_x_mrad * 1e-3
    theta_y = tilt_y_mrad * 1e-3
    phase = np.exp(1j * grid.k * (theta_x * grid.X + theta_y * grid.Y))
    return amp * phase


def elliptical_gaussian_field(
    grid: FieldGrid,
    wx_mm: float = 0.15,
    wy_mm: float = 0.70,
    angle_deg: float = 0.0,
    tilt_x_mrad: float = 0.0,
    tilt_y_mrad: float = 0.0,
) -> np.ndarray:
    """Elliptical Gaussian input; useful for laser-diode-like tests."""
    a = np.deg2rad(angle_deg)
    Xr = grid.X * np.cos(a) + grid.Y * np.sin(a)
    Yr = -grid.X * np.sin(a) + grid.Y * np.cos(a)
    amp = np.exp(-(Xr**2 / wx_mm**2 + Yr**2 / wy_mm**2))
    theta_x = tilt_x_mrad * 1e-3
    theta_y = tilt_y_mrad * 1e-3
    phase = np.exp(1j * grid.k * (theta_x * grid.X + theta_y * grid.Y))
    return amp * phase


def top_hat_field(grid: FieldGrid, radius_mm: float = 0.5, tilt_x_mrad: float = 0.0) -> np.ndarray:
    amp = (grid.R <= radius_mm).astype(float)
    theta_x = tilt_x_mrad * 1e-3
    phase = np.exp(1j * grid.k * theta_x * grid.X)
    return amp * phase


def lowpass_noise(noise: np.ndarray, corr_px: int) -> np.ndarray:
    N = noise.shape[0]
    fy = np.fft.fftfreq(N)
    fx = np.fft.fftfreq(N)
    FX, FY = np.meshgrid(fx, fy, indexing="xy")
    F2 = FX**2 + FY**2
    sigma = 1.0 / max(corr_px, 1)
    filt = np.exp(-F2 / (2 * sigma**2))
    return np.real(np.fft.ifft2(np.fft.fft2(noise) * filt))


def dirty_gaussian_field(
    grid: FieldGrid,
    waist_radius_mm: float = 0.35,
    amplitude_noise: float = 0.10,
    phase_noise_rad: float = 0.25,
    corr_px: int = 16,
    seed: int = 1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = gaussian_field(grid, waist_radius_mm=waist_radius_mm)
    amp_noise = lowpass_noise(rng.normal(size=(grid.N, grid.N)), corr_px)
    ph_noise = lowpass_noise(rng.normal(size=(grid.N, grid.N)), corr_px)
    amp_noise /= max(np.std(amp_noise), 1e-12)
    ph_noise /= max(np.std(ph_noise), 1e-12)
    amp_factor = np.clip(1.0 + amplitude_noise * amp_noise, 0.0, None)
    phase_factor = np.exp(1j * phase_noise_rad * ph_noise)
    return base * amp_factor * phase_factor


def recommended_input_factories() -> dict[str, Callable[[FieldGrid], np.ndarray]]:
    """Suggested uncertain-input test cases."""
    return {
        "small_clean_gaussian": lambda g: gaussian_field(g, waist_radius_mm=0.20),
        "large_clean_gaussian": lambda g: gaussian_field(g, waist_radius_mm=0.50),
        "elliptical_diode_like": lambda g: elliptical_gaussian_field(g, wx_mm=0.10, wy_mm=0.60),
        "elliptical_tilted_0p5mrad": lambda g: elliptical_gaussian_field(
            g, wx_mm=0.15, wy_mm=0.50, tilt_x_mrad=0.5
        ),
        "top_hat_aperture_filling": lambda g: top_hat_field(g, radius_mm=0.50),
        "dirty_gaussian_mild": lambda g: dirty_gaussian_field(
            g, waist_radius_mm=0.35, amplitude_noise=0.08, phase_noise_rad=0.15, seed=1
        ),
        "dirty_gaussian_strong": lambda g: dirty_gaussian_field(
            g, waist_radius_mm=0.35, amplitude_noise=0.20, phase_noise_rad=0.50, seed=2
        ),
    }


# ---------------------------------------------------------------------------
# Optical elements and propagation system
# ---------------------------------------------------------------------------

@dataclass
class ApertureElement:
    z_mm: float
    radius_mm: float
    name: str = "aperture"


@dataclass
class LensElement:
    z_mm: float
    f_mm: float
    clear_radius_mm: Optional[float] = None
    name: str = "lens"


@dataclass
class PinholeElement:
    z_mm: float
    radius_mm: Optional[float] = None
    name: str = "pinhole"

    @property
    def enabled(self) -> bool:
        return self.radius_mm is not None and self.radius_mm > 0


OpticalElement = ApertureElement | LensElement | PinholeElement


@dataclass
class PlaneSnapshot:
    """A saved copy of the field at an optical plane for diagnostics."""
    name: str
    z_mm: float
    U: np.ndarray
    throughput: float


@dataclass
class PropagationResult:
    U: np.ndarray
    grid: FieldGrid
    z_final_mm: float
    input_power: float
    output_power: float
    fraction_outside_target: float
    throughput: float
    rms_radius_mm: float
    log: list[str] = field(default_factory=list)
    snapshots: list[PlaneSnapshot] = field(default_factory=list)


@dataclass
class BeamConditioningSystem:
    """Sequential angular-spectrum propagation through lenses and apertures."""

    grid: FieldGrid
    elements: list[OpticalElement]
    z_start_mm: float = 0.0
    z_target_mm: float = 1000.0
    target_radius_mm: float = 1.5

    def sorted_elements(self) -> list[OpticalElement]:
        return sorted(self.elements, key=lambda e: e.z_mm)

    def propagate(self, U0: np.ndarray, capture_planes: bool = False) -> PropagationResult:
        U = U0.astype(complex, copy=True)
        p0 = power(U)
        z_current = self.z_start_mm
        log: list[str] = []
        snapshots: list[PlaneSnapshot] = []

        if capture_planes:
            snapshots.append(PlaneSnapshot("start", z_current, U.copy(), 1.0))

        for elem in self.sorted_elements():
            if elem.z_mm < z_current - 1e-12:
                raise ValueError(f"Element {elem.name} is behind current z={z_current} mm.")

            U = angular_spectrum_propagate(U, self.grid, elem.z_mm - z_current)
            z_current = elem.z_mm

            if isinstance(elem, ApertureElement):
                before = power(U)
                U = circular_aperture(U, self.grid, elem.radius_mm)
                after = power(U)
                log.append(
                    f"z={z_current:.3f} mm aperture {elem.name}, "
                    f"r={elem.radius_mm:.4g} mm, transmission={after/max(before,1e-300):.6g}"
                )

            elif isinstance(elem, PinholeElement):
                if elem.enabled:
                    before = power(U)
                    U = circular_aperture(U, self.grid, elem.radius_mm)
                    after = power(U)
                    log.append(
                        f"z={z_current:.3f} mm pinhole {elem.name}, "
                        f"r={elem.radius_mm:.4g} mm, transmission={after/max(before,1e-300):.6g}"
                    )
                else:
                    log.append(f"z={z_current:.3f} mm pinhole disabled")

            elif isinstance(elem, LensElement):
                if elem.clear_radius_mm is not None:
                    before = power(U)
                    U = circular_aperture(U, self.grid, elem.clear_radius_mm)
                    after = power(U)
                    log.append(
                        f"z={z_current:.3f} mm clear aperture {elem.name}, "
                        f"r={elem.clear_radius_mm:.4g} mm, transmission={after/max(before,1e-300):.6g}"
                    )
                U = thin_lens_phase(U, self.grid, elem.f_mm)
                log.append(f"z={z_current:.3f} mm lens {elem.name}, f={elem.f_mm:.6g} mm")

            if capture_planes:
                snapshots.append(
                    PlaneSnapshot(
                        getattr(elem, "name", "element"),
                        z_current,
                        U.copy(),
                        power(U) / max(p0, 1e-300),
                    )
                )

        if self.z_target_mm < z_current - 1e-12:
            raise ValueError("z_target_mm is behind the final optical element.")

        U = angular_spectrum_propagate(U, self.grid, self.z_target_mm - z_current)
        p_out = power(U)
        if capture_planes:
            snapshots.append(PlaneSnapshot("target plane", self.z_target_mm, U.copy(), p_out / max(p0, 1e-300)))
        return PropagationResult(
            U=U,
            grid=self.grid,
            z_final_mm=self.z_target_mm,
            input_power=p0,
            output_power=p_out,
            fraction_outside_target=fraction_outside_radius(U, self.grid, self.target_radius_mm),
            throughput=p_out / max(p0, 1e-300),
            rms_radius_mm=rms_radius(U, self.grid),
            log=log,
            snapshots=snapshots,
        )


# ---------------------------------------------------------------------------
# Layout builders
# ---------------------------------------------------------------------------

@dataclass
class TwoLensLayout:
    lens1: CementedDoublet
    lens2: CementedDoublet
    lens1_V1_z_mm: float
    lens2_V1_z_mm: float
    pinhole_z_mm: float
    lens1_H2_z_mm: float
    lens2_H1_z_mm: float
    lens1_props: dict
    lens2_props: dict

    @property
    def lens1_V3_z_mm(self) -> float:
        return self.lens1_V1_z_mm + self.lens1.L

    @property
    def lens2_V3_z_mm(self) -> float:
        return self.lens2_V1_z_mm + self.lens2.L

    @property
    def lens_separation_mm(self) -> float:
        return self.lens2_V1_z_mm - self.lens1_V3_z_mm

    @property
    def magnification(self) -> float:
        return self.lens2_props["EFL"] / self.lens1_props["EFL"]


def build_two_lens_common_focus_layout(
    lens1: CementedDoublet,
    lens2: CementedDoublet,
    lens1_V1_z_mm: float = 40.0,
    pinhole_z_offset_mm: float = 0.0,
    lens2_z_offset_mm: float = 0.0,
) -> TwoLensLayout:
    """
    Place two lenses near common focus.

    pinhole position = L1 V3 + BFL1 + pinhole_z_offset.
    L2 is placed so its front focus coincides with the pinhole, plus lens2_z_offset.
    """
    p1 = lens1.properties()
    p2 = lens2.properties()
    z_lens1 = lens1_V1_z_mm
    z_pinhole = z_lens1 + lens1.L + p1["BFL_from_V3"] + pinhole_z_offset_mm
    z_lens2 = z_pinhole - p2["FFL_from_V1"] + lens2_z_offset_mm

    return TwoLensLayout(
        lens1=lens1,
        lens2=lens2,
        lens1_V1_z_mm=z_lens1,
        lens2_V1_z_mm=z_lens2,
        pinhole_z_mm=z_pinhole,
        lens1_H2_z_mm=z_lens1 + p1["H2_from_V1"],
        lens2_H1_z_mm=z_lens2 + p2["H1_from_V1"],
        lens1_props=p1,
        lens2_props=p2,
    )


def default_input_apertures() -> list[ApertureElement]:
    """
    Two input-defining apertures:
        r = 0.50 mm at z = 5 mm
        r = 0.51 mm at z = 25 mm
    """
    return [
        ApertureElement(z_mm=5.0, radius_mm=0.50, name="input aperture 1"),
        ApertureElement(z_mm=25.0, radius_mm=0.51, name="input aperture 2"),
    ]

def student_default_input_field(grid):
    """
    Default input beam for the student-facing studies.

    Start with a circular Gaussian because it is easier to interpret.
    After the baseline is understood, replace this with one of the commented
    alternatives below.

    Suggested alternatives:
        return elliptical_gaussian_field(grid, wx_mm=0.15, wy_mm=0.50, tilt_x_mrad=0.2)
        return top_hat_field(grid, radius_mm=0.50)
        return dirty_gaussian_field(grid, waist_radius_mm=0.35, amplitude_noise=0.10,
                                    phase_noise_rad=0.25, seed=1)
    """
    return gaussian_field(
        grid,
        waist_radius_mm=0.35,
        tilt_x_mrad=0.0,
        tilt_y_mrad=0.0,
    )


def print_result_summary(label, result):
    """
    Compact one-line summary for comparing cases.
    """
    print(
        f"{label:40s}  "
        f"T={result.throughput:8.5f}  "
        f"outside={result.fraction_outside_target:12.4e}  "
        f"inside={100.0*(1.0-result.fraction_outside_target):9.5f}%  "
        f"rms={result.rms_radius_mm:8.4f} mm"
    )


def save_snapshots_pdf_basic(
    snapshots,
    final_result,
    filename,
    target_radius_mm=1.5,
    log_floor=1e-14,
):
    """
    Save a simple multi-page PDF showing the beam at each saved plane.

    This is intentionally simple and robust. It does not need the more elaborate
    PDF helper from the previous version of the script.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    grid = final_result.grid

    with PdfPages(filename) as pdf:
        for name, item in snapshots.items():
            z_mm, U = item

            I = np.abs(U)**2
            if I.max() > 0:
                I = I / I.max()

            Z = np.log10(np.maximum(I, log_floor))

            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            im = ax.imshow(
                Z,
                extent=[grid.x.min(), grid.x.max(), grid.y.min(), grid.y.max()],
                origin="lower",
                interpolation="nearest",
            )
            fig.colorbar(im, ax=ax, label="log10 normalized intensity")

            circ = plt.Circle((0, 0), target_radius_mm, fill=False, linewidth=2)
            ax.add_patch(circ)

            ax.set_aspect("equal")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.set_title(f"{name}\nz = {z_mm:.3f} mm")

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # Add final target plane as last page.
        U = final_result.U
        I = np.abs(U)**2
        if I.max() > 0:
            I = I / I.max()
        Z = np.log10(np.maximum(I, log_floor))

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(
            Z,
            extent=[grid.x.min(), grid.x.max(), grid.y.min(), grid.y.max()],
            origin="lower",
            interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, label="log10 normalized intensity")

        circ = plt.Circle((0, 0), target_radius_mm, fill=False, linewidth=2)
        ax.add_patch(circ)

        ax.text(
            0.02,
            0.98,
            (
                f"Target plane\n"
                f"outside r={target_radius_mm:.2f} mm: "
                f"{final_result.fraction_outside_target:.3e}\n"
                f"inside: {100.0*(1.0-final_result.fraction_outside_target):.5f}%\n"
                f"throughput: {final_result.throughput:.5f}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round", alpha=0.2),
        )

        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(f"Final target plane, z = {final_result.z_final_mm:.1f} mm")

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved {filename}")

def make_regular_downstream_apertures(
    z_start_mm: float,
    z_stop_mm: float,
    spacing_mm: float,
    radius_mm: float,
    name_prefix: str = "downstream",
) -> list[ApertureElement]:
    zs = np.arange(z_start_mm, z_stop_mm + 0.5 * spacing_mm, spacing_mm)
    return [
        ApertureElement(z_mm=float(z), radius_mm=radius_mm, name=f"{name_prefix}_{i:03d}")
        for i, z in enumerate(zs)
    ]


def build_system_from_layout(
    grid: FieldGrid,
    layout: TwoLensLayout,
    input_apertures: Sequence[ApertureElement],
    pinhole_radius_mm: Optional[float] = None,
    downstream_apertures: Sequence[ApertureElement] = (),
    z_target_mm: float = 1000.0,
    target_radius_mm: float = 1.5,
) -> BeamConditioningSystem:
    """
    Build a system using equivalent thin-lens phases.

    Lens 1 phase is placed at its rear principal plane H2.
    Lens 2 phase is placed at its front principal plane H1.
    """
    elements: list[OpticalElement] = []
    elements.extend(input_apertures)
    elements.append(
        LensElement(
            z_mm=layout.lens1_H2_z_mm,
            f_mm=layout.lens1_props["EFL"],
            clear_radius_mm=layout.lens1.aperture_radius,
            name=f"L1 {layout.lens1.name}",
        )
    )
    elements.append(PinholeElement(z_mm=layout.pinhole_z_mm, radius_mm=pinhole_radius_mm))
    elements.append(
        LensElement(
            z_mm=layout.lens2_H1_z_mm,
            f_mm=layout.lens2_props["EFL"],
            clear_radius_mm=layout.lens2.aperture_radius,
            name=f"L2 {layout.lens2.name}",
        )
    )
    elements.extend(downstream_apertures)

    return BeamConditioningSystem(
        grid=grid,
        elements=elements,
        z_start_mm=0.0,
        z_target_mm=z_target_mm,
        target_radius_mm=target_radius_mm,
    )


# ---------------------------------------------------------------------------
# Scan/optimization helpers
# ---------------------------------------------------------------------------

@dataclass
class ScanRow:
    case_name: str
    fraction_outside: float
    throughput: float
    rms_radius_mm: float
    objective: float
    pinhole_radius_mm: Optional[float]
    pinhole_z_offset_mm: float
    lens2_z_offset_mm: float
    downstream_aperture_radius_mm: Optional[float]


def objective_value(
    result: PropagationResult,
    throughput_soft_goal: float = 0.90,
    throughput_hard_floor: float = 0.75,
    penalty_weight: float = 10.0,
) -> float:
    """
    Primary objective is fraction outside the 1.5 mm target radius.
    Add penalties below 90% throughput and large penalties below 75% throughput.
    """
    f = result.fraction_outside_target
    T = result.throughput
    penalty = 0.0
    if T < throughput_soft_goal:
        penalty += penalty_weight * (throughput_soft_goal - T) ** 2
    if T < throughput_hard_floor:
        penalty += 1e3 * (throughput_hard_floor - T) ** 2
    return f + penalty


def run_parameter_scan(
    grid: FieldGrid,
    input_field_factory: Callable[[FieldGrid], np.ndarray],
    lens1: CementedDoublet,
    lens2: CementedDoublet,
    input_apertures: Sequence[ApertureElement],
    pinhole_radii_mm: Sequence[Optional[float]],
    pinhole_z_offsets_mm: Sequence[float],
    lens2_z_offsets_mm: Sequence[float],
    downstream_aperture_radii_mm: Sequence[Optional[float]],
    downstream_start_after_lens2_mm: float = 10.0,
    downstream_spacing_mm: float = 10.0,
    z_target_mm: float = 1000.0,
    target_radius_mm: float = 1.5,
    lens1_V1_z_mm: float = 40.0,
    throughput_soft_goal: float = 0.90,
    throughput_hard_floor: float = 0.75,
    stop_if_fraction_below: Optional[float] = None,
    max_cases: Optional[int] = None,
) -> list[ScanRow]:
    rows: list[ScanRow] = []
    U0 = input_field_factory(grid)
    n_done = 0

    for pinhole_r, pinhole_dz, lens2_dz, downstream_r in itertools.product(
        pinhole_radii_mm,
        pinhole_z_offsets_mm,
        lens2_z_offsets_mm,
        downstream_aperture_radii_mm,
    ):
        layout = build_two_lens_common_focus_layout(
            lens1=lens1,
            lens2=lens2,
            lens1_V1_z_mm=lens1_V1_z_mm,
            pinhole_z_offset_mm=float(pinhole_dz),
            lens2_z_offset_mm=float(lens2_dz),
        )

        downstream: list[ApertureElement] = []
        if downstream_r is not None:
            downstream = make_regular_downstream_apertures(
                z_start_mm=layout.lens2_V3_z_mm + downstream_start_after_lens2_mm,
                z_stop_mm=z_target_mm - downstream_spacing_mm,
                spacing_mm=downstream_spacing_mm,
                radius_mm=float(downstream_r),
            )

        system = build_system_from_layout(
            grid=grid,
            layout=layout,
            input_apertures=input_apertures,
            pinhole_radius_mm=pinhole_r,
            downstream_apertures=downstream,
            z_target_mm=z_target_mm,
            target_radius_mm=target_radius_mm,
        )
        result = system.propagate(U0)
        obj = objective_value(result, throughput_soft_goal, throughput_hard_floor)

        case_name = (
            f"pin={pinhole_r if pinhole_r is not None else 'none'}_"
            f"pdz={float(pinhole_dz):+.3f}_"
            f"l2dz={float(lens2_dz):+.3f}_"
            f"down={downstream_r if downstream_r is not None else 'none'}"
        )
        rows.append(
            ScanRow(
                case_name=case_name,
                fraction_outside=result.fraction_outside_target,
                throughput=result.throughput,
                rms_radius_mm=result.rms_radius_mm,
                objective=obj,
                pinhole_radius_mm=pinhole_r,
                pinhole_z_offset_mm=float(pinhole_dz),
                lens2_z_offset_mm=float(lens2_dz),
                downstream_aperture_radius_mm=downstream_r,
            )
        )

        n_done += 1
        if n_done % 10 == 0:
            best = min(rows, key=lambda r: r.objective)
            print(
                f"done {n_done:5d}; best outside={best.fraction_outside:.3e}, "
                f"T={best.throughput:.3f}, objective={best.objective:.3e}, {best.case_name}"
            )

        if stop_if_fraction_below is not None and result.fraction_outside_target < stop_if_fraction_below:
            print(f"Stopping early: reached fraction_outside < {stop_if_fraction_below:g}")
            break
        if max_cases is not None and n_done >= max_cases:
            print(f"Stopping early: reached max_cases={max_cases}")
            break

    rows.sort(key=lambda r: (r.objective, r.fraction_outside, -r.throughput))
    return rows


def print_scan_table(rows: Sequence[ScanRow], n: int = 20) -> None:
    print("\nBest scan results")
    print("-----------------")
    print(
        f"{'rank':>4} {'outside':>12} {'T':>8} {'rms[mm]':>9} {'obj':>12} "
        f"{'pin_r[mm]':>10} {'pin_dz':>9} {'L2_dz':>9} {'down_r':>9}  case"
    )
    for i, r in enumerate(rows[:n], start=1):
        pin = "None" if r.pinhole_radius_mm is None else f"{r.pinhole_radius_mm:.4g}"
        down = "None" if r.downstream_aperture_radius_mm is None else f"{r.downstream_aperture_radius_mm:.4g}"
        print(
            f"{i:4d} {r.fraction_outside:12.3e} {r.throughput:8.4f} "
            f"{r.rms_radius_mm:9.4f} {r.objective:12.3e} {pin:>10} "
            f"{r.pinhole_z_offset_mm:9.3f} {r.lens2_z_offset_mm:9.3f} {down:>9}  {r.case_name}"
        )



def print_system_summary(
    grid: FieldGrid,
    lens1: CementedDoublet,
    lens2: CementedDoublet,
    layout: TwoLensLayout,
    input_apertures: Sequence[ApertureElement],
    pinhole_radius_mm: Optional[float],
    downstream_apertures: Sequence[ApertureElement],
    z_target_mm: float,
    target_radius_mm: float,
    input_description: str = "unspecified input field",
):
    """Print a more complete human-readable description of the modeled system."""
    p1 = layout.lens1_props
    p2 = layout.lens2_props
    print("\nOptical layout summary")
    print("----------------------")
    print(f"Grid: N={grid.N}, window={grid.size_mm:g} mm, dx={grid.dx:.5g} mm, wavelength={grid.wavelength_mm*1e6:.3f} nm")
    print(f"Input field: {input_description}")
    print(f"Target: z={z_target_mm:g} mm, radius={target_radius_mm:g} mm (diameter={2*target_radius_mm:g} mm)")
    print("\nInput-defining apertures:")
    for ap in input_apertures:
        print(f"  z={ap.z_mm:9.3f} mm  r={ap.radius_mm:8.4f} mm  {ap.name}")
    if len(input_apertures) >= 2:
        a0, a1 = input_apertures[0], input_apertures[1]
        dz = a1.z_mm - a0.z_mm
        if dz != 0:
            approx_angle = (a1.radius_mm - a0.radius_mm) / dz
            print(f"  same-side aperture slope scale: {(approx_angle*1e3):.4f} mrad")
            print(f"  centre-to-opposite-edge full acceptance scale: {((a1.radius_mm+a0.radius_mm)/dz*1e3):.3f} mrad")

    print("\nLens 1:")
    print(f"  {lens1.name}")
    print(f"  V1={layout.lens1_V1_z_mm:.4f} mm, V3={layout.lens1_V3_z_mm:.4f} mm, H2={layout.lens1_H2_z_mm:.4f} mm")
    print(f"  EFL={p1['EFL']:.4f} mm, BFL={p1['BFL_from_V3']:.4f} mm, FFL={p1['FFL_from_V1']:.4f} mm")

    print("\nPinhole/common focus:")
    print(f"  z={layout.pinhole_z_mm:.4f} mm")
    if pinhole_radius_mm is None:
        print("  pinhole disabled")
    else:
        print(f"  pinhole radius={pinhole_radius_mm:.6g} mm, diameter={2*pinhole_radius_mm:.6g} mm")

    print("\nLens 2:")
    print(f"  {lens2.name}")
    print(f"  V1={layout.lens2_V1_z_mm:.4f} mm, V3={layout.lens2_V3_z_mm:.4f} mm, H1={layout.lens2_H1_z_mm:.4f} mm")
    print(f"  EFL={p2['EFL']:.4f} mm, BFL={p2['BFL_from_V3']:.4f} mm, FFL={p2['FFL_from_V1']:.4f} mm")
    print(f"  lens separation V3_1 to V1_2={layout.lens_separation_mm:.4f} mm")
    print(f"  nominal beam magnification EFL2/EFL1={layout.magnification:.4f}")

    print("\nDownstream apertures:")
    if not downstream_apertures:
        print("  none")
    else:
        print(f"  {len(downstream_apertures)} apertures")
        for ap in downstream_apertures[:8]:
            print(f"  z={ap.z_mm:9.3f} mm  r={ap.radius_mm:8.4f} mm  {ap.name}")
        if len(downstream_apertures) > 8:
            print("  ...")
            ap = downstream_apertures[-1]
            print(f"  z={ap.z_mm:9.3f} mm  r={ap.radius_mm:8.4f} mm  {ap.name}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_intensity(
    U: np.ndarray,
    grid: FieldGrid,
    title: str = "Intensity",
    target_radius_mm: Optional[float] = 1.5,
    log10: bool = True,
    savepath: Optional[str] = None,
    annotate_metrics: bool = True,
    throughput: Optional[float] = None,
):
    """Plot a 2D intensity map and optionally annotate target leakage metrics."""
    Iraw = np.abs(U)**2
    I = Iraw.copy()
    if I.max() > 0:
        I = I / I.max()
    if log10:
        Z = np.log10(np.maximum(I, 1e-14))
        label = "log10 normalized intensity"
    else:
        Z = I
        label = "normalized intensity"

    fig, ax = plt.subplots(figsize=(6.4, 5.3))
    im = ax.imshow(
        Z,
        extent=[grid.x.min(), grid.x.max(), grid.y.min(), grid.y.max()],
        origin="lower",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label=label)

    metric_lines = []
    if target_radius_mm is not None:
        ax.add_patch(plt.Circle((0, 0), target_radius_mm, fill=False, linewidth=2))
        fout = fraction_outside_radius(U, grid, target_radius_mm)
        metric_lines.append(f"outside r={target_radius_mm:g} mm: {fout:.3e}")
        metric_lines.append(f"inside: {(1.0 - fout) * 100.0:.5f}%")
    if throughput is not None:
        metric_lines.append(f"throughput: {throughput * 100.0:.3f}%")

    if annotate_metrics and metric_lines:
        ax.text(
            0.02,
            0.98,
            "\n".join(metric_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
        )

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)
        print(f"Saved {savepath}")
    return fig, ax


def save_plane_snapshots_pdf(
    result: PropagationResult,
    filename: str = "beam_planes_debug.pdf",
    target_radius_mm: Optional[float] = 1.5,
    log10: bool = True,
    max_pages: Optional[int] = None,
):
    """
    Save one intensity-map page per captured optical plane.

    Use system.propagate(U0, capture_planes=True) to populate result.snapshots.
    For systems with many downstream apertures, max_pages can keep the PDF manageable.
    """
    if not result.snapshots:
        print("No snapshots were captured. Run propagate(..., capture_planes=True).")
        return

    snapshots = result.snapshots if max_pages is None else result.snapshots[:max_pages]
    with PdfPages(filename) as pdf:
        for snap in snapshots:
            fig, ax = plot_intensity(
                snap.U,
                result.grid,
                title=f"{snap.name} at z={snap.z_mm:.3f} mm",
                target_radius_mm=target_radius_mm,
                log10=log10,
                savepath=None,
                annotate_metrics=True,
                throughput=snap.throughput,
            )
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Saved {filename}")

def plot_encircled_energy(
    U: np.ndarray,
    grid: FieldGrid,
    max_radius_mm: float = 5.0,
    savepath: Optional[str] = None,
):
    radii = np.linspace(0, max_radius_mm, 300)
    ee = encircled_energy(U, grid, radii)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(radii, ee)
    ax.axvline(1.5, linestyle="--", label="1.5 mm target radius")
    ax.set_xlabel("radius [mm]")
    ax.set_ylabel("encircled energy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)
        print(f"Saved {savepath}")
    return fig, ax



def optimize_continuous_configuration(
    grid,
    input_field_factory,
    lens1,
    lens2,
    input_apertures,
    use_pinhole=False,
    use_downstream_apertures=False,
    z_target_mm=1000.0,
    target_radius_mm=1.5,
    lens1_V1_z_mm=40.0,
    throughput_soft_goal=0.90,
    throughput_hard_floor=0.75,
    maxiter=30,
    popsize=8,
):
    """
    Continuous optimizer for the beam-conditioning system.

    This is intended to replace very large brute-force grid scans.

    Cases:
        use_pinhole=False, use_downstream_apertures=False
            variables = [lens2_z_offset_mm]

        use_pinhole=True, use_downstream_apertures=False
            variables = [pinhole_radius_mm, pinhole_z_offset_mm, lens2_z_offset_mm]

        use_pinhole=True, use_downstream_apertures=True
            variables = [pinhole_radius_mm, pinhole_z_offset_mm,
                         lens2_z_offset_mm, downstream_radius_mm]
    """

    U0 = input_field_factory(grid)

    variable_names = []

    if use_pinhole:
        variable_names.extend([
            "pinhole_radius_mm",
            "pinhole_z_offset_mm",
        ])

    variable_names.append("lens2_z_offset_mm")

    if use_downstream_apertures:
        variable_names.append("downstream_radius_mm")

    bounds_map = {
        "pinhole_radius_mm": (0.005, 0.200),
        "pinhole_z_offset_mm": (-2.0, 2.0),
        "lens2_z_offset_mm": (-25.0, 25.0),
        "downstream_radius_mm": (1.5, 5.0),
    }

    bounds = [bounds_map[name] for name in variable_names]

    history = []

    def unpack(x):
        params = {
            "pinhole_radius_mm": None,
            "pinhole_z_offset_mm": 0.0,
            "lens2_z_offset_mm": 0.0,
            "downstream_radius_mm": None,
        }

        for name, value in zip(variable_names, x):
            params[name] = float(value)

        if not use_pinhole:
            params["pinhole_radius_mm"] = None
            params["pinhole_z_offset_mm"] = 0.0

        if not use_downstream_apertures:
            params["downstream_radius_mm"] = None

        return params

    def evaluate(x):
        params = unpack(x)

        layout = build_two_lens_common_focus_layout(
            lens1=lens1,
            lens2=lens2,
            lens1_V1_z_mm=lens1_V1_z_mm,
            pinhole_z_offset_mm=params["pinhole_z_offset_mm"],
            lens2_z_offset_mm=params["lens2_z_offset_mm"],
        )

        downstream_apertures = []
        if params["downstream_radius_mm"] is not None:
            downstream_apertures = make_regular_downstream_apertures(
                z_start_mm=layout.lens2_V3_z_mm + 10.0,
                z_stop_mm=z_target_mm - 10.0,
                spacing_mm=10.0,
                radius_mm=params["downstream_radius_mm"],
            )

        system = build_system_from_layout(
            grid=grid,
            layout=layout,
            input_apertures=input_apertures,
            pinhole_radius_mm=params["pinhole_radius_mm"],
            downstream_apertures=downstream_apertures,
            z_target_mm=z_target_mm,
            target_radius_mm=target_radius_mm,
        )

        result = system.propagate(U0)

        score = objective_value(
            result,
            throughput_soft_goal=throughput_soft_goal,
            throughput_hard_floor=throughput_hard_floor,
            penalty_weight=10.0,
        )

        history.append({
            "score": score,
            "outside": result.fraction_outside_target,
            "throughput": result.throughput,
            "rms_radius_mm": result.rms_radius_mm,
            **params,
        })

        return score

    print()
    print("Starting differential evolution optimization")
    print("--------------------------------------------")
    print("Variables:", variable_names)
    print("Bounds:", bounds)

    de_result = differential_evolution(
        evaluate,
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        polish=False,
        updating="immediate",
        workers=1,
        tol=1e-3,
        disp=True,
    )

    print()
    print("Starting local Nelder-Mead refinement")
    print("-------------------------------------")

    nm_result = minimize(
        evaluate,
        de_result.x,
        method="Nelder-Mead",
        options={
            "maxiter": 80,
            "xatol": 1e-3,
            "fatol": 1e-5,
            "disp": True,
        },
    )

    best_x = nm_result.x
    best_params = unpack(best_x)

    # Re-evaluate best case and return full result.
    layout = build_two_lens_common_focus_layout(
        lens1=lens1,
        lens2=lens2,
        lens1_V1_z_mm=lens1_V1_z_mm,
        pinhole_z_offset_mm=best_params["pinhole_z_offset_mm"],
        lens2_z_offset_mm=best_params["lens2_z_offset_mm"],
    )

    downstream_apertures = []
    if best_params["downstream_radius_mm"] is not None:
        downstream_apertures = make_regular_downstream_apertures(
            z_start_mm=layout.lens2_V3_z_mm + 10.0,
            z_stop_mm=z_target_mm - 10.0,
            spacing_mm=10.0,
            radius_mm=best_params["downstream_radius_mm"],
        )

    system = build_system_from_layout(
        grid=grid,
        layout=layout,
        input_apertures=input_apertures,
        pinhole_radius_mm=best_params["pinhole_radius_mm"],
        downstream_apertures=downstream_apertures,
        z_target_mm=z_target_mm,
        target_radius_mm=target_radius_mm,
    )

    result = system.propagate(U0)

    history_sorted = sorted(history, key=lambda h: h["score"])

    print()
    print("Best optimized result")
    print("---------------------")
    print(f"score = {objective_value(result):.6e}")
    print(f"fraction outside target = {result.fraction_outside_target:.6e}")
    print(f"throughput = {result.throughput:.6f}")
    print(f"RMS radius at target = {result.rms_radius_mm:.6f} mm")
    print("parameters:")
    for k, v in best_params.items():
        print(f"  {k} = {v}")

    return {
        "result": result,
        "layout": layout,
        "system": system,
        "best_params": best_params,
        "history": history_sorted,
        "de_result": de_result,
        "nm_result": nm_result,
    }
# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_single_propagation():
    """Run one case and save diagnostic plots."""
    grid = FieldGrid(N=1024, size_mm=12.0, wavelength_mm=520e-6)
    lens1 = thorlabs_AC254_030_A()
    lens2 = thorlabs_AC254_050_A()

    U0 = elliptical_gaussian_field(
        grid, wx_mm=0.15, wy_mm=0.50, angle_deg=0.0, tilt_x_mrad=0.2
    )

    layout = build_two_lens_common_focus_layout(
        lens1=lens1,
        lens2=lens2,
        lens1_V1_z_mm=40.0,
        pinhole_z_offset_mm=0.0,
        lens2_z_offset_mm=0.0,
    )

    system = build_system_from_layout(
        grid=grid,
        layout=layout,
        input_apertures=default_input_apertures(),
        pinhole_radius_mm=None,
        downstream_apertures=[],
        z_target_mm=1000.0,
        target_radius_mm=1.5,
    )

    input_description = "elliptical Gaussian: wx=0.15 mm, wy=0.50 mm, tilt_x=0.2 mrad"
    print_system_summary(
        grid=grid,
        lens1=lens1,
        lens2=lens2,
        layout=layout,
        input_apertures=default_input_apertures(),
        pinhole_radius_mm=None,
        downstream_apertures=[],
        z_target_mm=1000.0,
        target_radius_mm=1.5,
        input_description=input_description,
    )

    result = system.propagate(U0, capture_planes=True)
    print("\nSingle-case result")
    print("------------------")
    print(f"L1 = {lens1.name}")
    print(f"L2 = {lens2.name}")
    print(f"Magnification f2/f1 = {layout.magnification:.4f}")
    print(f"Lens separation V3_1 to V1_2 = {layout.lens_separation_mm:.4f} mm")
    print(f"Pinhole/common focus z = {layout.pinhole_z_mm:.4f} mm")
    print(f"Throughput = {result.throughput:.6f}")
    print(f"Fraction outside 1.5 mm radius at 1 m = {result.fraction_outside_target:.3e}")
    print(f"RMS radius at 1 m = {result.rms_radius_mm:.4f} mm")
    print("\nElement log:")
    for line in result.log:
        print("  " + line)

    plot_intensity(
        result.U,
        grid,
        title="Target plane intensity at z=1 m",
        target_radius_mm=1.5,
        throughput=result.throughput,
        savepath="target_plane_intensity.png",
    )
    plot_encircled_energy(result.U, grid, savepath="target_plane_encircled_energy.png")
    save_plane_snapshots_pdf(result, filename="beam_planes_debug.pdf", target_radius_mm=1.5)
    plt.show()

def demo_single_propagation():
    """
    STUDENT STARTING POINT.

    This runs one fixed optical configuration and saves:
        target_plane_intensity.png
        target_plane_encircled_energy.png
        beam_planes_debug.pdf

    Purpose:
        Understand what the model is doing plane-by-plane.

    This is not an optimizer.
    """

    grid = FieldGrid(
        N=1024,
        size_mm=18.0,
        wavelength_mm=405e-6,
    )

    lens1 = thorlabs_AC254_030_A()
    lens2 = thorlabs_AC254_050_A()

    U0 = student_default_input_field(grid)

    input_apertures = default_input_apertures()

    layout = build_two_lens_common_focus_layout(
        lens1=lens1,
        lens2=lens2,
        lens1_V1_z_mm=40.0,
        pinhole_z_offset_mm=0.0,
        lens2_z_offset_mm=0.0,
    )

    system = build_system_from_layout(
        grid=grid,
        layout=layout,
        input_apertures=input_apertures,
        pinhole_radius_mm=None,
        downstream_apertures=[],
        z_target_mm=1000.0,
        target_radius_mm=1.5,
    )

    result, snapshots = system.propagate(U0, keep_snapshots=True)

    print()
    print("Single fixed propagation")
    print("------------------------")
    print(f"Grid: N={grid.N}, size={grid.size_mm:.3f} mm, dx={grid.dx:.5f} mm")
    print(f"Wavelength = {grid.wavelength_mm*1e6:.1f} nm")
    print()
    print(f"L1 = {lens1.name}")
    print(f"    EFL = {layout.lens1_props['EFL']:.4f} mm")
    print(f"    H2 z = {layout.lens1_H2_z_mm:.4f} mm")
    print()
    print(f"L2 = {lens2.name}")
    print(f"    EFL = {layout.lens2_props['EFL']:.4f} mm")
    print(f"    H1 z = {layout.lens2_H1_z_mm:.4f} mm")
    print()
    print(f"Magnification f2/f1 = {layout.magnification:.4f}")
    print(f"Lens separation V3_1 to V1_2 = {layout.lens_separation_mm:.4f} mm")
    print(f"Common focus / optional pinhole z = {layout.pinhole_z_mm:.4f} mm")
    print()
    print("Input apertures:")
    for ap in input_apertures:
        print(f"    {ap.name}: z={ap.z_mm:.3f} mm, r={ap.radius_mm:.4f} mm")

    if len(input_apertures) >= 2:
        dz = input_apertures[1].z_mm - input_apertures[0].z_mm
        dr = input_apertures[1].radius_mm - input_apertures[0].radius_mm
        if dz != 0:
            print(
                f"Approx. aperture-defined angular scale: "
                f"dr/dz = {dr/dz:.3e} rad = {1e3*dr/dz:.3f} mrad"
            )

    print()
    print_result_summary("fixed layout", result)

    print()
    print("Element log:")
    for line in result.log:
        print("  " + line)

    plot_intensity(
        result.U,
        grid,
        title="Single fixed propagation: target plane",
        target_radius_mm=1.5,
        savepath="target_plane_intensity.png",
    )

    plot_encircled_energy(
        result.U,
        grid,
        savepath="target_plane_encircled_energy.png",
    )

    save_snapshots_pdf_basic(
        snapshots,
        result,
        filename="beam_planes_debug.pdf",
        target_radius_mm=1.5,
    )

    plt.show()

def demo_subsystem_comparison():
    """
    STUDENT DIAGNOSTIC STUDY.

    This answers:
        Which part of the optical system helps or hurts?

    It compares increasingly complicated subsystems:
        1. input beam only
        2. input apertures only
        3. input apertures + L1 only
        4. input apertures + L1 + L2
        5. input apertures + L1 + L2 + pinhole
        6. input apertures + L1 + L2 + one post-L2 aperture

    The lenses are held fixed in the expected common-focus geometry.
    """

    grid = FieldGrid(
        N=1024,
        size_mm=18.0,
        wavelength_mm=405e-6,
    )

    lens1 = thorlabs_AC254_030_A()
    lens2 = thorlabs_AC254_050_A()

    U0 = student_default_input_field(grid)
    input_apertures = default_input_apertures()

    layout = build_two_lens_common_focus_layout(
        lens1=lens1,
        lens2=lens2,
        lens1_V1_z_mm=40.0,
        pinhole_z_offset_mm=0.0,
        lens2_z_offset_mm=0.0,
    )

    cases = []

    # Case 1: input only.
    cases.append((
        "input beam only",
        BeamConditioningSystem(
            grid=grid,
            elements=[],
            z_start_mm=0.0,
            z_target_mm=1000.0,
            target_radius_mm=1.5,
        ),
    ))

    # Case 2: input apertures only.
    cases.append((
        "input apertures only",
        BeamConditioningSystem(
            grid=grid,
            elements=list(input_apertures),
            z_start_mm=0.0,
            z_target_mm=1000.0,
            target_radius_mm=1.5,
        ),
    ))

    # Case 3: input apertures + L1 only.
    elements_L1_only = list(input_apertures)
    elements_L1_only.append(
        LensElement(
            z_H_mm=layout.lens1_H2_z_mm,
            f_mm=layout.lens1_props["EFL"],
            clear_radius_mm=lens1.aperture_radius,
            name=f"L1 {lens1.name}",
        )
    )

    cases.append((
        "input apertures + L1 only",
        BeamConditioningSystem(
            grid=grid,
            elements=elements_L1_only,
            z_start_mm=0.0,
            z_target_mm=1000.0,
            target_radius_mm=1.5,
        ),
    ))

    # Case 4: input apertures + L1 + L2, no pinhole.
    cases.append((
        "input apertures + L1 + L2",
        build_system_from_layout(
            grid=grid,
            layout=layout,
            input_apertures=input_apertures,
            pinhole_radius_mm=None,
            downstream_apertures=[],
            z_target_mm=1000.0,
            target_radius_mm=1.5,
        ),
    ))

    # Case 5: same, with a moderate pinhole.
    # This is a hypothesis test, not the default recommendation.
    cases.append((
        "L1 + L2 + 50 um radius pinhole",
        build_system_from_layout(
            grid=grid,
            layout=layout,
            input_apertures=input_apertures,
            pinhole_radius_mm=0.050,
            downstream_apertures=[],
            z_target_mm=1000.0,
            target_radius_mm=1.5,
        ),
    ))

    # Case 6: same, with one simple cleanup aperture after L2.
    post_l2_aperture = ApertureElement(
        z_mm=layout.lens2_V3_z_mm + 10.0,
        radius_mm=2.0,
        name="single post-L2 aperture",
    )

    cases.append((
        "L1 + L2 + post-L2 aperture r=2 mm",
        build_system_from_layout(
            grid=grid,
            layout=layout,
            input_apertures=input_apertures,
            pinhole_radius_mm=None,
            downstream_apertures=[post_l2_aperture],
            z_target_mm=1000.0,
            target_radius_mm=1.5,
        ),
    ))

    print()
    print("Subsystem comparison")
    print("--------------------")
    print(f"Grid: N={grid.N}, size={grid.size_mm:.3f} mm, dx={grid.dx:.5f} mm")
    print(f"L1: {lens1.name}, EFL={layout.lens1_props['EFL']:.4f} mm")
    print(f"L2: {lens2.name}, EFL={layout.lens2_props['EFL']:.4f} mm")
    print(f"Fixed lens separation = {layout.lens_separation_mm:.4f} mm")
    print(f"Target: r=1.5 mm at z=1000 mm")
    print()

    results = []

    for label, system in cases:
        result = system.propagate(U0)
        results.append((label, result))
        print_result_summary(label, result)

    # Save target-plane plots for the best and for the full lens system.
    best_label, best_result = min(results, key=lambda x: x[1].fraction_outside_target)

    print()
    print(f"Best subsystem by target containment: {best_label}")

    plot_intensity(
        best_result.U,
        grid,
        title=f"Best subsystem: {best_label}",
        target_radius_mm=1.5,
        savepath="subsystem_best_target_plane.png",
    )

    plot_encircled_energy(
        best_result.U,
        grid,
        savepath="subsystem_best_encircled_energy.png",
    )

    # Save a text summary.
    with open("subsystem_comparison_summary.txt", "w", encoding="utf-8") as f:
        f.write("Subsystem comparison\n")
        f.write("--------------------\n")
        for label, result in results:
            f.write(
                f"{label:40s}  "
                f"T={result.throughput:8.5f}  "
                f"outside={result.fraction_outside_target:12.4e}  "
                f"inside={100.0*(1.0-result.fraction_outside_target):9.5f}%  "
                f"rms={result.rms_radius_mm:8.4f} mm\n"
            )

    print("Saved subsystem_comparison_summary.txt")
    plt.show()

def demo_fixed_aperture_scan():
    """
    STUDENT APERTURE-ONLY SCAN.

    This keeps the lens geometry fixed and scans only aperture radii.

    Purpose:
        Find whether aperture-only cleanup improves the fraction of light inside
        the 3 mm diameter target, while keeping throughput reasonably high.

    This is deliberately small and fast. It is not a global optimizer.
    """

    grid = FieldGrid(
        N=1024,
        size_mm=18.0,
        wavelength_mm=405e-6,
    )

    lens1 = thorlabs_AC254_030_A()
    lens2 = thorlabs_AC254_050_A()

    U0 = student_default_input_field(grid)

    layout = build_two_lens_common_focus_layout(
        lens1=lens1,
        lens2=lens2,
        lens1_V1_z_mm=40.0,
        pinhole_z_offset_mm=0.0,
        lens2_z_offset_mm=0.0,
    )

    # Keep these paired so the second aperture is slightly larger than the first.
    input_aperture_pairs = [
        (0.30, 0.31),
        (0.40, 0.41),
        (0.50, 0.51),
        (0.60, 0.61),
        (0.75, 0.76),
    ]

    # One optional aperture just after lens 2.
    # None means no post-L2 aperture.
    post_l2_aperture_radii = [
        None,
        1.5,
        2.0,
        2.5,
        3.0,
    ]

    rows = []

    print()
    print("Fixed-layout aperture scan")
    print("--------------------------")
    print("Lens positions are fixed.")
    print("Only aperture radii are scanned.")
    print()
    print(f"L1: {lens1.name}, EFL={layout.lens1_props['EFL']:.4f} mm")
    print(f"L2: {lens2.name}, EFL={layout.lens2_props['EFL']:.4f} mm")
    print(f"Fixed lens separation = {layout.lens_separation_mm:.4f} mm")
    print()

    for r1, r2 in input_aperture_pairs:
        input_apertures = [
            ApertureElement(z_mm=5.0, radius_mm=r1, name="input aperture 1"),
            ApertureElement(z_mm=25.0, radius_mm=r2, name="input aperture 2"),
        ]

        for post_r in post_l2_aperture_radii:
            downstream_apertures = []

            if post_r is not None:
                downstream_apertures.append(
                    ApertureElement(
                        z_mm=layout.lens2_V3_z_mm + 10.0,
                        radius_mm=post_r,
                        name="post-L2 aperture",
                    )
                )

            system = build_system_from_layout(
                grid=grid,
                layout=layout,
                input_apertures=input_apertures,
                pinhole_radius_mm=None,
                downstream_apertures=downstream_apertures,
                z_target_mm=1000.0,
                target_radius_mm=1.5,
            )

            result = system.propagate(U0)

            rows.append({
                "r1": r1,
                "r2": r2,
                "post_r": post_r,
                "throughput": result.throughput,
                "outside": result.fraction_outside_target,
                "inside_percent": 100.0 * (1.0 - result.fraction_outside_target),
                "rms": result.rms_radius_mm,
                "result": result,
                "system": system,
                "input_apertures": input_apertures,
                "downstream_apertures": downstream_apertures,
            })

    # Sort primarily by outside fraction, but prefer higher throughput for ties.
    rows.sort(key=lambda r: (r["outside"], -r["throughput"]))

    print(
        f"{'rank':>4} {'r1':>7} {'r2':>7} {'post_r':>8} "
        f"{'T':>9} {'outside':>12} {'inside[%]':>11} {'rms[mm]':>9}"
    )

    for i, row in enumerate(rows, start=1):
        post_label = "None" if row["post_r"] is None else f"{row['post_r']:.3f}"
        print(
            f"{i:4d} "
            f"{row['r1']:7.3f} "
            f"{row['r2']:7.3f} "
            f"{post_label:>8} "
            f"{row['throughput']:9.5f} "
            f"{row['outside']:12.4e} "
            f"{row['inside_percent']:11.5f} "
            f"{row['rms']:9.4f}"
        )

    # Also save all rows to CSV for plotting later.
    with open("fixed_aperture_scan_summary.csv", "w", encoding="utf-8") as f:
        f.write("rank,r1_mm,r2_mm,post_l2_radius_mm,throughput,outside,inside_percent,rms_radius_mm\n")
        for i, row in enumerate(rows, start=1):
            post_value = "" if row["post_r"] is None else row["post_r"]
            f.write(
                f"{i},{row['r1']},{row['r2']},{post_value},"
                f"{row['throughput']},{row['outside']},"
                f"{row['inside_percent']},{row['rms']}\n"
            )

    print()
    print("Saved fixed_aperture_scan_summary.csv")

    best = rows[0]
    print()
    print("Best fixed-aperture case:")
    print(
        f"r1={best['r1']:.3f} mm, "
        f"r2={best['r2']:.3f} mm, "
        f"post-L2 aperture={best['post_r']}, "
        f"T={best['throughput']:.5f}, "
        f"outside={best['outside']:.4e}"
    )

    plot_intensity(
        best["result"].U,
        grid,
        title="Best fixed-aperture scan target plane",
        target_radius_mm=1.5,
        savepath="fixed_aperture_best_target_plane.png",
    )

    plot_encircled_energy(
        best["result"].U,
        grid,
        savepath="fixed_aperture_best_encircled_energy.png",
    )

    # Re-run best with snapshots so the student can inspect beam planes.
    best_system = build_system_from_layout(
        grid=grid,
        layout=layout,
        input_apertures=best["input_apertures"],
        pinhole_radius_mm=None,
        downstream_apertures=best["downstream_apertures"],
        z_target_mm=1000.0,
        target_radius_mm=1.5,
    )

    best_result, snapshots = best_system.propagate(U0, keep_snapshots=True)

    save_snapshots_pdf_basic(
        snapshots,
        best_result,
        filename="fixed_aperture_best_beam_planes.pdf",
        target_radius_mm=1.5,
    )

    plt.show()


def demo_scan():
    """
    Small scan intended to run reasonably quickly.

    Start with N=512. Increase to N=1024/2048 after finding promising regions.
    """
    grid = FieldGrid(N=1024, size_mm=12.0, wavelength_mm=520e-6)
    lens1 = thorlabs_AC254_030_A()
    lens2 = thorlabs_AC254_050_A()

    def input_factory(g: FieldGrid):
        return elliptical_gaussian_field(g, wx_mm=0.15, wy_mm=0.50, tilt_x_mrad=0.2)

    rows = run_parameter_scan(
        grid=grid,
        input_field_factory=input_factory,
        lens1=lens1,
        lens2=lens2,
        input_apertures=default_input_apertures(),
        pinhole_radii_mm=[None],
        pinhole_z_offsets_mm=np.linspace(-0.5, 0.5, 5),
        lens2_z_offsets_mm=np.linspace(-20.0, 20.0, 11),
        downstream_aperture_radii_mm=[2.0],
        downstream_start_after_lens2_mm=10.0,
        downstream_spacing_mm=10.0,
        z_target_mm=1000.0,
        target_radius_mm=1.5,
        lens1_V1_z_mm=40.0,
        throughput_soft_goal=0.90,
        throughput_hard_floor=0.75,
        stop_if_fraction_below=None,
    )
    print_scan_table(rows, n=25)

    # Re-run best case and save plots.
    best = rows[0]
    layout = build_two_lens_common_focus_layout(
        lens1=lens1,
        lens2=lens2,
        lens1_V1_z_mm=40.0,
        pinhole_z_offset_mm=best.pinhole_z_offset_mm,
        lens2_z_offset_mm=best.lens2_z_offset_mm,
    )

    if best.downstream_aperture_radius_mm is not None:
        downstream = make_regular_downstream_apertures(
            z_start_mm=layout.lens2_V3_z_mm + 10.0,
            z_stop_mm=1000.0 - 10.0,
            spacing_mm=10.0,
            radius_mm=best.downstream_aperture_radius_mm,
        )
    else:
        downstream = []

    system = build_system_from_layout(
        grid=grid,
        layout=layout,
        input_apertures=default_input_apertures(),
        pinhole_radius_mm=best.pinhole_radius_mm,
        downstream_apertures=downstream,
        z_target_mm=1000.0,
        target_radius_mm=1.5,
    )
    U0 = input_factory(grid)
    result = system.propagate(U0, capture_planes=True)
    plot_intensity(
        result.U,
        grid,
        title=f"Best scan target plane: {best.case_name}",
        target_radius_mm=1.5,
        throughput=result.throughput,
        savepath="best_scan_target_plane.png",
    )
    save_plane_snapshots_pdf(result, filename="best_scan_beam_planes_debug.pdf", target_radius_mm=1.5, max_pages=40)
    # Original line replaced below:
    plot_encircled_energy(result.U, grid, savepath="best_scan_encircled_energy.png")
    plt.show()


def demo_scan_all_input_cases():
    """
    Repeat a smaller scan for several input beam assumptions.
    This is the recommended way to check sensitivity to the unknown input beam.
    """
    grid = FieldGrid(N=512, size_mm=12.0, wavelength_mm=405e-6)
    lens1 = thorlabs_AC254_030_A()
    lens2 = thorlabs_AC254_050_A()

    for name, factory in recommended_input_factories().items():
        print(f"\n\n=== Input case: {name} ===")
        rows = run_parameter_scan(
            grid=grid,
            input_field_factory=factory,
            lens1=lens1,
            lens2=lens2,
            input_apertures=default_input_apertures(),
            pinhole_radii_mm=[None, 0.025, 0.050, 0.100],
            pinhole_z_offsets_mm=[-0.5, 0.0, 0.5],
            lens2_z_offsets_mm=np.linspace(-3.0, 3.0, 13),
            downstream_aperture_radii_mm=[None, 1.6, 2.0, 2.5],
            throughput_soft_goal=0.90,
            throughput_hard_floor=0.75,
        )
        print_scan_table(rows, n=5)

def demo_optimize():
    """
    Faster optimizer demo.

    This is usually much faster than a large grid scan.
    Start with N=512, then validate the best results with N=1024 or 2048.
    """

    grid = FieldGrid(N=512, size_mm=12.0, wavelength_mm=405e-6)

    lens1 = thorlabs_AC254_030_A()
    lens2 = thorlabs_AC254_050_A()

    def input_factory(g):
        return elliptical_gaussian_field(
            g,
            wx_mm=0.15,
            wy_mm=0.50,
            angle_deg=0.0,
            tilt_x_mrad=0.2,
            tilt_y_mrad=0.0,
        )

    opt = optimize_continuous_configuration(
        grid=grid,
        input_field_factory=input_factory,
        lens1=lens1,
        lens2=lens2,
        input_apertures=default_input_apertures(),

        # Try these combinations:
        use_pinhole=False,
        use_downstream_apertures=False,

        # Then:
        # use_pinhole=True,
        # use_downstream_apertures=False,

        # Then:
        # use_pinhole=True,
        # use_downstream_apertures=True,

        maxiter=25,
        popsize=6,
    )

    result = opt["result"]

    plot_intensity(
        result.U,
        grid,
        title="Optimized target plane intensity",
        savepath="optimized_target_plane.png",
    )

    plot_encircled_energy(
        result.U,
        grid,
        savepath="optimized_encircled_energy.png",
    )

    plt.show()

if __name__ == "__main__":

    # Start here.
    demo_single_propagation()

    # Then uncomment this.
    # demo_subsystem_comparison()

    # Then uncomment this only after understanding the subsystem comparison.
    # demo_fixed_aperture_scan()

    # Advanced only.
    # demo_optimize()