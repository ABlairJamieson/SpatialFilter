#!/usr/bin/env python3
"""
two_lens_spatial_filter_general.py

Two-cemented-doublet spatial filter / beam expander / beam reducer.

The code allows L1 and L2 to be different lenses.

Layout:
    collimated input beam
        -> lens 1
        -> focus / pinhole
        -> lens 2
        -> collimated output beam

The pinhole is placed at the rear focal point of lens 1.
Lens 2 is placed so that its front focal point coincides with the pinhole.

For two lenses in air, the paraxial beam magnification is approximately

    M = EFL_2 / EFL_1.

If M > 1, this is a beam expander.
If M < 1, this is a beam reducer.
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class CementedDoublet:
    R1: float
    R2: float
    R3: float
    n1: float
    n2: float
    t12: float
    t23: float
    aperture_radius: float = 12.7
    name: str = "doublet"

    @property
    def L(self):
        return self.t12 + self.t23

    @property
    def vertices(self):
        return {
            "V1": 0.0,
            "V2": self.t12,
            "V3": self.t12 + self.t23,
        }

    def reversed(self):
        """
        Return the same physical doublet flipped left-right.

        Original:
            air | R1 | n1, t12 | R2 | n2, t23 | R3 | air

        Reversed:
            air | -R3 | n2, t23 | -R2 | n1, t12 | -R1 | air
        """
        return CementedDoublet(
            R1=-self.R3,
            R2=-self.R2,
            R3=-self.R1,
            n1=self.n2,
            n2=self.n1,
            t12=self.t23,
            t23=self.t12,
            aperture_radius=self.aperture_radius,
            name=self.name + " reversed",
        )

    def surface_power(self, n_left, n_right, R):
        if np.isinf(R):
            return 0.0
        return (n_right - n_left) / R

    def S(self, phi):
        return np.array([
            [1.0, 0.0],
            [-phi, 1.0],
        ])

    def T(self, t, n):
        return np.array([
            [1.0, t / n],
            [0.0, 1.0],
        ])

    def matrix(self):
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

    def properties(self):
        M = self.matrix()
        A, B = M[0, 0], M[0, 1]
        C, D = M[1, 0], M[1, 1]

        if abs(C) < 1e-15:
            raise ValueError("C is approximately zero; infinite focal length.")

        efl = -1.0 / C
        ffl = D / C
        bfl = -A / C

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

    def spherical_surface_points(self, z_vertex, R, y_max, npts=300):
        y = np.linspace(-y_max, y_max, npts)

        if np.isinf(R):
            z = np.full_like(y, z_vertex)
            return z, y

        if abs(R) < y_max:
            raise ValueError(
                f"Drawing aperture radius {y_max} is larger than |R|={abs(R)}. "
                "Reduce aperture_radius."
            )

        zc = z_vertex + R
        root = np.sqrt(R**2 - y**2)

        if R > 0:
            z = zc - root
        else:
            z = zc + root

        return z, y

    def draw_at(self, ax, z0=0.0, label_prefix="", alpha=0.15):
        V = self.vertices
        props = self.properties()
        y_max = self.aperture_radius

        curves = []
        for label, z_vertex, R in [
            ("R1", V["V1"], self.R1),
            ("R2", V["V2"], self.R2),
            ("R3", V["V3"], self.R3),
        ]:
            z, y = self.spherical_surface_points(z0 + z_vertex, R, y_max)
            curves.append((z, y))
            ax.plot(z, y, linewidth=2)
            ax.text(
                z0 + z_vertex,
                1.05 * y_max,
                f"{label_prefix}{label}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        z1, y = curves[0]
        z2, _ = curves[1]
        z3, _ = curves[2]

        ax.fill(
            np.concatenate([z1, z2[::-1]]),
            np.concatenate([y, y[::-1]]),
            alpha=alpha,
        )
        ax.fill(
            np.concatenate([z2, z3[::-1]]),
            np.concatenate([y, y[::-1]]),
            alpha=alpha,
        )

        for name, z_local in V.items():
            z_abs = z0 + z_local
            ax.axvline(z_abs, linestyle=":", linewidth=0.8)
            ax.text(
                z_abs,
                -1.15 * y_max,
                f"{label_prefix}{name}",
                ha="center",
                va="top",
                fontsize=8,
            )

        z_H1 = z0 + props["H1_from_V1"]
        z_H2 = z0 + props["H2_from_V1"]

        ax.axvline(z_H1, linestyle="--", linewidth=1.5)
        ax.axvline(z_H2, linestyle="--", linewidth=1.5)

        ax.text(z_H1, -0.92 * y_max, f"{label_prefix}H1", ha="center", va="top", fontsize=8)
        ax.text(z_H2, -0.92 * y_max, f"{label_prefix}H2", ha="center", va="top", fontsize=8)

        return {
            "V1": z0 + V["V1"],
            "V2": z0 + V["V2"],
            "V3": z0 + V["V3"],
            "H1": z_H1,
            "H2": z_H2,
            "front_focus": z0 + props["front_focus_from_V1"],
            "rear_focus": z0 + props["rear_focus_from_V1"],
        }


def propagate_free_space(ray, distance):
    """
    Free-space propagation in air for ray vector [y, theta].
    """
    y, theta = ray
    return np.array([y + distance * theta, theta])


def refract_translate_lens_segments(lens, z0, ray_in):
    """
    Piecewise paraxial ray trace through a cemented doublet.

    ray_in is [y, theta] in air at the first vertex of the lens.
    Internally, the reduced vector [y, n theta] is used.
    """
    V1 = z0
    V2 = z0 + lens.t12
    V3 = z0 + lens.t12 + lens.t23

    q = np.array([ray_in[0], ray_in[1]])

    z_points = [V1]
    y_points = [q[0]]

    phi1 = lens.surface_power(1.0, lens.n1, lens.R1)
    phi2 = lens.surface_power(lens.n1, lens.n2, lens.R2)
    phi3 = lens.surface_power(lens.n2, 1.0, lens.R3)

    q = lens.S(phi1) @ q

    q = lens.T(lens.t12, lens.n1) @ q
    z_points.append(V2)
    y_points.append(q[0])

    q = lens.S(phi2) @ q

    q = lens.T(lens.t23, lens.n2) @ q
    z_points.append(V3)
    y_points.append(q[0])

    q = lens.S(phi3) @ q

    ray_out = np.array([q[0], q[1]])

    return np.array(z_points), np.array(y_points), ray_out


def draw_two_lens_spatial_filter(
    lens1,
    lens2,
    lens1_orientation="+",
    lens2_orientation="+",
    input_ray_heights=(-5.0, 0.0, 5.0),
    input_length=20.0,
    output_length=40.0,
    savepath=None,
):
    """
    Draw a two-lens spatial filter / telescope.

    Lens 1 starts at z=0.
    The pinhole is at the rear focal point of lens 1.
    Lens 2 is placed so that its front focal point coincides with the pinhole.
    """
    p1 = lens1.properties()
    p2 = lens2.properties()

    z_lens1 = 0.0
    z_pinhole = lens1.L + p1["BFL_from_V3"]

    # Lens 2 front focus location is z_lens2 + FFL_2.
    # Require z_lens2 + FFL_2 = z_pinhole.
    z_lens2 = z_pinhole - p2["FFL_from_V1"]

    lens_separation = z_lens2 - lens1.L
    magnification = p2["EFL"] / p1["EFL"]

    fig, ax = plt.subplots(figsize=(14, 5))

    lens1.draw_at(ax, z0=z_lens1, label_prefix="L1 ")
    lens2.draw_at(ax, z0=z_lens2, label_prefix="L2 ")

    # Pinhole.
    ax.axvline(z_pinhole, linestyle="-", linewidth=2)
    ax.plot([z_pinhole], [0.0], marker="o", markersize=8)
    ax.text(
        z_pinhole,
        0.12 * min(lens1.aperture_radius, lens2.aperture_radius),
        "pinhole",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    # Small pinhole block symbol.
    ax.plot(
        [z_pinhole, z_pinhole],
        [-0.9, 0.9],
        linewidth=5,
        solid_capstyle="round",
    )

    z_start = -input_length
    z_end = z_lens2 + lens2.L + output_length

    # Ray trace.
    for y0 in input_ray_heights:
        ray = np.array([y0, 0.0])

        ray_at_lens1 = propagate_free_space(ray, z_lens1 - z_start)

        z_ray = [z_start, z_lens1]
        y_ray = [y0, ray_at_lens1[0]]

        z_seg, y_seg, ray_after_lens1 = refract_translate_lens_segments(
            lens1,
            z_lens1,
            ray_at_lens1,
        )
        z_ray.extend(z_seg[1:].tolist())
        y_ray.extend(y_seg[1:].tolist())

        ray_at_pinhole = propagate_free_space(
            ray_after_lens1,
            z_pinhole - (z_lens1 + lens1.L),
        )
        z_ray.append(z_pinhole)
        y_ray.append(ray_at_pinhole[0])

        ray_at_lens2 = propagate_free_space(
            ray_at_pinhole,
            z_lens2 - z_pinhole,
        )
        z_ray.append(z_lens2)
        y_ray.append(ray_at_lens2[0])

        z_seg, y_seg, ray_after_lens2 = refract_translate_lens_segments(
            lens2,
            z_lens2,
            ray_at_lens2,
        )
        z_ray.extend(z_seg[1:].tolist())
        y_ray.extend(y_seg[1:].tolist())

        ray_at_end = propagate_free_space(
            ray_after_lens2,
            z_end - (z_lens2 + lens2.L),
        )
        z_ray.append(z_end)
        y_ray.append(ray_at_end[0])

        ax.plot(z_ray, y_ray, linewidth=1.5)

    # Show expected output beam radius from magnification.
    input_outer = max(abs(y) for y in input_ray_heights)
    output_outer = abs(magnification) * input_outer

    ax.text(
        0.02,
        0.98,
        (
            f"L1: {lens1.name} orientation {lens1_orientation}\n"
            f"L2: {lens2.name} orientation {lens2_orientation}\n"
            f"L1 EFL = {p1['EFL']:.2f} mm, BFL = {p1['BFL_from_V3']:.2f} mm\n"
            f"L2 EFL = {p2['EFL']:.2f} mm, FFL = {p2['FFL_from_V1']:.2f} mm\n"
            f"pinhole z = {z_pinhole:.2f} mm\n"
            f"L1 V3 to L2 V1 separation = {lens_separation:.2f} mm\n"
            f"beam magnification f2/f1 = {magnification:.3f}\n"
            f"outer ray: {input_outer:.2f} mm -> {output_outer:.2f} mm"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    # Distance annotations.
    y_annot = -1.45 * max(lens1.aperture_radius, lens2.aperture_radius)

    ax.annotate(
        "",
        xy=(lens1.L, y_annot),
        xytext=(z_pinhole, y_annot),
        arrowprops=dict(arrowstyle="<->", linewidth=1.2),
    )
    ax.text(
        0.5 * (lens1.L + z_pinhole),
        y_annot - 0.8,
        f"L1 BFL = {p1['BFL_from_V3']:.2f} mm",
        ha="center",
        va="top",
    )

    ax.annotate(
        "",
        xy=(z_pinhole, y_annot),
        xytext=(z_lens2, y_annot),
        arrowprops=dict(arrowstyle="<->", linewidth=1.2),
    )
    ax.text(
        0.5 * (z_pinhole + z_lens2),
        y_annot - 0.8,
        f"|L2 FFL| = {abs(p2['FFL_from_V1']):.2f} mm",
        ha="center",
        va="top",
    )

    ax.axhline(0, linewidth=1)
    ax.set_xlabel("z position [mm]")
    ax.set_ylabel("ray height y [mm]")
    ax.set_title("Two-lens spatial filter / beam telescope")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    y_max = 1.75 * max(lens1.aperture_radius, lens2.aperture_radius, output_outer)
    ax.set_xlim(z_start, z_end)
    ax.set_ylim(-y_max, y_max)

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=200)
        print(f"Saved {savepath}")

    return fig, ax, {
        "lens1_name": lens1.name,
        "lens2_name": lens2.name,
        "lens1_orientation": lens1_orientation,
        "lens2_orientation": lens2_orientation,
        "z_pinhole": z_pinhole,
        "z_lens2": z_lens2,
        "lens_separation": lens_separation,
        "magnification": magnification,
        "lens1": p1,
        "lens2": p2,
    }


def make_four_orientation_figures_for_two_lenses(lens1_base, lens2_base, basename="two_lens"):
    """
    Generate four orientation cases for two possibly different lenses:
        ++, +-, -+, --
    where + means nominal orientation and - means reversed.
    """
    lens1_plus = lens1_base
    lens1_minus = lens1_base.reversed()

    lens2_plus = lens2_base
    lens2_minus = lens2_base.reversed()

    cases = [
        ("++", lens1_plus, lens2_plus, "+", "+"),
        ("+-", lens1_plus, lens2_minus, "+", "-"),
        ("-+", lens1_minus, lens2_plus, "-", "+"),
        ("--", lens1_minus, lens2_minus, "-", "-"),
    ]

    summaries = []

    for label, L1, L2, o1, o2 in cases:
        fig, ax, summary = draw_two_lens_spatial_filter(
            L1,
            L2,
            lens1_orientation=o1,
            lens2_orientation=o2,
            input_ray_heights=(-5.0, 0.0, 5.0),
            savepath=f"{basename}_{label.replace('+', 'p').replace('-', 'm')}.png",
        )
        summaries.append(summary)

    print()
    print("Summary of four configurations")
    print("--------------------------------")
    for s in summaries:
        print(f"orientation {s['lens1_orientation']}{s['lens2_orientation']}:")
        print(f"  L1 = {s['lens1_name']}")
        print(f"  L2 = {s['lens2_name']}")
        print(f"  L1 EFL = {s['lens1']['EFL']:.4f} mm")
        print(f"  L2 EFL = {s['lens2']['EFL']:.4f} mm")
        print(f"  magnification = {s['magnification']:.4f}")
        print(f"  pinhole position from L1 V1 = {s['z_pinhole']:.4f} mm")
        print(f"  L2 V1 position from L1 V1 = {s['z_lens2']:.4f} mm")
        print(f"  lens separation V3_1 to V1_2 = {s['lens_separation']:.4f} mm")
        print(f"  L1 BFL = {s['lens1']['BFL_from_V3']:.4f} mm")
        print(f"  L2 FFL = {s['lens2']['FFL_from_V1']:.4f} mm")
        print()

    plt.show()


if __name__ == "__main__":

    AC254_030_A = CementedDoublet(
        R1=20.9,
        R2=-16.7,
        R3=-79.8,
        n1=1.67003,
        n2=1.80518,
        t12=12.0,
        t23=2.0,
        aperture_radius=12.7,
        name="AC254-030-A, f=30 mm",
    )

    AC254_050_A = CementedDoublet(
        R1=33.3,
        R2=-22.3,
        R3=-291.1,
        n1=1.67003,
        n2=1.80518,
        t12=9.0,
        t23=2.5,
        aperture_radius=12.7,
        name="AC254-050-A, f=50 mm",
    )

    make_four_orientation_figures_for_two_lenses(
        AC254_030_A,
        AC254_050_A,
        basename="AC254_030_to_AC254_050_spatial_filter",
    )

    make_four_orientation_figures_for_two_lenses(
        AC254_050_A,
        AC254_030_A,
        basename="AC254_050_to_AC254_030_spatial_filter",
    )