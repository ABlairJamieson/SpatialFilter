#!/usr/bin/env python3
"""
doublet_pair_spatial_filter.py

Draw two identical cemented doublets in all four orientation combinations.

The pinhole is placed at the rear focal point of lens 1.
Lens 2 is placed so that its front focal point is also at the pinhole.
Therefore, a collimated beam entering lens 1 focuses at the pinhole and exits
lens 2 collimated.

Conventions:
    z increases to the right.
    R > 0 if the centre of curvature is to the right of the surface.
    R < 0 if the centre of curvature is to the left of the surface.

Ray vector:
    [ y, n theta ]

For air-space outside the lenses, n = 1, so the plotted slopes are theta.
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
    def L(self) -> float:
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
        Return a new CementedDoublet corresponding to the same physical lens
        flipped left-right.

        Original:
            air | R1 | n1 thickness t12 | R2 | n2 thickness t23 | R3 | air

        Reversed:
            air | -R3 | n2 thickness t23 | -R2 | n1 thickness t12 | -R1 | air
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

        S1 = self.S(phi1)
        S2 = self.S(phi2)
        S3 = self.S(phi3)

        T12 = self.T(self.t12, self.n1)
        T23 = self.T(self.t23, self.n2)

        return S3 @ T23 @ S2 @ T12 @ S1

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
        """
        Draw the physical doublet at absolute axial offset z0.
        Returns a dictionary of absolute vertex and principal-plane positions.
        """
        V = self.vertices
        props = self.properties()
        y_max = self.aperture_radius

        surfaces = [
            ("R1", V["V1"], self.R1),
            ("R2", V["V2"], self.R2),
            ("R3", V["V3"], self.R3),
        ]

        surface_curves = []
        for label, z_vertex, R in surfaces:
            z, y = self.spherical_surface_points(z_vertex + z0, R, y_max)
            surface_curves.append((z, y))
            ax.plot(z, y, linewidth=2)
            ax.text(
                z_vertex + z0,
                1.05 * y_max,
                f"{label_prefix}{label}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        z1, y = surface_curves[0]
        z2, _ = surface_curves[1]
        z3, _ = surface_curves[2]

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

        # Vertices.
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

        # Principal planes.
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
    Build piecewise paraxial ray path through a cemented doublet.

    Input ray is [y, theta] in air at V1.
    Internally we convert to reduced coordinates [y, n theta].
    Returns arrays of z,y points including vertices.
    """
    V1 = z0
    V2 = z0 + lens.t12
    V3 = z0 + lens.t12 + lens.t23

    # Convert air ray [y, theta] to reduced vector [y, n theta].
    q = np.array([ray_in[0], ray_in[1]])

    z_points = [V1]
    y_points = [q[0]]

    phi1 = lens.surface_power(1.0, lens.n1, lens.R1)
    phi2 = lens.surface_power(lens.n1, lens.n2, lens.R2)
    phi3 = lens.surface_power(lens.n2, 1.0, lens.R3)

    # Refraction at S1.
    q = lens.S(phi1) @ q

    # Translation through glass 1.
    q = lens.T(lens.t12, lens.n1) @ q
    z_points.append(V2)
    y_points.append(q[0])

    # Refraction at S2.
    q = lens.S(phi2) @ q

    # Translation through glass 2.
    q = lens.T(lens.t23, lens.n2) @ q
    z_points.append(V3)
    y_points.append(q[0])

    # Refraction at S3. Back to air, so reduced angle equals theta.
    q = lens.S(phi3) @ q

    ray_out = np.array([q[0], q[1]])

    return np.array(z_points), np.array(y_points), ray_out


def draw_two_lens_spatial_filter(
    lens1,
    lens2,
    orientation_label="++",
    input_ray_heights=(-5.0, 0.0, 5.0),
    input_length=15.0,
    output_length=25.0,
    savepath=None,
):
    """
    Draw a two-lens spatial filter.

    Lens 1 starts at z = 0.
    Pinhole is at rear focal point of lens 1.
    Lens 2 is placed so that its front focal point is also at the pinhole.
    """
    p1 = lens1.properties()
    p2 = lens2.properties()

    z_lens1 = 0.0
    z_pinhole = lens1.L + p1["BFL_from_V3"]

    # Lens 2 front focal point absolute position:
    # z_front_focus_2 = z_lens2 + FFL_2.
    # We require z_front_focus_2 = z_pinhole.
    z_lens2 = z_pinhole - p2["FFL_from_V1"]

    lens_separation = z_lens2 - lens1.L

    fig, ax = plt.subplots(figsize=(13, 5))

    pos1 = lens1.draw_at(ax, z0=z_lens1, label_prefix="L1 ")
    pos2 = lens2.draw_at(ax, z0=z_lens2, label_prefix="L2 ")

    # Pinhole.
    ax.axvline(z_pinhole, linestyle="-", linewidth=2)
    ax.plot([z_pinhole], [0.0], marker="o", markersize=8)
    ax.text(
        z_pinhole,
        0.12 * lens1.aperture_radius,
        "pinhole",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    # Draw a small aperture/pinhole symbol.
    pinhole_half_height = 0.8
    ax.plot(
        [z_pinhole, z_pinhole],
        [-pinhole_half_height, pinhole_half_height],
        linewidth=5,
        solid_capstyle="round",
    )

    # Ray tracing.
    z_start = -input_length
    z_end = z_lens2 + lens2.L + output_length

    for y0 in input_ray_heights:
        # Incoming collimated ray.
        ray = np.array([y0, 0.0])

        # From z_start to L1 V1.
        ray_at_lens1 = propagate_free_space(ray, z_lens1 - z_start)

        z_ray = [z_start, z_lens1]
        y_ray = [y0, ray_at_lens1[0]]

        # Through lens 1.
        z_seg, y_seg, ray_after_lens1 = refract_translate_lens_segments(lens1, z_lens1, ray_at_lens1)
        z_ray.extend(z_seg[1:].tolist())
        y_ray.extend(y_seg[1:].tolist())

        # From lens 1 to pinhole.
        ray_at_pinhole = propagate_free_space(ray_after_lens1, z_pinhole - (z_lens1 + lens1.L))
        z_ray.append(z_pinhole)
        y_ray.append(ray_at_pinhole[0])

        # From pinhole to lens 2.
        # The ray should pass through y=0 at the pinhole.
        ray_at_lens2 = propagate_free_space(ray_at_pinhole, z_lens2 - z_pinhole)
        z_ray.append(z_lens2)
        y_ray.append(ray_at_lens2[0])

        # Through lens 2.
        z_seg, y_seg, ray_after_lens2 = refract_translate_lens_segments(lens2, z_lens2, ray_at_lens2)
        z_ray.extend(z_seg[1:].tolist())
        y_ray.extend(y_seg[1:].tolist())

        # To output.
        ray_at_end = propagate_free_space(ray_after_lens2, z_end - (z_lens2 + lens2.L))
        z_ray.append(z_end)
        y_ray.append(ray_at_end[0])

        ax.plot(z_ray, y_ray, linewidth=1.5)

    # Distance annotations.
    y_annot = -1.45 * lens1.aperture_radius

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

    ax.text(
        0.02,
        0.98,
        (
            f"orientation {orientation_label}\n"
            f"L1 EFL = {p1['EFL']:.2f} mm, BFL = {p1['BFL_from_V3']:.2f} mm\n"
            f"L2 EFL = {p2['EFL']:.2f} mm, FFL = {p2['FFL_from_V1']:.2f} mm\n"
            f"lens separation = {lens_separation:.2f} mm"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    ax.axhline(0, linewidth=1)
    ax.set_xlabel("z position [mm]")
    ax.set_ylabel("ray height y [mm]")
    ax.set_title(f"Two-doublet spatial filter, orientation {orientation_label}")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    z_min = z_start
    z_max = z_end
    y_max = 1.7 * lens1.aperture_radius
    ax.set_xlim(z_min, z_max)
    ax.set_ylim(-y_max, y_max)

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=200)
        print(f"Saved {savepath}")

    return fig, ax, {
        "orientation": orientation_label,
        "z_pinhole": z_pinhole,
        "z_lens2": z_lens2,
        "lens_separation": lens_separation,
        "lens1": p1,
        "lens2": p2,
    }


def make_all_four_orientation_figures(base_lens):
    """
    Generate all four orientation cases:
        ++, +-, -+, --
    """
    lens_plus = base_lens
    lens_minus = base_lens.reversed()

    cases = [
        ("++", lens_plus, lens_plus),
        ("+-", lens_plus, lens_minus),
        ("-+", lens_minus, lens_plus),
        ("--", lens_minus, lens_minus),
    ]

    summaries = []

    for label, L1, L2 in cases:
        fig, ax, summary = draw_two_lens_spatial_filter(
            L1,
            L2,
            orientation_label=label,
            input_ray_heights=(-5.0, 0.0, 5.0),
            savepath=f"two_doublet_spatial_filter_{label.replace('+', 'p').replace('-', 'm')}.png",
        )
        summaries.append(summary)

    print()
    print("Summary of four two-lens configurations")
    print("---------------------------------------")
    for s in summaries:
        print(f"orientation {s['orientation']}:")
        print(f"  pinhole position from L1 V1 = {s['z_pinhole']:.4f} mm")
        print(f"  L2 V1 position from L1 V1  = {s['z_lens2']:.4f} mm")
        print(f"  lens separation V3_1 to V1_2 = {s['lens_separation']:.4f} mm")
        print(f"  L1 BFL = {s['lens1']['BFL_from_V3']:.4f} mm")
        print(f"  L2 FFL = {s['lens2']['FFL_from_V1']:.4f} mm")
        print()

    plt.show()


if __name__ == "__main__":

    # Thorlabs AC254-030-A approximate prescription.
    # Units: mm.
    #
    # Orientation "+":
    #     air -> N-BAF10 -> N-SF6HT -> air
    #
    # Orientation "-" is generated automatically by lens.reversed().
    base_lens = CementedDoublet(
        R1=20.9,
        R2=-16.7,
        R3=-79.8,
        n1=1.67003,
        n2=1.80518,
        t12=12.0,
        t23=2.0,
        aperture_radius=12.7,
        name="AC254-030-A",
    )

    make_all_four_orientation_figures(base_lens)