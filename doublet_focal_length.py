#!/usr/bin/env python3
"""
cemented_doublet_principal_planes.py

Calculate and draw the principal planes of a cemented doublet.

Sign convention:
    z increases to the right.
    R > 0 if the centre of curvature is to the right of the surface.
    R < 0 if the centre of curvature is to the left of the surface.

Geometry:
    V1 at z = 0
    V2 at z = t12
    V3 at z = t12 + t23

Optical system:
    air -> n1 -> n2 -> air

Ray vector:
    [ y, n theta ]

Surface matrix:
    S = [[1, 0],
         [-Phi, 1]]

Translation matrix:
    T = [[1, t/n],
         [0, 1]]

For system matrix

    M = [[A, B],
         [C, D]]

in air:

    EFL = -1 / C
    BFL = -A / C

    z_H1 relative to V1 = (D - 1) / C
    z_H2 relative to V3 = (1 - A) / C
    z_H2 absolute      = L + (1 - A) / C

where L = t12 + t23.
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
    aperture_radius: float = 12.5  # drawing aperture radius, same units as R/t

    def surface_power(self, n_left: float, n_right: float, R: float) -> float:
        """Surface optical power using reduced ray coordinates."""
        if np.isinf(R):
            return 0.0
        return (n_right - n_left) / R

    def S(self, phi: float) -> np.ndarray:
        """Refraction matrix for reduced ray vector [y, n theta]."""
        return np.array([
            [1.0, 0.0],
            [-phi, 1.0]
        ])

    def T(self, t: float, n: float) -> np.ndarray:
        """Translation matrix for reduced ray vector [y, n theta]."""
        return np.array([
            [1.0, t / n],
            [0.0, 1.0]
        ])

    @property
    def L(self) -> float:
        """Total centre thickness."""
        return self.t12 + self.t23

    @property
    def vertices(self):
        """Physical surface vertex z positions."""
        return {
            "V1": 0.0,
            "V2": self.t12,
            "V3": self.t12 + self.t23,
        }

    def matrix(self) -> np.ndarray:
        """Full system matrix from V1 to V3."""
        phi1 = self.surface_power(1.0, self.n1, self.R1)
        phi2 = self.surface_power(self.n1, self.n2, self.R2)
        phi3 = self.surface_power(self.n2, 1.0, self.R3)

        S1 = self.S(phi1)
        S2 = self.S(phi2)
        S3 = self.S(phi3)

        T12 = self.T(self.t12, self.n1)
        T23 = self.T(self.t23, self.n2)

        # Rays encounter S1, then T12, then S2, then T23, then S3.
        # Matrix multiplication acts right-to-left on column vectors.
        return S3 @ T23 @ S2 @ T12 @ S1

    def properties(self) -> dict:
        """Return EFL, FFL, BFL, principal plane locations, and focal points."""
        M = self.matrix()
        A, B = M[0, 0], M[0, 1]
        C, D = M[1, 0], M[1, 1]

        if abs(C) < 1e-15:
            raise ValueError("C is approximately zero; system has infinite focal length.")

        efl = -1.0 / C

        # Signed distances from physical lens surfaces.
        # FFL is measured from V1 to the front focal point.
        # For a positive lens this is usually negative, i.e. to the left of V1.
        ffl = D / C

        # BFL is measured from V3 to the rear focal point.
        bfl = -A / C

        z_H1 = (D - 1.0) / C
        z_H2_relative_to_V3 = (1.0 - A) / C
        z_H2 = self.L + z_H2_relative_to_V3

        z_front_focus = ffl
        z_rear_focus = self.L + bfl

        return {
            "A": A,
            "B": B,
            "C": C,
            "D": D,
            "EFL": efl,

            # Signed distances from physical surfaces
            "FFL_from_V1": ffl,
            "BFL_from_V3": bfl,

            # Absolute focal point locations measured from V1
            "front_focus_from_V1": z_front_focus,
            "rear_focus_from_V1": z_rear_focus,

            # Principal plane locations
            "H1_from_V1": z_H1,
            "H2_from_V3": z_H2_relative_to_V3,
            "H2_from_V1": z_H2,

            "matrix": M,
        }

    def spherical_surface_points(self, z_vertex: float, R: float, y_max: float, npts: int = 300):
        """
        Return z(y), y points for a spherical surface.

        The spherical surface has vertex at z_vertex and centre at z_vertex + R.

        Equation:
            (z - zc)^2 + y^2 = R^2

        The branch is chosen so that z = z_vertex at y = 0.
        """
        y = np.linspace(-y_max, y_max, npts)

        if np.isinf(R):
            z = np.full_like(y, z_vertex)
            return z, y

        if abs(R) < y_max:
            raise ValueError(
                f"Aperture radius {y_max} is larger than |R|={abs(R)}. "
                "Reduce aperture_radius for drawing."
            )

        zc = z_vertex + R

        # Choose the branch that gives z = z_vertex at y = 0.
        # For R > 0, vertex is left of centre, so use z = zc - sqrt(...)
        # For R < 0, vertex is right of centre, so use z = zc + sqrt(...)
        root = np.sqrt(R**2 - y**2)
        if R > 0:
            z = zc - root
        else:
            z = zc + root

        return z, y

    def draw(
        self,
        ax=None,
        show_front_focus=True,
        show_rear_focus=True,
        title="Cemented doublet principal planes and focal points",
    ):
        """Draw lens surfaces, principal planes, and front/rear focal points."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure

        props = self.properties()
        V = self.vertices
        y_max = self.aperture_radius

        # Draw the three lens surfaces.
        for label, z_vertex, R in [
            ("R1", V["V1"], self.R1),
            ("R2", V["V2"], self.R2),
            ("R3", V["V3"], self.R3),
        ]:
            z, y = self.spherical_surface_points(z_vertex, R, y_max)
            ax.plot(z, y, linewidth=2)
            ax.text(z_vertex, 1.08 * y_max, label, ha="center", va="bottom")

        # Fill a simple outline of the two elements.
        z1, y = self.spherical_surface_points(V["V1"], self.R1, y_max)
        z2, _ = self.spherical_surface_points(V["V2"], self.R2, y_max)
        z3, _ = self.spherical_surface_points(V["V3"], self.R3, y_max)

        # First glass outline: surface 1 plus reversed surface 2.
        ax.fill(
            np.concatenate([z1, z2[::-1]]),
            np.concatenate([y, y[::-1]]),
            alpha=0.15,
            label=f"glass 1, n={self.n1:.4g}",
        )

        # Second glass outline: surface 2 plus reversed surface 3.
        ax.fill(
            np.concatenate([z2, z3[::-1]]),
            np.concatenate([y, y[::-1]]),
            alpha=0.15,
            label=f"glass 2, n={self.n2:.4g}",
        )

        # Draw physical vertices.
        for name, z in V.items():
            ax.axvline(z, linestyle=":", linewidth=1)
            ax.text(z, -1.16 * y_max, name, ha="center", va="top")

        # Principal planes.
        z_H1 = props["H1_from_V1"]
        z_H2 = props["H2_from_V1"]

        ax.axvline(z_H1, linestyle="--", linewidth=2, label="H1")
        ax.axvline(z_H2, linestyle="--", linewidth=2, label="H2")

        ax.text(z_H1, -0.95 * y_max, "H1", ha="center", va="top")
        ax.text(z_H2, -0.95 * y_max, "H2", ha="center", va="top")

        # Front and rear focal points.
        if show_front_focus:
            z_front_focus = props["front_focus_from_V1"]
            ax.axvline(z_front_focus, linestyle="-.", linewidth=2, label="front focus")
            ax.plot([z_front_focus], [0], marker="o")
            ax.text(
                z_front_focus,
                0.08 * y_max,
                "front focus",
                ha="center",
                va="bottom",
            )

        if show_rear_focus:
            z_rear_focus = props["rear_focus_from_V1"]
            ax.axvline(z_rear_focus, linestyle="-.", linewidth=2, label="rear focus")
            ax.plot([z_rear_focus], [0], marker="o")
            ax.text(
                z_rear_focus,
                0.08 * y_max,
                "rear focus",
                ha="center",
                va="bottom",
            )

        # Draw distance arrows along the optical axis.
        arrow_y_front = -0.55 * y_max
        arrow_y_rear = -0.72 * y_max

        if show_front_focus:
            z_front_focus = props["front_focus_from_V1"]
            ax.annotate(
                "",
                xy=(z_front_focus, arrow_y_front),
                xytext=(V["V1"], arrow_y_front),
                arrowprops=dict(arrowstyle="<->", linewidth=1.5),
            )
            ax.text(
                0.5 * (z_front_focus + V["V1"]),
                arrow_y_front - 0.08 * y_max,
                f"FFL = {props['FFL_from_V1']:.3g}",
                ha="center",
                va="top",
            )

        if show_rear_focus:
            z_rear_focus = props["rear_focus_from_V1"]
            ax.annotate(
                "",
                xy=(V["V3"], arrow_y_rear),
                xytext=(z_rear_focus, arrow_y_rear),
                arrowprops=dict(arrowstyle="<->", linewidth=1.5),
            )
            ax.text(
                0.5 * (V["V3"] + z_rear_focus),
                arrow_y_rear - 0.08 * y_max,
                f"BFL = {props['BFL_from_V3']:.3g}",
                ha="center",
                va="top",
            )

        ax.axhline(0, linewidth=1)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("z position")
        ax.set_ylabel("ray height y")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # Make sure focal points are visible even if far from the lens.
        z_values = [V["V1"], V["V2"], V["V3"], z_H1, z_H2]
        if show_front_focus:
            z_values.append(props["front_focus_from_V1"])
        if show_rear_focus:
            z_values.append(props["rear_focus_from_V1"])

        z_min = min(z_values)
        z_max = max(z_values)
        z_pad = 0.08 * max(z_max - z_min, self.L, 1.0)
        ax.set_xlim(z_min - z_pad, z_max + z_pad)
        ax.set_ylim(-1.35 * y_max, 1.25 * y_max)

        return fig, ax

    def print_summary(self):
        """Print a readable summary."""
        props = self.properties()

        print("System matrix M =")
        print(props["matrix"])
        print()
        print(f"A = {props['A']:.8g}")
        print(f"B = {props['B']:.8g}")
        print(f"C = {props['C']:.8g}")
        print(f"D = {props['D']:.8g}")
        print()
        print(f"EFL = {props['EFL']:.8g}")
        print()
        print("Focal distances from physical lens surfaces:")
        print(f"FFL from V1 = {props['FFL_from_V1']:.8g}")
        print(f"BFL from V3 = {props['BFL_from_V3']:.8g}")
        print()
        print("Absolute focal point positions measured from V1:")
        print(f"front focus from V1 = {props['front_focus_from_V1']:.8g}")
        print(f"rear focus from V1  = {props['rear_focus_from_V1']:.8g}")
        print()
        print("Principal plane positions:")
        print(f"H1 from V1 = {props['H1_from_V1']:.8g}")
        print(f"H2 from V3 = {props['H2_from_V3']:.8g}")
        print(f"H2 from V1 = {props['H2_from_V1']:.8g}")

if __name__ == "__main__":

    # Example values.
    # Replace these with actual Thorlabs prescription values if available.
    #
    # Units can be mm, as long as all lengths use the same units.
    
    lens = CementedDoublet(
        R1=20.9,
        R2=-16.7,
        R3=-79.8,
        n1=1.67003,   # N-BAF10 at d-line, 587.6 nm
        n2=1.80518,   # N-SF6HT at d-line, 587.6 nm
        t12=12.0,
        t23=2.0,
        aperture_radius=12.7,  # half of 25.4 mm diameter
    )

    lens.print_summary()
    fig, ax = lens.draw()
    print('***** Lens in orientation 1 done, close graphics to show reverse *****')
    
    plt.show()

    lens_rev = CementedDoublet(
        R1=79.8, 
        R2=16.7,
        R3=-20.9,
        n1=1.80518,   # N-BAF10 at d-line, 587.6 nm
        n2=1.67003,   # N-SF6HT at d-line, 587.6 nm
        t12=2.0,
        t23=12.0,
        aperture_radius=12.7,  # half of 25.4 mm diameter
    )

    lens_rev.print_summary()

    fig, ax = lens_rev.draw()
    print('***** Lens in orientation 2 done, close graphics to exit *****')
    plt.show()