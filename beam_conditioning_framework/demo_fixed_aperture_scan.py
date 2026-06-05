"""
Student aperture-only scan with fixed lens positions.

This deliberately avoids moving the lenses. It scans only aperture radii to ask
whether aperture-only cleanup improves target containment without excessive loss
of throughput.
"""
import matplotlib.pyplot as plt

from beam_conditioning_core import (
    FieldGrid,
    ApertureElement,
    thorlabs_AC254_030_A,
    thorlabs_AC254_050_A,
    build_two_lens_common_focus_layout,
    build_system_from_layout,
    plot_intensity,
    plot_encircled_energy,
    save_plane_snapshots_pdf,
)
from student_common import OUTPUT_DIR, student_default_input_field


def main():
    # Important numerical limitation:
    #
    # This code uses one fixed transverse grid for all planes. That means the same
    # dx and same field of view are used at the input apertures, lenses, focus plane,
    # pinhole plane, and target plane.
    #
    # This is simple, but it can be inefficient or inaccurate when the beam becomes
    # very small at a focus or when a small pinhole is inserted.
    #
    # Rule of thumb:
    #     pinhole_radius_mm should be at least ~20*dx for exploratory work
    #     and preferably ~50*dx for low-tail studies.
    #
    # For N=4096, size_mm=9:
    #     dx = 9/4096 = 0.0022 mm = 2.2 um
    #     20*dx = 44 um
    #     50*dx = 110 um
    #
    # Therefore, pinholes with radii of 10--25 um are not well resolved on this grid.
    # A future version should use zoom/resampled propagation near the focus.
    grid = FieldGrid(N=4096, size_mm=9.0, wavelength_mm=520e-6)

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

    input_aperture_pairs = [
        (0.30, 0.31),
        (0.40, 0.41),
        (0.50, 0.51),
        (0.60, 0.61),
        (0.75, 0.76),
    ]
    post_l2_aperture_radii = [None, 1.5, 2.0, 2.5, 3.0]

    rows = []

    print("\nFixed-layout aperture scan")
    print("--------------------------")
    print("Lens positions are fixed. Only aperture radii are scanned.\n")
    print(f"L1: {lens1.name}, EFL={layout.lens1_props['EFL']:.4f} mm")
    print(f"L2: {lens2.name}, EFL={layout.lens2_props['EFL']:.4f} mm")
    print(f"Fixed lens separation = {layout.lens_separation_mm:.4f} mm\n")

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
                "input_apertures": input_apertures,
                "downstream_apertures": downstream_apertures,
            })

    rows.sort(key=lambda r: (r["outside"], -r["throughput"]))

    print(f"{'rank':>4} {'r1':>7} {'r2':>7} {'post_r':>8} {'T':>9} {'outside':>12} {'inside[%]':>11} {'rms[mm]':>9}")
    for i, row in enumerate(rows, start=1):
        post_label = "None" if row["post_r"] is None else f"{row['post_r']:.3f}"
        print(
            f"{i:4d} {row['r1']:7.3f} {row['r2']:7.3f} {post_label:>8} "
            f"{row['throughput']:9.5f} {row['outside']:12.4e} {row['inside_percent']:11.5f} {row['rms']:9.4f}"
        )

    csv_path = OUTPUT_DIR / "fixed_aperture_scan_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("rank,r1_mm,r2_mm,post_l2_radius_mm,throughput,outside,inside_percent,rms_radius_mm\n")
        for i, row in enumerate(rows, start=1):
            post_value = "" if row["post_r"] is None else row["post_r"]
            f.write(f"{i},{row['r1']},{row['r2']},{post_value},{row['throughput']},{row['outside']},{row['inside_percent']},{row['rms']}\n")
    print(f"\nSaved {csv_path}")

    best = rows[0]
    print("\nBest fixed-aperture case:")
    print(
        f"r1={best['r1']:.3f} mm, r2={best['r2']:.3f} mm, "
        f"post-L2 aperture={best['post_r']}, T={best['throughput']:.5f}, outside={best['outside']:.4e}"
    )

    plot_intensity(
        best["result"].U,
        grid,
        title="Best fixed-aperture scan target plane",
        target_radius_mm=1.5,
        savepath=str(OUTPUT_DIR / "fixed_aperture_best_target_plane.png"),
    )

    plot_encircled_energy(
        best["result"].U,
        grid,
        savepath=str(OUTPUT_DIR / "fixed_aperture_best_encircled_energy.png"),
    )

    best_system = build_system_from_layout(
        grid=grid,
        layout=layout,
        input_apertures=best["input_apertures"],
        pinhole_radius_mm=None,
        downstream_apertures=best["downstream_apertures"],
        z_target_mm=1000.0,
        target_radius_mm=1.5,
    )
    best_result = best_system.propagate(U0, capture_planes=True)

    save_plane_snapshots_pdf(
        best_result,
        filename=str(OUTPUT_DIR / "fixed_aperture_best_beam_planes.pdf"),
        target_radius_mm=1.5,
    )

    plt.close("all")


if __name__ == "__main__":
    main()
