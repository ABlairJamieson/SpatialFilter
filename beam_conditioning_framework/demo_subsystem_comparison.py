"""
Student diagnostic study: compare increasingly complicated subsystems.

Question: does each added optical element improve or degrade the fraction of
light inside the 3 mm diameter target at 1 m?
"""
import matplotlib.pyplot as plt

from beam_conditioning_core import (
    FieldGrid,
    BeamConditioningSystem,
    ApertureElement,
    LensElement,
    thorlabs_AC254_030_A,
    thorlabs_AC254_050_A,
    build_two_lens_common_focus_layout,
    build_system_from_layout,
    default_input_apertures,
    plot_intensity,
    plot_encircled_energy,
)
from student_common import OUTPUT_DIR, student_default_input_field, print_result_summary


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
    grid = FieldGrid(N=9148, size_mm=5.0, wavelength_mm=520e-6)

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

    cases.append((
        "input beam only",
        BeamConditioningSystem(grid=grid, elements=[], z_start_mm=0.0, z_target_mm=1000.0, target_radius_mm=1.5),
    ))

    cases.append((
        "input apertures only",
        BeamConditioningSystem(grid=grid, elements=list(input_apertures), z_start_mm=0.0, z_target_mm=1000.0, target_radius_mm=1.5),
    ))

    elements_L1_only = list(input_apertures)
    elements_L1_only.append(
        LensElement(
            z_mm=layout.lens1_H2_z_mm,
            f_mm=layout.lens1_props["EFL"],
            clear_radius_mm=lens1.aperture_radius,
            name=f"L1 {lens1.name}",
        )
    )

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

    post_l2_aperture = ApertureElement(
        z_mm=layout.lens2_V3_z_mm + 10.0,
        radius_mm=1.1,
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

    print("\nSubsystem comparison")
    print("--------------------")
    print(f"Grid: N={grid.N}, size={grid.size_mm:.3f} mm, dx={grid.dx:.5f} mm")
    print(f"L1: {lens1.name}, EFL={layout.lens1_props['EFL']:.4f} mm")
    print(f"L2: {lens2.name}, EFL={layout.lens2_props['EFL']:.4f} mm")
    print(f"Fixed lens separation = {layout.lens_separation_mm:.4f} mm")
    print("Target: r=1.5 mm at z=1000 mm\n")

    results = []
    for label, system in cases:
        result = system.propagate(U0)
        results.append((label, result))
        print_result_summary(label, result)

    best_label, best_result = min(results, key=lambda x: x[1].fraction_outside_target)
    print(f"\nBest subsystem by target containment: {best_label}")

    plot_intensity(
        best_result.U,
        grid,
        title=f"Best subsystem: {best_label}",
        target_radius_mm=1.5,
        savepath=str(OUTPUT_DIR / "subsystem_best_target_plane.png"),
    )

    plot_encircled_energy(
        best_result.U,
        grid,
        savepath=str(OUTPUT_DIR / "subsystem_best_encircled_energy.png"),
    )

    summary_path = OUTPUT_DIR / "subsystem_comparison_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Subsystem comparison\n")
        f.write("--------------------\n")
        for label, result in results:
            f.write(
                f"{label:45s}  "
                f"T={result.throughput:8.5f}  "
                f"outside={result.fraction_outside_target:12.4e}  "
                f"inside={100.0*(1.0-result.fraction_outside_target):9.5f}%  "
                f"rms={result.rms_radius_mm:8.4f} mm\n"
            )
    print(f"Saved {summary_path}")
    plt.close("all")


if __name__ == "__main__":
    main()
