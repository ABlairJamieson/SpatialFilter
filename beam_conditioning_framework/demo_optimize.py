"""
ADVANCED exploratory optimizer.

Do not start here. First run and understand:
    demo_single_propagation.py
    demo_subsystem_comparison.py
    demo_fixed_aperture_scan.py

This optimizer can find mathematically improved configurations that are not
necessarily physically meaningful. Validate any candidate with fixed-layout and
subsystem comparisons, and rerun using a larger FFT grid.
"""
import matplotlib.pyplot as plt

from beam_conditioning_core import (
    FieldGrid,
    thorlabs_AC254_030_A,
    thorlabs_AC254_050_A,
    default_input_apertures,
    optimize_continuous_configuration,
    plot_intensity,
    plot_encircled_energy,
    save_plane_snapshots_pdf,
)
from student_common import OUTPUT_DIR, student_default_input_field


def main():
    # Coarse grid for speed. Validate best candidates with N=1024 or 2048.
    grid = FieldGrid(N=512, size_mm=18.0, wavelength_mm=405e-6)
    lens1 = thorlabs_AC254_030_A()
    lens2 = thorlabs_AC254_050_A()

    opt = optimize_continuous_configuration(
        grid=grid,
        input_field_factory=student_default_input_field,
        lens1=lens1,
        lens2=lens2,
        input_apertures=default_input_apertures(),
        use_pinhole=False,
        use_downstream_apertures=False,
        maxiter=25,
        popsize=6,
    )

    result = opt["result"]
    plot_intensity(
        result.U,
        grid,
        title="Optimized target plane intensity",
        target_radius_mm=1.5,
        savepath=str(OUTPUT_DIR / "optimized_target_plane.png"),
    )
    plot_encircled_energy(
        result.U,
        grid,
        savepath=str(OUTPUT_DIR / "optimized_encircled_energy.png"),
    )
    save_plane_snapshots_pdf(
        result,
        filename=str(OUTPUT_DIR / "optimized_beam_planes.pdf"),
        target_radius_mm=1.5,
        max_pages=40,
    )
    plt.close("all")


if __name__ == "__main__":
    main()
