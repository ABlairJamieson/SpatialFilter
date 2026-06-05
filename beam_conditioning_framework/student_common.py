"""Shared helpers for the student-facing beam-conditioning demos."""
from __future__ import annotations

from pathlib import Path
from beam_conditioning_core import gaussian_field

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def student_default_input_field(grid):
    """
    Conservative starting beam.

    The waist is chosen small enough that the default 0.50 mm input aperture
    does not strongly clip the field. This avoids making the first example
    dominated by hard-aperture diffraction.
    """
    return gaussian_field(
        grid,
        waist_radius_mm=0.20,
        tilt_x_mrad=0.0,
        tilt_y_mrad=0.0,
    )
def print_result_summary(label, result):
    """Compact one-line summary for comparing cases."""
    print(
        f"{label:45s}  "
        f"T={result.throughput:8.5f}  "
        f"outside={result.fraction_outside_target:12.4e}  "
        f"inside={100.0*(1.0-result.fraction_outside_target):9.5f}%  "
        f"rms={result.rms_radius_mm:8.4f} mm"
    )
