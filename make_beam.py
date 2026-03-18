from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def str2bool(s: str) -> bool:
    return s.strip().lower() in {"1", "true", "yes", "y", "on"}


def main():
    parser = argparse.ArgumentParser(description="Make a CODE V BCM complex-amplitude file.")
    parser.add_argument("--outfile", default="dirty_beam.dat")
    parser.add_argument("--n", type=int, default=512, help="Grid size")
    parser.add_argument("--dx", type=float, default=0.08, help="Grid spacing in mm")
    parser.add_argument("--wl_nm", type=float, default=520.0, help="Wavelength in nm")

    parser.add_argument("--w_main", type=float, default=1.0, help="Main beam 1/e field radius in mm")
    parser.add_argument("--w_tail", type=float, default=10.0, help="Tail beam 1/e field radius in mm")
    parser.add_argument("--tail_frac", type=float, default=0.01, help="Tail power fraction")
    parser.add_argument("--tail_x_mm", type=float, default=0.2, help="Tail x offset in mm")
    parser.add_argument("--tail_y_mm", type=float, default=0.0, help="Tail y offset in mm")

    parser.add_argument("--normalize_power", type=str2bool, default=True,
                        help="Normalize total power to 1")
    args = parser.parse_args()

    n = args.n
    dx = args.dx

    x = (np.arange(n) - (n - 1) / 2.0) * dx
    y = (np.arange(n) - (n - 1) / 2.0) * dx
    xx, yy = np.meshgrid(x, y)

    # Power fractions (interpreted as FRACTION OF TOTAL POWER)
    main_frac = 1.0 - args.tail_frac
    tail_frac = args.tail_frac
    
    # Correct amplitude normalization:
    # A ∝ sqrt(power_fraction) / beam_width
    A_main = np.sqrt(main_frac) / args.w_main
    A_tail = np.sqrt(tail_frac) / args.w_tail

    # Gaussian fields
    E_main = A_main * np.exp(-(xx**2 + yy**2) / (args.w_main**2))

    E_tail = A_tail * np.exp(-(((xx - args.tail_x_mm)**2 +
                                (yy - args.tail_y_mm)**2) / (args.w_tail**2)))
    E = E_main + E_tail

    if args.normalize_power:
        power = np.sum(np.abs(E)**2)
        if power > 0:
            E /= np.sqrt(power)

    out = Path(args.outfile)
    with out.open("w", encoding="utf-8") as f:
        f.write("! CODE V - complex optical field data\n")
        f.write("Datatype:  Complex\n")
        f.write(f"Wavelength:  {args.wl_nm:.6f} nm\n")
        f.write(f"Grid spacing:  {dx:.8g}  {dx:.8g} mm\n")
        f.write(f"Array size:  {n}  {n}\n")
        f.write("Coordinates:  0  0  0 mm\n")
        f.write("Direction:  0  0  1\n")

        for i in range(n):
            row_vals = []
            for j in range(n):
                row_vals.append(f"{E[i, j].real:.10e}")
                row_vals.append(f"{E[i, j].imag:.10e}")
            f.write("  ".join(row_vals) + "\n")

    print(f"Saved {out}")
    print(f"Grid size: {n} x {n}")
    print(f"Grid spacing: {dx} mm")
    print(f"Main width: {args.w_main} mm")
    print(f"Tail width: {args.w_tail} mm")
    print(f"Tail power fraction: {args.tail_frac}")
    print(f"Tail offset: ({args.tail_x_mm}, {args.tail_y_mm}) mm")


if __name__ == "__main__":
    main()