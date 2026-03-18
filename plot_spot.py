from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

def enclosed_fraction_at_radius(xx, yy, data, radius_mm, x0=0.0, y0=0.0):
    rr = np.sqrt((xx - x0)**2 + (yy - y0)**2)
    mask = rr <= radius_mm
    total = np.sum(data)
    if total <= 0:
        return np.nan
    return float(np.sum(data[mask]) / total)

def enclosed_fraction_at_radius(xx, yy, data, radius_mm, x0=0.0, y0=0.0):
    rr = np.sqrt((xx - x0)**2 + (yy - y0)**2)
    mask = rr <= radius_mm
    total = np.sum(data)
    if total <= 0:
        return np.nan
    return float(np.sum(data[mask]) / total)

def encircled_energy_radius(xx, yy, data, frac=0.99, x0=0.0, y0=0.0):
    rr = np.sqrt((xx - x0)**2 + (yy - y0)**2)
    r = rr.ravel()
    I = data.ravel()

    order = np.argsort(r)
    r_sorted = r[order]
    I_sorted = I[order]

    csum = np.cumsum(I_sorted)
    total = csum[-1]

    if total <= 0:
        return np.nan

    cfrac = csum / total
    idx = np.searchsorted(cfrac, frac, side="left")
    idx = min(idx, len(r_sorted) - 1)
    return float(r_sorted[idx])

def radial_encircled_fraction_table(xx, yy, data, x0=0.0, y0=0.0, rmax=0.02, npts=10):
    import numpy as np

    rr = np.sqrt((xx - x0)**2 + (yy - y0)**2)
    total = np.sum(data)
    if total <= 0:
        print("Total intensity is zero.")
        return

    radii = np.linspace(rmax / npts, rmax, npts)

    print("\nEncircled energy table:")
    print("Radius (mm)    Fraction enclosed")
    print("---------------------------------")

    for r in radii:
        frac = np.sum(data[rr <= r]) / total
        print(f"{r:10.6f}    {frac:10.6f}")

def str2bool(val: str) -> bool:
    v = val.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret boolean value: {val}")


def parse_range_text(text: str) -> Tuple[float, float]:
    """
    Accept forms like:
      "(-1,1)"
      "(-1, 1)"
      "-1,1"
      "-1 1"
      "[-1,1]"
    """
    s = text.strip()
    s = s.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    s = s.replace(";", ",")
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
    else:
        parts = [p.strip() for p in s.split() if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Could not parse range from {text!r}")
    return float(parts[0]), float(parts[1])


def extract_float(pattern: str, text: str, default: Optional[float] = None) -> Optional[float]:
    m = re.search(pattern, text, re.IGNORECASE)
    return float(m.group(1)) if m else default


def extract_int(pattern: str, text: str, default: Optional[int] = None) -> Optional[int]:
    m = re.search(pattern, text, re.IGNORECASE)
    return int(m.group(1)) if m else default


def load_codev_grid(path: Path) -> dict:
    text = path.read_text(errors="ignore")

    plane_label_match = re.search(r"These data represent the Beam Intensity at (.+)", text)
    plane_label = plane_label_match.group(1).strip() if plane_label_match else "Unknown"

    frac = extract_float(r"Fraction of Starting Energy:\s*([0-9Ee+\-\.]+)", text)
    dx = extract_float(r"Grid spacing:\s*([0-9Ee+\-\.]+)\s*mm", text)
    n = extract_int(r"Array Size:\s*(\d+)", text)

    off_match = re.search(
        r"Offset of grid center from beam origin \(X,Y\):\s*\(\s*([0-9Ee+\-\.]+)\s+([0-9Ee+\-\.]+)\s*\)",
        text,
    )
    off_x = float(off_match.group(1)) if off_match else 0.0
    off_y = float(off_match.group(2)) if off_match else 0.0

    if dx is None or n is None:
        raise ValueError(f"Could not parse grid spacing or array size from {path}")

    numeric_line_re = re.compile(r"^\s*[-+0-9Ee\.]+\s*(?:[\t ]+[-+0-9Ee\.]+\s*)+$")
    lines = text.splitlines()

    start_idx = None
    for i, line in enumerate(lines):
        if numeric_line_re.match(line):
            start_idx = i
            break

    if start_idx is None:
        raise ValueError(f"Could not find numeric data block in {path}")

    rows = []
    for line in lines[start_idx:]:
        s = line.strip()
        if not s:
            continue
        if not numeric_line_re.match(s):
            break
        row = [float(x) for x in re.split(r"[\t ]+", s) if x]
        rows.append(row)

    data = np.array(rows, dtype=float)
    if data.shape != (n, n):
        print(f"Warning: parsed shape {data.shape}, header says {n}x{n}")

    ny, nx = data.shape
    x = (np.arange(nx) - (nx - 1) / 2.0) * dx + off_x
    y = (np.arange(ny) - (ny - 1) / 2.0) * dx + off_y
    xx, yy = np.meshgrid(x, y)

    return {
        "plane_label": plane_label,
        "fraction_of_starting_energy": frac,
        "grid_spacing_mm": dx,
        "array_size": n,
        "offset_x_mm": off_x,
        "offset_y_mm": off_y,
        "data": data,
        "x": x,
        "y": y,
        "xx": xx,
        "yy": yy,
    }


def highest_density_threshold(data: np.ndarray, fraction: float) -> float:
    """
    Returns the intensity threshold T such that the sum of pixels with I >= T
    contains 'fraction' of the total beam energy.

    This is not a radial 99% contour; it is a highest-intensity-region contour.
    """
    flat = np.ravel(data)
    total = float(np.sum(flat))
    if total <= 0:
        return 0.0

    order = np.argsort(flat)[::-1]
    sorted_vals = flat[order]
    csum = np.cumsum(sorted_vals)
    idx = np.searchsorted(csum, fraction * total, side="left")
    idx = min(idx, len(sorted_vals) - 1)
    return float(sorted_vals[idx])


def preprocess_argv(argv: list[str]) -> list[str]:
    """
    Support convenient user forms like:
      -x=(-1,1)
      -y=(-1,1)
      -l=false
      -c=0.99
    by rewriting them to argparse-friendly long options.
    """
    out: list[str] = []
    for arg in argv:
        if arg.startswith("-x="):
            out.extend(["--xrange_text", arg.split("=", 1)[1]])
        elif arg.startswith("-y="):
            out.extend(["--yrange_text", arg.split("=", 1)[1]])
        elif arg.startswith("-l="):
            out.extend(["--log", arg.split("=", 1)[1]])
        elif arg.startswith("-c="):
            out.extend(["--contour", arg.split("=", 1)[1]])
        else:
            out.append(arg)
    return out


def main() -> None:
    argv = preprocess_argv(sys.argv[1:])

    parser = argparse.ArgumentParser(
        description="Plot a 2D colormap from a CODE V exported surface-map text file."
    )
    parser.add_argument("filepath", type=Path, help="Input CODE V surface-map text file")

    # User-friendly compact form:
    parser.add_argument("--xrange_text", type=str, default=None,
                        help='x range text, e.g. "(-1,1)"')
    parser.add_argument("--yrange_text", type=str, default=None,
                        help='y range text, e.g. "(-1,1)"')

    # Standard explicit form:
    parser.add_argument("--xrange", nargs=2, type=float, default=None,
                        metavar=("XMIN", "XMAX"), help="x range in mm")
    parser.add_argument("--yrange", nargs=2, type=float, default=None,
                        metavar=("YMIN", "YMAX"), help="y range in mm")

    parser.add_argument("--log", "-l", type=str2bool, default=False,
                        help="Use log10 color scale (true/false)")
    parser.add_argument("--contour", "-c", type=float, default=None,
                        help="Draw highest-intensity energy contour, e.g. 0.99")
    parser.add_argument("--cmap", type=str, default="viridis",
                        help="Matplotlib colormap name")
    parser.add_argument("--dpi", type=int, default=140,
                        help="Save DPI if --save is used")
    parser.add_argument("--save", type=str, default=None, help="Output PNG filename")

    args = parser.parse_args(argv)

    if args.save:
        output_png = args.save
    else:
        output_png = args.filepath.with_suffix(".png")

    g = load_codev_grid(args.filepath)

    x = g["x"]
    y = g["y"]
    xx = g["xx"]
    yy = g["yy"]
    data = g["data"]

    # Defaults
    xlim = (-1.0, 1.0)
    ylim = (-1.0, 1.0)

    if args.xrange_text is not None:
        xlim = parse_range_text(args.xrange_text)
    elif args.xrange is not None:
        xlim = (args.xrange[0], args.xrange[1])

    if args.yrange_text is not None:
        ylim = parse_range_text(args.yrange_text)
    elif args.yrange is not None:
        ylim = (args.yrange[0], args.yrange[1])

    plot_data = data.copy()
    cbar_label = "Intensity"

    if args.log:
        maxval = float(np.max(plot_data))
        floor = max(maxval * 1e-12, 1e-300)
        plot_data = np.log10(np.maximum(plot_data, floor))
        cbar_label = "log10(Intensity)"

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        plot_data,
        extent=[x[0], x[-1], y[0], y[-1]],
        origin="lower",
        aspect="equal",
        cmap=args.cmap,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    title = args.filepath.name
    if g["plane_label"] != "Unknown":
        title += f"\n{g['plane_label']}"
    ax.set_title(title)

    # Optional highest-density contour
    if args.contour is not None:
        frac = args.contour
        if not (0.0 < frac < 1.0):
            raise ValueError("--contour must be between 0 and 1, e.g. 0.99")

        thr = highest_density_threshold(data, frac)
        #cs = ax.contour(xx, yy, data, levels=[thr], colors="white", linewidths=1.2)
        # --- Estimate effective radius of contour ---
        # choose beam center
        # simplest: use geometric center (0,0)
        # better: use intensity-weighted centroid
        I_sum = np.sum(data)
        if I_sum > 0:
            x0 = np.sum(xx * data) / I_sum
            y0 = np.sum(yy * data) / I_sum
        else:
            x0, y0 = 0.0, 0.0

        radial_encircled_fraction_table(xx, yy, data, x0=x0, y0=y0, rmax=xlim[1], npts=10)
        r_enc = encircled_energy_radius(xx, yy, data, frac=frac, x0=x0, y0=y0)

        print(f"Encircled-energy radius for {frac:.3f}: {r_enc:.6f} mm")
        print(f"Beam centroid: x0={x0:.6e} mm, y0={y0:.6e} mm")

        circle = Circle((x0, y0), r_enc, fill=False, edgecolor="white", linewidth=1.5)
        ax.add_patch(circle)

        ax.text(
            0.02, 0.98,
            f"r{int(frac*100)} ≈ {r_enc:.3f} mm",
            transform=ax.transAxes,
            va="top",
            ha="left",
            color="white",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none")
        )

    print(f"File: {args.filepath}")
    print(f"Plane label: {g['plane_label']}")
    print(f"Grid spacing: {g['grid_spacing_mm']:.8g} mm")
    print(f"Array size: {g['array_size']}")
    print(f"Fraction of starting energy (header): {g['fraction_of_starting_energy']}")
    print(f"Plot x-range: {xlim}")
    print(f"Plot y-range: {ylim}")
    print(f"Log scale: {args.log}")

    plt.tight_layout()

    if args.save is not None:
        fig.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved plot to: {args.save}")

    # --- Auto save PNG ---
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    print(f"Saved image to: {output_png}")
    plt.show()


if __name__ == "__main__":
    main()