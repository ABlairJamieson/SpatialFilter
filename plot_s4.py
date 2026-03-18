from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


FILEPATH = Path("s4.txt")

# zoom radius for close-up plots, in mm
ZOOM_RADIUS_MM = 0.10   # 0.10 mm = 100 um

# radii to annotate on enclosed-energy plot, in mm
REPORT_RADII_MM = [0.005, 0.0075, 0.010, 0.0125, 0.015, 0.020, 0.025, 0.050, 0.100]


def extract_float(pattern: str, text: str, default=None):
    m = re.search(pattern, text, re.IGNORECASE)
    return float(m.group(1)) if m else default


def extract_int(pattern: str, text: str, default=None):
    m = re.search(pattern, text, re.IGNORECASE)
    return int(m.group(1)) if m else default


def load_codev_grid(path: Path):
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
        raise ValueError("Could not parse grid spacing or array size.")

    lines = text.splitlines()
    numeric_line_re = re.compile(r"^\s*[-+0-9Ee\.]+\s*(?:[\t ]+[-+0-9Ee\.]+\s*)+$")

    start_idx = None
    for i, line in enumerate(lines):
        if numeric_line_re.match(line):
            start_idx = i
            break

    if start_idx is None:
        raise ValueError("Could not find numeric data block.")

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

    return {
        "plane_label": plane_label,
        "fraction_of_starting_energy": frac,
        "grid_spacing_mm": dx,
        "array_size": n,
        "offset_x_mm": off_x,
        "offset_y_mm": off_y,
        "data": data,
    }


def make_xy(data: np.ndarray, dx_mm: float, off_x_mm: float, off_y_mm: float):
    ny, nx = data.shape
    x = (np.arange(nx) - (nx - 1) / 2.0) * dx_mm + off_x_mm
    y = (np.arange(ny) - (ny - 1) / 2.0) * dx_mm + off_y_mm
    xx, yy = np.meshgrid(x, y)
    return x, y, xx, yy


def radial_analysis(data: np.ndarray, xx: np.ndarray, yy: np.ndarray):
    rr = np.sqrt(xx**2 + yy**2)

    r = rr.ravel()
    I = data.ravel()

    order = np.argsort(r)
    r_sorted = r[order]
    I_sorted = I[order]

    csum = np.cumsum(I_sorted)
    total = csum[-1] if len(csum) else 0.0
    enclosed = csum / total if total > 0 else csum

    return r_sorted, I_sorted, enclosed, total


def enclosed_fraction_at_radius(r_sorted, enclosed, radius_mm: float):
    idx = np.searchsorted(r_sorted, radius_mm, side="right") - 1
    if idx < 0:
        return 0.0
    idx = min(idx, len(enclosed) - 1)
    return float(enclosed[idx])


def radial_mean_profile(data: np.ndarray, xx: np.ndarray, yy: np.ndarray, dr_mm: float):
    rr = np.sqrt(xx**2 + yy**2)
    rmax = rr.max()

    edges = np.arange(0.0, rmax + dr_mm, dr_mm)
    centers = 0.5 * (edges[:-1] + edges[1:])

    r_flat = rr.ravel()
    i_flat = data.ravel()

    which_bin = np.digitize(r_flat, edges) - 1

    prof = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        mask = which_bin == i
        if np.any(mask):
            prof[i] = np.mean(i_flat[mask])

    return centers, prof


def main():
    g = load_codev_grid(FILEPATH)
    data = g["data"]
    dx_mm = g["grid_spacing_mm"]
    off_x_mm = g["offset_x_mm"]
    off_y_mm = g["offset_y_mm"]

    x, y, xx, yy = make_xy(data, dx_mm, off_x_mm, off_y_mm)
    r_sorted, I_sorted, enclosed, total = radial_analysis(data, xx, yy)

    print(f"File: {FILEPATH}")
    print(f"Plane label: {g['plane_label']}")
    print(f"Grid spacing: {dx_mm:.8f} mm")
    print(f"Array size: {g['array_size']}")
    print(f"Fraction of starting energy from header: {g['fraction_of_starting_energy']}")
    print(f"Grid sum: {total:.8e}")

    print("\nTransmitted fraction inside radius:")
    for rmm in REPORT_RADII_MM:
        fin = enclosed_fraction_at_radius(r_sorted, enclosed, rmm)
        print(f"  r = {rmm*1000:7.2f} um  ->  inside = {fin:10.6f}   outside = {1-fin:10.6f}")

    # full 2D image
    plt.figure(figsize=(7, 6))
    plt.imshow(
        data,
        extent=[x[0], x[-1], y[0], y[-1]],
        origin="lower",
        aspect="equal",
    )
    plt.colorbar(label="Intensity")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Beam intensity at pinhole plane (full grid)")
    plt.tight_layout()

    # zoomed 2D image
    plt.figure(figsize=(7, 6))
    plt.imshow(
        data,
        extent=[x[0], x[-1], y[0], y[-1]],
        origin="lower",
        aspect="equal",
    )
    plt.colorbar(label="Intensity")
    plt.xlim(-ZOOM_RADIUS_MM, ZOOM_RADIUS_MM)
    plt.ylim(-ZOOM_RADIUS_MM, ZOOM_RADIUS_MM)
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title(f"Beam intensity at pinhole plane (zoom to ±{ZOOM_RADIUS_MM*1000:.0f} um)")
    plt.tight_layout()

    # zoomed 2D image with log color
    plt.figure(figsize=(7, 6))
    plt.imshow(
        np.log10(np.maximum(data, np.max(data) * 1e-12)),
        extent=[x[0], x[-1], y[0], y[-1]],
        origin="lower",
        aspect="equal",
    )
    plt.colorbar(label="log10(Intensity)")
    plt.xlim(-ZOOM_RADIUS_MM, ZOOM_RADIUS_MM)
    plt.ylim(-ZOOM_RADIUS_MM, ZOOM_RADIUS_MM)
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title(f"Pinhole plane log-intensity (zoom to ±{ZOOM_RADIUS_MM*1000:.0f} um)")
    plt.tight_layout()

    # radial mean profile
    dr_mm = dx_mm
    r_centers, prof = radial_mean_profile(data, xx, yy, dr_mm)

    plt.figure(figsize=(7, 5))
    plt.plot(r_centers * 1000.0, prof, marker=".", linewidth=1)
    plt.xlabel("Radius (um)")
    plt.ylabel("Mean intensity in annulus")
    plt.title("Radial mean intensity profile")
    plt.xlim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # log radial profile
    plt.figure(figsize=(7, 5))
    plt.plot(r_centers * 1000.0, np.maximum(prof, np.nanmax(prof) * 1e-12), marker=".", linewidth=1)
    plt.yscale("log")
    plt.xlabel("Radius (um)")
    plt.ylabel("Mean intensity in annulus")
    plt.title("Radial mean intensity profile (log scale)")
    plt.xlim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # enclosed energy
    plt.figure(figsize=(7, 5))
    plt.plot(r_sorted * 1000.0, enclosed, linewidth=1.2)
    for rmm in REPORT_RADII_MM:
        fin = enclosed_fraction_at_radius(r_sorted, enclosed, rmm)
        plt.axvline(rmm * 1000.0, linestyle="--", linewidth=0.8, alpha=0.5)
        plt.text(rmm * 1000.0, min(fin + 0.03, 0.98), f"{rmm*1000:.1f} um", rotation=90, va="bottom", ha="right")
    plt.xlabel("Radius (um)")
    plt.ylabel("Enclosed fraction")
    plt.title("Encircled energy at pinhole plane")
    plt.xlim(0, 100)
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()