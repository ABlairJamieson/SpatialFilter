from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATA_DIR = Path(".")   # change if needed
FILE_GLOB = "s*um_d*s*.txt"

# Simple fixed radii (mm) to report
REPORT_RADII_MM = [0.25, 0.5, 1.0, 2.0]

# Outside-fraction targets for "how much radius do I need?"
TARGET_OUTSIDE_FRACTIONS = [1e-3, 1e-5]

# Whether to normalize enclosed energy to the sum of the grid
NORMALIZE_TO_GRID_SUM = True

# Main radius used in paired s8/s9 halo comparison
PAIR_COMPARE_RADIUS_MM = 0.5

# Enclosed-energy targets used as alternative beam-size measures
TARGET_ENCLOSED_FRACTIONS = [0.90, 0.95, 0.99, 0.999, 0.99999]


# ------------------------------------------------------------
# Data container
# ------------------------------------------------------------

@dataclass
class BeamGrid:
    path: Path
    slit_um: Optional[float]
    dist_mm: Optional[float]
    plane: Optional[str]          # "s8" or "s9"
    plane_label: str
    grid_spacing_mm: float
    array_size: int
    fraction_of_starting_energy: Optional[float]
    offset_x_mm: float
    offset_y_mm: float
    data: np.ndarray

    @property
    def total_grid_sum(self) -> float:
        return float(np.sum(self.data))


# ------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------

def parse_filename(path: Path) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Expected examples:
      s25um_d30p0_s8.txt
      s25um_d30p45_s9.txt
    """
    name = path.stem
    m = re.match(r"s(?P<slit>\d+(?:p\d+)?)um_d(?P<dist>\d+(?:p\d+)?)_(?P<plane>s\d+)", name)
    if not m:
        return None, None, None

    slit_um = float(m.group("slit").replace("p", "."))
    dist_mm = float(m.group("dist").replace("p", "."))
    plane = m.group("plane").lower()
    return slit_um, dist_mm, plane


def extract_float(pattern: str, text: str, default: Optional[float] = None) -> Optional[float]:
    m = re.search(pattern, text, re.IGNORECASE)
    return float(m.group(1)) if m else default


def extract_int(pattern: str, text: str, default: Optional[int] = None) -> Optional[int]:
    m = re.search(pattern, text, re.IGNORECASE)
    return int(m.group(1)) if m else default


def load_codev_grid(path: Path) -> BeamGrid:
    text = path.read_text(errors="ignore")

    slit_um, dist_mm, plane = parse_filename(path)

    plane_label_match = re.search(r"These data represent the Beam Intensity at (.+)", text)
    plane_label = plane_label_match.group(1).strip() if plane_label_match else "Unknown"

    frac = extract_float(r"Fraction of Starting Energy:\s*([0-9Ee+\-\.]+)", text)
    dx = extract_float(r"Grid spacing:\s*([0-9Ee+\-\.]+)\s*mm", text)
    n = extract_int(r"Array Size:\s*(\d+)", text)

    off_match = re.search(
        r"Offset of grid center from beam origin \(X,Y\):\s*\(\s*([0-9Ee+\-\.]+)\s*([0-9Ee+\-\.]+)\s*\)",
        text,
    )
    off_x = float(off_match.group(1)) if off_match else 0.0
    off_y = float(off_match.group(2)) if off_match else 0.0

    if dx is None or n is None:
        raise ValueError(f"Could not parse grid spacing or array size from {path}")

    lines = text.splitlines()
    start_idx = None
    numeric_line_re = re.compile(r"^\s*[-+0-9Ee\.]+\s*(?:[\t ]+[-+0-9Ee\.]+\s*)+$")

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

    if data.shape[0] != n or data.shape[1] != n:
        print(f"Warning: {path.name} parsed shape {data.shape}, header says {n}x{n}")

    return BeamGrid(
        path=path,
        slit_um=slit_um,
        dist_mm=dist_mm,
        plane=plane,
        plane_label=plane_label,
        grid_spacing_mm=dx,
        array_size=n,
        fraction_of_starting_energy=frac,
        offset_x_mm=off_x,
        offset_y_mm=off_y,
        data=data,
    )


# ------------------------------------------------------------
# Coordinate / radial helpers
# ------------------------------------------------------------

def make_xy(grid: BeamGrid) -> Tuple[np.ndarray, np.ndarray]:
    ny, nx = grid.data.shape
    x = (np.arange(nx) - (nx - 1) / 2.0) * grid.grid_spacing_mm + grid.offset_x_mm
    y = (np.arange(ny) - (ny - 1) / 2.0) * grid.grid_spacing_mm + grid.offset_y_mm
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def radial_profile_and_cumulative(grid: BeamGrid) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    xx, yy = make_xy(grid)
    rr = np.sqrt(xx**2 + yy**2)

    r = rr.ravel()
    I = grid.data.ravel()

    order = np.argsort(r)
    r_sorted = r[order]
    I_sorted = I[order]

    csum = np.cumsum(I_sorted)
    total = float(csum[-1]) if csum.size else 0.0

    if NORMALIZE_TO_GRID_SUM and total > 0:
        cfrac = csum / total
    else:
        cfrac = csum

    return r_sorted, I_sorted, cfrac, total


def enclosed_fraction_at_radius(grid: BeamGrid, radius_mm: float) -> float:
    r_sorted, _, cfrac, _ = radial_profile_and_cumulative(grid)
    idx = np.searchsorted(r_sorted, radius_mm, side="right") - 1
    if idx < 0:
        return 0.0
    idx = min(idx, len(cfrac) - 1)
    return float(cfrac[idx])


def outside_fraction_at_radius(grid: BeamGrid, radius_mm: float) -> float:
    return 1.0 - enclosed_fraction_at_radius(grid, radius_mm)


def radius_for_enclosed_fraction(grid: BeamGrid, target_fraction: float) -> float:
    """
    Smallest radius whose enclosed fraction reaches target_fraction.
    Example: target_fraction=0.999 gives radius containing 99.9% of grid energy.
    """
    if not (0.0 <= target_fraction <= 1.0):
        raise ValueError("target_fraction must be between 0 and 1")

    r_sorted, _, cfrac, _ = radial_profile_and_cumulative(grid)
    if len(r_sorted) == 0:
        return np.nan

    idx = np.searchsorted(cfrac, target_fraction, side="left")
    idx = min(idx, len(r_sorted) - 1)
    return float(r_sorted[idx])


def radius_for_outside_fraction(grid: BeamGrid, target_outside_fraction: float) -> float:
    """
    Smallest radius whose outside fraction is <= target_outside_fraction.
    Example: target_outside_fraction=1e-3 gives radius containing 99.9%.
    """
    if not (0.0 <= target_outside_fraction <= 1.0):
        raise ValueError("target_outside_fraction must be between 0 and 1")
    return radius_for_enclosed_fraction(grid, 1.0 - target_outside_fraction)


# ------------------------------------------------------------
# Reporting helpers
# ------------------------------------------------------------

def summarize_grid(grid: BeamGrid) -> None:
    print(f"\nFile: {grid.path.name}")
    print(f"  plane_label = {grid.plane_label}")
    print(f"  slit_um     = {grid.slit_um}")
    print(f"  dist_mm     = {grid.dist_mm}")
    print(f"  plane       = {grid.plane}")
    print(f"  grid dx     = {grid.grid_spacing_mm:.8g} mm")
    print(f"  size        = {grid.data.shape}")
    print(f"  start frac  = {grid.fraction_of_starting_energy}")
    print(f"  grid sum    = {grid.total_grid_sum:.8g}")

    print("  fixed-radius metrics:")
    for rmm in REPORT_RADII_MM:
        fin = enclosed_fraction_at_radius(grid, rmm)
        fout = 1.0 - fin
        print(f"    inside {rmm:>7.4f} mm : {fin:12.8f}   outside: {fout:12.8f}")

    print("  radius for target enclosed fractions:")
    for fin_target in TARGET_ENCLOSED_FRACTIONS:
        r = radius_for_enclosed_fraction(grid, fin_target)
        print(f"    enclosed {fin_target:>9.6f} -> r = {r:10.6f} mm")

    print("  radius for target outside fractions:")
    for fout_target in TARGET_OUTSIDE_FRACTIONS:
        r = radius_for_outside_fraction(grid, fout_target)
        print(f"    outside < {fout_target:>8.1e} -> r = {r:10.6f} mm")


def compare_pair(g8: BeamGrid, g9: BeamGrid) -> Dict[str, float]:
    """
    Compare s8 and s9 as a two-plane pair for one optical case.
    """
    result: Dict[str, float] = {
        "slit_um": g8.slit_um if g8.slit_um is not None else np.nan,
        "dist_mm": g8.dist_mm if g8.dist_mm is not None else np.nan,
    }

    # Halo growth at one chosen radius
    f8 = outside_fraction_at_radius(g8, PAIR_COMPARE_RADIUS_MM)
    f9 = outside_fraction_at_radius(g9, PAIR_COMPARE_RADIUS_MM)
    result["outside_s8_at_pair_radius"] = f8
    result["outside_s9_at_pair_radius"] = f9
    result["delta_outside_at_pair_radius"] = f9 - f8

    # Enclosed growth at one chosen radius
    result["inside_s8_at_pair_radius"] = 1.0 - f8
    result["inside_s9_at_pair_radius"] = 1.0 - f9

    # r(f_enclosed) growth
    for fin_target in TARGET_ENCLOSED_FRACTIONS:
        key = f"{fin_target:.6f}".replace(".", "p")
        r8 = radius_for_enclosed_fraction(g8, fin_target)
        r9 = radius_for_enclosed_fraction(g9, fin_target)
        result[f"r_enclosed_{key}_s8"] = r8
        result[f"r_enclosed_{key}_s9"] = r9
        result[f"r_enclosed_{key}_growth"] = r9 - r8

    # r(f_outside) growth
    for fout_target in TARGET_OUTSIDE_FRACTIONS:
        key = f"{fout_target:.0e}".replace("-", "m")
        r8 = radius_for_outside_fraction(g8, fout_target)
        r9 = radius_for_outside_fraction(g9, fout_target)
        result[f"r_outside_{key}_s8"] = r8
        result[f"r_outside_{key}_s9"] = r9
        result[f"r_outside_{key}_growth"] = r9 - r8

    return result


def print_pair_report(g8: BeamGrid, g9: BeamGrid) -> None:
    pair = compare_pair(g8, g9)

    slit = pair["slit_um"]
    dist = pair["dist_mm"]

    print("\n" + "=" * 72)
    print(f"PAIR COMPARISON: slit={slit} um, spacing={dist} mm")
    print("=" * 72)

    print(f"  At radius {PAIR_COMPARE_RADIUS_MM:.4f} mm:")
    print(f"    outside(s8) = {pair['outside_s8_at_pair_radius']:.8f}")
    print(f"    outside(s9) = {pair['outside_s9_at_pair_radius']:.8f}")
    print(f"    delta       = {pair['delta_outside_at_pair_radius']:+.8f}")
    print("    interpretation: positive delta means more halo at s9 than s8")

    print("  Growth in containment radii:")
    for fin_target in TARGET_ENCLOSED_FRACTIONS:
        key = f"{fin_target:.6f}".replace(".", "p")
        r8 = pair[f"r_enclosed_{key}_s8"]
        r9 = pair[f"r_enclosed_{key}_s9"]
        dg = pair[f"r_enclosed_{key}_growth"]
        print(f"    enclosed {fin_target:>9.6f}: s8={r8:10.6f} mm  s9={r9:10.6f} mm  growth={dg:+10.6f} mm")

    print("  Radius needed for small outside fractions:")
    for fout_target in TARGET_OUTSIDE_FRACTIONS:
        key = f"{fout_target:.0e}".replace("-", "m")
        r8 = pair[f"r_outside_{key}_s8"]
        r9 = pair[f"r_outside_{key}_s9"]
        dg = pair[f"r_outside_{key}_growth"]
        print(f"    outside < {fout_target:>8.1e}: s8={r8:10.6f} mm  s9={r9:10.6f} mm  growth={dg:+10.6f} mm")


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def plot_radial_enclosed_energy(grids: List[BeamGrid], title: str = "Radial enclosed energy") -> None:
    plt.figure(figsize=(7, 5))
    for g in grids:
        r_sorted, _, cfrac, _ = radial_profile_and_cumulative(g)
        label = f"{g.path.stem} ({g.plane_label})"
        plt.plot(r_sorted, cfrac, label=label)

    plt.xlabel("Radius from beam center (mm)")
    plt.ylabel("Enclosed fraction of grid energy")
    plt.title(title)
    plt.xlim(left=0)
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()


def plot_outside_fraction_vs_radius(grids: List[BeamGrid], radii_mm: List[float]) -> None:
    plt.figure(figsize=(7, 5))
    for g in grids:
        vals = [outside_fraction_at_radius(g, r) for r in radii_mm]
        plt.plot(radii_mm, vals, label=g.path.stem)

    plt.xlabel("Chosen beam radius (mm)")
    plt.ylabel("Fraction outside radius")
    plt.title("Light outside chosen beam radius")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()


def plot_case_scan(cases: Dict[Tuple[Optional[float], Optional[float]], Dict[str, BeamGrid]],
                   report_radius_mm: float = 0.5) -> None:
    """
    For all cases, plot fixed-radius outside fraction vs spacing for s8 and s9.
    """
    xs8, ys8 = [], []
    xs9, ys9 = [], []

    for (slit_um, dist_mm), planes in sorted(cases.items(), key=lambda kv: ((kv[0][0] or -1), (kv[0][1] or -1))):
        if dist_mm is None:
            continue
        if "s8" in planes:
            xs8.append(dist_mm)
            ys8.append(outside_fraction_at_radius(planes["s8"], report_radius_mm))
        if "s9" in planes:
            xs9.append(dist_mm)
            ys9.append(outside_fraction_at_radius(planes["s9"], report_radius_mm))

    plt.figure(figsize=(7, 5))
    if xs8:
        plt.plot(xs8, ys8, marker="o", label=f"S8 outside {report_radius_mm:.3f} mm")
    if xs9:
        plt.plot(xs9, ys9, marker="o", label=f"S9 outside {report_radius_mm:.3f} mm")

    plt.xlabel("Pinhole/slit to lens2 distance (mm)")
    plt.ylabel("Fraction outside chosen radius")
    plt.title("Single-plane halo metric vs spacing")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


def plot_pair_growth_vs_spacing(cases: Dict[Tuple[Optional[float], Optional[float]], Dict[str, BeamGrid]],
                                target_outside_fraction: float = 1e-3) -> None:
    """
    Plot how the radius needed to contain all but target_outside_fraction grows from s8 to s9.
    """
    xs = []
    growths = []

    for (slit_um, dist_mm), planes in sorted(cases.items(), key=lambda kv: ((kv[0][0] or -1), (kv[0][1] or -1))):
        if dist_mm is None or "s8" not in planes or "s9" not in planes:
            continue
        r8 = radius_for_outside_fraction(planes["s8"], target_outside_fraction)
        r9 = radius_for_outside_fraction(planes["s9"], target_outside_fraction)
        xs.append(dist_mm)
        growths.append(r9 - r8)

    if not xs:
        return

    plt.figure(figsize=(7, 5))
    plt.plot(xs, growths, marker="o")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Pinhole/slit to lens2 distance (mm)")
    plt.ylabel(f"Growth in r(outside<{target_outside_fraction:.0e}) from s8 to s9 (mm)")
    plt.title("Two-plane halo-growth metric vs spacing")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_pair_curves(g8: BeamGrid, g9: BeamGrid, title_prefix: str = "") -> None:
    """
    Plot the s8 and s9 enclosed-energy curves for one matched case.
    """
    plt.figure(figsize=(7, 5))
    for g in [g8, g9]:
        r_sorted, _, cfrac, _ = radial_profile_and_cumulative(g)
        plt.plot(r_sorted, cfrac, label=g.path.stem)

    slit = g8.slit_um
    dist = g8.dist_mm
    title = f"{title_prefix} slit={slit} um, spacing={dist} mm"
    plt.xlabel("Radius from beam center (mm)")
    plt.ylabel("Enclosed fraction")
    plt.title(title.strip())
    plt.xlim(left=0)
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    files = sorted(DATA_DIR.glob(FILE_GLOB))
    if not files:
        print(f"No files found matching {FILE_GLOB} in {DATA_DIR.resolve()}")
        return

    grids: List[BeamGrid] = []
    for f in files:
        try:
            g = load_codev_grid(f)
            grids.append(g)
        except Exception as e:
            print(f"Skipping {f.name}: {e}")

    if not grids:
        print("No valid grids loaded.")
        return

    print(f"Loaded {len(grids)} files.")

    # Single-file summaries
    for g in grids:
        summarize_grid(g)

    # Group by optical case
    cases: Dict[Tuple[Optional[float], Optional[float]], Dict[str, BeamGrid]] = {}
    for g in grids:
        key = (g.slit_um, g.dist_mm)
        cases.setdefault(key, {})
        if g.plane is not None:
            cases[key][g.plane] = g

    # Paired s8/s9 reports
    print("\n" + "#" * 72)
    print("PAIRED CASE REPORTS")
    print("#" * 72)
    for key in sorted(cases.keys(), key=lambda k: ((k[0] or -1), (k[1] or -1))):
        planes = cases[key]
        if "s8" in planes and "s9" in planes:
            print_pair_report(planes["s8"], planes["s9"])

    # Global plots
    plot_radial_enclosed_energy(grids, title="Radial enclosed energy for exported CODE V planes")

    dense_radii = np.linspace(0.01, 3.0, 250)
    plot_outside_fraction_vs_radius(grids, dense_radii.tolist())

    if len(cases) > 1:
        plot_case_scan(cases, report_radius_mm=PAIR_COMPARE_RADIUS_MM)
        for fout in TARGET_OUTSIDE_FRACTIONS:
            plot_pair_growth_vs_spacing(cases, target_outside_fraction=fout)

    # Also plot paired s8/s9 curves case by case
    for key in sorted(cases.keys(), key=lambda k: ((k[0] or -1), (k[1] or -1))):
        planes = cases[key]
        if "s8" in planes and "s9" in planes:
            plot_pair_curves(planes["s8"], planes["s9"], title_prefix="Two-plane enclosed-energy comparison:")

    plt.show()


if __name__ == "__main__":
    main()