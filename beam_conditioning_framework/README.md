# Beam Conditioning Student Framework

This is a simplified, split version of the beam-conditioning / spatial-filter simulation.

## Files

- `beam_conditioning_core.py` — reusable library code: fields, lenses, propagation, layouts, plotting, scans, and optional optimizer.
- `student_common.py` — shared student-facing helpers and default input beam.
- `demo_single_propagation.py` — start here. Runs one fixed configuration and saves plots/PDFs.
- `demo_subsystem_comparison.py` — compares increasingly complicated subsystems to see what helps or hurts.
- `demo_fixed_aperture_scan.py` — fixed lens positions; scans only aperture radii.
- `demo_optimize.py` — advanced only; continuous optimizer requiring SciPy.
- `beam_conditioning_manual.tex` — short LaTeX manual / study plan.
- `outputs/` — scripts write plots, PDFs, CSVs, and summaries here.

## Suggested order

```powershell
python .\demo_single_propagation.py
python .\demo_subsystem_comparison.py
python .\demo_fixed_aperture_scan.py
```

Do not start with the optimizer. It is intentionally separated because it can find mathematically improved but physically confusing configurations.

## Dependencies

Core demos require:

```powershell
python -m pip install numpy matplotlib
```

The advanced optimizer also requires:

```powershell
python -m pip install scipy
```

## Main metric

The primary metric is the fraction of transmitted optical power outside a radius of 1.5 mm at z = 1000 mm. The secondary metric is throughput. Prefer configurations with throughput above 90%, or above 75% only if the containment improvement is large.
