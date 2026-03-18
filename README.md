# Spatial Filter Study (CODE V + Python)

## Overview

This project models a two-lens spatial filter using CODE V's Beam Propagation Method (BPM) and Python-based analysis tools.

The goal is to understand how:
- input beam size
- pinhole diameter
- and optical configuration

affect the filtered beam quality.

---

## Method

We simulate full wave optics propagation (not ray tracing):

- The input beam is defined as a **complex field** (Gaussian + optional wide tail)
- CODE V propagates the field using **Beam Propagation Method (BPM)**
- Propagation between planes uses **FFT-based Fresnel diffraction**
- Optical elements (lenses, pinhole) modify the field at each surface

This allows realistic modeling of:
- diffraction
- spatial filtering
- beam cleanup

---

## Repository Structure

### Core scripts

- `make_beam.py`  
  Generates input beam files (CODE V BCM format)

- `analyze_beam_after_spatial_filter.py`  
  Main analysis script for computing:
  - encircled energy
  - r99 (99.9% power radius)
  - beam profiles

- `plot_spot.py`  
  Visualizes beam intensity at selected planes

- `plot_s4.py`  
  Specialized plotting for pinhole plane (surface 4)

- `plot_waist_scan.py`  
  Used for scanning and comparing beam waist / pinhole sizes

---

### CODE V model

- `spatial_filter_520nm_pass1.len`  
  Main optical design file

- `spatial_filter_520nm_pass1.env`  
  CODE V environment / settings

---

## How to Use

### 1. Generate input beam

Example:

```bash
python make_beam.py --w_main 1.0 --w_tail 10.0 --tail_frac 0.01