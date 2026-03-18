import numpy as np
import matplotlib.pyplot as plt

# Distances after lens 1 exit surface [mm]
coarse_z_mm = np.array([19.0, 20.0, 21.0, 21.5, 21.7, 22.0, 22.5, 23.5, 24.5])
coarse_w_mm = np.array([0.107496, 0.0737128, 0.0400094, 0.0232829, 0.0166904, 0.00735055, 0.0121909, 0.0451466, 0.0788607])
coarse_rms_fit = np.array([0.000576683, 0.000879719, 0.00163459, 0.00277837, 0.00368865, 0.00222371, 0.00309181, 0.00157799, 0.000979995])

fine_z_mm = np.array([21.90, 21.95, 22.00, 22.05, 22.10, 22.15, 22.20, 22.25, 22.30])
fine_w_mm = np.array([0.0103123, 0.00878896, 0.0073505, 0.00609973, 0.00519967, 0.0048969, 0.00515343, 0.00581793, 0.00677916])
fine_rms_fit = np.array([0.00372617, 0.00364325, 0.00222419, 0.00381715, 0.00372879, 0.000701476, 0.00226434, 0.00261292, 0.000430036])

# Convert beam radii to microns for plotting
coarse_w_um = 1000.0 * coarse_w_mm
fine_w_um = 1000.0 * fine_w_mm

# CODE V RMS fit error is dimensionless and not a true uncertainty.
# As a qualitative visualization only, scale it by the fitted radius.
coarse_proxy_err_um = coarse_w_um * coarse_rms_fit
fine_proxy_err_um = fine_w_um * fine_rms_fit

# Combine all points to find the global minimum
all_z = np.concatenate([coarse_z_mm, fine_z_mm])
all_w_um = np.concatenate([coarse_w_um, fine_w_um])

imin = np.argmin(all_w_um)
z_best = all_z[imin]
w_best_um = all_w_um[imin]

plt.figure(figsize=(8, 5))

# Coarse scan
plt.plot(coarse_z_mm, coarse_w_um, 'o-', label='Coarse scan')
# Fine scan
plt.plot(fine_z_mm, fine_w_um, 's-', label='Fine scan')

# Optional proxy error bars
plt.errorbar(
    coarse_z_mm, coarse_w_um, yerr=coarse_proxy_err_um,
    fmt='none', capsize=3, alpha=0.5
)
plt.errorbar(
    fine_z_mm, fine_w_um, yerr=fine_proxy_err_um,
    fmt='none', capsize=3, alpha=0.5
)

# Highlight best point
plt.plot(z_best, w_best_um, 'r*', markersize=14,
         label=f'Best waist: z={z_best:.2f} mm, w={w_best_um:.2f} µm')

plt.xlabel('Distance after lens 1 exit surface [mm]')
plt.ylabel(r'Gaussian beam radius $w$ [$\mu$m]')
plt.title('Spatial filter waist scan from CODE V FFT propagation')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print(f'Best waist location: {z_best:.2f} mm after lens 1 exit surface')
print(f'Best waist radius:   {w_best_um:.2f} µm')
print(f'Best waist diameter: {2*w_best_um:.2f} µm')