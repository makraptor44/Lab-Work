import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SYSTEMATIC ANGLE UNCERTAINTY
# ============================================================

dtheta_deg = 0.66667                     # degrees
dtheta = np.deg2rad(dtheta_deg)          # radians

# ============================================================
# GRATING SPACINGS
# ============================================================

d_300 = 1 / 300e3   # metres
d_80  = 1 / 80e3    # metres

# ============================================================
# 300 lines/mm DATA
# ============================================================

m_300 = np.array([-3, -2, -1, 0, 1, 2, 3])

sin_red_300 = np.array([
    -0.5877852523,
    -0.3930688213,
    -0.1972304460,
    0,
    0.1996519441,
    0.39362533495,
    0.5901306972
])

sin_mag_300 = np.array([
    -0.3894183423,
    -0.2602233652,
    -0.13110297745,
    0,
    0.1286433422,
    0.2588190451,
    0.3911273986
])

sin_cyan_300 = np.array([
    -0.4379067434,
    -0.2923717741,
    -0.14680209965,
    0,
    0.1430654192,
    0.29028464855,
    0.4369335405
])

# ============================================================
# 80 lines/mm DATA (REPLACED WITH YOUR NEW VALUES)
# ============================================================

m_80 = np.array([-3, -2, -1, 0, 1, 2, 3])

# ---- RED ----
sin_red_80 = np.array([
    -0.155572485,     # m = -3
    -0.106264071,     # m = -2
    -0.053788348,     # m = -1
    0,                # m = 0
    0.059887116,      # m = +1
    0.107420964,      # m = +2
    0.173648178       # m = +3
])

# ---- MAGENTA (VIOLET) ----
sin_mag_80 = np.array([
    -0.104239164,     # m = -3
    -0.072659710,     # m = -2
    -0.034899497,     # m = -1
    0,                # m = 0
    0.034899497,      # m = +1
    0.069756474,      # m = +2
    0.103660539       # m = +3
])

# ---- CYAN ----
sin_cyan_80 = np.array([
    -0.114359210,     # m = -3
    -0.076719028,     # m = -2
    -0.039259816,     # m = -1
    0,                # m = 0
    0.037225088,      # m = +1
    0.078459096,      # m = +2
    0.117537397       # m = +3
])

# ============================================================
# UNCERTAINTY PROPAGATION: Δsin(θ)
# ============================================================

def compute_dsin(sin_theta):
    theta = np.arcsin(sin_theta)
    cos_theta = np.cos(theta)
    return np.abs(cos_theta) * dtheta

# ============================================================
# WEIGHTED LINEAR REGRESSION
# ============================================================

def weighted_fit(x, y, dy):
    w = 1 / dy**2
    W = np.sum(w)
    Wx = np.sum(w * x)
    Wy = np.sum(w * y)
    Wxx = np.sum(w * x * x)
    Wxy = np.sum(w * x * y)

    denom = W * Wxx - Wx**2

    slope = (W * Wxy - Wx * Wy) / denom
    intercept = (Wxx * Wy - Wx * Wxy) / denom

    slope_err = np.sqrt(W / denom)

    return slope, intercept, slope_err

# ============================================================
# ANALYSIS FUNCTION
# ============================================================

def analyse_colour(m, sin_theta, name, d, label):
    dsin = compute_dsin(sin_theta)

    slope, intercept, slope_err = weighted_fit(m, sin_theta, dsin)

    wavelength = slope * d
    wavelength_err = slope_err * d

    print(f"{name} ({label})")
    print(f"Slope = {slope:.6f} ± {slope_err:.6f}")
    print(f"Wavelength = {wavelength:.3e} ± {wavelength_err:.3e}\n")

    plt.figure()
    plt.errorbar(m, sin_theta, yerr=dsin, fmt='o', capsize=5)
    plt.plot(m, slope*m + intercept)
    plt.xlabel("Diffraction order m")
    plt.ylabel("sin(θ)")
    plt.title(f"{name}: sin(θ) vs m ({label})")
    plt.grid()
    plt.show()

    return wavelength, wavelength_err

# ============================================================
# RUN ANALYSIS — 300 lines/mm
# ============================================================

lambda_red_300, err_red_300 = analyse_colour(m_300, sin_red_300, "Red", d_300, "300 lines/mm")
lambda_mag_300, err_mag_300 = analyse_colour(m_300, sin_mag_300, "Magenta", d_300, "300 lines/mm")
lambda_cyan_300, err_cyan_300 = analyse_colour(m_300, sin_cyan_300, "Cyan", d_300, "300 lines/mm")

# Rydberg fit
x_300 = np.array([-5/36, -3/16, -21/100])
inv_lambda_300 = 1 / np.array([lambda_red_300, lambda_cyan_300, lambda_mag_300])
inv_lambda_err_300 = np.array([err_red_300, err_cyan_300, err_mag_300]) / np.array([lambda_red_300, lambda_cyan_300, lambda_mag_300])**2

slope_R_300, intercept_R_300, slope_R_err_300 = weighted_fit(x_300, inv_lambda_300, inv_lambda_err_300)
rydberg_300 = -slope_R_300
rydberg_err_300 = slope_R_err_300

print("\n===== RYDBERG CONSTANT (300 lines/mm) =====")
print(f"R = {rydberg_300:.3e} ± {rydberg_err_300:.3e}\n")

# ============================================================
# RUN ANALYSIS — 80 lines/mm
# ============================================================

lambda_red_80, err_red_80 = analyse_colour(m_80, sin_red_80, "Red", d_80, "80 lines/mm")
lambda_mag_80, err_mag_80 = analyse_colour(m_80, sin_mag_80, "Magenta", d_80, "80 lines/mm")
lambda_cyan_80, err_cyan_80 = analyse_colour(m_80, sin_cyan_80, "Cyan", d_80, "80 lines/mm")

x_80 = np.array([-5/36, -3/16, -21/100])
inv_lambda_80 = 1 / np.array([lambda_red_80, lambda_cyan_80, lambda_mag_80])
inv_lambda_err_80 = np.array([err_red_80, err_cyan_80, err_mag_80]) / np.array([lambda_red_80, lambda_cyan_80, lambda_mag_80])**2

slope_R_80, intercept_R_80, slope_R_err_80 = weighted_fit(x_80, inv_lambda_80, inv_lambda_err_80)
rydberg_80 = -slope_R_80
rydberg_err_80 = slope_R_err_80

# ============================================================
# COMBINED PLOT: 1/λ vs x FOR BOTH GRATINGS
# ============================================================

plt.figure()

plt.errorbar(x_300, inv_lambda_300, yerr=inv_lambda_err_300, fmt='o', capsize=5, label='300 lines/mm')
plt.plot(x_300, slope_R_300 * x_300 + intercept_R_300, label='Fit (300 lines/mm)')

plt.errorbar(x_80, inv_lambda_80, yerr=inv_lambda_err_80, fmt='s', capsize=5, label='80 lines/mm')
plt.plot(x_80, slope_R_80 * x_80 + intercept_R_80, label='Fit (80 lines/mm)')

plt.xlabel("x value")
plt.ylabel("1 / wavelength (m⁻¹)")
plt.title("Comparison of 1/λ vs x for 300 and 80 lines/mm")
plt.grid()
plt.legend()
plt.show()

print("\n===== RYDBERG CONSTANT (80 lines/mm) =====")
print(f"R = {rydberg_80:.3e} ± {rydberg_err_80:.3e}\n")




