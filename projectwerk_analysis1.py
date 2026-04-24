import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import re
from scipy.ndimage import gaussian_filter1d
import matplotlib.patches as mpatches

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =========================
# ROI INTENSITY (mean + SEM from pixel variation)
# =========================
def roi_intensity(file):
    img = tiff.imread(file)
    h, w = img.shape
    roi = img[h//3:2*h//3, w//3:2*w//3]
    roi = roi - np.min(roi)
    mean_intensity = np.mean(roi)
    sem_intensity = np.std(roi) / np.sqrt(roi.size)  # SEM from pixels
    return mean_intensity, sem_intensity

# =========================
# CONCENTRATION EXTRACTION
# =========================
def extract_low(file):
    match = re.search(r"(\d+)mul", file)
    return float(match.group(1)) if match else np.nan

def extract_high(file):
    nums = re.findall(r"(\d+)mul", file)
    nums = [float(n) for n in nums]
    return nums[-1] if len(nums) >= 2 else nums[0]

# =========================
# FILES
# =========================
low_files = [
    "zuiver_C307.tif",
    "C307_Rh110_5mul.tif",
    "C307_Rh110_10mul.tif",
    "C307_Rh110_15mul.tif",
    "C307_Rh110_25mul.tif",
    "C307_Rh110_45mul.tif",
    "C307_Rh110_65mul.tif",
    "C307_Rh110_85mul.tif"
]

high_files = [
    "C307_Rh110_85mul_5mul.tif",
    "C307_Rh110_85mul_10mul.tif",
    "C307_Rh110_85mul_20mul.tif",
    "C307_Rh110_85mul_30mul.tif",
    "C307_Rh110_85mul_50mul.tif",
    "C307_Rh110_85mul_70mul.tif"
]

# =========================
# DATA (mean + SEM)
# =========================
I_low, sem_low = zip(*[roi_intensity(f) for f in low_files])
I_high, sem_high = zip(*[roi_intensity(f) for f in high_files])

I_low = np.array(I_low)
sem_low = np.array(sem_low)
I_high = np.array(I_high)
sem_high = np.array(sem_high)

x_low = np.array([extract_low(f) for f in low_files])
x_high = np.array([extract_high(f) for f in high_files])

# =========================
# FRET EFFICIENCY + propagated SEM
# =========================
intensity_map = {f: roi_intensity(f)[0] for f in low_files}
I0 = intensity_map["zuiver_C307.tif"]
sigma_I0 = sem_low[0]  # uncertainty in reference

E_low = 1 - I_low / I0
E_high = 1 - I_high / I0

# Propagation of error: sqrt((σ_I / I0)^2 + (I * σ_I0 / I0^2)^2)
sigma_E_low = np.sqrt((sem_low / I0)**2 + (I_low * sigma_I0 / I0**2)**2)
sigma_E_high = np.sqrt((sem_high / I0)**2 + (I_high * sigma_I0 / I0**2)**2)

# =========================
# Smoothed trend
# =========================
E_low_s = gaussian_filter1d(E_low, 0.8)
E_high_s = gaussian_filter1d(E_high, 0.8)

# =========================
# Sorting
# =========================
low_sorted = sorted(zip(x_low, E_low, E_low_s, sigma_E_low))
high_sorted = sorted(zip(x_high, E_high, E_high_s, sigma_E_high))

x_low, E_low, E_low_s, sigma_E_low = map(np.array, zip(*low_sorted))
x_high, E_high, E_high_s, sigma_E_high = map(np.array, zip(*high_sorted))

# =========================
# Y-LIMITS
# =========================
def nice_ylim(data, pad=0.03):
    return np.min(data) - pad, np.max(data) + pad

# --- Normalized Hill function ---
def hill_norm(x, EC50, n):
    return x**n / (EC50**n + x**n)

# --- Fit normalized data ---
popt_low_norm, _ = curve_fit(hill_norm, x_low[1:], E_low[1:], p0=[50, 2], bounds=(0, np.inf))
EC50_low, n_low = popt_low_norm

popt_high_norm, _ = curve_fit(hill_norm, x_high[1:], E_high[1:], p0=[50, 1], bounds=(0, np.inf))
EC50_high, n_high = popt_high_norm

# --- Smooth x-values for plotting ---
x_smooth_low = np.linspace(min(x_low[1:]), max(x_low[1:]), 200)
x_smooth_high = np.linspace(min(x_high[1:]), max(x_high[1:]), 200)

# =========================
# BOLTZMANN SIGMOID FUNCTION
# =========================
def boltzmann_sigmoid(x, x0, k):
    """x0 = midpoint, k = slope factor"""
    return 1 / (1 + np.exp(-(x - x0)/k))

# --- Fit Boltzmann ---
popt_low_bolt, _ = curve_fit(boltzmann_sigmoid, x_low[1:], E_low[1:], p0=[50, 5], bounds=(0, np.inf))
x0_low, k_low = popt_low_bolt

popt_high_bolt, _ = curve_fit(boltzmann_sigmoid, x_high[1:], E_high[1:], p0=[50, 5], bounds=(0, np.inf))
x0_high, k_high = popt_high_bolt

# --- Smooth curves (for plotting) ---
y_bolt_low_smooth = boltzmann_sigmoid(x_smooth_low, *popt_low_bolt)
y_bolt_high_smooth = boltzmann_sigmoid(x_smooth_high, *popt_high_bolt)

# --- Fit evaluated at data points (for R²) ---
y_bolt_low_fit = boltzmann_sigmoid(x_low[1:], *popt_low_bolt)
y_bolt_high_fit = boltzmann_sigmoid(x_high[1:], *popt_high_bolt)

# =========================
# R² FUNCTION
# =========================
def compute_r2(y_data, y_fit):
    ss_res = np.sum((y_data - y_fit)**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    return 1 - ss_res / ss_tot

# --- R² for LOW ---
y_hill_low = hill_norm(x_low[1:], *popt_low_norm)
y_bolt_low = boltzmann_sigmoid(x_low[1:], *popt_low_bolt)

r2_hill_low = compute_r2(E_low[1:], y_hill_low)
r2_bolt_low = compute_r2(E_low[1:], y_bolt_low_fit)

# --- R² for HIGH ---
y_hill_high = hill_norm(x_high[1:], *popt_high_norm)
y_bolt_high = boltzmann_sigmoid(x_high[1:], *popt_high_bolt)

r2_hill_high = compute_r2(E_high[1:], y_hill_high)
r2_bolt_high = compute_r2(E_high[1:], y_bolt_high_fit)

r2_smooth_low = compute_r2(E_low, E_low_s)
r2_smooth_high = compute_r2(E_high, E_high_s)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# =========================
# LOW RMSE
# =========================
rmse_hill_low = rmse(E_low[1:], y_hill_low)
rmse_bolt_low = rmse(E_low[1:], y_bolt_low_fit)
rmse_smooth_low = rmse(E_low, E_low_s)

# =========================
# HIGH RMSE
# =========================
rmse_hill_high = rmse(E_high[1:], y_hill_high)
rmse_bolt_high = rmse(E_high[1:], y_bolt_high_fit)
rmse_smooth_high = rmse(E_high, E_high_s)

# =========================
# PLOT
# =========================
plt.style.use("seaborn-v0_8-whitegrid")
fig, axs = plt.subplots(1, 2, figsize=(13, 5))

# LOW SERIES
ax = axs[0]
ax.axhspan(0, 1, color="green", alpha=0.08)
ax.axhspan(-1, 0, color="red", alpha=0.08)
ax.errorbar(x_low, E_low, yerr=sigma_E_low, fmt='o-', linewidth=2, capsize=3, label="FRET efficiency")
ax.plot(
    x_low, E_low_s,
    "--",
    linewidth=2,
    label=f"Trend (smoothed, R²={r2_smooth_low:.3f})"
)
ax.plot(x_smooth_low, hill_norm(x_smooth_low, *popt_low_norm), ':', color='black',
        linewidth=2, label=f"Hill (EC50={EC50_low:.1f}, n={n_low:.2f}, R²={r2_hill_low:.3f})")
bolt_line_low, = ax.plot(x_smooth_low, y_bolt_low_smooth, '-.', color='purple',
                         linewidth=2, label=f"Boltzmann (x0={x0_low:.1f}, k={k_low:.1f}, R²={r2_bolt_low:.3f})")
#ax.axvline(EC50_low, color='black', linestyle=':', alpha=0.5)
#ax.axvline(x0_low, color='purple', linestyle=':', alpha=0.5)
ax.text(
    0.02, 0.98,
    f"RMSE\n"
    f"Hill: {rmse_hill_low:.3f}\n"
    f"Boltz: {rmse_bolt_low:.3f}\n"
    f"Smooth: {rmse_smooth_low:.3f}",
    transform=ax.transAxes,
    va="top",
    fontsize=9,
    bbox=dict(facecolor="white", edgecolor="black", alpha=0.85)
)
ax.set_title("Low Rh110 concentration series")
ax.set_xlabel("Low Rh110 added (µL)")
ax.set_ylabel("FRET efficiency")
ax.set_ylim(*nice_ylim(E_low))

# HIGH SERIES
ax = axs[1]
ax.axhspan(0, 1, color="green", alpha=0.08)
ax.axhspan(-1, 0, color="red", alpha=0.08)
ax.errorbar(x_high, E_high, yerr=sigma_E_high, fmt='o-', linewidth=2, capsize=3, label="FRET efficiency")
ax.plot(
    x_high, E_high_s,
    "--",
    linewidth=2,
    label=f"Trend (smoothed, R²={r2_smooth_high:.3f})"
)
ax.plot(x_smooth_high, hill_norm(x_smooth_high, *popt_high_norm), ':', color='black',
        linewidth=2, label=f"Hill (EC50={EC50_high:.1f}, n={n_high:.2f}, R²={r2_hill_high:.3f})")
bolt_line_high, = ax.plot(x_smooth_high, y_bolt_high_smooth, '-.', color='purple',
                          linewidth=2, label=f"Boltzmann (x0={x0_high:.1f}, k={k_high:.1f}, R²={r2_bolt_high:.3f})")
#ax.axvline(EC50_high, color='black', linestyle=':', alpha=0.5)
#ax.axvline(x0_high, color='purple', linestyle=':', alpha=0.5)
ax.text(
    0.02, 0.98,
    f"RMSE\n"
    f"Hill: {rmse_hill_high:.3f}\n"
    f"Boltz: {rmse_bolt_high:.3f}\n"
    f"Smooth: {rmse_smooth_high:.3f}",
    transform=ax.transAxes,
    va="top",
    fontsize=9,
    bbox=dict(facecolor="white", edgecolor="black", alpha=0.85)
)
ax.set_title("High Rh110 concentration series")
ax.set_xlabel("High Rh110 added (µL)")
ax.set_ylim(*nice_ylim(E_high))

# LEGEND
zone_pos = mpatches.Patch(color="green", alpha=0.08, label="Favorable (E > 0)")
zone_neg = mpatches.Patch(color="red", alpha=0.08, label="Unfavorable (E < 0)")

for ax in axs:
    ax.legend(handles=[zone_pos, zone_neg, *ax.get_legend_handles_labels()[0]],
              title="Legend", loc="lower right", frameon=True, facecolor="white",
              edgecolor="black", framealpha=0.9)

# GLOBAL TITLE
fig.suptitle("FRET efficiency vs Rh110 concentration", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
# =========================
# RESIDUALS
# =========================
res_hill_low = E_low[1:] - y_hill_low
res_bolt_low = E_low[1:] - y_bolt_low_fit
res_smooth_low = E_low - E_low_s

res_hill_high = E_high[1:] - y_hill_high
res_bolt_high = E_high[1:] - y_bolt_high_fit
res_smooth_high = E_high - E_high_s

# rmse_hill_low = rmse(E_low[1:], y_hill_low)
# rmse_bolt_low = rmse(E_low[1:], y_bolt_low_fit)
# rmse_smooth_low = rmse(E_low, E_low_s)

# rmse_hill_high = rmse(E_high[1:], y_hill_high)
# rmse_bolt_high = rmse(E_high[1:], y_bolt_high_fit)
# rmse_smooth_high = rmse(E_high, E_high_s)

def add_rmse_box(ax, rmse_hill, rmse_bolt, rmse_smooth):
    text = (
        f"RMSE:\n"
        f"Hill: {rmse_hill:.4f}\n"
        f"Boltz: {rmse_bolt:.4f}\n"
        f"Smooth: {rmse_smooth:.4f}"
    )

    ax.text(
        0.02, 0.98,
        text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="black",
            alpha=0.85
        )
    )

# =========================
# FIGURE
# =========================
fig, axs = plt.subplots(1, 2, figsize=(13, 4))

pos_patch = mpatches.Patch(color="green", alpha=0.10,
                            label="Positive residual")
neg_patch = mpatches.Patch(color="red", alpha=0.10,
                            label="Negative residual")
zero_line = Line2D(
    [0], [0],
    color="black",
    linestyle="--",
    label="No residual"
)

# helper: force full background coverage
def set_full_background(ax, ymin, ymax):
    ax.set_ylim(ymin, ymax)
    ax.axhspan(ymin, 0, color="red", alpha=0.10, zorder=0)
    ax.axhspan(0, ymax, color="green", alpha=0.10, zorder=0)
    ax.axhline(0, color='black', linestyle='--', alpha=0.6)

# =========================
# LOW
# =========================
ax = axs[0]

all_low = np.concatenate([res_hill_low, res_bolt_low, res_smooth_low])

ymin, ymax = np.nanmin(all_low), np.nanmax(all_low)

# EXPAND RANGE (THIS IS THE FIX)
pad = 0.05 * (ymax - ymin)
ymin -= pad
ymax += pad

set_full_background(ax, ymin, ymax)

h1, = ax.plot(x_low[1:], res_hill_low, 'o-', label="Hill")
h2, = ax.plot(x_low[1:], res_bolt_low, 's-', label="Boltzmann")
h3, = ax.plot(
    x_low,
    res_smooth_low,
    marker='^',
    linestyle='-',
    linewidth=2,
    label="Smoothed"
)
# add_rmse_box(ax, rmse_hill_low, rmse_bolt_low, rmse_smooth_low)
ax.set_title("Low Rh110 concentration series")
ax.set_xlabel("Low Rh110 added (µL)")
ax.set_ylabel("Residual")
ax.grid(alpha=0.3)

ax.legend(
    handles=[pos_patch, neg_patch, zero_line, h1, h2, h3],
    title="Legend",
    loc="lower right",
    frameon=True,
    facecolor="white",
    edgecolor="black",
    framealpha=0.9
)

# =========================
# HIGH
# =========================
ax = axs[1]

all_high = np.concatenate([res_hill_high, res_bolt_high, res_smooth_high])

ymin, ymax = np.nanmin(all_high), np.nanmax(all_high)

pad = 0.05 * (ymax - ymin)
ymin -= pad
ymax += pad

set_full_background(ax, ymin, ymax)

h1, = ax.plot(x_high[1:], res_hill_high, 'o-', label="Hill")
h2, = ax.plot(x_high[1:], res_bolt_high, 's-', label="Boltzmann")
h3, = ax.plot(
    x_high,
    res_smooth_high,
    marker='^',
    linestyle='-',
    linewidth=2,
    label="Smoothed"
)
# add_rmse_box(ax, rmse_hill_high, rmse_bolt_high, rmse_smooth_high)
ax.set_title("High Rh110 concentration series")
ax.set_xlabel("High Rh110 added (µL)")
ax.grid(alpha=0.3)

ax.legend(
    handles=[pos_patch, neg_patch, zero_line, h1, h2, h3],
    title="Legend",
    loc="lower right",
    frameon=True,
    facecolor="white",
    edgecolor="black",
    framealpha=0.9
)

# =========================
# TITLE
# =========================
fig.suptitle(
    "Residual analysis",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline

# =========================
# SAFE SPLINE PREP
# =========================
def prepare_for_spline(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    x_unique, unique_idx = np.unique(x, return_index=True)
    y_unique = y[unique_idx]

    return x_unique, y_unique


# =========================
# FINITE DIFFERENCE
# =========================
x_low_mid = (x_low[:-1] + x_low[1:]) / 2
x_high_mid = (x_high[:-1] + x_high[1:]) / 2

dx_low = np.diff(x_low)
dx_high = np.diff(x_high)

dx_low[np.isclose(dx_low, 0)] = np.nan
dx_high[np.isclose(dx_high, 0)] = np.nan

dE_low_fd = np.diff(E_low) / dx_low
dE_high_fd = np.diff(E_high) / dx_high

sigma_dE_low_fd = np.sqrt(sigma_E_low[:-1]**2 + sigma_E_low[1:]**2) / dx_low
sigma_dE_high_fd = np.sqrt(sigma_E_high[:-1]**2 + sigma_E_high[1:]**2) / dx_high

valid_low = np.isfinite(dE_low_fd)
valid_high = np.isfinite(dE_high_fd)

x_low_mid, dE_low_fd, sigma_dE_low_fd = x_low_mid[valid_low], dE_low_fd[valid_low], sigma_dE_low_fd[valid_low]
x_high_mid, dE_high_fd, sigma_dE_high_fd = x_high_mid[valid_high], dE_high_fd[valid_high], sigma_dE_high_fd[valid_high]


# =========================
# GAUSSIAN DERIVATIVE
# =========================
E_low_s = gaussian_filter1d(E_low, sigma=0.8)
E_high_s = gaussian_filter1d(E_high, sigma=0.8)

dE_low_gauss = np.gradient(E_low_s, x_low)
dE_high_gauss = np.gradient(E_high_s, x_high)


# =========================
# SPLINE DERIVATIVE
# =========================
x_low_c, E_low_c = prepare_for_spline(x_low, E_low)
x_high_c, E_high_c = prepare_for_spline(x_high, E_high)

if len(x_low_c) < 4 or len(x_high_c) < 4:
    raise ValueError("Not enough unique points for spline")

spline_low = UnivariateSpline(x_low_c, E_low_c, s=0.001)
spline_high = UnivariateSpline(x_high_c, E_high_c, s=0.001)

x_low_smooth = np.linspace(x_low_c.min(), x_low_c.max(), 300)
x_high_smooth = np.linspace(x_high_c.min(), x_high_c.max(), 300)

dE_low_spline = spline_low.derivative()(x_low_smooth)
dE_high_spline = spline_high.derivative()(x_high_smooth)


# =========================
# Y LIMIT FUNCTION
# =========================
def nice_ylim(data, pad=0.003):
    data = data[np.isfinite(data)]
    return np.min(data) - pad, np.max(data) + pad


# =========================
# PLOT
# =========================
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
plt.style.use("seaborn-v0_8-whitegrid")

zone_pos = mpatches.Patch(color="green", alpha=0.08, label="Positive sensitivity")
zone_neg = mpatches.Patch(color="red", alpha=0.08, label="Negative sensitivity")


# =========================
# LOW
# =========================
ax = axs[0]

all_low = np.concatenate([dE_low_fd, dE_low_gauss, dE_low_spline])
ymin, ymax = nice_ylim(all_low)

ax.axhspan(0, ymax, color="green", alpha=0.08)
ax.axhspan(ymin, 0, color="red", alpha=0.08)

fd_line = ax.errorbar(
    x_low_mid, dE_low_fd,
    yerr=sigma_dE_low_fd,
    fmt='o-',
    capsize=3,
    label="Finite difference"
)

gauss_line, = ax.plot(
    x_low, dE_low_gauss,
    '--',
    linewidth=2,
    label="Gaussian derivative"
)

spline_line, = ax.plot(
    x_low_smooth, dE_low_spline,
    '-',
    linewidth=2,
    label="Spline derivative"
)
ax.axhline(0, linestyle="--", color="black", linewidth=1, label="No sensitivity")
ax.set_title("Low Rh110 concentration series")
ax.set_xlabel("Low Rh110 added (µL)")
ax.set_ylabel("Sensitivity")
ax.set_ylim(ymin, ymax)
ax.grid(alpha=0.3)

no_sense_line = Line2D(
    [0], [0],
    color="black",
    linestyle="--",
    linewidth=1,
    label="No sensitivity"
)

ax.legend(
    handles=[
        zone_pos,
        zone_neg,
        no_sense_line,
        fd_line,
        gauss_line,
        spline_line
    ],
    title="Legend",
    loc="upper right",
    frameon=True,
    facecolor="white",
    edgecolor="black",
    framealpha=0.9
)


# =========================
# HIGH
# =========================
ax = axs[1]

all_high = np.concatenate([dE_high_fd, dE_high_gauss, dE_high_spline])
ymin, ymax = nice_ylim(all_high)

ax.axhspan(0, ymax, color="green", alpha=0.08)
ax.axhspan(ymin, 0, color="red", alpha=0.08)

fd_line = ax.errorbar(
    x_high_mid, dE_high_fd,
    yerr=sigma_dE_high_fd,
    fmt='o-',
    capsize=3,
    label="Finite difference"
)

gauss_line, = ax.plot(
    x_high, dE_high_gauss,
    '--',
    linewidth=2,
    label="Gaussian derivative"
)

spline_line, = ax.plot(
    x_high_smooth, dE_high_spline,
    '-',
    linewidth=2,
    label="Spline derivative"
)
ax.axhline(0, linestyle="--", color="black", linewidth=1, label="No sensitivity")
ax.set_title("High Rh110 concentration series")
ax.set_xlabel("High Rh110 added (µL)")
ax.set_ylim(ymin, ymax)
ax.grid(alpha=0.3)

ax.legend(
    handles=[
        zone_pos,
        zone_neg,
        no_sense_line,
        fd_line,
        gauss_line,
        spline_line
    ],
    title="Legend",
    loc="upper right",
    frameon=True,
    facecolor="white",
    edgecolor="black",
    framealpha=0.9
)


# =========================
# FINAL TITLE
# =========================
fig.suptitle("Sensitivity analysis", fontsize=15, fontweight="bold")

plt.tight_layout()
plt.show()
