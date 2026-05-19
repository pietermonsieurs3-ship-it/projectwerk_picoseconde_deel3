# Code for step 2 and onward

from pathlib import Path

import numpy as np
import tifffile
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import tifffile

import re
import numpy as np

def power_model(x, A, m):
    return A * x**m

def extract_volumes(filename):
    numbers = re.findall(r"(\d+)\s*mul", filename)

    if len(numbers) == 1:
        return float(numbers[0]), 0
    elif len(numbers) >= 2:
        return float(numbers[0]), float(numbers[1])
    else:
        return 0, 0

def exponential_decay(t, A, tau):
    return A * np.exp(-t / tau)

def r_squared(y, y_pred):
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res / ss_tot

# --------------------------------------------------
# Settings
# --------------------------------------------------
DATA_DIR = Path(".")

# Files to analyze
all_files = sorted(DATA_DIR.glob("*.tif"))

donor_file = "zuiver_C307.tif"
acceptor_file = "zuiver_Rh110.tif"
dimmer_file = "C307_Rh110_85mul_70mul_dimmer.tif"
accident_file = "C307_Rh110_35mul_accident.tif"

# --------------------------------------------------
# Donor-only lifetime (C307)
# --------------------------------------------------

donor_image = tifffile.imread(donor_file)

roi = donor_image[20:460, :]
decay = np.sum(roi, axis=0)
# TIME ROI (important fix)
# Not allowed!!!!!!!!!!!!!
# decay = decay[400:600]
decay = decay / np.max(decay)

t = np.arange(len(decay))

peak_index = np.argmax(decay)
fit_start = peak_index + 15

t_fit = t[fit_start:]
y_fit = decay[fit_start:]

mask = y_fit > 1e-4
t_fit = t_fit[mask]
y_fit = y_fit[mask]

t_fit = t_fit - t_fit[0]

popt, _ = curve_fit(exponential_decay, t_fit, y_fit, p0=[1.0, 50])

DT_NS = 2.455115e-3  # ns/pixel
tau_D = popt[1] * DT_NS
print(popt[1])
print(f"Donor lifetime τ_D = {tau_D:.3f} ns")

tif_files = [
    f for f in all_files
    if f.name != acceptor_file
    and f.name != dimmer_file
    and f.name != accident_file
    and f.name != donor_file 
]

# --------------------------------------------------
# Exponential model
# --------------------------------------------------

# --------------------------------------------------
# Storage
# --------------------------------------------------

results = []

# --------------------------------------------------
# Loop over files
# --------------------------------------------------
donor_image = tifffile.imread(donor_file)

roi = donor_image[20:460, :]
F_D = np.sum(roi)

t = np.arange(len(decay))

DT_NS = 2.455115e-3  # ns per pixel

time_axis_ns = t * DT_NS
print("Mean dt (ns):", np.mean(np.diff(time_axis_ns)))
print("dt unique values:", np.unique(np.diff(time_axis_ns)))
print(np.allclose(np.diff(time_axis_ns), DT_NS))

peak_index = np.argmax(decay)
fit_start = peak_index + 15

t_fit = t[fit_start:]
y_fit = decay[fit_start:]

mask = y_fit > 1e-4
t_fit = t_fit[mask]
y_fit = y_fit[mask]

t_fit = t_fit - t_fit[0]

popt, pcov = curve_fit(exponential_decay, t_fit, y_fit, p0=[1.0, 50])

DT_NS = 2.455115e-3
tau_D = popt[1] * DT_NS
tau_D_err = np.sqrt(np.diag(pcov))[1] * DT_NS
# --------------------------------------------------
# Store plots instead of showing immediately
# --------------------------------------------------

plot_data = []

for file in tif_files:

    print(f"\nProcessing: {file.name}")

    image = tifffile.imread(file)

    roi = image[20:460, :]
    decay = np.sum(roi, axis=0)
    # TIME ROI (important fix)
    # Not allowed!!!! bias
    # decay = decay[400:600]
    decay = decay / np.max(decay)
    
    roi = image[20:460, :]
    raw_decay = np.sum(roi, axis=0)

    F_DA = np.sum(raw_decay)

    if F_D <= 0:
        continue

    E_intensity = 1 - (F_DA / F_D)
    
    E_intensity_err = (F_DA / F_D) * np.sqrt(
        (1 / F_DA) + (1 / F_D)
    )

    peak_index = np.argmax(decay)
    fit_start = peak_index + 15
    fit_end = len(decay)

    t = np.arange(len(decay))

    t_fit = t[fit_start:fit_end]
    y_fit = decay[fit_start:fit_end]

    mask = y_fit > 1e-4
    t_fit = t_fit[mask]
    y_fit = y_fit[mask]

    t_fit = t_fit - t_fit[0]

    popt, pcov = curve_fit(
        exponential_decay,
        t_fit,
        y_fit,
        p0=[1.0, 50]
    )

    tau_fit = popt[1]
    
    DT_NS = 2.455115e-3  # ns/pixel, from triggering file

    tau_pixels = tau_fit
    tau_ns = tau_pixels * DT_NS

    tau_err_pixels = np.sqrt(np.diag(pcov))[1]
    tau_err_ns = tau_err_pixels * DT_NS
    
    tau_FRET = np.nan
    tau_FRET_err = np.nan
    
    E = 1 - tau_ns / tau_D
    
    E_err = np.sqrt(
        (tau_err_ns / tau_D)**2 +
        ((tau_ns * tau_D_err) / (tau_D**2))**2  
    )

    if file.name != donor_file:
  
        denom = tau_D - tau_ns

        if denom > 0 and tau_ns < tau_D: # fixed to be "and"
 
            tau_FRET = (tau_ns * tau_D) / denom

            # better error propagation (includes tau_D too)
            dF_dDA = (tau_D**2) / (denom**2)
            dF_dD  = (tau_ns**2) / (denom**2)

            tau_FRET_err = np.sqrt(
                (dF_dDA * tau_err_ns)**2 +
                (dF_dD  * tau_D_err)**2
            )

    k_FRET = (1 / tau_ns) - (1 / tau_D)
    k_FRET_err = tau_err_ns / (tau_ns**2)

    v_low, v_high = extract_volumes(file.name)
    
    results.append({
        "file": file.name,
        
        "v_low": v_low,
        "v_high": v_high,

        "tau_ns": tau_ns,
        "tau_error_ns": tau_err_ns,

        "tau_pixels": tau_pixels,
        "tau_error_pixels": tau_err_pixels,

        "tau_FRET_ns": tau_FRET,
        "tau_FRET_error_ns": tau_FRET_err,

        "FRET_efficiency": E,
        "FRET_efficiency_error": E_err,
        
        "k_FRET": k_FRET,
        "k_FRET_error": k_FRET_err,
        
        "E_intensity": E_intensity,
        "E_intensity_error": E_intensity_err

    })

    # store for plotting later
    plot_data.append((file.name, t, decay, t_fit, fit_start, popt))

for r in results:
    v1 = r["v_low"]
    v2 = r["v_high"]

    C_LOW  = 0.005
    C_HIGH = 0.05

    uL_to_L = 1e-6
    mL_to_L = 1e-3

    V_low  = v1 * uL_to_L
    V_high = v2 * uL_to_L
    V_C307 = 1.0 * mL_to_L

    V_total = V_C307 + V_low + V_high

    r["c_eff"] = (C_LOW*V_low + C_HIGH*V_high) / V_total
    
    # -------------------------------
    # Simplified uncertainty on c_eff
    # (V_total treated as constant)
    # -------------------------------

    sigma_VL = 0.05 * V_low   # example 5%
    sigma_VH = 0.05 * V_high

    V_T = V_total

    c_err = (1 / V_T) * np.sqrt(
        (C_LOW * sigma_VL)**2 +
        (C_HIGH * sigma_VH)**2
    )

    r["c_eff_error"] = c_err

    # -------------------------------
    # Propagate c_eff → r uncertainty
    # -------------------------------

    r_val = r["c_eff"]

    # Model 1: r = c^(-1/3)
    r1 = r_val ** (-1/3)
    r1_err = (1/3) * c_err / r_val * r1

    # Model 2: r = (3/(4πc))^(1/3)
    r2 = (3 / (4 * np.pi * r_val)) ** (1/3)
    r2_err = (1/3) * c_err / r_val * r2

    r["r_model1"] = r1
    r["r_model1_error"] = r1_err

    r["r_model2"] = r2
    r["r_model2_error"] = r2_err

# --------------------------------------------------
# Find donor-only reference lifetime
# --------------------------------------------------

# --------------------------------------------------
# Split datasets into two regimes
# --------------------------------------------------
from collections import defaultdict

label_intensity = defaultdict(list)

for item in plot_data:
    name = item[0]
    decay = item[2]

    v_low, v_high = extract_volumes(name)

    if v_high == 0:
        label = f"{v_low:.0f} µ"
    else:
        label = f"{v_low:.0f}–{v_high:.0f} µ"

    total_intensity = np.sum(decay)

    label_intensity[label].append(total_intensity)

dimmer_flags = {}

for label, vals in label_intensity.items():
    if len(vals) > 1:
        threshold = np.mean(vals)  # or median (better)
        dimmer_flags[label] = threshold


# --------------------------------------------------
# FIGURE: All fluorescence decay fits (no regime split)
# --------------------------------------------------

all_groups = plot_data  # combine everything

n = len(all_groups)
cols = 3
rows = int(np.ceil(n / cols))

fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 1.6 * rows))
axs = axs.flatten()

for i, (name, t, decay, t_fit, fit_start, popt) in enumerate(all_groups):

    v_low, v_high = extract_volumes(name)

    if v_high == 0:
        title = rf"$V_{{low,\ Rh110}} = {v_low:.0f}\ \mu L$"
    else:
        title = rf"$V_{{low,\ Rh110}} = {v_low:.0f}\ \mu L,\ V_{{high,\ Rh110}} = {v_high:.0f}\ \mu L$"

    axs[i].plot(t, decay, label="data")

    axs[i].plot(
        t_fit + fit_start,
        exponential_decay(t_fit, *popt),
        label="fit"
    )

    axs[i].set_yscale("log")
    axs[i].set_title(title, fontsize=9)
    # axs[i].set_xlabel("Time pixel")
    # axs[i].set_ylabel("Norm. intensity")
    axs[i].grid(True, which="both", alpha=0.3)
    axs[i].legend(fontsize=7)
    # Only visually limited
    axs[i].set_xlim(left=400)
    axs[i].set_ylim(bottom=5*10**-1)

# remove empty subplots
for j in range(len(all_groups), len(axs)):
    fig.delaxes(axs[j])

fig.supxlabel("Pixels", fontsize=10)
fig.supylabel("log(Normalized intensity)", fontsize=10)
fig.suptitle("Fluorescence decay fits: all Rh110 concentrations", fontsize=14, fontweight = "bold")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("all_rh110_added_samples_data_with_fit.png",dpi=600,bbox_inches="tight")
plt.show()

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n======================")
print("FITTED LIFETIMES")
print("======================")

print(f"{'File':40s} || {'Low vol (µL)':12s} {'High vol (µL)':12s} {'c_eff (M)':14s}  {'±':14s} || {'tau (pix)':>12s} {'±':>8s} || {'tau (ns)':>12s} {'±':>8s} || {'tau_FRET (ns)':>14s} {'±':>10s} || {'r_model1 (arb)':>14s} {'±':>10s} || {'r_model2 (arb)':>14s} {'±':>10s}")

print("-" * 220)

for r in results:

    print(
        f"{r['file']:40s} || "
        
        f"{r['v_low']:10.1f} "
        f"{r['v_high']:10.1f} "
        
        f"{r['c_eff']:10.3e} "
        f"{r['c_eff_error']:10.3e} || "

        f"{r['tau_pixels']:16.3f} "
        f"{r['tau_error_pixels']:12.3f} || "
        
        f"{r['tau_ns']:16.3f} "
        f"{r['tau_error_ns']:12.3f} || "
        
        f"{r['tau_FRET_ns']:17.3f} "
        f"{r['tau_FRET_error_ns']:12.3f} || "
        
        f"{r['r_model1']:10.3e} "
        f"{r['r_model1_error']:10.3e} || "
        
        f"{r['r_model2']:10.3e} "
        f"{r['r_model2_error']:10.3e}"
    )

#####################################################################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

BLUE = "tab:blue"
GREEN = "tab:green"

# --------------------------------------------------
# Extract data
# --------------------------------------------------

tau_fret = []
tau_fret_err = []
c_eff = []

for r in results:
    if np.isnan(r["tau_FRET_ns"]) or np.isnan(r["c_eff"]):
        continue

    tau_fret.append(r["tau_FRET_ns"])
    tau_fret_err.append(r["tau_FRET_error_ns"])
    c_eff.append(r["c_eff"])

tau_fret = np.array(tau_fret)
tau_fret_err = np.array(tau_fret_err)
c_eff = np.array(c_eff)

log_tau = np.log(tau_fret)
log_tau_err = tau_fret_err / tau_fret

# --------------------------------------------------
# Model 1: r ∝ c^(-1/3)
# --------------------------------------------------

r1 = c_eff ** (-1/3)
log_r1 = np.log(r1)

slope1, intercept1, r_value1, _, _ = linregress(log_r1, log_tau)
r2_1 = r_value1**2

xfit1 = np.linspace(np.min(log_r1), np.max(log_r1), 200)
yfit1 = slope1 * xfit1 + intercept1

# --------------------------------------------------
# Model 2: Poisson / sphere model
# r = (3 / (4πc))^(1/3)
# --------------------------------------------------

r2 = (3 / (4 * np.pi * c_eff)) ** (1/3)
log_r2 = np.log(r2)

slope2, intercept2, r_value2, _, _ = linregress(log_r2, log_tau)
r2_2 = r_value2**2

xfit2 = np.linspace(np.min(log_r2), np.max(log_r2), 200)
yfit2 = slope2 * xfit2 + intercept2

# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.figure(figsize=(6, 5))

# Model 1 data + fit
plt.errorbar(
    log_r1,
    log_tau,
    yerr=log_tau_err,
    fmt='o',
    capsize=3,
    color=BLUE,
    label="Aanname 1 data: c⁻¹ᐟ³"
)

plt.plot(
    xfit1,
    yfit1,
    color=BLUE,
    label=f"Aanname 1 fit: slope={slope1:.2f}, R²={r2_1:.3f}"
)

# Model 2 data + fit
plt.errorbar(
    log_r2,
    log_tau,
    yerr=log_tau_err,
    fmt='s',
    capsize=3,
    color=GREEN,
    label="Aanname 2 data: spherical model"
)

plt.plot(
    xfit2,
    yfit2,
    color=GREEN,
    label=f"Aanname 2 fit: slope={slope2:.2f}, R²={r2_2:.3f}"
)


plt.xlabel("log(r)")
plt.ylabel("log(τ_FRET)")
plt.title("Förster scaling test: two distance models")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("fret_scaling_two_models_separate_data.jpg", dpi=600, bbox_inches="tight")
plt.show()

# --------------------------------------------------
# NONLINEAR CURVE_FIT VALIDATION (NEW PLOT)
# --------------------------------------------------

from scipy.optimize import curve_fit

# clean arrays
tau = tau_fret
c = c_eff

# avoid invalid values
mask = (tau > 0) & (c > 0)
tau = tau[mask]
c = c[mask]

# -------------------------
# Model 1: r = c^(-1/3)
# -------------------------
r1 = c ** (-1/3)

popt1, pcov1 = curve_fit(
    power_model,
    r1,
    tau,
    p0=[np.mean(tau), 1.0]
)

A1, m1 = popt1

tau_pred1 = power_model(r1, A1, m1)
r2_val_1 = r_squared(tau, tau_pred1)

# -------------------------
# Model 2: sphere model
# -------------------------
r2 = (3 / (4 * np.pi * c)) ** (1/3)

popt2, pcov2 = curve_fit(
    power_model,
    r2,
    tau,
    p0=[np.mean(tau), 1.0]
)

A2, m2 = popt2

tau_pred2 = power_model(r2, A2, m2)
r2_val_2 = r_squared(tau, tau_pred2)

# -------------------------
# plotting
# -------------------------
plt.figure(figsize=(6,5))

# scatter + fit Model 1
plt.scatter(r1, tau, color=BLUE, label="Aanname 1 data: c⁻¹ᐟ³")
x1 = np.linspace(np.min(r1), np.max(r1), 200)
plt.plot(x1, power_model(x1, A1, m1),
         color=BLUE,
         label=f"Aanname 1 fit: m={m1:.2f}, R²={r2_val_1:.3f}")

# scatter + fit Model 2
plt.scatter(r2, tau, color=GREEN, label="Aanname 2 data: spherical model")
x2 = np.linspace(np.min(r2), np.max(r2), 200)
plt.plot(x2, power_model(x2, A2, m2),
         color=GREEN,
         label=f"Aanname 2 fit: m={m2:.2f}, R²={r2_val_2:.3f}")

plt.errorbar(
    r1,
    tau,
    yerr=tau_fret_err,
    fmt='none',
    ecolor=BLUE,
    capsize=3
)

plt.errorbar(
    r2,
    tau,
    yerr=tau_fret_err,
    fmt='none',
    ecolor=GREEN,
    capsize=3
)

plt.xlabel("r")
plt.ylabel("τ_FRET (ns)")
plt.title("Nonlinear curve_fit validation of scaling law")
plt.legend(loc = "lower right")
plt.grid(True)

plt.tight_layout()
plt.savefig("fret_curvefit_validation.png", dpi=600, bbox_inches="tight")
plt.show()
