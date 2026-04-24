import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import re
from scipy.stats import linregress

# =========================
# ROI INTENSITY
# =========================
def roi_intensity(file):
    img = tiff.imread(file)
    h, w = img.shape

    roi = img[h//3:2*h//3, w//3:2*w//3]
    roi = roi - np.min(roi)

    return np.mean(roi)


# =========================
# CONCENTRATION EXTRACTION
# =========================
def extract_low(file):
    m = re.search(r"(\d+)mul", file)
    return float(m.group(1)) if m else np.nan

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
# LOAD DATA
# =========================
I_low = np.array([roi_intensity(f) for f in low_files])
I_high = np.array([roi_intensity(f) for f in high_files])

x_low = np.array([extract_low(f) for f in low_files])
x_high = np.array([extract_high(f) for f in high_files])


# =========================
# FRET EFFICIENCY
# =========================
I0 = I_low[0]

E_low = 1 - I_low / I0
E_high = 1 - I_high / I0


# =========================
# TRANSFORM
# =========================
def y_transform(E):
    E = np.clip(E, 1e-6, 1 - 1e-6)
    return np.log10((1 / E) - 1)


# =========================
# GLOBAL FIT
# =========================
def fit(x, E):
    mask = x > 0

    lx = np.log10(x[mask])
    ly = y_transform(E[mask])

    slope, intercept, r, _, _ = linregress(lx, ly)

    return slope, intercept, r**2, lx, ly


# =========================
# BOOTSTRAP UNCERTAINTY
# =========================
def bootstrap_slope(x, E, n_boot=2000):
    mask = x > 0

    lx = np.log10(x[mask])
    ly = y_transform(E[mask])

    slopes = []

    for _ in range(n_boot):
        idx = np.random.randint(0, len(lx), len(lx))

        if np.std(lx[idx]) < 1e-12:
            continue

        s, _, _, _, _ = linregress(lx[idx], ly[idx])
        slopes.append(s)

    return np.std(slopes)


# =========================
# RESULTS
# =========================
mL, bL, r2L, lxL, lyL = fit(x_low, E_low)
mH, bH, r2H, lxH, lyH = fit(x_high, E_high)

smL = bootstrap_slope(x_low, E_low)
smH = bootstrap_slope(x_high, E_high)

IDEAL = -6

print("\n===== R⁻⁶ SCALING TEST =====\n")

print("LOW REGIME")
print(f"slope = {mL:.3f} ± {smL:.3f}")
print(f"R²    = {r2L:.3f}")
print(f"dev   = {mL - IDEAL:.3f}\n")

print("HIGH REGIME")
print(f"slope = {mH:.3f} ± {smH:.3f}")
print(f"R²    = {r2H:.3f}")
print(f"dev   = {mH - IDEAL:.3f}")


# =========================
# PLOT 1: GLOBAL FIT
# =========================
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Global slope analysis", fontsize=14, fontweight="bold")

# Low regime
fit_x = np.linspace(lxL.min(), lxL.max(), 200)
axs[0].scatter(lxL, lyL, label="transformed data")
axs[0].plot(fit_x, mL * fit_x + bL, '--',
            label=f"slope={mL:.2f} ± {smL:.2f}")
axs[0].plot(fit_x, -6 * fit_x + bL, ':', label="ideal slope = -6")
axs[0].set_title("Low Rh110 concentration series")
axs[0].set_xlabel("log([low Rh110])")
axs[0].set_ylabel("log(1/E - 1)")
axs[0].grid(alpha=0.3)
axs[0].legend(loc="lower left")

# High regime
fit_x = np.linspace(lxH.min(), lxH.max(), 200)
axs[1].scatter(lxH, lyH, label="transformed data")
axs[1].plot(fit_x, mH * fit_x + bH, '--',
            label=f"slope={mH:.2f} ± {smH:.2f}")
axs[1].plot(fit_x, -6 * fit_x + bH, ':', label="ideal slope = -6")
axs[1].set_title("High Rh110 concentration series")
axs[1].set_xlabel("log([high Rh110])")
axs[1].grid(alpha=0.3)
axs[1].legend(loc="lower left")

plt.tight_layout()
plt.show()

# =========================
# LOCAL SLOPES (INTERPRETATION PLOT)
# =========================
def local_slopes(x, E, window=3):
    mask = x > 0
    x = x[mask]
    lx = np.log10(x)
    ly = y_transform(E[mask])

    slopes, centers = [], []

    for i in range(len(lx) - window + 1):
        xw = lx[i:i+window]
        yw = ly[i:i+window]

        if np.std(xw) < 1e-12:
            continue

        m, _, _, _, _ = linregress(xw, yw)
        slopes.append(m)
        
        # Use geometric mean to prevent "looping" on log-scale
        centers.append(np.exp(np.mean(np.log(x[i:i+window]))))

    return np.array(centers), np.array(slopes)


xL_loc, nL = local_slopes(x_low, E_low)
xH_loc, nH = local_slopes(x_high, E_high)


# =========================
# PLOT 2: MODEL BREAKDOWN
# =========================
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Local slope analysis", fontsize=14, fontweight="bold")

# Low regime
axs[0].axhline(-6, linestyle=":", color="orange", label="ideal slope = -6")
axs[0].plot(xL_loc, nL, "--o", label="local exponent")
axs[0].set_title("Low Rh110 concentration series")
axs[0].set_xlabel("[low Rh110]")
axs[0].set_ylabel("local exponent n(x)")
axs[0].set_xscale("log")
axs[0].grid(alpha=0.3)
axs[0].legend(loc="lower right")

# High regime
axs[1].axhline(-6, linestyle=":", color="orange", label="ideal slope = -6")
axs[1].plot(xH_loc, nH, "--o", label="local exponent")
axs[1].set_title("High Rh110 concentration series")
axs[1].set_xlabel("[high Rh110]")
axs[1].set_xscale("log")
axs[1].grid(alpha=0.3)
axs[1].legend(loc="lower right")

plt.tight_layout()
plt.show()
