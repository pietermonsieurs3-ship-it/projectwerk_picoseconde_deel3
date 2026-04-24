import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import re
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import median_abs_deviation
from scipy.stats import poisson, norm, skellam
from scipy.stats import normaltest
from scipy.stats import chisquare
from scipy.stats import probplot

class txt:
    # For the different files
    LOW = "\033[94m"      # blue
    HIGH = "\033[91m"     # red
    # For the different tests
    NORMAL = "\033[37m"    # light grey (true soft silver)
    POISSON = "\033[92m"  # green
    SKELLAM = "\033[95m"  # purple
    # To reset
    END = "\033[0m"
    
# =========================
# ROI STATS
# =========================
def roi_robust_stats(file):
    img = tiff.imread(file)
    h, w = img.shape

    roi = img[h//3:2*h//3, w//3:2*w//3].astype(float)
    roi = roi - np.min(roi)

    flat = roi.ravel()

    mean = np.mean(flat)
    std = np.std(flat)

    median = np.median(flat)
    mad = median_abs_deviation(flat, scale='normal')

    n = flat.size

    return mean, std, median, mad, n, flat


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
# TIFF VALIDITY CHECK
# =========================
import pandas as pd
import warnings

def check_tiff_file(file):
    result = {
        "file": file,
        "status": "OK",
        "shape": None,
        "dtype": None,
        "min": None,
        "max": None,
        "nan_count": None,
        "finite": True,
        "error": None
    }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = tiff.imread(file)

        result["shape"] = img.shape
        result["dtype"] = str(img.dtype)
        result["min"] = np.nanmin(img)
        result["max"] = np.nanmax(img)

        nan_count = np.isnan(img).sum() if np.issubdtype(img.dtype, np.floating) else 0
        result["nan_count"] = int(nan_count)
        result["finite"] = bool(np.isfinite(img).all())

        if img.size == 0:
            result["status"] = "EMPTY"
        elif nan_count > 0:
            result["status"] = "HAS_NAN"
        elif not result["finite"]:
            result["status"] = "NONFINITE"

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)

    return result


all_files = low_files + high_files
df_check = pd.DataFrame([check_tiff_file(f) for f in all_files])

priority = {"ERROR": 0, "EMPTY": 1, "HAS_NAN": 2, "NONFINITE": 3, "OK": 4}
df_check["priority"] = df_check["status"].map(priority)
df_check = df_check.sort_values("priority")

print("\n===== TIFF DATA QUALITY REPORT =====\n")
print(df_check.drop(columns=["priority"]))

# =========================
# LOAD DATA
# =========================
low_results = [roi_robust_stats(f) for f in low_files]
high_results = [roi_robust_stats(f) for f in high_files]

(I_low, I_low_std, I_low_med, I_low_mad, _, low_pixel_data) = zip(*low_results)
(I_high, I_high_std, I_high_med, I_high_mad, _, high_pixel_data) = zip(*high_results)

I_low = np.array(I_low)
I_high = np.array(I_high)

I_low_std = np.array(I_low_std)
I_high_std = np.array(I_high_std)

I_low_med = np.array(I_low_med)
I_high_med = np.array(I_high_med)

I_low_mad = np.array(I_low_mad)
I_high_mad = np.array(I_high_mad)


# =========================
# CONCENTRATIONS
# =========================
x_low = np.array([extract_low(f) for f in low_files])
x_high = np.array([extract_high(f) for f in high_files])


# =========================
# SNR DEFINITIONS
# =========================
SNR_mean_std_low = I_low / I_low_std
SNR_mean_std_high = I_high / I_high_std

SNR_robust_low = I_low_med / I_low_mad
SNR_robust_high = I_high_med / I_high_mad

SNR_poisson_low = I_low / np.sqrt(I_low + I_low_std**2)
SNR_poisson_high = I_high / np.sqrt(I_high + I_high_std**2)


# =========================
# HELPERS
# =========================
def nice_ylim(*arrays, pad=0.15):
    data = np.concatenate([np.ravel(a) for a in arrays])
    data = data[np.isfinite(data)]

    lo = np.nanpercentile(data, 5)
    hi = np.nanpercentile(data, 95)

    span = hi - lo
    lo -= pad * span
    hi += pad * span

    if lo == hi:
        lo -= 0.1
        hi += 0.1

    return lo, hi


def nice_log_xlim(x, pad_factor=0.15):
    x = np.array(x)
    x = x[np.isfinite(x) & (x > 0)]

    logx = np.log10(x)
    lo = 10**(logx.min() - pad_factor)
    hi = 10**(logx.max() + pad_factor)

    return lo, hi

def quick_dataset_diagnostics(files, datasets, name):
    print(f"\n===== {name} DATA CHECK =====")

    for file, data in zip(files, datasets):

        data = np.asarray(data)

        print(f"\n{file}")
        print(f"  size        : {data.size}")
        print(f"  mean        : {np.mean(data):.3f}")
        print(f"  std         : {np.std(data):.3f}")
        print(f"  skew-ish     : {(np.mean((data - np.mean(data))**3) / (np.std(data)**3)):.3f}")
        print(f"  zero fraction: {np.mean(data == 0):.3f}")
        print(f"  saturation   : {np.mean(data == np.max(data)):.3e}")

# run it
quick_dataset_diagnostics(low_files, low_pixel_data, "LOW")
quick_dataset_diagnostics(high_files, high_pixel_data, "HIGH")

# =========================
# PLOT
# =========================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

green = mpatches.Patch(color="green", alpha=0.12, label="SNR > 1")
red = mpatches.Patch(color="red", alpha=0.12, label="SNR < 1")
ref_line = Line2D([0], [0], color="black", linestyle="--", linewidth=1, label="SNR = 1")


# =========================
# LOW
# =========================
ax1.set_xscale("log")

l1, = ax1.plot(x_low, SNR_mean_std_low, 'o-', label="Classic")
l2, = ax1.plot(x_low, SNR_robust_low, '^-', label="Robust")
l3, = ax1.plot(x_low, SNR_poisson_low, 's-', label="Shot-noise")

ax1.axhline(1, color="black", linestyle="--", linewidth=1)

ax1.set_title("Low Rh110 concentration series")
ax1.set_xlabel("[low Rh110]")
ax1.set_ylabel("SNR")
ax1.grid(alpha=0.3)

ax1.set_xlim(*nice_log_xlim(x_low))

ymin, ymax = nice_ylim(
    SNR_mean_std_low,
    SNR_robust_low,
    SNR_poisson_low
)

ax1.set_ylim(ymin, ymax)

ax1.axhspan(1, ymax, color="green", alpha=0.10)
ax1.axhspan(ymin, 1, color="red", alpha=0.10)

ax1.legend(handles=[green, red, ref_line, l1, l2, l3],
           loc="lower left",
           frameon=True,
           facecolor="white",
           edgecolor="black",
           title="Legend")


# =========================
# HIGH
# =========================
ax2.set_xscale("log")

r1, = ax2.plot(x_high, SNR_mean_std_high, 'o-', label="Classic")
r2, = ax2.plot(x_high, SNR_robust_high, '^-', label="Robust")
r3, = ax2.plot(x_high, SNR_poisson_high, 's-', label="Shot-noise")

ax2.axhline(1, color="black", linestyle="--", linewidth=1)

ax2.set_title("High Rh110 concentration series")
ax2.set_xlabel("[high Rh110]")
ax2.grid(alpha=0.3)

ax2.set_xlim(*nice_log_xlim(x_high))

ymin, ymax = nice_ylim(
    SNR_mean_std_high,
    SNR_robust_high,
    SNR_poisson_high
)

ax2.set_ylim(ymin, ymax)

ax2.axhspan(1, ymax, color="green", alpha=0.10)
ax2.axhspan(ymin, 1, color="red", alpha=0.10)

ax2.legend(handles=[green, red, ref_line, r1, r2, r3],
           loc="lower left",
           frameon=True,
           facecolor="white",
           edgecolor="black",
           title="Legend")


# =========================
# TITLE
# =========================
fig.suptitle(
    "Reliability analysis",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()
plt.show()

# =========================
# HISTOGRAMS: LOW REGIME
# =========================

# =========================
# HISTOGRAMS: LOW REGIME
# =========================
n_low = len(low_files)

fig_low, axes_low = plt.subplots(
    nrows=int(np.ceil(n_low / 3)),
    ncols=3,
    figsize=(15, 2.2 * int(np.ceil(n_low / 3)))
)

axes_low = axes_low.flatten()

low_min = min(np.min(d) for d in low_pixel_data)
low_max = max(np.max(d) for d in low_pixel_data)

for i, (file, data) in enumerate(zip(low_files, low_pixel_data)):

    axes_low[i].hist(
        data,
        bins=50,
        range=(low_min, low_max),
        color='blue',
        alpha=0.7
    )

    axes_low[i].set_title(
        file.replace(".tif", "").replace("C307_Rh110_", ""),
        fontweight="bold"
    )
    axes_low[i].set_xlabel("Data")
    axes_low[i].set_ylabel("Frequency")
    axes_low[i].grid(alpha=0.3)

    # =========================
    # Gaussian
    # =========================
    mu = np.mean(data)
    sigma = np.std(data)

    x = np.linspace(low_min, low_max, 300)
    bin_width = (low_max - low_min) / 50

    gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    gauss_scaled = gauss * len(data) * bin_width

    axes_low[i].plot(x, gauss_scaled, color='black', linewidth=2, label='Gaussian')

    # =========================
    # Poisson
    # =========================
    lam = np.mean(data)
    x_pois = np.arange(int(np.floor(low_min)), int(np.ceil(low_max)) + 1)

    pois = poisson.pmf(x_pois, mu=lam)
    pois_scaled = pois * len(data)

    axes_low[i].plot(x_pois, pois_scaled, color='green', linewidth=2, label='Poisson')

    # =========================
    # Skellam (FIXED)
    # =========================
    lam1 = lam
    lam2 = lam * 0.8

    x_sk = np.arange(int(np.floor(low_min)), int(np.ceil(low_max)) + 1)

    mu_center = np.round(np.mean(data)).astype(int)
    sk = skellam.pmf(x_sk - mu_center, lam1, lam2)
    sk_scaled = sk * len(data)

    axes_low[i].plot(x_sk, sk_scaled, color='purple', linewidth=2, label='Skellam')

    # =========================
    # Legend
    # =========================
    intensity_patch = mpatches.Patch(color='blue', alpha=0.7, label='Data')
    gauss_line = Line2D([0], [0], color='black', linewidth=2, label='Gaussian')
    poisson_line = Line2D([0], [0], color='green', linewidth=2, label='Poisson')
    sk_line = Line2D([0], [0], color='purple', linewidth=2, label='Skellam')

    axes_low[i].legend(
        handles=[intensity_patch, gauss_line, poisson_line, sk_line],
        loc="upper right",
        frameon=True
    )
    
    # =========================
    # NORMALITY TEST
    # =========================
    k2, p_norm = normaltest(data)

    print(f"{txt.LOW} {file}{txt.END}")
    print(f"{txt.NORMAL}Normaltest:{txt.END}")
    print(f"  K2 = {k2:.3f}")
    print(f"  p = {p_norm:.3e}")

    # =========================
    # POISSON GOODNESS OF FIT
    # =========================
  
    lam = np.mean(data)

    x_vals = np.arange(
        int(np.min(data)),
        int(np.max(data)) + 1
    )

    obs_counts, _ = np.histogram(
        data,
        bins = np.arange(x_vals[0], x_vals[-1] + 2) - 0.5,
        range=(x_vals[0], x_vals[-1])
    )

    exp_probs = poisson.pmf(x_vals, mu=lam)

    # first normalize full distribution  
    exp_probs = exp_probs / np.sum(exp_probs)

    exp_counts = exp_probs * np.sum(obs_counts)

    # remove tiny expected bins
    mask = exp_counts > 5

    obs_filtered = obs_counts[mask]
    exp_filtered = exp_counts[mask]

    # renormalize AFTER masking
    exp_filtered = exp_filtered * (obs_filtered.sum() / exp_filtered.sum())

    chi_pois = chisquare(
        f_obs=obs_filtered,
        f_exp=exp_filtered
    )

    print(f"{txt.POISSON}Poisson:{txt.END}")
    print(f"  chi2 = {chi_pois.statistic:.3f}")
    print(f"  p = {chi_pois.pvalue:.3e}")
    
    # =========================
    # SKELLAM GOODNESS OF FIT
    # =========================
    lam1 = lam
    lam2 = lam * 0.8

    x_vals = np.arange(
        int(np.min(data)),
        int(np.max(data)) + 1 
    )

    obs_counts, _ = np.histogram(
        data,
        bins = np.arange(x_vals[0], x_vals[-1] + 2) - 0.5,
        range=(x_vals[0], x_vals[-1])
    )

    mu_center = np.round(np.mean(data)).astype(int)

    exp_probs = skellam.pmf(
        x_vals - mu_center,
        lam1,
        lam2
    )

    # normalize full distribution
    exp_probs = exp_probs / np.sum(exp_probs)

    exp_counts = exp_probs * np.sum(obs_counts)
     
    mask = exp_counts > 5

    obs_filtered = obs_counts[mask]
    exp_filtered = exp_counts[mask]

    # renormalize after masking
    exp_filtered = exp_filtered * (obs_filtered.sum() / exp_filtered.sum())

    chi_sk = chisquare(
        f_obs=obs_filtered,
        f_exp=exp_filtered
    )

    print(f"{txt.SKELLAM}Skellam:{txt.END}")
    print(f"  chi2 = {chi_sk.statistic:.3f}")
    print(f"  p = {chi_sk.pvalue:.3e}")

for j in range(i + 1, len(axes_low)):
    axes_low[j].axis("off")

fig_low.suptitle("Histograms of Intensities (Low Regime)", fontsize=16)
fig_low.tight_layout()
plt.show()

from scipy.stats import probplot

# =========================
# QQ PLOTS: LOW REGIME
# =========================
n_low = len(low_files)

fig_qq_low, axes_qq_low = plt.subplots(
    nrows=int(np.ceil(n_low / 3)),
    ncols=3,
    figsize=(16, 5 * int(np.ceil(n_low / 3)))
)

axes_qq_low = axes_qq_low.flatten()

for i, (file, data) in enumerate(zip(low_files, low_pixel_data)):

    ax = axes_qq_low[i]

    # theoretical + observed quantiles
    (osm, osr), (slope, intercept, r) = probplot(
        data
    )

    # scatter points
    ax.scatter(
        osm,
        osr,
        s=8,
        alpha=0.5,
        color="blue",
        label="Observed data"
    )

    # fitted line
    ax.plot(
        osm,
        slope * np.array(osm) + intercept,
        color="black",
        linewidth=2,
        label=f"Normal fit"
    )

    ax.set_title(
        file.replace(".tif", "").replace("C307_Rh110_", ""),
        fontsize = 10,
        fontweight="bold"
    )

    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Observed intensities")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

# Remove empty panels
for j in range(i + 1, len(axes_qq_low)):
    axes_qq_low[j].axis("off")

# Main title
fig_qq_low.suptitle(
    "Q-Q Plots (Low Regime)",
    fontsize=16,
    y=0.98   # pushes title slightly higher
)

# Manual spacing control
fig_qq_low.subplots_adjust(
    hspace=0.65,   # more vertical spacing
    wspace=0.35,   # more horizontal spacing
    top=0.88       # reserves room for suptitle
)

plt.show()

# =========================
# HISTOGRAMS: HIGH REGIME
# =========================

# =========================
# HISTOGRAMS: HIGH REGIME
# =========================
n_high = len(high_files)

fig_high, axes_high = plt.subplots(
    nrows=int(np.ceil(n_high / 3)),
    ncols=3,
    figsize=(15, 3.5 * int(np.ceil(n_high / 3)))
)

axes_high = axes_high.flatten()

high_min = min(np.min(d) for d in high_pixel_data)
high_max = max(np.max(d) for d in high_pixel_data)

for i, (file, data) in enumerate(zip(high_files, high_pixel_data)):

    axes_high[i].hist(
        data,
        bins=50,
        range=(high_min, high_max),
        color='red',
        alpha=0.7
    )

    axes_high[i].set_title(
        file.replace(".tif", "").replace("C307_Rh110_", ""),
        fontweight="bold"
    )

    axes_high[i].set_xlabel("Intensity")
    axes_high[i].set_ylabel("Frequency")
    axes_high[i].grid(alpha=0.3)

    # Gaussian fit
    mu = np.mean(data)
    sigma = np.std(data)

    x = np.linspace(high_min, high_max, 300)
    bin_width = (high_max - high_min) / 50

    gauss = norm.pdf(x, mu, sigma)
    gauss_scaled = gauss * len(data) * bin_width

    axes_high[i].plot(
        x,
        gauss_scaled,
        color='black',
        linewidth=2,
        label='Gaussian'
    )

    # Poisson fit
    lam = np.mean(data)
    x_pois = np.arange(
        int(np.floor(high_min)),
        int(np.ceil(high_max)) + 1
    )

    pois = poisson.pmf(x_pois, mu=lam)
    pois_scaled = pois * len(data)

    axes_high[i].plot(
        x_pois,
        pois_scaled,
        color='green',
        linewidth=2,
        label='Poisson'
    )

    # Skellam fit
    lam1 = lam
    lam2 = lam * 0.8

    x_sk = np.arange(
        int(np.floor(high_min)),
        int(np.ceil(high_max)) + 1
    )

    mu_center = np.round(np.mean(data)).astype(int)

    sk = skellam.pmf(
        x_sk - mu_center,
        lam1,
        lam2
    )

    sk_scaled = sk * len(data)

    axes_high[i].plot(
        x_sk,
        sk_scaled,
        color='purple',
        linewidth=2,
        label='Skellam'
    )

    intensity_patch = mpatches.Patch(color='red', alpha=0.7, label='Data')
    gauss_line = Line2D([0], [0], color='black', linewidth=2, label='Gaussian')
    poisson_line = Line2D([0], [0], color='green', linewidth=2, label='Poisson')
    sk_line = Line2D([0], [0], color='purple', linewidth=2, label='Skellam')

    axes_high[i].legend(
        handles=[intensity_patch, gauss_line, poisson_line, sk_line],
        loc="upper right",
        frameon=True
    )
    
    # =========================
    # NORMALITY TEST
    # =========================
    k2, p_norm = normaltest(data)

    print(f"{txt.HIGH} {file}{txt.END}")
    print(f"{txt.NORMAL}Normaltest:{txt.END}")
    print(f"  K2 = {k2:.3f}")
    print(f"  p = {p_norm:.3e}")
    
    # =========================
    # POISSON GOODNESS OF FIT
    # =========================
  
    lam = np.mean(data)

    x_vals = np.arange(
        int(np.min(data)),
        int(np.max(data)) + 1
    )

    obs_counts, _ = np.histogram(
        data,
        bins = np.arange(x_vals[0], x_vals[-1] + 2) - 0.5,
        range=(x_vals[0], x_vals[-1])
    )

    exp_probs = poisson.pmf(x_vals, mu=lam)

    # first normalize full distribution  
    exp_probs = exp_probs / np.sum(exp_probs)

    exp_counts = exp_probs * np.sum(obs_counts)

    # remove tiny expected bins
    mask = exp_counts > 5

    obs_filtered = obs_counts[mask]
    exp_filtered = exp_counts[mask]

    # renormalize AFTER masking
    exp_filtered = exp_filtered * (obs_filtered.sum() / exp_filtered.sum())

    chi_pois = chisquare(
        f_obs=obs_filtered,
        f_exp=exp_filtered
    )

    print(f"{txt.POISSON}Poisson:{txt.END}")
    print(f"  chi2 = {chi_pois.statistic:.3f}")
    print(f"  p = {chi_pois.pvalue:.3e}")
    
    # =========================
    # SKELLAM GOODNESS OF FIT
    # =========================
    lam1 = lam
    lam2 = lam * 0.8

    x_vals = np.arange(
        int(np.min(data)),
        int(np.max(data)) + 1 
    )

    obs_counts, _ = np.histogram(
        data,
        bins = np.arange(x_vals[0], x_vals[-1] + 2) - 0.5,
        range=(x_vals[0], x_vals[-1])
    )

    mu_center = np.round(np.mean(data)).astype(int)

    exp_probs = skellam.pmf(
        x_vals - mu_center,
        lam1,
        lam2
    )

    # normalize full distribution
    exp_probs = exp_probs / np.sum(exp_probs)

    exp_counts = exp_probs * np.sum(obs_counts)
     
    mask = exp_counts > 5

    obs_filtered = obs_counts[mask]
    exp_filtered = exp_counts[mask]

    # renormalize after masking
    exp_filtered = exp_filtered * (obs_filtered.sum() / exp_filtered.sum())

    chi_sk = chisquare(
        f_obs=obs_filtered,
        f_exp=exp_filtered
    )

    print(f"{txt.SKELLAM}Skellam:{txt.END}")
    print(f"  chi2 = {chi_sk.statistic:.3f}")
    print(f"  p = {chi_sk.pvalue:.3e}")

# Remove empty panels
for j in range(i + 1, len(axes_high)):
    axes_high[j].axis("off")

fig_high.suptitle(
    "Histograms of Intensities (High Regime)",
    fontsize=16
)

fig_high.subplots_adjust(
    hspace=0.55,
    wspace=0.35,
    top=0.88
)

plt.show()

# =========================
# QQ PLOTS: HIGH REGIME
# =========================

# =========================
# QQ PLOTS: HIGH REGIME
# =========================
n_high = len(high_files)

fig_qq_high, axes_qq_high = plt.subplots(
    nrows=int(np.ceil(n_high / 3)),
    ncols=3,
    figsize=(16, 5 * int(np.ceil(n_high / 3)))
)

axes_qq_high = axes_qq_high.flatten()

for i, (file, data) in enumerate(zip(high_files, high_pixel_data)):

    ax = axes_qq_high[i]

    (osm, osr), (slope, intercept, r) = probplot(
        data
    )

    # Data points
    ax.scatter(
        osm,
        osr,
        s=8,
        alpha=0.5,
        color="red",
        label="Observed data"
    )

    # Fit line
    ax.plot(
        osm,
        slope*np.array(osm) + intercept,
        color="black",
        linewidth=2,
        label=f"Normal fit"
    )

    ax.set_title(
        file.replace(".tif", "").replace("C307_Rh110_", ""),
        fontsize = 10,
        fontweight="bold"
    )

    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Observed intensities")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

# Remove empty panels
for j in range(i + 1, len(axes_qq_high)):
    axes_qq_high[j].axis("off")

fig_qq_high.suptitle(
    "Q-Q Plots (High Regime)",
    fontsize=16
)

fig_qq_high.subplots_adjust(
    hspace=0.6,
    wspace=0.35,
    top=0.88
)

plt.show()
