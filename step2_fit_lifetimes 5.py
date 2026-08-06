# Code for step 2 and onward
# 1507, 1707, 1907, 1907, 2107

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

def tau_fret(tau_da, tau_d):
    return (tau_da * tau_d) / (tau_d - tau_da)

# --------------------------------------------------
# Load IRF
# --------------------------------------------------

def load_hipic_prf(filename):

    rows = []

    with open(filename, "r", errors="ignore") as f:

        for line in f:

            # remove HiPic markers
            clean = line.replace(";", "").strip()

            if clean == "":
                continue

            # split commas/spaces
            parts = clean.replace(",", " ").split()

            numbers = []

            for p in parts:
                try:
                    numbers.append(float(p))
                except:
                    pass

            # keep only real numerical rows
            if len(numbers) >= 2:
                rows.append(numbers[:2])


    return np.array(rows)

irf_data = load_hipic_prf(
    "triggering_1ns_vertical.prf"
)

irf_time = irf_data[:,0]
irf_counts = irf_data[:,1]

# normalize
irf_counts = irf_counts / np.sum(irf_counts)

# first peak only
peak_index = np.argmax(irf_counts)
# Back up 20:56, we got this brother.

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

def biexponential_decay(t, A1, tau1, A2, tau2):
    return (
        A1*np.exp(-t/tau1)
        +
        A2*np.exp(-t/tau2)
    )
    
def exponential_background_decay(t, A, tau, B):
    return A * np.exp(-t / tau) + B

from scipy.special import expit

def irf_mono_decay_causal(t, A, tau, t0, irf):
    shifted_t = t - t0
    decay = np.zeros_like(t, dtype=float)
    mask = shifted_t >= 0
    decay[mask] = A * np.exp(-shifted_t[mask] / tau)
   
    # Convolutie uitvoeren # we are here in 8:11
    conv = np.convolve(decay, irf, mode="full")

    # Trek de index van de IRF-piek af om de horizontale rechtsverschuiving op te heffen: # We are here 19;45
    irf_peak_idx = np.argmax(irf)
    return conv[irf_peak_idx : irf_peak_idx + len(t)]

# NEW: Het ultieme Mono + IRF + BG Model
def irf_mono_decay_causal_bg(t, A, tau, t0, B, irf):
    """
    Causale IRF-convolutie gecorrigeerd voor IRF-piek offset EN
    inclusief achtergrondruisvloer B.
    """
    shifted_t = t - t0
    decay = np.zeros_like(t, dtype=float)
    mask = shifted_t >= 0
    decay[mask] = A * np.exp(-shifted_t[mask] / tau)
   
    conv = np.convolve(decay, irf, mode="full")
    irf_peak_idx = np.argmax(irf)
    return conv[irf_peak_idx : irf_peak_idx + len(t)] + B

def r_squared(y, y_pred):
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res / ss_tot

def reduced_chi_squared(y, y_fit, sigma, n_params):
    """
    Reduced chi-squared.

    Parameters
    ----------
    y : measured data
    y_fit : fitted model
    sigma : uncertainty of each point
    n_params : number of fitted parameters

    Returns
    -------
    reduced chi^2
    """

    chi2 = np.sum(((y - y_fit) / sigma) ** 2)

    dof = len(y) - n_params

    return chi2 / dof

def calculate_1over_e_lifetime(decay, dt_ns):
    """
    Finds the 1/e decay time relative to the peak index.
    Uses linear interpolation between pixels for sub-pixel precision.
    """
    peak_idx = np.argmax(decay)
    I_0 = decay[peak_idx]
    target_intensity = I_0 / np.e
   
    # Analyze only the decaying tail after the peak
    decay_tail = decay[peak_idx:]
   
    # Find the first index where the decay drops below the 1/e target
    below_target_indices = np.where(decay_tail <= target_intensity)[0]
    if len(below_target_indices) == 0:
        return np.nan, np.nan
   
    idx_below = below_target_indices[0]
    if idx_below == 0:
        return 0.0, 0.0
       
    # Linear interpolation between index right before and right after crossing target
    x1 = idx_below - 1
    x2 = idx_below
    y1 = decay_tail[x1]
    y2 = decay_tail[x2]
   
    # --- VEILIGHEIDSCHECK ---
    # Controleer of y1 en y2 normale getallen zijn en geen overflow veroorzaken
    if not np.isfinite(y1) or not np.isfinite(y2):
        return np.nan, np.nan

    denominator = y2 - y1

    # Voorkom dat de noemer te klein is OF een overflow veroorzaakt
    if abs(denominator) < 1e-12 or not np.isfinite(denominator):
        fractional_pixel = x1
    else:
        # Extra check: is het resultaat van de deling te groot?
        try:
            fractional_pixel = x1 + (target_intensity - y1) / denominator
        except OverflowError:
            fractional_pixel = x1
    # ------------------------ we are back, no regrets

    tau_pixels = fractional_pixel
    tau_ns = tau_pixels * dt_ns
   
    # Apparent error propagation from shot-noise approximation
    relative_uncertainty = 1.0 / np.sqrt(target_intensity) if target_intensity > 0 else 0.1
    tau_err_ns = tau_ns * relative_uncertainty
   
    return tau_ns, tau_err_ns
    
# --------------------------------------------------
# NEW: Rapid Lifetime Determination (RLD) Shortcut Finder
# --------------------------------------------------
def calculate_rld_lifetime(decay, dt_ns, bg, gate_width_pixels=40):
    """
    Calculates the RLD lifetime by integrating two adjacent gates of width W pixels
    starting directly after the convoluted peak, pre-correcting for background level.
    """
    # Subtract background first to eliminate the constant offset systematic error
    decay_corr = decay - bg
   
    peak_idx = np.argmax(decay)
    # Start integration gate 15 pixels after peak (matching fitting start window)
    start_idx = peak_idx + 15
    W = gate_width_pixels
   
    if start_idx + 2 * W > len(decay):
        return np.nan, np.nan
       
    D1 = np.sum(decay_corr[start_idx : start_idx + W])
    D2 = np.sum(decay_corr[start_idx + W : start_idx + 2 * W])
   
    if D1 <= 0 or D2 <= 0 or D1 <= D2:
        return np.nan, np.nan
       
    tau_pixels = W / np.log(D1 / D2)
    tau_ns = tau_pixels * dt_ns
   
    # Error propagation including Poisson shot noise of raw counts
    ln_ratio = np.log(D1 / D2)
    raw_D1 = np.sum(decay[start_idx : start_idx + W])
    raw_D2 = np.sum(decay[start_idx + W : start_idx + 2 * W])
   
    var_ln_ratio = (raw_D1 / D1**2) + (raw_D2 / D2**2)
    sigma_ln = np.sqrt(var_ln_ratio)
    tau_err_ns = tau_ns * (sigma_ln / ln_ratio)
   
    return tau_ns, tau_err_ns
    
# --------------------------------------------------
# Centroid (First Moment) Shortcut Finder # 16:15 now
# --------------------------------------------------
def calculate_centroid_lifetime(decay, dt_ns, bg):
    """
    Calculates mean lifetime using the centroid of arrival times (First Moment).
    """
    decay_corr = decay - bg
    peak_idx = np.argmax(decay)
    start_idx = peak_idx + 15
   
    tail_y = decay_corr[start_idx:]
    tail_t = np.arange(len(tail_y)) * dt_ns
   
    mask = tail_y > 0
    if np.sum(mask) == 0:
        return np.nan, np.nan
       
    y_filtered = tail_y[mask]
    t_filtered = tail_t[mask]
   
    numerator = np.sum(t_filtered * y_filtered)
    denominator = np.sum(y_filtered)
   
    if denominator <= 0:
        return np.nan, np.nan
       
    tau_ns = numerator / denominator
   
    mean_t = tau_ns
    var_t = np.sum(((t_filtered - mean_t)**2) * y_filtered) / denominator
    tau_err_ns = np.sqrt(var_t / denominator) if denominator > 0 else 0.1
   
    return tau_ns, tau_err_ns

def compute_chi2_red_and_sigma(y_data, y_fit, sigma_data, p):
    """
    Berekent chi_r^2 en bijbehorende sigma_chi_r^2 met de meegegeven sigma.
    """
    valid = (y_data > 0) & (sigma_data > 0)
    y_d = y_data[valid]
    y_f = y_fit[valid]
    sig = sigma_data[valid]
   
    N = len(y_d)
    nu = N - p
   
    if nu <= 0:
        return np.nan, np.nan
   
    # Gebruik de daadwerkelijk gebruikte gewichten!
    chi2 = np.sum(((y_d - y_f) / sig) ** 2)
    chi2_red = chi2 / nu
    sigma_chi2_red = np.sqrt(2.0 / nu)
   
    return chi2_red, sigma_chi2_red

# --------------------------------------------------
# NEW: Phasor Method for Frequency-Domain Analysis
# --------------------------------------------------
# -------------------------------------------------- #New
# NEW: Robuuste Phasor berekening met hoek-correctie #New
# -------------------------------------------------- #New
def calculate_phasor_coordinates(t_array, y_array, frequency_hz=100e6): #New
    t_sec = t_array * 1e-9 #New
    omega = 2 * np.pi * frequency_hz #New
    total_intensity = np.sum(y_array) #New
   
    if total_intensity <= 0: #New
        return 0.0, 0.0, 0.0 #New
       
    G = np.sum(y_array * np.cos(omega * t_sec)) / total_intensity #New
    S = np.sum(y_array * np.sin(omega * t_sec)) / total_intensity #New
   
    # Gebruik atan2 en zorg dat de hoek netjes in het bereik [-pi, pi] valt #New
    theta = np.arctan2(S, G) #New
   
    return G, S, theta #New

# -------------------------------------------------- # 22:55 now time, yes.
# Settings
# --------------------------------------------------
DATA_DIR = Path(".")

# Files to analyze
all_files = sorted(DATA_DIR.glob("*.tif"))

donor_file = "zuiver_C307.tif"
acceptor_file = "zuiver_Rh110.tif"
dimmer_file = "C307_Rh110_85mul_70mul_dimmer.tif"
accident_file = "C307_Rh110_35mul_accident.tif"
exception_file = "C307_Rh110_5mul.tif"
last_sample_file = "C307_Rh110_85mul_70mul.tif"

DT_NS = 2.083507e-2  # ns/pixel
CUTOFF_NS = 477 * DT_NS # based on this delta t, this am i am sure about, yeah, i am, yes baby we got this
# We are now 18:47, nice man, yeah
CUTOFF_PIXEL = int(CUTOFF_NS / DT_NS)

# --------------------------------------------------
# Donor-only lifetime (C307)
# --------------------------------------------------

donor_image = tifffile.imread(donor_file)

print("Shape of donor image")
print(donor_image.shape)
# This gives (480,640), as for any tiff image
# This is the shape of the tiff image across the board

ROI_START = 50
ROI_END = 150

roi = donor_image[:, ROI_START:ROI_END]

raw_decay = np.sum(roi, axis=1)

background = np.percentile(raw_decay, 5)

decay = raw_decay.copy()

t = np.arange(len(decay))

peak_index = np.argmax(decay)

# --- TAIL FIT START (Ná de piek, bijv. +15 pixels) ---
fit_start_tail = peak_index + 15
fit_end        = min(len(decay), CUTOFF_PIXEL)

# Data voor Tail-fits (Mono, Mono+BG, Bi-exp)
t_fit_tail     = t[fit_start_tail:fit_end]
y_fit_tail     = decay[fit_start_tail:fit_end]
mask_tail      = y_fit_tail > 0

t_fit_tail     = t_fit_tail[mask_tail] - t_fit_tail[0]
y_fit_tail     = y_fit_tail[mask_tail]
sigma_fit_tail = np.sqrt(y_fit_tail)

# --- IRF CONVOLUTIE START (Vóór de piek, bijv. -30 pixels op de baseline!) ---
fit_start_irf  = max(0, peak_index - 50)

# Data voor IRF Reconvolutie fit
t_fit_irf      = t[fit_start_irf:fit_end]
y_fit_irf      = decay[fit_start_irf:fit_end]
mask_irf       = y_fit_irf > 0

t_fit_irf      = t_fit_irf[mask_irf] - t_fit_irf[0]
y_fit_irf      = y_fit_irf[mask_irf]
sigma_fit_irf  = np.sqrt(y_fit_irf)

t_fit_tail = t_fit_tail - t_fit_tail[0]

popt, pcov = curve_fit(
    exponential_decay,
    t_fit_tail,
    y_fit_tail,
    sigma=sigma_fit_tail,
    absolute_sigma=True,
    p0=[
        np.max(y_fit_tail),
        50
    ],
    bounds=(
        [0, 0],             # Ondergrenzen: amplitude >= 0, levensduur >= 0
        [np.inf, np.inf]    # Bovengrenzen: geen oneindige restricties, maar wel $> 0$
    ),
    maxfev=50000
)

initial_background = np.percentile(y_fit_tail, 5)

popt_bg, pcov_bg = curve_fit(
    exponential_background_decay,
    t_fit_tail,
    y_fit_tail,
    sigma=sigma_fit_tail,
    absolute_sigma=True,
    p0=[
        np.max(y_fit_tail),
        50,
        initial_background
    ],
    bounds=(
        [0,0,0],
        [np.inf,np.inf,np.inf]
    ),
    maxfev=50000
)

A_bg_D, tau_bg_D_pixels, B_D = popt_bg

tau_D_bg = tau_bg_D_pixels * DT_NS

print(
    f"Donor lifetime mono+BG = {tau_D_bg:.3f} ns"
)

print(
    f"Background level = {B_D:.5f}"
)

popt_irf, pcov_irf = curve_fit(
    lambda t,A,tau,t0:
        irf_mono_decay_causal(
            t,
            A,
            tau,
            t0,
            irf_counts
        ),
    t_fit_irf,
    y_fit_irf,
    sigma=sigma_fit_irf,
    absolute_sigma=True,
    p0=[
        np.max(y_fit_irf),
        100,
        50
    ],
    bounds=(
        [0, 0, -np.inf],      # Ondergrenzen: A >= 0, tau >= 0, t0 mag eventueel negatief zijn (afhankelijk van je tijdsschaal)
        [np.inf, np.inf, np.inf] # Bovengrenzen
    ),
    maxfev=50000
)

A_irf, tau_irf, t0_irf = popt_irf
tau_D_irf = tau_irf * DT_NS # correct name now 6:32
tau_D_irf_err = np.sqrt(np.diag(pcov_irf))[1] * DT_NS

popt_bi, pcov_bi = curve_fit(
    biexponential_decay,
    t_fit_tail,
    y_fit_tail,
    sigma=sigma_fit_tail,
    absolute_sigma=True,
    p0=[
        0.7, 30,
        0.3, 80
    ],
    maxfev=20000
)

A1,tau1,A2,tau2 = popt_bi

tau_bi_pixels = (
    A1*tau1**2 +
    A2*tau2**2
) / (
    A1*tau1 +
    A2*tau2
)

tau_bi_ns = tau_bi_pixels * DT_NS

tau_bi_err_ns = (
    np.sqrt(np.diag(pcov_bi)[1] + np.diag(pcov_bi)[3])
    * DT_NS / 2
)

tau_D_bi = tau_bi_ns

tau_D_bi_err_ns = tau_bi_err_ns

tau_D_pixels = popt[1]

tau_D = tau_D_pixels * DT_NS

tau_D_err = np.sqrt(np.diag(pcov))[1] * DT_NS

# 5. NEW: 1/e Shortcut for Donor
tau_D_shortcut, tau_D_shortcut_err = calculate_1over_e_lifetime(decay, DT_NS)

# 6. NEW: RLD Shortcut for Donor
tau_D_rld, tau_D_rld_err = calculate_rld_lifetime(decay, DT_NS, background)

# 5. Shortcuts for Donor
tau_D_centroid, tau_D_centroid_err = calculate_centroid_lifetime(decay, DT_NS, background)

# Donor Fit Mono + IRF + BG
peak_val_D = np.max(y_fit_irf)
initial_bg_D = np.percentile(y_fit_irf, 5)
A_guess_D = 1.2 * (peak_val_D - initial_bg_D)
rel_peak_D = peak_index

popt_irf_bg, pcov_irf_bg = curve_fit(
    lambda t_arr, A, tau, t0, B: irf_mono_decay_causal_bg(t_arr, A, tau, t0, B, irf_counts),
    t_fit_irf, y_fit_irf, sigma=sigma_fit_irf, absolute_sigma=True,
    p0=[A_guess_D, 100, rel_peak_D, initial_bg_D],
    bounds=([0, 0, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf]),
    maxfev=50000
)

A_irf_bg_D, tau_irf_bg_D, t0_irf_bg_D, B_irf_bg_D = popt_irf_bg
tau_D_irf_bg = tau_irf_bg_D * DT_NS
tau_D_irf_bg_err = np.sqrt(np.diag(pcov_irf_bg))[1] * DT_NS

# 5. NEW: Phasor Reference for Donor
t_phasor = t * DT_NS #New
G_donor, S_donor, theta_donor = calculate_phasor_coordinates(t_phasor, decay, frequency_hz=100e6) #New

print(f"Donor Reference Lifetimes:")
print(f"  τD Mono:            {tau_D:.3f} ns")
print(f"  τD Mono + BG:       {tau_D_bg:.3f} ns")
print(f"  τD Mono + IRF:      {tau_D_irf:.3f} ns")
print(f"  τD Mono + IRF + BG: {tau_D_irf_bg:.3f} ns")
print(f"  Sub-pixel 1/e:      {tau_D_shortcut:.3f} ns")
print(f"  RLD Integrated:     {tau_D_rld:.3f} ns")
print(f"  Centroid (1st Mom): {tau_D_centroid:.3f} ns")

# Berekende modellen voor het donorstaal (globaal)
model_donor_mono = exponential_decay(t_fit_tail, *popt)
model_donor_bg   = exponential_background_decay(t_fit_tail, *popt_bg)
model_donor_irf  = irf_mono_decay_causal(t_fit_irf, *popt_irf, irf_counts)
model_donor_irf_bg = irf_mono_decay_causal_bg(t_fit_irf, *popt_irf_bg, irf_counts)

# Chi2 en sigma berekenen voor de 4 donor-modellen, voor de donor D
chi2_mono_d, sig_mono_d     = compute_chi2_red_and_sigma(y_fit_tail, model_donor_mono, sigma_fit_tail, p=2)
chi2_bg_d,   sig_bg_d       = compute_chi2_red_and_sigma(y_fit_tail, model_donor_bg, sigma_fit_tail, p=3)
chi2_irf_d,  sig_irf_d      = compute_chi2_red_and_sigma(y_fit_irf, model_donor_irf, sigma_fit_irf, p=3)
chi2_irf_bg_d, sig_irf_bg_d = compute_chi2_red_and_sigma(y_fit_irf, model_donor_irf_bg, sigma_fit_irf, p=4)

# Black-listed files
# These files will not be analysed.
tif_files = [
    f for f in all_files
    if f.name != acceptor_file
    and f.name != dimmer_file
    and f.name != accident_file
    and f.name != donor_file 
    and f.name != exception_file 
    # and f.name != last_sample_file
]
# It's 14:21
# --------------------------------------------------
# NEW: Plot Donor C307 met Harde Fitlijn en Gestippelde Extrapolatie
# --------------------------------------------------
time_ns_D = t[:CUTOFF_PIXEL] * DT_NS
decay_plot_D = decay[:CUTOFF_PIXEL]

# 1. Fitbereik (Harde lijn)
t_tail_fit_D = t[fit_start_tail:fit_end]
t_tail_fit_shift_D = t_tail_fit_D - t[fit_start_tail]
time_fit_tail_ns_D = t_tail_fit_D * DT_NS

# 2. Extrapolatiebereik van piek tot fit_start_tail (Gestippelde lijn)
t_tail_ext_D = t[peak_index:fit_start_tail + 1]
t_tail_ext_shift_D = t_tail_ext_D - t[fit_start_tail]
time_ext_tail_ns_D = t_tail_ext_D * DT_NS

time_fit_irf_ns_D = (t_fit_irf + fit_start_irf) * DT_NS

plt.figure(figsize=(7, 4.5))

# Ruwe Data in ZWART
plt.plot(time_ns_D, decay_plot_D, color='black', alpha=0.6, linewidth=1.2, label="Data (Zuiver C307)")

# Fit Mono (BLAUW: hard voor fit, gestippeld voor extrapolatie)
plt.plot(time_fit_tail_ns_D, exponential_decay(t_tail_fit_shift_D, *popt),
         color='blue', linewidth=1.5, label="Mono fit")
plt.plot(time_ext_tail_ns_D, exponential_decay(t_tail_ext_shift_D, *popt),
         color='blue', linewidth=1.5, linestyle=':')

# Fit Mono + BG (ROOD: hard voor fit, gestippeld voor extrapolatie)
plt.plot(time_fit_tail_ns_D, exponential_background_decay(t_tail_fit_shift_D, *popt_bg),
         color='red', linewidth=1.5, label="Mono + BG fit")
plt.plot(time_ext_tail_ns_D, exponential_background_decay(t_tail_ext_shift_D, *popt_bg),
         color='red', linewidth=1.5, linestyle=':')

# Fit Mono + IRF (MAGENTA)
plt.plot(time_fit_irf_ns_D, irf_mono_decay_causal(t_fit_irf, *popt_irf, irf_counts),
         color='magenta', linewidth=1.2, alpha=0.7, label="Mono + IRF fit")

# Fit Mono + IRF + BG (DONKERGROEN)
plt.plot(time_fit_irf_ns_D, irf_mono_decay_causal_bg(t_fit_irf, *popt_irf_bg, irf_counts),
         color='darkgreen', linewidth=1.5, label="Mono + IRF + BG fit")

plt.xlabel("Tijd (ns)", fontsize=11)
plt.ylabel("Fluorescentie-intensiteit (counts)", fontsize=11)
plt.title(r"Fluorescentie vervalfits: Donor staal (pure $C307$)", fontsize=12, fontweight="bold")
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(fontsize=8, loc="best")
plt.tight_layout()
plt.savefig("donor_C307_decay_with_4_fits.png", dpi=600, bbox_inches="tight")
plt.show()

# Nice right place, no overwirte (yet) This is the pure dye
# 19:50, its now time. You are strong.
# --------------------------------------------------
# Exponential model
# --------------------------------------------------
# Back-up 31/07. We are here to siwtch
# Its now 19:39
# --------------------------------------------------
# Storage
# --------------------------------------------------

results = []

# --------------------------------------------------
# Loop over files
# --------------------------------------------------
donor_image = tifffile.imread(donor_file)

# roi = donor_image[20:460, :]
F_D = np.sum(raw_decay)

t = np.arange(len(decay))

time_axis_ns = t * DT_NS
print("Mean dt (ns):", np.mean(np.diff(time_axis_ns)))
print("dt unique values:", np.unique(np.diff(time_axis_ns)))
print(np.allclose(np.diff(time_axis_ns), DT_NS))

# --------------------------------------------------
# Store plots instead of showing immediately DA
# --------------------------------------------------

plot_data = []

tau_D_bg = tau_bg_D_pixels * DT_NS
tau_D_bg_err_ns = np.sqrt(np.diag(pcov_bg))[1] * DT_NS

for file in tif_files:

    print(f"\nProcessing: {file.name}")

    image = tifffile.imread(file)

    ROI_START = 50
    ROI_END = 150

    roi = image[:, ROI_START:ROI_END]
    raw_decay = np.sum(roi, axis=1)

    background = np.percentile(raw_decay, 5)

    decay = raw_decay.copy()

    # roi = image[20:460, :]
    # raw_decay = np.sum(roi, axis=1)

    F_DA = np.sum(raw_decay)

    if F_D <= 0:
        continue

    E_intensity = 1 - (F_DA / F_D)
    
    E_intensity_err = (F_DA / F_D) * np.sqrt(
        (1 / F_DA) + (1 / F_D)
    )

    peak_index = np.argmax(decay)
    
    # --- TAIL FIT START (Ná de piek, bijv. +15 pixels) ---
    fit_start_tail = peak_index + 15
    fit_end        = min(len(decay), CUTOFF_PIXEL)

    # Data voor Tail-fits (Mono, Mono+BG, Bi-exp)
    t_fit_tail     = t[fit_start_tail:fit_end]
    y_fit_tail     = decay[fit_start_tail:fit_end]
    mask_tail      = y_fit_tail > 0

    t_fit_tail     = t_fit_tail[mask_tail] - t_fit_tail[0]
    y_fit_tail     = y_fit_tail[mask_tail]
    sigma_fit_tail = np.sqrt(y_fit_tail)

    # --- IRF CONVOLUTIE START (Vóór de piek, bijv. -30 pixels op de baseline!) ---
    fit_start_irf  = max(0, peak_index - 50)

    # Data voor IRF Reconvolutie fit
    t_fit_irf      = t[fit_start_irf:fit_end]
    y_fit_irf      = decay[fit_start_irf:fit_end]
    mask_irf       = y_fit_irf > 0

    t_fit_irf      = t_fit_irf[mask_irf] - t_fit_irf[0]
    y_fit_irf      = y_fit_irf[mask_irf]
    sigma_fit_irf  = np.sqrt(y_fit_irf)

    t_fit_tail = t_fit_tail - t_fit_tail[0]

    popt, pcov = curve_fit(
        exponential_decay,
        t_fit_tail,
        y_fit_tail,
        sigma=sigma_fit_tail,
        absolute_sigma=True,
        p0=[
            np.max(y_fit_tail),
            50
        ],
        bounds=(
            [0, 0],             # Ondergrenzen: amplitude >= 0, levensduur >= 0
            [np.inf, np.inf]    # Bovengrenzen: geen oneindige restricties, maar wel $> 0$
        ),
        maxfev=50000
    )

    initial_background = np.percentile(y_fit_tail, 5)
    
    popt_bg, pcov_bg = curve_fit(
        exponential_background_decay,
        t_fit_tail,
        y_fit_tail,
        sigma=sigma_fit_tail,
        absolute_sigma=True,
        p0=[
            np.max(y_fit_tail),
            50,
            initial_background
        ],
        bounds=(
            [0,0,0],
            [np.inf,np.inf,np.inf]
        ),
        maxfev=50000
    )

    A_bg, tau_bg_pixels, B = popt_bg  # Background term for bg

    tau_bg_ns = tau_bg_pixels * DT_NS
    
    tau_bg_err_pixels = np.sqrt(
        np.diag(pcov_bg)
    )[1]
 
    tau_bg_err_ns = tau_bg_err_pixels * DT_NS
    
    popt_irf, pcov_irf = curve_fit(
        lambda t,A,tau,t0:
            irf_mono_decay_causal(
               t,
               A,
               tau,
               t0,
               irf_counts
            ),
         t_fit_irf,
         y_fit_irf,
         sigma=sigma_fit_irf,
         absolute_sigma=True,
         p0=[
             np.max(y_fit_irf),
             100,
             50
         ],
         bounds=(
             [0, 0, -np.inf],      # Ondergrenzen: A >= 0, tau >= 0, t0 mag eventueel negatief zijn (afhankelijk van je tijdsschaal)
             [np.inf, np.inf, np.inf] # Bovengrenzen
         ),
         maxfev=50000
    )

    A_irf, tau_irf, t0_irf = popt_irf
    tau_irf_ns = tau_irf * DT_NS
    
    # Sample Fit Mono + IRF + BG
    peak_val_sample = np.max(y_fit_irf)
    initial_bg_sample = np.percentile(y_fit_irf, 5)
    A_guess_sample = 1.2 * (peak_val_sample - initial_bg_sample)
    rel_peak_offset = peak_index

    popt_irf_bg, pcov_irf_bg = curve_fit(
        lambda t_arr, A, tau, t0, B_val: irf_mono_decay_causal_bg(t_arr, A, tau, t0, B_val, irf_counts),
        t_fit_irf, y_fit_irf, sigma=sigma_fit_irf, absolute_sigma=True,
        p0=[A_guess_sample, 100, rel_peak_offset, initial_bg_sample],
        bounds=([0, 0, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf]),
        maxfev=50000 
    ) # Yes, we are here.
    # No more changes to the code
    A_irf_bg, tau_irf_bg, t0_irf_bg, B_irf_bg = popt_irf_bg  # Background term for irf bg
    tau_irf_bg_ns = tau_irf_bg * DT_NS
    tau_irf_bg_err_pixels = np.sqrt(np.diag(pcov_irf_bg))[1]
    tau_irf_bg_err_ns = tau_irf_bg_err_pixels * DT_NS
    # Code is done, i believe, we got this.
    popt_bi, pcov_bi = curve_fit(
        biexponential_decay,
        t_fit_tail,
        y_fit_tail,
        sigma=sigma_fit_tail,
        absolute_sigma=True,
        p0=[
           0.7, 30,
           0.3, 80
        ],
        maxfev=20000
    )
    
    A1,tau1,A2,tau2 = popt_bi

    tau_bi_pixels = (
        A1*tau1**2 +
        A2*tau2**2
        ) / (
        A1*tau1 +
        A2*tau2
    )

    tau_bi_ns = tau_bi_pixels * DT_NS
    
    tau_bi_err_ns = np.sqrt(
        np.diag(pcov_bi)[1] +
        np.diag(pcov_bi)[3]
    ) * DT_NS / 2

    model = exponential_decay(t_fit_tail, *popt)
 
    chi2_red = reduced_chi_squared(
        y_fit_tail,
        model,
        sigma_fit_tail,
        n_params=2
    )

    tau_fit = popt[1]

    tau_pixels = tau_fit
    tau_ns = tau_pixels * DT_NS

    tau_err_pixels = np.sqrt(np.diag(pcov))[1]
    tau_err_ns = tau_err_pixels * DT_NS
    
    # NEW: Calculate 1/e Shortcut for Sample
    tau_shortcut_ns, tau_shortcut_err_ns = calculate_1over_e_lifetime(decay, DT_NS)
    
    # 3. NEW: Heuristic Shortcut FRET & Efficiency
    denom_shortcut = tau_D_shortcut - tau_shortcut_ns
    if denom_shortcut > 0 and tau_shortcut_ns < tau_D_shortcut:
        tau_FRET_shortcut = (tau_shortcut_ns * tau_D_shortcut) / denom_shortcut
        dF_dDA_sc = (tau_D_shortcut**2) / (denom_shortcut**2)
        dF_dD_sc  = (tau_shortcut_ns**2) / (denom_shortcut**2)
        tau_FRET_shortcut_err = np.sqrt((dF_dDA_sc * tau_shortcut_err_ns)**2 + (dF_dD_sc * tau_D_shortcut_err)**2)
    else:
        tau_FRET_shortcut, tau_FRET_shortcut_err = np.nan, np.nan
    
    E_shortcut = 1 - (tau_shortcut_ns / tau_D_shortcut)
    E_shortcut_err = np.sqrt((-1/tau_D_shortcut * tau_shortcut_err_ns)**2 + ((tau_shortcut_ns * tau_D_shortcut_err) / (tau_D_shortcut**2))**2)
    
    # NEW: Calculate RLD Shortcut for Sample
    tau_rld_ns, tau_rld_err_ns = calculate_rld_lifetime(decay, DT_NS, background)
    
    
    # 4. NEW: RLD Shortcut Model
    denom_rld = tau_D_rld - tau_rld_ns
    if denom_rld > 0 and tau_rld_ns < tau_D_rld:
        tau_FRET_rld = (tau_rld_ns * tau_D_rld) / denom_rld
        dF_dDA_rld = (tau_D_rld**2) / (denom_rld**2)
        dF_dD_rld  = (tau_rld_ns**2) / (denom_rld**2)
        tau_FRET_rld_err = np.sqrt((dF_dDA_rld * tau_rld_err_ns)**2 + (dF_dD_rld * tau_D_rld_err)**2)
    else:
        tau_FRET_rld, tau_FRET_rld_err = np.nan, np.nan

    E_rld = 1 - (tau_rld_ns / tau_D_rld)
    E_rld_err = np.sqrt((-1/tau_D_rld * tau_rld_err_ns)**2 + ((tau_rld_ns * tau_D_rld_err) / (tau_D_rld**2))**2)

    # Centroid Shortcut for Sample
    tau_centroid_ns, tau_centroid_err_ns = calculate_centroid_lifetime(decay, DT_NS, background)
    
    # Centroid Shortcut FRET & Efficiency
    denom_centroid = tau_D_centroid - tau_centroid_ns
    if denom_centroid > 0 and tau_centroid_ns < tau_D_centroid:
        tau_FRET_centroid = (tau_centroid_ns * tau_D_centroid) / denom_centroid
        dF_dDA_cen = (tau_D_centroid**2) / (denom_centroid**2)
        dF_dD_cen  = (tau_centroid_ns**2) / (denom_centroid**2)
        tau_FRET_centroid_err = np.sqrt((dF_dDA_cen * tau_centroid_err_ns)**2 + (dF_dD_cen * tau_D_centroid_err)**2)
    else:
        tau_FRET_centroid, tau_FRET_centroid_err = np.nan, np.nan
    E_centroid = 1 - (tau_centroid_ns / tau_D_centroid)
    E_centroid_err = np.sqrt((-1/tau_D_centroid * tau_centroid_err_ns)**2 + ((tau_centroid_ns * tau_D_centroid_err) / (tau_D_centroid**2))**2)
    
    # Calculate the denominator safely
    denom = tau_D - tau_ns

    # Only calculate if the donor-acceptor lifetime is shorter than the donor-only lifetime
    if denom > 0 and tau_ns < tau_D:
        tau_FRET = (tau_ns * tau_D) / denom
    else:
        tau_FRET = np.nan  # Fallback to avoid script crashes

    tau_FRET_err = np.sqrt(
       tau_D_err**2 +
       tau_err_ns**2
    )
    
    # Safe and correct FRET lifetime calculation for the Background Model
    denom_bg = tau_D_bg - tau_bg_ns
    if denom_bg > 0 and tau_bg_ns < tau_D_bg:
        tau_FRET_bg = (tau_bg_ns * tau_D_bg) / denom_bg

        # Proper partial derivative error propagation
        dF_dDA_bg = (tau_D_bg**2) / (denom_bg**2)
        dF_dD_bg  = (tau_bg_ns**2) / (denom_bg**2)
        tau_FRET_bg_err = np.sqrt((dF_dDA_bg * tau_bg_err_ns)**2 + (dF_dD_bg * tau_D_err)**2)
    else:
        tau_FRET_bg = np.nan
        tau_FRET_bg_err = np.nan
    
    E = 1 - tau_ns / tau_D
    
    E_bg = 1 - tau_bg_ns / tau_D_bg
    
    dE_dtau_bg = -1 / tau_D_bg

    dE_dtau_D_bg = (
        tau_bg_ns /
        (tau_D_bg**2)
    )
    
    E_bg_err = np.sqrt(
        (dE_dtau_bg * tau_bg_err_ns)**2 +
        (dE_dtau_D_bg * tau_D_bg_err_ns)**2
    )

    E_err = np.sqrt(
        (tau_err_ns / tau_D)**2 +
        ((tau_ns * tau_D_err) / (tau_D**2))**2  
    )
    
    E_bi = 1 - tau_bi_ns / tau_D_bi
    
    E_bi_err = np.sqrt(
       (tau_bi_err_ns / tau_D_bi)**2 +
       (tau_bi_ns * tau_D_err / tau_D_bi**2)**2 
    )
    
    tau_irf_err_pixels = np.sqrt(np.diag(pcov_irf))[1]
    
    tau_irf_err_ns = tau_irf_err_pixels * DT_NS
    
    # Use convolved donor reference instead of standard mono reference:
    E_irf = 1 - tau_irf_ns / tau_D_irf
    E_irf_err = np.sqrt(
        (tau_irf_err_ns / tau_D_irf)**2 +
        ((tau_irf_ns * tau_D_irf_err) / (tau_D_irf**2))**2
    )
    
    # NEW: Mono + IRF + BG Efficiëntie & FRET Overdrachtstijd
    E_irf_bg = 1 - tau_irf_bg_ns / tau_D_irf_bg
    E_irf_bg_err = np.sqrt((tau_irf_bg_err_ns / tau_D_irf_bg)**2 + ((tau_irf_bg_ns * tau_D_irf_bg_err) / (tau_D_irf_bg**2))**2)

    if file.name != donor_file:
  
        denom = tau_D - tau_ns

        print(f"\n{file.name}")
        print(f"tau_D = {tau_D:.4f} ns") 
        print(f"tau_DA = {tau_ns:.4f} ns") 
        print(f"denom = {denom:.4f}")

        if denom > 0 and tau_ns < tau_D: # fixed to be "and"
 
            tau_FRET = (tau_ns * tau_D) / denom

            # better error propagation (includes tau_D too)
            dF_dDA = (tau_D**2) / (denom**2)
            dF_dD  = (tau_ns**2) / (denom**2)

            tau_FRET_err = np.sqrt(
                (dF_dDA * tau_err_ns)**2 +
                (dF_dD  * tau_D_err)**2
            )
        
        denom_bi = tau_D_bi - tau_bi_ns
        
        print(f"tau_D_bi = {tau_D_bi:.4f} ns") 
        print(f"tau_bi_ns = {tau_bi_ns:.4f} ns") 
        print(f"denom_bi = {denom_bi:.4f}")

        if denom_bi > 0:

            tau_FRET_bi = (
                tau_bi_ns * tau_D_bi
            ) / denom_bi
        else:
            tau_FRET_bi = np.nan
        
        eps = 1e-8

        d_tau_da = (
            tau_fret(tau_bi_ns + eps, tau_D_bi)
    -       tau_fret(tau_bi_ns - eps, tau_D_bi)
        ) / (2 * eps)

        d_tau_d = (
            tau_fret(tau_bi_ns, tau_D_bi + eps)
    -       tau_fret(tau_bi_ns, tau_D_bi - eps)
        ) / (2 * eps)
    
        tau_FRET_bi_err = np.sqrt(
            (d_tau_da * tau_bi_err_ns)**2 +
            (d_tau_d * tau_D_bi_err_ns)**2
        )

        ###
        
        tau_FRET_irf = np.nan
        tau_FRET_irf_err = np.nan


        denom_irf = tau_D_irf - tau_irf_ns
        if denom_irf > 0 and tau_irf_ns < tau_D_irf:
            tau_FRET_irf = (tau_irf_ns * tau_D_irf) / denom_irf
            # Error propagation using convolved derivatives:
            dF_dDA_irf = (tau_D_irf**2) / (denom_irf**2)
            dF_dD_irf  = (tau_irf_ns**2) / (denom_irf**2)
            tau_FRET_irf_err = np.sqrt((dF_dDA_irf * tau_irf_err_ns)**2 + (dF_dD_irf * tau_D_irf_err)**2)
        
        # NEW: Mono + IRF + BG FRET Overdrachtstijd
        tau_FRET_irf_bg = np.nan
        tau_FRET_irf_bg_err = np.nan
        denom_irf_bg = tau_D_irf_bg - tau_irf_bg_ns
        if denom_irf_bg > 0 and tau_irf_bg_ns < tau_D_irf_bg:
            tau_FRET_irf_bg = (tau_irf_bg_ns * tau_D_irf_bg) / denom_irf_bg
            dF_dDA_irf_bg = (tau_D_irf_bg**2) / (denom_irf_bg**2)
            dF_dD_irf_bg  = (tau_irf_bg_ns**2) / (denom_irf_bg**2)
            tau_FRET_irf_bg_err = np.sqrt((dF_dDA_irf_bg * tau_irf_bg_err_ns)**2 + (dF_dD_irf_bg * tau_D_irf_bg_err)**2)
    
    k_FRET = (1 / tau_ns) - (1 / tau_D)
    k_FRET_err = tau_err_ns / (tau_ns**2)

    v_low, v_high = extract_volumes(file.name)
    
    # For chi squared and sigma
    model_mono = exponential_decay(t_fit_tail, *popt)
    model_bg   = exponential_background_decay(t_fit_tail, *popt_bg)
    model_irf  = irf_mono_decay_causal(t_fit_irf, *popt_irf, irf_counts)
    model_irf_bg = irf_mono_decay_causal_bg(t_fit_irf, *popt_irf_bg, irf_counts) # NEW

    chi2_mono, sig_mono = compute_chi2_red_and_sigma(y_fit_tail, model_mono, sigma_fit_tail, p=2)
    chi2_bg,   sig_bg   = compute_chi2_red_and_sigma(y_fit_tail, model_bg, sigma_fit_tail, p=3)
    chi2_irf,  sig_irf  = compute_chi2_red_and_sigma(y_fit_irf, model_irf, sigma_fit_irf, p=3) # We are here 18:22
    chi2_irf_bg, sig_irf_bg = compute_chi2_red_and_sigma(y_fit_irf, model_irf_bg, sigma_fit_irf, p=4) # NEW
    
    import numpy as np
    import tifffile

    # =====================================================================
    # STAP 1: DEFINIEER DE BEKENDE WAARDEN VOOR C307 (ALS JE REFERENTIE)
    # =====================================================================
    # De literatuurwaarde / verwachte levensduur van jouw C307 in nanoseconden
    tau_calib_ns = 5.27                    
    frequency_hz = 100e6                    # Meetfrequentie (100 MHz)
    omega = 2 * np.pi * frequency_hz

    # Bereken de theoretische, ware hoek die hoort bij die 5.27 ns
    theta_theoretisch = np.arctan(omega * (tau_calib_ns * 1e-9))

    # =====================================================================
    # STAP 2: LAAD JOUW METINGEN IN EN BEREKEN DE RUWE HOEKEN
    # =====================================================================
    t_axis = t * DT_NS

    # A. Jouw schone C307-meting (gebruikt om de hardware-offset te vinden)
    raw_c307 = tifffile.imread("zuiver_C307.tif")
    decay_c307 = np.mean(raw_c307, axis=(1)) # Ruimtelijk gemiddelde
    _, _, theta_raw_calib = calculate_phasor_coordinates(t_axis, decay_c307, frequency_hz=frequency_hz)

    # C. Jouw FRET-sample meting (C307 + Rh110 acceptor)
    G_sample, S_sample, theta_sample = calculate_phasor_coordinates(t_axis, decay, frequency_hz=frequency_hz)

    # =====================================================================
    # STAP 3: BEREKEN DE HARDWARE-OFFSET EN CORRIGEER ALLE HOEKEN
    # =====================================================================
    # Hoeveel graden wijkt jouw microscoop af van de theoretische werkelijkheid?
    theta_offset = theta_raw_calib - theta_theoretisch

    # Trek de offset van al je hoeken af zodat ze wiskundig kloppen
    theta_donor_calibrated = theta_donor - theta_offset
    theta_sample_calibrated = theta_sample - theta_offset

    # =====================================================================
    # STAP 4: BEREKEN DE FRET-EFFICIËNTIE MET DE GECORRIGEERDE HOEKEN
    # =====================================================================
    # =====================================================================
    # BEREKENING HOEK-GEBASEERDE EFFICIËNTIE + FOUTENVOORTPLANTING
    # =====================================================================
    tan_donor = np.tan(theta_donor_calibrated)
    tan_sample = np.tan(theta_sample_calibrated)
    # we got this babay in the bag.
    print("THIS FILES HAS ...")
    print("Theta donor:", theta_donor_calibrated)
    print("Theta sample:", theta_sample_calibrated)

    if tan_donor != 0 and not np.isnan(tan_donor) and not np.isnan(tan_sample):
        E_taylor = 1.0 - (tan_sample / tan_donor)
   
        # Schatten van de hoekfout (bijv. op basis van intensiteit / shot-noise of standaard deviatie)
        # Als je een geschatte hoekfout d_theta hebt (bijv. 0.01 rad of afgeleid uit S/G spreiding):
        # Normally, first error on G and S, error on angles, and then error on Efficiency_taylor.
        total_counts_sample = np.sum(decay)
        d_theta_sample = max(1.0 / np.sqrt(total_counts_sample), 0.01)
        d_theta_donor = 0.01  # Vaste reële schatting voor je stabiele donor-referentie

        # Afgeleiden via de kettingregel (d/dtheta [tan(theta)] = 1 / cos^2(theta))
        dE_dtheta_sample = -1.0 / (tan_donor * (np.cos(theta_sample_calibrated)**2))
        dE_dtheta_donor  = (tan_sample / (tan_donor**2)) * (1.0 / (np.cos(theta_donor_calibrated)**2))
   
        E_taylor_err = np.sqrt(
           (dE_dtheta_sample * d_theta_sample)**2 +
           (dE_dtheta_donor * d_theta_donor)**2
        )
        print("E taylor error has been calculated")
    else:
        E_taylor = np.nan
        E_taylor_err = np.nan

    # =====================================================================
    # STAP 5: FYSISCHE CONTROLE (Vangnet)
    # =====================================================================
    if E_taylor < 0.0 or E_taylor > 1.0 or np.isnan(E_taylor) or np.isinf(E_taylor):
        E_taylor = np.nan
    
    print("CHECKER")
    print(f"File: {file.name} | E_taylor: {E_taylor:.4f} | Err: {E_taylor_err:.4f}")

    print(f"Berekende Gecorrigeerde FRET-efficiëntie voor dit staal: {E_taylor:.4f}")
 
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
        "E_intensity_error": E_intensity_err,
        
        "chi2_red": chi2_red,
        
        "tau_bi_ns": tau_bi_ns,
        "tau_bi_error_ns": tau_bi_err_ns,

        "tau_FRET_bi_ns": tau_FRET_bi,
        "tau_FRET_bi_error_ns": tau_FRET_bi_err,

        "FRET_efficiency_bi": E_bi,
        "FRET_efficiency_bi_error": E_bi_err,
        
        "tau_irf_ns": tau_irf_ns,
        "tau_irf_error_ns": tau_irf_err_ns,

        "tau_FRET_irf_ns": tau_FRET_irf,
        "tau_FRET_irf_error_ns": tau_FRET_irf_err,

        "FRET_efficiency_irf": E_irf,
        "FRET_efficiency_irf_error": E_irf_err,
         
        "tau_bg_ns": tau_bg_ns,
        "tau_bg_error_ns": tau_bg_err_ns,

        "background": B,

        "tau_FRET_bg_ns": tau_FRET_bg,
        "tau_FRET_bg_error_ns": tau_FRET_bg_err,

        "FRET_efficiency_bg": E_bg,
        "FRET_efficiency_bg_error": E_bg_err,
        
        "tau_shortcut_ns": tau_shortcut_ns,
        "tau_shortcut_err_ns": tau_shortcut_err_ns,
        "tau_FRET_shortcut_ns": tau_FRET_shortcut,
        "tau_FRET_shortcut_error_ns": tau_FRET_shortcut_err,
        "FRET_efficiency_shortcut": E_shortcut,
        "FRET_efficiency_shortcut_error": E_shortcut_err,
        
        "tau_rld_ns": tau_rld_ns,
        "tau_rld_err_ns": tau_rld_err_ns,
        "tau_FRET_rld_ns": tau_FRET_rld,
        "tau_FRET_rld_error_ns": tau_FRET_rld_err,
        "FRET_efficiency_rld": E_rld,
        "FRET_efficiency_rld_error": E_rld_err,
        
        "tau_centroid_ns": tau_centroid_ns,
        "tau_centroid_err_ns": tau_centroid_err_ns,
        "tau_FRET_centroid_ns": tau_FRET_centroid,
        "tau_FRET_centroid_error_ns": tau_FRET_centroid_err,
        "FRET_efficiency_centroid": E_centroid,
        "FRET_efficiency_centroid_error": E_centroid_err,
        
        # NIEUW: Chi-kwadraat resultaten opslaan!
        "chi2_mono": chi2_mono,
        "sig_chi2_mono": sig_mono,
        
        "chi2_bg": chi2_bg,
        "sig_chi2_bg": sig_bg,
        
        "chi2_irf": chi2_irf,
        "sig_chi2_irf": sig_irf,
        
        # NEW: Mono + IRF + BG Resultaten opslaan
        "tau_irf_bg_ns": tau_irf_bg_ns, 
        "tau_irf_bg_error_ns": tau_irf_bg_err_ns,
        "tau_FRET_irf_bg_ns": tau_FRET_irf_bg, 
        "tau_FRET_irf_bg_error_ns": tau_FRET_irf_bg_err,
        "FRET_efficiency_irf_bg": E_irf_bg,
        "FRET_efficiency_irf_bg_error": E_irf_bg_err,
        "chi2_irf_bg": chi2_irf_bg,
        "sig_chi2_irf_bg": sig_irf_bg, # NEW,
        
        "FRET_efficiency_phasor": E_taylor, #New
        "theta_sample": theta_sample,
        "E_taylor": E_taylor,
        "E_taylor_err": E_taylor_err,

        # Background terms stored in dictionary
        'B_bg': B,  # This is mono + BG
        'B_irf_bg': B_irf_bg  # This is mono + IRF + BG
    })
    
    # --- DEBUG CHECK VOOR ALLE HOEKEN ---
    print("DEBUG VALIDITY")
    # 1. Check of de losse hoeken binnen een logisch bereik vallen (bijv. tussen 0 en 3.14 radialen / 0 en 180 graden)
    donor_ok = "Goed" if 0 < theta_donor < 3.14 else "Vreemd!"
    sample_ok = "Goed" if 0 < theta_sample < 3.14 else "Vreemd!"

    # 2. Check of delta_theta klein genoeg is voor de Taylor-benadering
    delta_theta = theta_donor - theta_sample
    is_geldig = "Ja (< 0.35)" if abs(delta_theta) < 0.35 else "Nee (Te groot!)"

    # 3. Print alles overzichtelijk naast elkaar met een duidelijke kop
    # (Zet deze print-kop eenmalig VÓÓR je loop)
    # print(f"{'Donor (rad)':<12} | {'Check':<8} | {'Sample (rad)':<12} | {'Check':<8} | {'Delta (rad)':<12} | {'Taylor Geldig?':<15}")
    print(f"{theta_donor:<12.4f} | {donor_ok:<8} | {theta_sample:<12.4f} | {sample_ok:<8} | {delta_theta:<12.4f} | {is_geldig:<15}")

    
    # NEW: Sla peak_index en fit_end op zodat we in de subplots de tail-fits tot de piek kunnen extrapoleren
    plot_data.append((
      file.name, t, decay,
      t_fit_tail, fit_start_tail, popt, popt_bg,
      t_fit_irf, fit_start_irf, popt_irf, popt_irf_bg,
      peak_index, fit_end
    ))
    
    # 3. ZET DE CHECK HIER:
    # Verzamel eerst alle berekende hoeken uit je resultaten (of zorg dat je ze in een lijst bewaart)
    theta_sample_array = [res['theta_sample'] for res in results if 'theta_sample' in res]

    # Om te controleren of alle hoeken kleiner zijn dan bijv. 1.0 radiaal (~57 graden)
    max_hoek_rad = np.max([theta_donor, np.max(theta_sample_array)])
    max_hoek_deg = np.rad2deg(max_hoek_rad)

    print(f"Maximale fasehoek in de dataset: {max_hoek_rad:.3f} rad ({max_hoek_deg:.1f}°)")

    if max_hoek_rad < 1.0:
        print("Conclusie: Aan de voorwaarde voor kleine hoeken (theta < 1.0 rad) is voldaan!")
    else:
        print("Let op: Sommige hoeken zijn groter dan 1.0 rad, de Taylor-benadering kan hier iets afwijken.")

for r in results:
    v1 = r["v_low"]
    v2 = r["v_high"]

    C_LOW  = 0.005
    C_HIGH = 0.05

    uL_to_L = 1e-6
    mL_to_L = 1e-3

    V_low  = v1 * uL_to_L
    V_high = v2 * uL_to_L # 21:21 now
    V_C307 = 1.0 * mL_to_L

    V_total = V_C307 + V_low + V_high

    r["c_eff"] = (C_LOW*V_low + C_HIGH*V_high) / V_total
    
    # -------------------------------
    # new uncertainty on c_eff with correct essay formula
    # -------------------------------

    sigma_VL = 0.05 * V_low   # example 5%
    sigma_VH = 0.05 * V_high

    # Noemer in het kwadraat voor de partiële afgeleiden
    D_squared = V_total ** 2

    # Partiële afgeleide naar V_low
    d_dV_low = (C_LOW * V_C307 + C_LOW * V_high - C_HIGH * V_high) / D_squared

    # Partiële afgeleide naar V_high
    d_dV_high = (C_HIGH * V_C307 + C_HIGH * V_low - C_LOW * V_low) / D_squared

    # Volledige foutpropagatie formule
    c_err = np.sqrt(
        (d_dV_low * sigma_VL) ** 2 +
        (d_dV_high * sigma_VH) ** 2
    )

    r["c_eff_error"] = c_err

    # -------------------------------
    # Propagate c_eff → r uncertainty
    # ------------------------------- # 1844

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
# NEW: Plot de evolutie van de achtergrondterm B per DA-staal
# --------------------------------------------------
from matplotlib.lines import Line2D
# --------------------------------------------------
# SORTEER DE DATA NUMERIEK OP V_LOW EN V_HIGH
# --------------------------------------------------
# Sorting, because (85,5) has to go way before (85,10)
# Computer doesn't know this, so we code this in.
results_sorted = sorted(results, key=lambda r: (r['v_low'], r['v_high']))

v_lows = [r['v_low'] for r in results_sorted]
v_highs = [r['v_high'] for r in results_sorted]
b_values_bg = np.array([r['B_bg'] for r in results_sorted])
b_values_irf_bg = np.array([r['B_irf_bg'] for r in results_sorted])

# When dimmer file is applied, we check for its value
# This is a clever trick to get that.
print("MINIMUM IRF + BG")
print(np.min(b_values_irf_bg))

x_labels = [f"({v_l:.0f}, {v_h:.0f})" for v_l, v_h in zip(v_lows, v_highs)]
x_indices = np.arange(len(results_sorted))
# 22:50, time for chilling, this is backup return key
# --- (Je data en plt.plot lijnen staan hier al) ---
plt.figure(figsize=(9, 5))

plt.plot(x_indices, b_values_bg, marker='D', color='red', linestyle='-', label='Mono + BG (Lijn)')
plt.plot(x_indices, b_values_irf_bg, marker='s', color='darkgreen', linestyle='-', label='Mono + IRF + BG (Lijn)')

# --- MARKERINGEN (zonder label in de scatter zelf) ---
# Mono + BG (Rood)
max_bg_idx = np.argmax(b_values_bg)
min_bg_idx = np.argmin(b_values_bg)
plt.scatter(x_indices[max_bg_idx], b_values_bg[max_bg_idx], marker='*', color='red', s=110, edgecolor='black', linewidth=0.8, zorder=5)
plt.scatter(x_indices[min_bg_idx], b_values_bg[min_bg_idx], marker='x', color='black', s=130, linewidths=3.5, zorder=4)
plt.scatter(x_indices[min_bg_idx], b_values_bg[min_bg_idx], marker='x', color='red', s=110, linewidths=1.8, zorder=5)

# Mono + IRF + BG (Donkergroen)
max_irf_idx = np.argmax(b_values_irf_bg)
min_irf_idx = np.argmin(b_values_irf_bg)
plt.scatter(x_indices[max_irf_idx], b_values_irf_bg[max_irf_idx], marker='*', color='darkgreen', s=110, edgecolor='black', linewidths=0.8, zorder=5)
plt.scatter(x_indices[min_irf_idx], b_values_irf_bg[min_irf_idx], marker='x', color='black', s=130, linewidths=3.5, zorder=4)
plt.scatter(x_indices[min_irf_idx], b_values_irf_bg[min_irf_idx], marker='x', color='darkgreen', s=110, linewidths=1.8, zorder=5)


# --- SLIMME LEGENDA AANMAKEN ---
# Hiermee dwingen we af dat de legenda de juiste symbolen toont zonder dat matplotlib ze verknoeit:
legend_elements = [
    # --- Categorie 1: Mono + IRF + BG ---
    # Dit werkt als een "titel" (onzichtbaar lijntje of marker, alleen tekst)
    plt.Line2D([], [], color='none', label=r'$\mathbf{Mono+IRF+BG}$'),
    plt.Line2D([0], [0], color='darkgreen', marker='s', linestyle='-', markersize=7, label='Data'),
    plt.Line2D([0], [0], color='darkgreen', marker='*', linestyle='None', markersize=9, markeredgecolor='black', markeredgewidth=0.8, label='Maximum'),
    plt.Line2D([0], [0], color='darkgreen', marker='x', linestyle='None', markersize=8, markeredgewidth=1.8, label='Minimum'),
   
    # --- Categorie 2: Mono + BG ---
    plt.Line2D([], [], color='none', label=r'$\mathbf{Mono+BG}$'),
    plt.Line2D([0], [0], color='red', marker='D', linestyle='-', markersize=7, label='Data'),
    plt.Line2D([0], [0], color='red', marker='*', linestyle='None', markersize=9, markeredgecolor='black', markeredgewidth=0.8, label='Maximum'),
    plt.Line2D([0], [0], color='red', marker='x', linestyle='None', markersize=8, markeredgewidth=1.8, label='Minimum')
]

# Opmaak van de plot
plt.xlabel("Volume combinatie ($V_{\\text{low}}$, $V_{\\text{high}}$ in $\\mu$L)", fontsize=11)
plt.ylabel("Gefitte achtergrondwaarde $B$ (counts)", fontsize=11)
plt.title("Evolutie van de achtergrondcorrectieterm $B$ voor elk donnor-acceptorstaal", fontsize=12, fontweight="bold")

plt.xticks(x_indices, x_labels, rotation=45, ha='right', fontsize=9)
plt.grid(True, linestyle='--', alpha=0.4)

# Roep de handgemaakte legenda aan in 2 kolommen voor een compacte look
# Om te zorgen dat het in een mooie strakke structuur komt te staan
# (bijvoorbeeld in 2 kolommen waarbij de kopjes en items netjes vallen),
# kun je de legenda als volgt aanroepen:
plt.legend(handles=legend_elements, fontsize=9, loc="upper right", frameon=True, ncol=1)
plt.tight_layout()

plt.savefig("background_evolution_min_max_perfect_legend.png", dpi=600, bbox_inches="tight")
plt.show()

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

# --------------------------------------------------
# NEW: FIGURE All fluorescence decay fits (Met Gestippelde Extrapolatie op alle subplots)
# --------------------------------------------------
all_groups = plot_data  
n = len(all_groups)
cols = 3
rows = int(np.ceil(n / cols))

fig, axs = plt.subplots(rows, cols, figsize=(3.5 * cols, 2.2 * rows))
axs = axs.flatten()

for i, (name, t_all, decay, t_fit_tail, fit_start_tail, popt_mono, popt_bg, t_fit_irf, fit_start_irf, popt_irf, popt_irf_bg, peak_idx_sample, fit_end_sample) in enumerate(plot_data):

    v_low, v_high = extract_volumes(name)
    if v_high == 0:
        title = rf"$V_{{laag,\ Rh110}} = {v_low:.0f}\ \mu L$"
    else:
        title = rf"$V_{{laag}} = {v_low:.0f}\ \mu L,\ V_{{hoog}} = {v_high:.0f}\ \mu L$"

    time_ns = t_all[:CUTOFF_PIXEL] * DT_NS
    decay_plot = decay[:CUTOFF_PIXEL]
   
    # 1. Harde fitlijn (op daadwerkelijk fitbereik fit_start_tail tot fit_end_sample)
    t_tail_fit_s = t_all[fit_start_tail:fit_end_sample]
    t_tail_fit_shift_s = t_tail_fit_s - t_all[fit_start_tail]
    time_fit_tail_ns_s = t_tail_fit_s * DT_NS

    # 2. Gestippelde extrapolatie (van peak_idx_sample tot fit_start_tail)
    t_tail_ext_s = t_all[peak_idx_sample:fit_start_tail + 1]
    t_tail_ext_shift_s = t_tail_ext_s - t_all[fit_start_tail]
    time_ext_tail_ns_s = t_tail_ext_s * DT_NS

    time_fit_irf_ns = (t_fit_irf + fit_start_irf) * DT_NS

    # Ruwe Data in ZWART
    axs[i].plot(time_ns, decay_plot, color='black', alpha=0.6, linewidth=1.2, label="Data")
   
    # Fit Mono (BLAUW: hard voor fitbereik, gestippeld voor geëxtrapoleerd deel)
    axs[i].plot(time_fit_tail_ns_s, exponential_decay(t_tail_fit_shift_s, *popt_mono),
                color='blue', linewidth=1.5, label="Mono fit")
    axs[i].plot(time_ext_tail_ns_s, exponential_decay(t_tail_ext_shift_s, *popt_mono),
                color='blue', linewidth=1.5, linestyle=':')

    # Fit Mono + BG (ROOD: hard voor fitbereik, gestippeld voor geëxtrapoleerd deel)
    axs[i].plot(time_fit_tail_ns_s, exponential_background_decay(t_tail_fit_shift_s, *popt_bg),
                color='red', linewidth=1.5, label="Mono + BG fit")
    axs[i].plot(time_ext_tail_ns_s, exponential_background_decay(t_tail_ext_shift_s, *popt_bg),
                color='red', linewidth=1.5, linestyle=':')

    # Fit Mono + IRF (MAGENTA)
    axs[i].plot(time_fit_irf_ns, irf_mono_decay_causal(t_fit_irf, *popt_irf, irf_counts),
                color='magenta', linewidth=1.2, alpha=0.7, label="Mono + IRF fit")

    # Fit Mono + IRF + BG (DONKERGROEN)
    axs[i].plot(time_fit_irf_ns, irf_mono_decay_causal_bg(t_fit_irf, *popt_irf_bg, irf_counts),
                color='darkgreen', linewidth=1.5, label="Mono + IRF + BG fit")

    axs[i].set_title(title, fontsize=9)
    axs[i].grid(True, linestyle='--', alpha=0.4)
    axs[i].legend(fontsize=5.5, loc="best")

for j in range(len(plot_data), len(axs)):
    fig.delaxes(axs[j])

fig.supxlabel("Tijd (ns)", fontsize=10)
fig.supylabel("Fluorescentie-intensiteit (counts)")
fig.suptitle(r"Fluorescentie vervalfits: samples ($C307$ met $Rh110$ toegevoegd)", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("all_rh110_added_samples_data_with_fit.png", dpi=600, bbox_inches="tight")
plt.show()

##### DEBUG

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n======================")
print("FITTED LIFETIMES")
print("======================")

print(f"{'File':40s} || {'Low vol (µL)':12s} {'High vol (µL)':12s} {'c_eff (M)':14s}  {'±':14s} || {'tau (pix)':>12s} {'±':>8s} || {'tau (ns)':>12s} {'±':>8s} || {'tau_FRET (ns)':>14s} {'±':>10s} || {'r_model1 (arb)':>14s} {'±':>10s} || {'r_model2 (arb)':>14s} {'±':>10s}")

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
        
        f"{r['chi2_red']:10.3f}"
    )

chi = [r["chi2_red"] for r in results]

plt.figure(figsize=(5,4))

plt.hist(chi, bins=8)

plt.axvline(
    1,
    color="green",
    linestyle="--",
    label=r"Ideal $\chi^2_\nu = 1$"
)

plt.xlabel(r"Reduced $\chi^2$")
plt.ylabel("Number of fits")

plt.legend()
plt.tight_layout()

#####################################################################################################################################################################

##duplicate ###################################################################################################################################################################

# plot 1

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# --------------------------------------------------
# Physical constants
# --------------------------------------------------

NA = 6.02214076e23      # mol^-1

# --------------------------------------------------
# Extract valid data
# --------------------------------------------------
# back up 15:50
# --------------------------------------------------
# Extract valid data safely (Use unique names to avoid cross-contamination)
# --------------------------------------------------

# Replace the entire extraction block with this:
c_raw = np.array([r["c_eff"] for r in results])

# 1. Mono-exponential data
tau_mono_raw = np.array([r["tau_FRET_ns"] for r in results])
tau_mono_err_raw = np.array([r["tau_FRET_error_ns"] for r in results])
mask_mono = (tau_mono_raw > 0) & (~np.isnan(tau_mono_raw)) & (c_raw > 0)
tau = tau_mono_raw[mask_mono]
tau_err = tau_mono_err_raw[mask_mono]
c = c_raw[mask_mono]
log_tau_err = tau_err / tau

# 2. Background lifetime data
tau_bg_raw = np.array([r["tau_FRET_bg_ns"] for r in results])
tau_bg_err_raw = np.array([r["tau_FRET_bg_error_ns"] for r in results])
mask_bg = (tau_bg_raw > 0) & (~np.isnan(tau_bg_raw)) & (c_raw > 0)
tau_bg = tau_bg_raw[mask_bg]
tau_bg_err = tau_bg_err_raw[mask_bg]
c_bg = c_raw[mask_bg]
log_tau_bg_err = tau_bg_err / tau_bg

# 3. Biexponential lifetime data
tau_bi_raw = np.array([r["tau_FRET_bi_ns"] for r in results])
tau_bi_err_raw = np.array([r["tau_FRET_bi_error_ns"] for r in results])
mask_bi = (tau_bi_raw > 0) & (~np.isnan(tau_bi_raw)) & (c_raw > 0)
tau_bi = tau_bi_raw[mask_bi]
tau_bi_err = tau_bi_err_raw[mask_bi]
c_bi = c_raw[mask_bi]
log_tau_bi_err = tau_bi_err / tau_bi

# 4. IRF corrected lifetime data
tau_irf_raw = np.array([r["tau_FRET_irf_ns"] for r in results])
tau_irf_err_raw = np.array([r["tau_FRET_irf_error_ns"] for r in results])
mask_irf = (tau_irf_raw > 0) & (~np.isnan(tau_irf_raw)) & (c_raw > 0)
tau_irf = tau_irf_raw[mask_irf]
tau_irf_err = tau_irf_err_raw[mask_irf]
c_irf = c_raw[mask_irf]
log_tau_irf_err = tau_irf_err / tau_irf

# 4. NEW: Mono + IRF + BG Data Extraction
tau_irf_bg_raw = np.array([r["tau_FRET_irf_bg_ns"] for r in results])
tau_irf_bg_err_raw = np.array([r["tau_FRET_irf_bg_error_ns"] for r in results])
mask_irf_bg = (tau_irf_bg_raw > 0) & (~np.isnan(tau_irf_bg_raw)) & (c_raw > 0)
tau_irf_bg = tau_irf_bg_raw[mask_irf_bg]
tau_irf_bg_err = tau_irf_bg_err_raw[mask_irf_bg]
c_irf_bg = c_raw[mask_irf_bg]
log_tau_irf_bg_err = tau_irf_bg_err / tau_irf_bg

# NEW: Regressie voor Mono + IRF + BG
r_nm_irf_bg = ((c_irf_bg*1000*NA))**(-1/3) * 1e9
log_r_irf_bg = np.log(r_nm_irf_bg)
log_tau_irf_bg = np.log(tau_irf_bg)
slope_irf_bg, intercept_irf_bg, rvalue_irf_bg, _, std_err_irf_bg = linregress(log_r_irf_bg, log_tau_irf_bg)
R2_irf_bg = rvalue_irf_bg**2
xfit_irf_bg = np.linspace(log_r_irf_bg.min(), log_r_irf_bg.max(), 200)
yfit_irf_bg = slope_irf_bg*xfit_irf_bg + intercept_irf_bg

# 3. NEW: Shortcut lifetime data Filtering
tau_sc_raw = np.array([r["tau_FRET_shortcut_ns"] for r in results])
tau_sc_err_raw = np.array([r["tau_FRET_shortcut_error_ns"] for r in results])
mask_sc = (tau_sc_raw > 0) & (~np.isnan(tau_sc_raw)) & (c_raw > 0)
tau_sc = tau_sc_raw[mask_sc]
tau_sc_err = tau_sc_err_raw[mask_sc]
c_sc = c_raw[mask_sc]
log_tau_sc_err = tau_sc_err / tau_sc

# 4. NEW: RLD Shortcut Filtering
tau_rld_raw = np.array([r["tau_FRET_rld_ns"] for r in results])
tau_rld_err_raw = np.array([r["tau_FRET_rld_error_ns"] for r in results])
mask_rld = (tau_rld_raw > 0) & (~np.isnan(tau_rld_raw)) & (c_raw > 0)
tau_rld = tau_rld_raw[mask_rld]
tau_rld_err = tau_rld_err_raw[mask_rld]
c_rld = c_raw[mask_rld]
log_tau_rld_err = tau_rld_err / tau_rld

# 6. Shortcut Centroid data
tau_cen_raw = np.array([r["tau_FRET_centroid_ns"] for r in results])
tau_cen_err_raw = np.array([r["tau_FRET_centroid_error_ns"] for r in results])
mask_cen = (tau_cen_raw > 0) & (~np.isnan(tau_cen_raw)) & (c_raw > 0)
tau_cen = tau_cen_raw[mask_cen]
tau_cen_err = tau_cen_err_raw[mask_cen]
c_cen = c_raw[mask_cen]
log_tau_cen_err = tau_cen_err / tau_cen


r_nm_cen = (c_cen*1000*NA)**(-1/3)*1e9
log_r_cen = np.log(r_nm_cen)
log_tau_cen = np.log(tau_cen)


slope_cen, intercept_cen, rvalue_cen, _, std_err_cen = linregress(log_r_cen, log_tau_cen)
R2_cen = rvalue_cen**2
xfit_cen = np.linspace(log_r_cen.min(), log_r_cen.max(), 200)
yfit_cen = slope_cen*xfit_cen + intercept_cen

r_nm_rld = (c_rld*1000*NA)**(-1/3)*1e9
log_r_rld = np.log(r_nm_rld)
log_tau_rld = np.log(tau_rld)

slope_rld, intercept_rld, rvalue_rld, _, std_err_rld = linregress(log_r_rld, log_tau_rld)
R2_rld = rvalue_rld**2
xfit_rld = np.linspace(log_r_rld.min(), log_r_rld.max(), 200)
yfit_rld = slope_rld*xfit_rld + intercept_rld

r_nm_sc = (c_sc*1000*NA)**(-1/3)*1e9
log_r_sc = np.log(r_nm_sc)
log_tau_sc = np.log(tau_sc)

slope_sc, intercept_sc, rvalue_sc, _, std_err_sc = linregress(log_r_sc, log_tau_sc)
R2_sc = rvalue_sc**2
xfit_sc = np.linspace(log_r_sc.min(), log_r_sc.max(), 200)
yfit_sc = slope_sc*xfit_sc + intercept_sc

r_nm_bg = (
    c_bg*1000*NA
)**(-1/3)*1e9

log_r_bg = np.log(r_nm_bg)
log_tau_bg = np.log(tau_bg)

slope_bg, intercept_bg, rvalue_bg, _, std_err_bg = linregress(
    log_r_bg,
    log_tau_bg
)

R2_bg = rvalue_bg**2


xfit_bg = np.linspace(
    log_r_bg.min(),
    log_r_bg.max(),
    200
)

yfit_bg = slope_bg*xfit_bg + intercept_bg

r_nm_bi = (
    (c_bi*1000*NA)
)**(-1/3) * 1e9

log_r_bi = np.log(r_nm_bi)
log_tau_bi = np.log(tau_bi)

#slope_bi, intercept_bi, rvalue_bi, _, std_err_bi = linregress(
    #log_r_bi,
    #log_tau_bi
#)

#R2_bi = rvalue_bi**2

#xfit_bi = np.linspace(
 #   log_r_bi.min(),
 #   log_r_bi.max(),
 #   200
#)

# yfit_bi = slope_bi*xfit_bi + intercept_bi

# --------------------------------------------------
# IRF corrected lifetime data
# --------------------------------------------------


r_nm_irf = (
    (c_irf*1000*NA)
)**(-1/3) * 1e9


log_r_irf = np.log(r_nm_irf)
log_tau_irf = np.log(tau_irf)


slope_irf, intercept_irf, rvalue_irf, _, std_err_irf = linregress(
    log_r_irf,
    log_tau_irf
)

R2_irf = rvalue_irf**2


xfit_irf = np.linspace(
    log_r_irf.min(),
    log_r_irf.max(),
    200
)

yfit_irf = slope_irf*xfit_irf + intercept_irf

# --------------------------------------------------
# Cubic lattice distance estimate
# --------------------------------------------------

# mol/L  --> mol/m^3
c_m3 = c * 1000

# molecules/m^3
number_density = c_m3 * NA

# average spacing (m)
r = number_density**(-1/3)

# convert to nm
r_nm = r * 1e9

# --------------------------------------------------
# Logarithmic quantities
# --------------------------------------------------

log_r = np.log(r_nm)
log_tau = np.log(tau)

# uncertainty on log(tau)
log_tau_err = tau_err / tau

# --------------------------------------------------
# Linear regression
# --------------------------------------------------

slope, intercept, r_value, p_value, std_err = linregress(
    log_r,
    log_tau
)

R2 = r_value**2

xfit = np.linspace(log_r.min(), log_r.max(), 200)
yfit = slope * xfit + intercept

# --------------------------------------------------
# Theoretical Förster slope = 6
# --------------------------------------------------

# force theory line to pass through average experimental offset
theory_intercept = np.mean(log_tau - 6*log_r)

y_theory = 6*xfit + theory_intercept

y_theory3 = 3 * xfit + np.mean(log_tau - 3 * log_r)

# --------------------------------------------------
# Plot
# --------------------------------------------------

from matplotlib.lines import Line2D

plt.figure(figsize=(10,7))


# --------------------------------------------------
# Plot data and fits
# --------------------------------------------------

# Mono experimental data
plt.errorbar(
    log_r,
    log_tau,
    yerr=log_tau_err,
    fmt='o',
    color='blue',
    capsize=3
)


# Mono regression
plt.plot(
    xfit,
    yfit,
    color="blue",
    linewidth=2
)


# Bi-exp experimental data
#plt.errorbar(
    #log_r_bi,
   # log_tau_bi,
   # yerr=log_tau_bi_err,
   # fmt="s",
   # color="red",
   # capsize=3
#)


# Bi-exp regression
#plt.plot(
   # xfit_bi,
   # yfit_bi,
   # color="red",
   # linewidth=2
#)

# IRF mono data

plt.errorbar(
    log_r_irf,
    log_tau_irf,
    yerr=log_tau_irf_err,
    fmt="h",
    color="magenta",
    capsize=3
)

# IRF regression

plt.plot(
    xfit_irf,
    yfit_irf,
    color="magenta",
    linewidth=2
)

# NEW: Plot Mono + IRF + BG Data & Fit
plt.errorbar(log_r_irf_bg, log_tau_irf_bg, yerr=log_tau_irf_bg_err, fmt="s", color="darkgreen", capsize=3)
plt.plot(xfit_irf_bg, yfit_irf_bg, color="darkgreen", linewidth=2)

# Background mono

plt.errorbar(
    log_r_bg,
    log_tau_bg,
    yerr=log_tau_bg_err,
    fmt="D",
    color="red",
    capsize=3
)

plt.errorbar(log_r_cen, log_tau_cen, yerr=log_tau_cen_err, fmt="p", color="#8c564b", capsize=3)
plt.plot(xfit_cen, yfit_cen, color="#8c564b", linewidth=2)


plt.plot(
    xfit_bg,
    yfit_bg,
    color="red",
    linewidth=2
)

# Plot NEW Heuristic Shortcut in Forest Green
plt.errorbar(log_r_sc, log_tau_sc, yerr=log_tau_sc_err, fmt="^", color="darkorange", capsize=3)
plt.plot(xfit_sc, yfit_sc, color="darkorange", linewidth=2)

# Plot NEW RLD Shortcut in Dark Violet
plt.errorbar(log_r_rld, log_tau_rld, yerr=log_tau_rld_err, fmt="*", color="darkviolet", capsize=3)
plt.plot(xfit_rld, yfit_rld, color="darkviolet", linewidth=2)

# Theory line m = 6
plt.plot(
    xfit,
    y_theory,
    linestyle="--",
    linewidth=2,
    color="slategrey"
)

# Theory line m = 3
#plt.plot(
 #   xfit,
  #  y_theory3,
  #  linestyle="--",
  #  linewidth=2,
  #  color="maroon"
#)

# ==============================================================================
# ANNOTATIES PLOT 1 (Strakke opmaak: zwarte pijlen, witte kaders, scherpe hoeken)
# ==============================================================================

# 1. CENTROID UITSCHIETER (Rechtsboven geplaatst)
if len(tau_cen) > 0:
    idx_max_cen = np.argmax(tau_cen)
    x_cent = log_r_cen[idx_max_cen]
    y_cent = log_tau_cen[idx_max_cen]

    # plt.annotate(
      #  "Centroid-uitschieter:\nGevoeligheid voor tail-ruis\noverschat $\\tau_{\\mathrm{FRET}}$ systematisch",
       # xy=(x_cent, y_cent),
       # xytext=(x_cent - 0.5, y_cent -1.7),  # Veel meer naar rechtsboven
      #  fontsize=8.5,
      #  bbox=dict(boxstyle="square,pad=0.4", fc="white", ec="black", lw=0.8)  # Scherpe hoeken, neutraal
    #)

# 2. BEGIN DIVERGENTIE (Naar rechts verschoven)
if len(tau_irf_bg) > 0:
    idx_min_r = np.argmin(r_nm_irf_bg)
    x_div = log_r_irf_bg[idx_min_r]
    y_div = log_tau_irf_bg[idx_min_r]

    plt.annotate(
        "Zone van divergentie:\nSterke spreiding van shortcut-methodes\ndoor gevoeligheid voor IRF-vervorming.",
        xy=(x_div, y_div),
        xytext=(x_div + 0.25, y_div - 1.4),  # Meer naar rechts verschoven
        fontsize=8.5,
        bbox=dict(boxstyle="square,pad=0.4", fc="white", ec="black", lw=0.8)
    )

# 3. EINDE CONVERGENTIE (Naar beneden verschoven)
if len(tau_irf_bg) > 0:
    idx_max_r = np.argmax(r_nm_irf_bg)
    x_conv = log_r_irf_bg[idx_max_r]
    y_conv = log_tau_irf_bg[idx_max_r]

   # plt.annotate(
       # "Convergentiepunt:\nIRF-loze fits overlappen volledig.\ndoordat B = 0 gekozen is.",
       # xy=(x_conv, y_conv),
       # xytext=(x_conv - 0.3, y_conv - 2.9),  # Duidelijk naar beneden verschoven
       # fontsize=8.5,
       # bbox=dict(boxstyle="square,pad=0.4", fc="white", ec="black", lw=0.8)
   # )

# ==============================================================================
# INSET SUBPLOT: PLOT 1 (LOG-LOG TAU_FRET) - SMALLER CONVERGENTIEGEBIED
# ==============================================================================
ax_main1 = plt.gca()
# Positionering rechtsonder
ax_inset1 = ax_main1.inset_axes([0.615, 0.06, 0.35, 0.25])

# 1. Theoretische m=3 referentielijn in de inset
x_fit_in1 = np.linspace(np.min(log_r), np.max(log_r), 100)
# y_theory3_in1 = 3 * x_fit_in1 + np.mean(log_tau - 3 * log_r)
# ax_inset1.plot(x_fit_in1, y_theory3_in1, color="maroon", linestyle="--", linewidth=1.2, alpha=0.8)

# 2. Exact dezelfde methoden met identieke kleuren en markers
methods_plot1 = [
    # Four fitmethodes
    (log_r,        log_tau,        log_tau_err,        'o', 'blue',      'Mono'),
    (log_r_bg,     log_tau_bg,     log_tau_bg_err,     'D', 'red',       'Mono+BG'),
    (log_r_irf,    log_tau_irf,    log_tau_irf_err,    'h', 'magenta',   'Mono+IRF'),
    (log_r_irf_bg, log_tau_irf_bg, log_tau_irf_bg_err, 's', 'darkgreen', 'Mono+IRF+BG'),
    # Three shortcuts
    (log_r_sc,     log_tau_sc,     log_tau_sc_err,     '^', 'darkorange', '1/e'),
    (log_r_rld,    log_tau_rld,    log_tau_rld_err,    '*', 'purple',    'RLD'),
    (log_r_cen,    log_tau_cen,    log_tau_cen_err,    'H', 'brown',     'Centroid'),
]

for x_data, y_data, y_err, marker, col, label_name in methods_plot1:
    if len(x_data) > 0:
        s_idx = np.argsort(x_data)
        xs, ys, es = x_data[s_idx], y_data[s_idx], y_err[s_idx]
       
        # Datapunten met foutbalken (geen lijnen tussen datapunten)
        ax_inset1.errorbar(
            xs, ys, yerr=es,
            fmt=marker, color=col, ecolor=col, ms=3.5, capsize=2,
            linestyle='none', elinewidth=0.7, label=label_name
        )
       
        # Lineaire regressie fitlijn met exact dezelfde kleur
        if len(xs) > 1:
            m_fit, c_fit = np.polyfit(xs, ys, 1)
            x_range = np.linspace(np.min(xs), np.max(xs), 50)
            ax_inset1.plot(x_range, m_fit * x_range + c_fit, color=col, linestyle='-', linewidth=0.9, alpha=0.8)

# 3. Smaller convergentiegebied (focus op de laatste 3 punten i.p.v. 5)
if len(log_r_irf_bg) >= 3:
    s_idx = np.argsort(log_r_irf_bg)
    x_conv = log_r_irf_bg[s_idx][-3:]
    y_conv = log_tau_irf_bg[s_idx][-3:]
   
    # Smaller x-bereik voor sterkere zoom
    ax_inset1.set_xlim(np.min(x_conv) - 0.008, np.max(x_conv) + 0.008)
    ax_inset1.set_ylim(np.min(y_conv) - 0.15, np.max(y_conv) + 0.15)

ax_inset1.tick_params(labelsize=6)
ax_inset1.grid(False)

# Verbind het zoomvak met de convergentiezone
# ax_main1.indicate_inset_zoom(ax_inset1, edgecolor="gray", alpha=0.6)

# --------------------------------------------------
# Custom legend (fixed order)
# --------------------------------------------------

legend_elements = [

    # --------------------------------------------------
    # THEORY
    # --------------------------------------------------

    Line2D(
        [],
        [],
        linestyle="None",
        label="THEORY"
    ),

    Line2D(
        [0],
        [0],
        color="slategrey",
        linestyle="--",
        linewidth=1,
        label="Theory: Isolated single pair model: m = 6"
    ),
    
    # Line2D(
      #  [0],
      #  [0],
      #  color="maroon",
      #  linestyle="--",
      #  linewidth=1,
      #  label="Theory: 3D fluid continuum model: m = 3"
    #),

    # --------------------------------------------------
    # REGRESSION FITS
    # --------------------------------------------------

    Line2D(
        [],
        [],
        linestyle="None",
        label="REGRESSION FITS"
    ),


    # Mono fit
    Line2D(
        [0],
        [0],
        color="blue",
        linewidth=2,
        label=f"Mono exp fit: m={slope:.2f} ± {std_err:.2f}, R²={R2:.3f}"
    ),


    # Mono + BG fit
    Line2D(
        [0],
        [0],
        color="red",
        linewidth=2,
        label=f"Mono exp + BG fit: m={slope_bg:.2f} ± {std_err_bg:.2f}, R²={R2_bg:.3f}"
    ),


    # IRF mono fit
    Line2D(
        [0],
        [0],
        color="magenta",
        linewidth=2,
        label=f"Mono exp + IRF fit: m={slope_irf:.2f} ± {std_err_irf:.2f}, R²={R2_irf:.3f}"
    ),


    # Bi-exp fit
    #Line2D(
     #   [0],
     #   [0],
     #   color="red",
     #   linewidth=2,
     #   label=f"Bi-exp fit: m={slope_bi:.2f} ± {std_err_bi:.2f}, R²={R2_bi:.3f}"
    #),
    
    Line2D([0], [0], color="darkgreen", linewidth=2, label=f"Mono exp + IRF + BG fit: m={slope_irf_bg:.2f} ± {std_err_irf_bg:.2f}, R²={R2_irf_bg:.3f}"), # NEW
    Line2D([0], [0], color="darkorange", linewidth=2, label=f"1/e fit (shortcut): m={slope_sc:.2f} ± {std_err_sc:.2f}, R²={R2_sc:.3f}"),
    Line2D([0], [0], color="darkviolet", linewidth=2, label=f"RLD Shortcut fit (shortcut): m={slope_rld:.2f} ± {std_err_rld:.2f}, R²={R2_rld:.3f}"),
    Line2D([0], [0], color="#8c564b", linewidth=2, label=f"Centroid fit (shortcut): m={slope_cen:.2f} ± {std_err_cen:.2f}, R²={R2_cen:.3f}"),

    # --------------------------------------------------
    # EXPERIMENTAL DATA
    # --------------------------------------------------

    Line2D(
        [],
        [],
        linestyle="None",
        label="EXPERIMENTAL DATA"
    ),


    # Mono data
    Line2D(
        [0],
        [0],
        marker="o",
        color="blue",
        linestyle="None",
        label="Mono exp data"
    ),


    # Mono + BG data
    Line2D(
        [0],
        [0],
        marker="D",
        color="red",
        linestyle="None",
        label="Mono exp + BG data"
    ),


    # IRF mono data
    Line2D(
        [0],
        [0],
        marker="h",
        color="magenta",
        linestyle="None",
        label="Mono exp + IRF data"
    ),

    # Bi-exp data
    #Line2D(
      #  [0],
      #  [0],
      #  marker="s",
      #  color="red",
       # linestyle="None",
       # label="Bi-exp data"
   # )

    Line2D([0], [0], marker="s", color="darkgreen", linestyle="None", label="Mono exp + IRF + BG data"), # NEW
    Line2D([0], [0], marker="^", color="darkorange", linestyle="None", label="1/e data (shortcut)"),
    Line2D([0], [0], marker="*", color="darkviolet", linestyle="None", label="RLD data (shortcut)"),
    Line2D([0], [0], marker="p", color="#8c564b", linestyle="None", label="Centroid data (shortcut)")
]

leg = plt.legend(
    handles=legend_elements,
    loc="best",
    fontsize=8,
    ncol=2,
    frameon=True,
    framealpha=0.9,
    handlelength=1.5,
    handletextpad=0.5,
    labelspacing=0.4,
    borderpad=0.5
)

for text in leg.get_texts():
    if text.get_text().strip() in [
        "THEORY",
        "REGRESSION FITS",
        "EXPERIMENTAL DATA"
    ]:
        text.set_weight("bold")

leg._legend_box.align = "left"

# --------------------------------------------------
# Axis formatting
# --------------------------------------------------

plt.xlabel(
    r"$\log(r\,\mathrm{(nm)})$"
)

plt.ylabel(
    r"$\log(\tau_{\mathrm{FRET}}\mathrm{\ (ns)})$"
)


plt.title(
    "Förster law validation using FRET-lifetimes"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "bob1",
    dpi=600,
    bbox_inches="tight"
)


plt.show()


# --------------------------------------------------
# Print results
# --------------------------------------------------

print(f"Slope = {slope:.3f} ± {std_err:.3f}")
print(f"R² = {R2:.4f}")
print("Theoretical Förster exponent = 6")

print(f"{'File':<40} {'r (nm)':>10} {'σ_r (nm)':>12}")
print("-" * 65)

for sample in results:

    if np.isnan(sample["c_eff"]):
        continue

    c = sample["c_eff"]
    sigma_c = sample["c_eff_error"]

    r_nm = 1e8 * (NA * c) ** (-1/3)

    sigma_r = (
        (1/3)
        * 1e8
        * (NA ** (-1/3))
        * (c ** (-4/3))
        * sigma_c
    )

    print(
        f"{sample['file']:<40}"
        f"{r_nm:10.2f}"
        f"{sigma_r:12.2f}"
    )
    
# -------------------------------------------------- # Plot 2
# Förster law validation using lifetime efficiency -----------------------------------------------------------------------------------------------
# --------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


NA = 6.02214076e23   # Avogadro constant


r_values = []
E_values = []
E_errors = []

r_values_sc = []
E_values_sc = []
E_errors_sc = []

r_values_rld = []
E_values_rld = []
E_errors_rld = []

r_values_cen = []
E_values_cen = []
E_errors_cen = []

# NEW: Mono + IRF + BG Efficiency Arrays
r_values_irf_bg, E_values_irf_bg, E_errors_irf_bg = [], [], []


for sample in results:

    # Lifetime-derived FRET efficiency
    E = sample["FRET_efficiency"]

    if np.isnan(E):
        print("File skipped")
        print(f"Skipped file: {sample['file']}")
        print("its lifetime mono")
        continue

    c = sample["c_eff"]

    if c <= 0:
        print("File skipped")
        print(f"Skipped file: {sample['file']}")
        print("its lifetime mono")
        continue

    
    if (E <= 0) or (E >= 1):
        print("File skipped")
        print(f"Skipped file: {sample['file']}")
        print("its lifetime mono")
        continue
    
    if np.isnan(E):
        print("File skipped")
        print(f"Skipped file: {sample['file']}")
        print("its lifetime mono")
        continue

    if not (0 < E < 1):
        print("File skipped")
        print(f"Skipped file: {sample['file']}")
        print("its lifetime mono")
        continue

    # --------------------------------------------------
    # concentration -> intermolecular distance
    # --------------------------------------------------

    number_density = c * 1000 * NA   # molecules/m^3

    r = number_density ** (-1/3)

    r_nm = r * 1e9

    r_values.append(r_nm)
    E_values.append(E)
    E_errors.append(sample["FRET_efficiency_error"])
    
    E_s = sample["FRET_efficiency_shortcut"]
    if 0 < E_s < 1 and not np.isnan(E_s):
        r_values_sc.append(r_nm)
        E_values_sc.append(E_s)
        E_errors_sc.append(sample["FRET_efficiency_shortcut_error"])
        
    # NEW: RLD Shortcut
    E_rld_val = sample["FRET_efficiency_rld"]
    if 0 < E_rld_val < 1 and not np.isnan(E_rld_val):
        r_values_rld.append(r_nm)
        E_values_rld.append(E_rld_val)
        E_errors_rld.append(sample["FRET_efficiency_rld_error"])
        
    # 4. Centroid Shortcut efficiency
    E_cen_val = sample["FRET_efficiency_centroid"]
    if 0 < E_cen_val < 1 and not np.isnan(E_cen_val):
        r_values_cen.append(r_nm)
        E_values_cen.append(E_cen_val)
        E_errors_cen.append(sample["FRET_efficiency_centroid_error"])
    
    # NEW: Mono + IRF + BG Filtering
    E_irf_bg_val = sample["FRET_efficiency_irf_bg"]
    if 0 < E_irf_bg_val < 1 and not np.isnan(E_irf_bg_val):
        r_values_irf_bg.append(r_nm)
        E_values_irf_bg.append(E_irf_bg_val)
        E_errors_irf_bg.append(sample["FRET_efficiency_irf_bg_error"])

# NEW: Mono + IRF + BG Efficiëntie Regressie
r_values_irf_bg = np.array(r_values_irf_bg)
E_values_irf_bg = np.array(E_values_irf_bg)
E_errors_irf_bg = np.array(E_errors_irf_bg)
x_irf_bg = np.log(r_values_irf_bg)
y_irf_bg = np.log((1/E_values_irf_bg) - 1)
sigma_y_irf_bg = E_errors_irf_bg / (E_values_irf_bg * (1 - E_values_irf_bg))
slope_irf_bg_E, intercept_irf_bg_E, rvalue_irf_bg_E, _, std_err_irf_bg_E = linregress(x_irf_bg, y_irf_bg)
R2_irf_bg_E = rvalue_irf_bg_E**2
xfit_irf_bg_E = np.linspace(x_irf_bg.min(), x_irf_bg.max(), 200)
yfit_irf_bg_E = slope_irf_bg_E * xfit_irf_bg_E + intercept_irf_bg_E

r_values_cen = np.array(r_values_cen)
E_values_cen = np.array(E_values_cen)
E_errors_cen = np.array(E_errors_cen)


x_cen = np.log(r_values_cen)
y_cen = np.log((1/E_values_cen) - 1)
sigma_y_cen = E_errors_cen / (E_values_cen * (1 - E_values_cen))


slope_cen_E, intercept_cen_E, rvalue_cen_E, _, std_err_cen_E = linregress(x_cen, y_cen)
R2_cen_E = rvalue_cen_E**2
xfit_cen_E = np.linspace(x_cen.min(), x_cen.max(), 200)
yfit_cen_E = slope_cen_E*xfit_cen_E + intercept_cen_E

r_values_rld = np.array(r_values_rld)
E_values_rld = np.array(E_values_rld)
E_errors_rld = np.array(E_errors_rld)

x_rld = np.log(r_values_rld)
y_rld = np.log((1/E_values_rld) - 1)
sigma_y_rld = E_errors_rld / (E_values_rld * (1 - E_values_rld))

slope_rld_E, intercept_rld_E, rvalue_rld_E, _, std_err_rld_E = linregress(x_rld, y_rld)
R2_rld_E = rvalue_rld_E**2
xfit_rld_E = np.linspace(x_rld.min(), x_rld.max(), 200)
yfit_rld_E = slope_rld_E*xfit_rld_E + intercept_rld_E

r_values_sc = np.array(r_values_sc)
E_values_sc = np.array(E_values_sc)
E_errors_sc = np.array(E_errors_sc)

x_sc = np.log(r_values_sc)
y_sc = np.log((1/E_values_sc) - 1)
sigma_y_sc = E_errors_sc / (E_values_sc * (1 - E_values_sc))

slope_sc_E, intercept_sc_E, rvalue_sc_E, _, std_err_sc_E = linregress(x_sc, y_sc)
R2_sc_E = rvalue_sc_E**2
xfit_sc_E = np.linspace(x_sc.min(), x_sc.max(), 200)
yfit_sc_E = slope_sc_E*xfit_sc_E + intercept_sc_E

r_values = np.array(r_values)
E_values = np.array(E_values)
E_errors = np.array(E_errors)

# --------------------------------------------------
# Intensity-based efficiency
# --------------------------------------------------

r_values_int = []
E_values_int = []
E_errors_int = []

for sample in results:

    E = sample["E_intensity"]

    if np.isnan(E):
        print("File skipped")
        print(f"Skipped file: {sample['file']}")
        print("its intense")
        continue

    if not (0 < E < 1):
        print("File skipped")
        print(f"Skipped file: {sample['file']}")
        print("its intense")
        continue

    c = sample["c_eff"]

    if c <= 0:
        print("File skipped")
        print(f"Skipped file: {sample['file']}")
        print("its intense") # This is time 6:16
        continue

    number_density = c * 1000 * NA

    r = number_density**(-1/3)

    r_nm = r * 1e9

    r_values_int.append(r_nm)
    E_values_int.append(E)
    E_errors_int.append(sample["E_intensity_error"])

r_values_int = np.array(r_values_int)
E_values_int = np.array(E_values_int)
E_errors_int = np.array(E_errors_int)

print("Number of efficiency points:", len(E_values))


if len(E_values) < 2:
    raise RuntimeError(
        "Not enough valid efficiency points for Förster validation"
    )

r_values_bi = []
E_values_bi = []
E_errors_bi = []

for sample in results:

    E = sample["FRET_efficiency_bi"]

    if np.isnan(E):
        print("File skipped")
        print(f"Skipped file: {sample['file']}")
        print("its bi")
        continue

    if not (0 < E < 1):
        print("File skipped")
        print(f"Skipped file: {sample['file']}")
        print("its bi")
        continue

    c = sample["c_eff"]

    number_density = c*1000*NA

    r_nm = number_density**(-1/3)*1e9

    r_values_bi.append(r_nm)
    E_values_bi.append(E)
    
    E_errors_bi.append(
        sample["FRET_efficiency_bi_error"]
    )

E_errors_bi = np.array(E_errors_bi)

sigma_y_bi = (
    E_errors_bi /
    (np.array(E_values_bi)*(1-np.array(E_values_bi)))
)

x_bi = np.log(r_values_bi)

y_bi = np.log(
    (1/np.array(E_values_bi))-1
)

slope_bi, intercept_bi, rvalue_bi, _, std_err_bi = linregress(
    x_bi,
    y_bi
)

xfit_bi = np.linspace(
    x_bi.min(),
    x_bi.max(),
    200
)

yfit_bi = slope_bi*xfit_bi + intercept_bi

# --------------------------------------------------
# IRF corrected efficiency
# --------------------------------------------------

r_values_irf = []
E_values_irf = []
E_errors_irf = []


for sample in results:

    E = sample["FRET_efficiency_irf"]

    if np.isnan(E):
        print("File skipped")
        print(f"Skipped file: {sample['file']}")
        print("its irf")
        continue

    if not (0 < E < 1):
        print("File skipped")
        print(f"Skipped file: {sample['file']}")
        print("its irf")
        continue

    c = sample["c_eff"]

    if c <= 0:
        print("File skipped")
        print(f"Skipped file: {sample['file']}")
        print("its irf")
        continue


    number_density = c*1000*NA

    r_nm = number_density**(-1/3)*1e9


    r_values_irf.append(r_nm)
    E_values_irf.append(E)
    E_errors_irf.append(
        sample["FRET_efficiency_irf_error"]
    )


r_values_irf = np.array(r_values_irf)
E_values_irf = np.array(E_values_irf)
E_errors_irf = np.array(E_errors_irf)

x_irf = np.log(r_values_irf)

y_irf = np.log(
    (1/E_values_irf)-1
)


sigma_y_irf = (
    E_errors_irf /
    (E_values_irf*(1-E_values_irf))
)

slope_irf, intercept_irf, rvalue_irf, _, std_err_irf = linregress(
    x_irf,
    y_irf
)

R2_irf = rvalue_irf**2


x_fit_irf = np.linspace(
    x_irf.min(),
    x_irf.max(),
    200
)

y_fit_irf = slope_irf*x_fit_irf + intercept_irf

# --------------------------------------------------
# Förster linearization
# --------------------------------------------------

x = np.log(r_values)

y = np.log(
    (1/E_values) - 1
)


# uncertainty propagation

sigma_y = (
    E_errors /
    (E_values * (1-E_values))
)


# --------------------------------------------------
# Linear regression
# --------------------------------------------------

slope, intercept, r_value, p_value, std_err = linregress(
    x,
    y
)

R2 = r_value**2



# fitted experimental line

x_fit = np.linspace(
    x.min(),
    x.max(),
    200
)

y_fit = slope*x_fit + intercept

# --------------------------------------------------
# Intensity regression
# --------------------------------------------------

x_int = np.log(r_values_int)

y_int = np.log(
    (1/E_values_int) - 1
)

sigma_y_int = (
    E_errors_int /
    (E_values_int * (1-E_values_int))
)

slope_int, intercept_int, r_value_int, p_value_int, std_err_int = linregress(
    x_int,
    y_int
)

R2_int = r_value_int**2

x_fit_int = np.linspace(
    x_int.min(),
    x_int.max(),
    200
)

y_fit_int = slope_int * x_fit_int + intercept_int

# theoretical slope = 6

theory_intercept = np.mean(
    y - 6*x
)

y_theory = (
    6*x_fit +
    theory_intercept
)

# Theory m = 3

theory_intercept_theory3 = np.mean(
    y - 3*x
)

y_theory_theory3 = (
    3*x_fit +
    theory_intercept_theory3
)

r_values_bg=[]
E_values_bg=[]
E_errors_bg=[]

for sample in results:

    E = sample["FRET_efficiency_bg"]

    if np.isnan(E):
        print("File skipped")
        print(f"Skipped file: {sample['file']}")
        print("its bg")
        continue

    if not (0<E<1):
        print("File skipped")
        print(f"Skipped file: {sample['file']}")
        print("its bg")
        continue

    c = sample["c_eff"]

    r_nm = (
        c*1000*NA
    )**(-1/3)*1e9

    r_values_bg.append(r_nm)
    E_values_bg.append(E)
    
    E_errors_bg.append(
        sample["FRET_efficiency_bg_error"]
    )

E_errors_bg=np.array(E_errors_bg)
r_values_bg=np.array(r_values_bg)
E_values_bg=np.array(E_values_bg)

# We are here now

sigma_y_bg = (
    E_errors_bg /
    (E_values_bg*(1-E_values_bg))
)

x_bg=np.log(r_values_bg)

y_bg=np.log(
    (1/E_values_bg)-1
)

slope_bg, intercept_bg, rvalue_bg, _, std_err_bg = linregress(
    x_bg,
    y_bg
)

R2_bg=rvalue_bg**2


xfit_bg=np.linspace(
    x_bg.min(),
    x_bg.max(),
    200
)

yfit_bg=slope_bg*xfit_bg+intercept_bg

# -------------------------------------------------- #New
# NEW: Phasor FRET Efficiency log-log arrays & fit #New
# -------------------------------------------------- #New
# -------------------------------------------------- #New
# NEW: Phasor data toevoegen aan de log-log plot arrays #New
# -------------------------------------------------- #New
r_values_phasor = [] #New
E_values_phasor = [] #New
E_errors_phasor = [] #New #New

for sample in results: #New
    E_phasor_val = sample.get("FRET_efficiency_phasor", np.nan) #New
    if np.isnan(E_phasor_val) or not (0 < E_phasor_val < 1): #New
        continue #New
   
    c = sample.get("c_eff", 0) #New
    if c <= 0: #New
        continue #New
       
    number_density = c * 1000 * NA #New
    r_nm = number_density**(-1/3) * 1e9 #New
   
    r_values_phasor.append(r_nm) #New
    E_values_phasor.append(E_phasor_val) #New
    E_errors_phasor.append(0.05) #New #New
    
    # --- DEBUG CHECK VOOR PHASOR --- #New
    print("****************************************** helo world *************************************")
    E_phasor_debug = sample.get("FRET_efficiency_phasor", "SLEUTEL_BESTAAT_NIET") #New
    print(f"Bestand: {sample.get('file', 'onbekend')} | Phasor E: {E_phasor_debug}") #New

# Omzetten naar numpy arrays en lineaire regressie #New
r_values_phasor = np.array(r_values_phasor) #New
E_values_phasor = np.array(E_values_phasor) #New
E_errors_phasor = np.array(E_errors_phasor) #New

if len(r_values_phasor) > 1: #New
    x_phasor = np.log(r_values_phasor) #New
    y_phasor = np.log((1 / E_values_phasor) - 1) #New
    sigma_y_phasor = E_errors_phasor / (E_values_phasor * (1 - E_values_phasor)) #New
   
    slope_phasor, intercept_phasor, rvalue_phasor, _, std_err_phasor = linregress(x_phasor, y_phasor) #New
    R2_phasor = rvalue_phasor**2 #New
   
    xfit_phasor = np.linspace(x_phasor.min(), x_phasor.max(), 200) #New
    yfit_phasor = slope_phasor * xfit_phasor + intercept_phasor #New
else: #New
    # --- VEILIGHEIDSNET: Maak lege arrays aan als er geen phasor-data is --- #New
    x_phasor = np.array([]) #New
    y_phasor = np.array([]) #New
    sigma_y_phasor = np.array([]) #New
    slope_phasor, intercept_phasor, R2_phasor, std_err_phasor = 0, 0, 0, 0 #New
    xfit_phasor, yfit_phasor = np.array([]), np.array([]) #New

y_errs = [r['E_taylor_err'] for r in results if not np.isnan(r.get('E_taylor', np.nan))]

# --------------------------------------------------
# Plot
# --------------------------------------------------

from matplotlib.lines import Line2D

plt.figure(figsize=(10,7))

# --------------------------------------------------
# Plot data and fits
# --------------------------------------------------

# Lifetime mono data
# plt.errorbar(
  #  x,
  #  y,
  #  yerr=sigma_y,
  #  fmt="o",
  #  color="blue",
  #  capsize=3
#)

# Lifetime mono fit
#plt.plot(
  #  x_fit,
  #  y_fit,
  #  color="blue",
  #  linewidth=2
#)


# Bi-exp data
# plt.errorbar(
  #  x_bi,
  #  y_bi,
   # yerr=sigma_y_bi,
   # fmt="p",
   # color="purple",
    #capsize=3
#)

# Bi-exp fit
#plt.plot(
   # xfit_bi,
   # yfit_bi,
   # color="purple",
   # linewidth=2
#)

# IRF data

#plt.errorbar(
  #  x_irf,
  #  y_irf,
  #  yerr=sigma_y_irf,
  #  fmt="h",
  #  color="magenta",
  #  capsize=2
#)


# IRF fit

#plt.plot(
 #   x_fit_irf,
 #   y_fit_irf,
 #   color="magenta",
 #   linewidth=2
#)

# NEW: Plot Mono + IRF + BG Efficiënties
#plt.errorbar(x_irf_bg, y_irf_bg, yerr=sigma_y_irf_bg, fmt="s", color="darkgreen", capsize=3)
#plt.plot(xfit_irf_bg_E, yfit_irf_bg_E, color="darkgreen", linewidth=2)

# Plot NEW Heuristic Shortcut in Forest Green
#plt.errorbar(x_sc, y_sc, yerr=sigma_y_sc, fmt="^", color="teal", capsize=3)
#plt.plot(xfit_sc_E, yfit_sc_E, color="teal", linewidth=2)

#plt.errorbar(x_rld, y_rld, yerr=sigma_y_rld, fmt="*", color="darkviolet", capsize=3)
#plt.plot(xfit_rld_E, yfit_rld_E, color="darkviolet", linewidth=2)

#plt.errorbar(x_cen, y_cen, yerr=sigma_y_cen, fmt="p", color="#8c564b", capsize=3)
#plt.plot(xfit_cen_E, yfit_cen_E, color="#8c564b", linewidth=2)

# NEW: Plot Phasor Data and Fit #New
# plt.errorbar( #New
   # x_phasor, y_phasor, yerr=y_errs, #New
  #  fmt="8", color="dodgerblue", capsize=3 #New
#) #New

# plt.plot( #New
   #  xfit_phasor, yfit_phasor, #New
   #  color="dodgerblue", linewidth=2, linestyle="-" #New
# ) #New

# Intensity data
plt.errorbar(
    x_int,
    y_int,
    yerr=sigma_y_int,
    fmt="s",
    color="dodgerblue",
    capsize=3
)

# Intensity fit
plt.plot(
    x_fit_int,
    y_fit_int,
    color="dodgerblue",
    linewidth=2
)

# Background mono

#plt.errorbar(
 #   x_bg,
 #   y_bg,
 #   yerr=sigma_y_bg,
 #   fmt="D",
 #   color="red",
 #   capsize=3
#)

#plt.plot(
  #  xfit_bg,
  #  yfit_bg,
  #  color="red",
  #  linewidth=2
# )


# Theory
plt.plot(
    x_fit,
    y_theory,
    "--",
    color="slategrey",
    linewidth=2
)

# Theory
# plt.plot(
   # x_fit,
   # y_theory_theory3,
   # "--",
   # color="maroon",
   # linewidth=2
#)

# ==============================================================================
# ANNOTATIES PLOT 2 (Strakke opmaak: zwarte pijlen, witte kaders, scherpe hoeken)
# ==============================================================================

# 1. CENTROID UITSCHIETER (Rechtsboven geplaatst)
if len(E_values_cen) > 0:
    x_cen_log = np.log(r_values_cen)
    y_cen_log = np.log((1.0 / np.array(E_values_cen)) - 1.0)
   
    idx_cen = np.argmin(x_cen_log)
    x_cent_E = x_cen_log[idx_cen]
    y_cent_E = y_cen_log[idx_cen]

    #plt.annotate(
        #"Centroid-afwijking:\nOverschatting van $\\tau$ geeft\nsystematische afwijking van E",
        #xy=(x_cent_E, y_cent_E),
        #xytext=(x_cent_E + 0.1, y_cent_E + 1.8),  # Meer naar rechtsboven
        #fontsize=8.5,
       # bbox=dict(boxstyle="square,pad=0.4", fc="white", ec="black", lw=0.8)
    #)

# 2. BEGIN DIVERGENTIE (Naar rechts verschoven)
if len(E_values_irf_bg) > 0:
    x_irf_bg_log = np.log(r_values_irf_bg)
    y_irf_bg_log = np.log((1.0 / np.array(E_values_irf_bg)) - 1.0)
   
    idx_div = np.argmin(x_irf_bg_log)
    x_div_E = x_irf_bg_log[idx_div]
    y_div_E = y_irf_bg_log[idx_div]

    #plt.annotate(
     #   "Zone van divergentie:\nSterke spreiding van shortcut-methodes\ndoor gevoeligheid voor IRF-vervorming.",
     #   xy=(x_div_E, y_div_E),
     #   xytext=(x_div_E + 0.25, y_div_E - 1.8),  # Meer naar rechts verschoven
     #   fontsize=8.5,
     #   bbox=dict(boxstyle="square,pad=0.4", fc="white", ec="black", lw=0.8)
    #)

# 3. EINDE CONVERGENTIE (Naar beneden verschoven)
if len(E_values_irf_bg) > 0:
    idx_conv = np.argmax(x_irf_bg_log)
    x_conv_E = x_irf_bg_log[idx_conv]
    y_conv_E = y_irf_bg_log[idx_conv]

    #plt.annotate(
     #   "Convergentiepunt:\nIRF-loze fits overlappen volledig.\ndoordat B = 0 gekozen is.",
     #   xy=(x_conv_E, y_conv_E),
     #   xytext=(x_conv_E-0.25, y_conv_E - 3.2),  # Duidelijk naar beneden verschoven
     #   fontsize=8.5,
     #   bbox=dict(boxstyle="square,pad=0.4", fc="white", ec="black", lw=0.8)
    #)

# ==============================================================================
# INSET SUBPLOT: PLOT 2 (LOG-LOG EFFICIENCY) - SMALLER CONVERGENTIEGEBIED
# ==============================================================================

# --------------------------------------------------
# Custom legend (controlled order)
# --------------------------------------------------
# This shows the order

legend_elements = [

    # --------------------------------------------------
    # THE THEORY
    # --------------------------------------------------

    Line2D(
        [],
        [],
        linestyle="None",
        label="THEORY"
    ),
    
    # theory
    Line2D(
        [0],
        [0],
        color="slategrey",
        linestyle="--",
        linewidth=1,
        label="Theory: Isolated single-pair model: m = 6"
    ),
    
    # theory
    #Line2D(
     #   [0],
     #   [0],
     #   color="maroon",
     #   linestyle="--",
     #   linewidth=1,
     #   label="Theory: 3D fluid continuum model: m = 3"
    #),

    # --------------------------------------------------
    # FITS (ordered by model)
    # --------------------------------------------------
    
    Line2D(
        [],
        [],
        linestyle="None",
        label="REGRESSION FITS"
    ),

    # mono fit
    #Line2D(
     #   [0],
     #   [0],
     #   color="blue",
     #   linewidth=2,
     #   label=f"Lifetime mono exp fit: m={slope:.2f} ± {std_err:.2f}, R²={R2:.3f}"
    #),

    # mono + background fit
   # Line2D(
    #    [0],
    #    [0],
    #    color="red",
    #    linewidth=2,
    #    label=f"Lifetime mono exp + BG fit: m={slope_bg:.2f} ± {std_err_bg:.2f}, R²={R2_bg:.3f}"
    #),

    # mono + IRF fit
   # Line2D(
    #    [0],
    #    [0],
    #    color="magenta",
    #    linewidth=2,
    #    label=f"Lifetime mono exp + IRF fit: m={slope_irf:.2f} ± {std_err_irf:.2f}, R²={R2_irf:.3f}"
    #),

    # bi exponential fit
   # Line2D(
       # [0],
       # [0],
       # color="purple",
        #linewidth=2,
        #label=f"Lifetime bi-exp fit: m={slope_bi:.2f} ± {std_err_bi:.2f}, R²={R2_bi:.3f}"
    #),

    #Line2D([0], [0], color="darkgreen", linewidth=2, label=f"Lifetime mono exp + IRF + BG fit: m={slope_irf_bg_E:.2f} ± {std_err_irf_bg_E:.2f}, R²={R2_irf_bg_E:.3f}"), # NEW
   # Line2D([0], [0], color="teal", linewidth=2, label=f"Lifetime 1/e Heuristic fit (shortcut): m={slope_sc_E:.2f} ± {std_err_sc_E:.2f}, R²={R2_sc_E:.3f}"),
   # Line2D([0], [0], color="darkviolet", linewidth=2, label=f"Lifetime RLD fit (shortcut): m={slope_rld_E:.2f} ± {std_err_rld_E:.2f}, R²={R2_rld_E:.3f}"),
   # Line2D([0], [0], color="#8c564b", linewidth=2, label=f"Lifetime Centroid fit (shortcut): m={slope_cen_E:.2f} ± {std_err_cen_E:.2f}, R²={R2_cen_E:.3f}"),


    # intensity fit
    Line2D(
        [0],
        [0],
        color="dodgerblue",
        linewidth=2,
        label=f"Intensity fit: m={slope_int:.2f} ± {std_err_int:.2f}, R²={R2_int:.3f}"
    ),
    
    # Line2D([0], [0], color="dodgerblue", linewidth=2, linestyle="-", label=f"Angle fit: m={slope_phasor:.2f} ± {std_err_phasor:.2f}, R²={R2_phasor:.3f}"), #New

    # --------------------------------------------------
    # DATA POINTS (same order)
    # --------------------------------------------------
    
    Line2D(
        [],
        [],
        linestyle="None",
        label="EXPERIMENTAL DATA"
    ),

    # mono data
    #Line2D(
     #   [0],
     #   [0],
     #   marker="o",
     #   color="blue",
     #   linestyle="None",
     #   label="Lifetime mono exp data"
    #),

    # mono + background data
    #Line2D(
     #   [0],
     #   [0],
     #   marker="D",
     #   color="red",
      #  linestyle="None",
      #  label="Lifetime mono exp + BG data"
    #),

    # mono + IRF data
    #Line2D(
     #   [0],
     #   [0],
     #   marker="h",
     #   color="magenta",
     #   linestyle="None",
     #   label="Lifetime mono exp + IRF data"
    #),

    # bi exponential data
   # Line2D(
      #  [0],
      #  [0],
       # marker="p",
       # color="purple",
       # linestyle="None",
       # label="Lifetime bi-exp data"
   # ),

   # Line2D([0], [0], marker="s", color="darkgreen", linestyle="None", label="Lifetime mono exp + IRF + BG data"), # NEW
   # Line2D([0], [0], marker="^", color="teal", linestyle="None", label="Lifetime 1/e Heuristic data (shortcut)"),
   # Line2D([0], [0], marker="*", color="darkviolet", linestyle="None", label="Lifetime RLD data (shortcut)"),
   # Line2D([0], [0], marker="p", color="#8c564b", linestyle="None", label="Lifetime Centroid data (shortcut)"),

    # intensity data
    Line2D(
        [0],
        [0],
        marker="s",
        color="dodgerblue",
        linestyle="None",
        label="Intensity data"
    ),

    # Line2D([0], [0], marker="8", color="dodgerblue", linestyle="None", label="Angle data") #New

]

# --------------------------------------------------
# Clean ordered legend inside plot
# --------------------------------------------------

leg = plt.legend(
    handles=legend_elements,
    loc="best",
    fontsize=8,
    ncol=1,
    frameon=True,
    framealpha=0.9,
    handlelength=1.5,
    handletextpad=0.5,
    labelspacing=0.4,
    borderpad=0.5
)

for text in leg.get_texts():
    if text.get_text().strip() in [
        "THEORY",
        "REGRESSION FITS",
        "EXPERIMENTAL DATA"
    ]:
        text.set_weight("bold")

# --------------------------------------------------
# Axis formatting
# --------------------------------------------------

plt.xlabel(
    r"$\log(r\,\mathrm{(nm)})$",
    fontsize=12
)

plt.ylabel(
    r"$\log(1/E-1)$",
    fontsize=12
)

plt.title( # we are here you baby
    "Förster law validation using intensity-based efficiencies",  # Last edit: 23:22 now, we are almost done, just p0 en boudns refine.
    fontsize=13
)

plt.grid(True)

plt.tight_layout()


plt.savefig(
    "bob2",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

############# result printing

print("")
print("Results printed on screen for essay")
print("")

import numpy as np

def print_latex_pixel_tables(results, DT_NS, donor_dict):
    """
    Prints pre-formatted LaTeX tables for donor-acceptor and donor-only lifetimes in pixels (Tables 1-4).
    Uses defensive dictionary fetching (.get) to prevent KeyError crashes.
    """
   
    # ------------------------------------------------------------------
    # TABEL 1: Fitmodellen voor Sample Donor-Acceptor Lifetimes (tau_DA in pixels)
    # ------------------------------------------------------------------
    print("\n% " + "="*60)
    print("% TABEL 1: Fitmodellen Donor-Acceptor (tau_DA in pixels)")
    print("% " + "="*60)
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\caption{Skelettabel van de gemeten en gefitte donor-acceptorlevensduren ($\tau_{DA}$) en bijbehorende standaardfouten ($\sigma$), bepaald met de exponentiële fitmodellen. Alle levensduren zijn uitgedrukt in pixels.}")
    print(r"\label{tab:tau_da_fitmodels}")
    print(r"\begin{adjustbox}{max width=\textwidth}")
    print(r"\begin{tabular}{cc|cc|cc|cc}")
    print(r"\toprule")
    print(r"$\mathbf{V_{\mathrm{laag}}}$ & $\mathbf{V_{\mathrm{hoog}}}$ & $\mathbf{\tau_{DA,\mathrm{mono}}}$ & $\mathbf{\sigma}$ & $\mathbf{\tau_{DA,\mathrm{BG}}}$ & $\mathbf{\sigma}$ & $\mathbf{\tau_{DA,\mathrm{IRF}}}$ & $\mathbf{\sigma}$ & $\mathbf{\tau_{DA,\mathrm{IRF+BG}}}$ & $\mathbf{\sigma}$ \\") # NEW
    print(r"$(\mu\mathrm{L})$ & $(\mu\mathrm{L})$ & (px) & (px) & (px) & (px) & (px) & (px) & (px) & (px) \\")
    print(r"\midrule")
   
    for r in results:
        v_low = f"{r.get('v_low', 0):.0f}"
        v_high = f"{r.get('v_high', 0):.0f}"

        tau_mono = f"{r['tau_pixels']:.2f}" if 'tau_pixels' in r else f"{r.get('tau_ns', 0)/DT_NS:.2f}"
        err_mono = f"{r['tau_error_pixels']:.2f}" if 'tau_error_pixels' in r else f"{r.get('tau_error_ns', 0)/DT_NS:.2f}"
       
        tau_bg = f"{r['tau_bg_ns']/DT_NS:.2f}" if ('tau_bg_ns' in r and not np.isnan(r['tau_bg_ns'])) else "..."
        err_bg = f"{r['tau_bg_error_ns']/DT_NS:.2f}" if ('tau_bg_error_ns' in r and not np.isnan(r['tau_bg_error_ns'])) else "..."

        tau_irf = f"{r['tau_irf_ns']/DT_NS:.2f}" if ('tau_irf_ns' in r and not np.isnan(r['tau_irf_ns'])) else "..."
        err_irf = f"{r['tau_irf_error_ns']/DT_NS:.2f}" if ('tau_irf_error_ns' in r and not np.isnan(r['tau_irf_error_ns'])) else "..."
        
        tau_irf_bg = f"{r['tau_irf_bg_ns']/DT_NS:.2f}" if ('tau_irf_bg_ns' in r and not np.isnan(r['tau_irf_bg_ns'])) else "..."
        err_irf_bg = f"{r['tau_irf_bg_error_ns']/DT_NS:.2f}" if ('tau_irf_bg_error_ns' in r and not np.isnan(r['tau_irf_bg_error_ns'])) else "..."
       
        print(f"{v_low} & {v_high} & {tau_mono} & {err_mono} & {tau_bg} & {err_bg} & {tau_irf} & {err_irf} & {tau_irf_bg} & {err_irf_bg} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{adjustbox}")
    print(r"\end{table}")

    # ------------------------------------------------------------------
    # TABEL 2: Shortcuts voor Sample Donor-Acceptor Lifetimes (tau_DA in pixels)
    # ------------------------------------------------------------------
    print("\n% " + "="*60)
    print("% TABEL 2: Shortcuts Donor-Acceptor (tau_DA in pixels)")
    print("% " + "="*60)
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\caption{Skelettabel van de gemeten donor-acceptorlevensduren ($\tau_{DA}$) en bijbehorende standaardfouten ($\sigma$), bepaald met de modelvrije levensduurschatters. Alle levensduren zijn uitgedrukt in pixels.}")
    print(r"\label{tab:tau_da_shortcuts}")
    print(r"\begin{adjustbox}{max width=\textwidth}")
    print(r"\begin{tabular}{cc|cc|cc|cc}")
    print(r"\toprule")
    print(r"$\mathbf{V_{\mathrm{laag}}}$ & $\mathbf{V_{\mathrm{hoog}}}$ & $\mathbf{\tau_{DA,1/e}}$ & $\mathbf{\sigma_{\tau_{DA,1/e}}}$ & $\mathbf{\tau_{DA,\mathrm{RLD}}}$ & $\mathbf{\sigma_{\tau_{DA,\mathrm{RLD}}}}$ & $\mathbf{\tau_{DA,\mathrm{cen}}}$ & $\mathbf{\sigma_{\tau_{DA,\mathrm{cen}}}}$ \\")
    print(r"$(\mu\mathrm{L})$ & $(\mu\mathrm{L})$ & (px) & (px) & (px) & (px) & (px) & (px) \\")
    print(r"\midrule")
   
    for r in results:
        v_low = f"{r.get('v_low', 0):.0f}"
        v_high = f"{r.get('v_high', 0):.0f}"
       
        tau_sc = f"{r['tau_shortcut_ns']/DT_NS:.2f}" if ('tau_shortcut_ns' in r and not np.isnan(r['tau_shortcut_ns'])) else "..."
        err_sc = f"{r['tau_shortcut_err_ns']/DT_NS:.2f}" if ('tau_shortcut_err_ns' in r and not np.isnan(r['tau_shortcut_err_ns'])) else "..."
       
        tau_rld = f"{r['tau_rld_ns']/DT_NS:.2f}" if ('tau_rld_ns' in r and not np.isnan(r['tau_rld_ns'])) else "..."
        err_rld = f"{r['tau_rld_err_ns']/DT_NS:.2f}" if ('tau_rld_err_ns' in r and not np.isnan(r['tau_rld_err_ns'])) else "..."
       
        tau_cen = f"{r['tau_centroid_ns']/DT_NS:.2f}" if ('tau_centroid_ns' in r and not np.isnan(r['tau_centroid_ns'])) else "..."
        err_cen = f"{r['tau_centroid_err_ns']/DT_NS:.2f}" if ('tau_centroid_err_ns' in r and not np.isnan(r['tau_centroid_err_ns'])) else "..."
       
        print(f"{v_low} & {v_high} & {tau_sc} & {err_sc} & {tau_rld} & {err_rld} & {tau_cen} & {err_cen} \\\\")
       
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{adjustbox}")
    print(r"\end{table}")

    # ------------------------------------------------------------------
    # TABEL 3: Fitmodellen voor Donor Referentie (tau_D in pixels)
    # ------------------------------------------------------------------
    print("\n% " + "="*60)
    print("% TABEL 3: Fitmodellen Donor Referentie (tau_D in pixels)")
    print("% " + "="*60)
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\caption{Skelettabel van de gemeten donorlevensduren ($\tau_D$) en bijbehorende standaardfouten ($\sigma$), bepaald met de exponentiële fitmodellen. Alle levensduren zijn uitgedrukt in pixels.}")
    print(r"\label{tab:tau_d_fitmodels}")
    print(r"\begin{adjustbox}{max width=\textwidth}")
    print(r"\begin{tabular}{cc|cc|cc|cc}")
    print(r"\toprule")
    print(r"$\mathbf{V_{\mathrm{laag}}}$ & $\mathbf{V_{\mathrm{hoog}}}$ & $\mathbf{\tau_{D,\mathrm{mono}}}$ & $\mathbf{\sigma}$ & $\mathbf{\tau_{D,\mathrm{BG}}}$ & $\mathbf{\sigma}$ & $\mathbf{\tau_{D,\mathrm{IRF}}}$ & $\mathbf{\sigma}$ & $\mathbf{\tau_{D,\mathrm{IRF+BG}}}$ & $\mathbf{\sigma}$ \\") # NEW
    print(r"$(\mu\mathrm{L})$ & $(\mu\mathrm{L})$ & (px) & (px) & (px) & (px) & (px) & (px) & (px) & (px) \\")
    print(r"\midrule")
   
    t_d_mono = f"{donor_dict['tau_D_pixels']:.2f}" if 'tau_D_pixels' in donor_dict else "..."
    s_d_mono = f"{donor_dict['tau_D_err_pixels']:.2f}" if 'tau_D_err_pixels' in donor_dict else "..."
   
    t_d_bg = f"{donor_dict['tau_D_bg_ns']/DT_NS:.2f}" if 'tau_D_bg_ns' in donor_dict else "..."
    s_d_bg = f"{donor_dict['tau_D_bg_err_ns']/DT_NS:.2f}" if 'tau_D_bg_err_ns' in donor_dict else "..."
   
    t_d_irf = f"{donor_dict['tau_D_irf_ns']/DT_NS:.2f}" if 'tau_D_irf_ns' in donor_dict else "..."
    s_d_irf_val = donor_dict.get('tau_D_irf_err', donor_dict.get('donor_dict_tau_D_irf_err', None))
    s_d_irf = f"{s_d_irf_val/DT_NS:.2f}" if s_d_irf_val is not None else "..."
    
    t_d_irf_bg = f"{donor_dict['tau_D_irf_bg_ns']/DT_NS:.2f}" if 'tau_D_irf_bg_ns' in donor_dict else "..."
    s_d_irf_bg_val = donor_dict.get('tau_D_irf_bg_err', None)
    s_d_irf_bg = f"{s_d_irf_bg_val/DT_NS:.2f}" if s_d_irf_bg_val is not None else "..."
   
    print(f"0 & 0 & {t_d_mono} & {s_d_mono} & {t_d_bg} & {s_d_bg} & {t_d_irf} & {s_d_irf} & {t_d_irf_bg} & {s_d_irf_bg} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{adjustbox}")
    print(r"\end{table}")

    # ------------------------------------------------------------------
    # TABEL 4: Shortcuts voor Donor Referentie (tau_D in pixels)
    # ------------------------------------------------------------------
    print("\n% " + "="*60)
    print("% TABEL 4: Shortcuts Donor Referentie (tau_D in pixels)")
    print("% " + "="*60)
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\caption{Skelettabel van de gemeten donorlevensduren ($\tau_D$) en bijbehorende standaardfouten ($\sigma$), bepaald met de modelvrije levensduurschatters. Alle levensduren zijn uitgedrukt in pixels.}")
    print(r"\label{tab:tau_d_shortcuts}")
    print(r"\begin{adjustbox}{max width=\textwidth}")
    print(r"\begin{tabular}{cc|cc|cc|cc}")
    print(r"\toprule")
    print(r"$\mathbf{V_{\mathrm{laag}}}$ & $\mathbf{V_{\mathrm{hoog}}}$ & $\mathbf{\tau_{D,1/e}}$ & $\mathbf{\sigma_{\tau_{D,1/e}}}$ & $\mathbf{\tau_{D,\mathrm{RLD}}}$ & $\mathbf{\sigma_{\tau_{D,\mathrm{RLD}}}}$ & $\mathbf{\tau_{D,\mathrm{cen}}}$ & $\mathbf{\sigma_{\tau_{D,\mathrm{cen}}}}$ \\")
    print(r"$(\mu\mathrm{L})$ & $(\mu\mathrm{L})$ & (px) & (px) & (px) & (px) & (px) & (px) \\")
    print(r"\midrule")
   
    t_d_sc = f"{donor_dict['tau_D_shortcut']/DT_NS:.2f}" if 'tau_D_shortcut' in donor_dict else "..."
    s_d_sc = f"{donor_dict['tau_D_shortcut_err']/DT_NS:.2f}" if 'tau_D_shortcut_err' in donor_dict else "..."
   
    t_d_rld = f"{donor_dict['tau_D_rld']/DT_NS:.2f}" if 'tau_D_rld' in donor_dict else "..."
    s_d_rld = f"{donor_dict['tau_D_rld_err']/DT_NS:.2f}" if 'tau_D_rld_err' in donor_dict else "..."
   
    t_d_cen = f"{donor_dict['tau_D_centroid']/DT_NS:.2f}" if 'tau_D_centroid' in donor_dict else "..."
    s_d_cen = f"{donor_dict['tau_D_centroid_err']/DT_NS:.2f}" if 'tau_D_centroid_err' in donor_dict else "..."
   
    print(f"0 & 0 & {t_d_sc} & {s_d_sc} & {t_d_rld} & {s_d_rld} & {t_d_cen} & {s_d_cen} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{adjustbox}")
    print(r"\end{table}")


def print_latex_ns_tables(results, donor_dict):
    """
    Prints pre-formatted LaTeX tables for FRET transfer times (tau_FRET) and donor lifetimes in nanoseconds (Tables 5-8).
    """
   
    # ------------------------------------------------------------------
    # TABEL 5: Fitmodellen voor FRET Overdrachtstijden (tau_FRET in ns)
    # ------------------------------------------------------------------
    print("\n% " + "="*60)
    print("% TABEL 5: Fitmodellen FRET Overdrachtstijden (tau_FRET in ns)")
    print("% " + "="*60)
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\caption{Skelettabel van de berekende FRET-overdrachtstijden ($\tau_{\mathrm{FRET}}$) en bijbehorende standaardfouten ($\sigma$), bepaald met de exponentiële fitmodellen. Alle tijden zijn uitgedrukt in nanoseconden.}")
    print(r"\label{tab:tau_fret_fitmodels}")
    print(r"\begin{adjustbox}{max width=\textwidth}")
    print(r"\begin{tabular}{cc|cc|cc|cc}")
    print(r"\toprule")
    print(r"$\mathbf{V_{\mathrm{laag}}}$ & $\mathbf{V_{\mathrm{hoog}}}$ & $\mathbf{\tau_{\mathrm{FRET},\mathrm{mono}}}$ & $\mathbf{\sigma}$ & $\mathbf{\tau_{\mathrm{FRET},\mathrm{BG}}}$ & $\mathbf{\sigma}$ & $\mathbf{\tau_{\mathrm{FRET},\mathrm{IRF}}}$ & $\mathbf{\sigma}$ & $\mathbf{\tau_{\mathrm{FRET},\mathrm{IRF+BG}}}$ & $\mathbf{\sigma}$ \\") # NEW
    print(r"$(\mu\mathrm{L})$ & $(\mu\mathrm{L})$ & (ns) & (ns) & (ns) & (ns) & (ns) & (ns) & (ns) & (ns) \\")
    print(r"\midrule")
   
    for r in results:
        v_low = f"{r.get('v_low', 0):.0f}"
        v_high = f"{r.get('v_high', 0):.0f}"
       
        fret_mono = f"{r['tau_FRET_ns']:.3f}" if ('tau_FRET_ns' in r and not np.isnan(r['tau_FRET_ns'])) else "..."
        err_mono  = f"{r['tau_FRET_error_ns']:.3f}" if ('tau_FRET_error_ns' in r and not np.isnan(r['tau_FRET_error_ns'])) else "..."
       
        fret_bg = f"{r['tau_FRET_bg_ns']:.3f}" if ('tau_FRET_bg_ns' in r and not np.isnan(r['tau_FRET_bg_ns'])) else "..."
        err_bg  = f"{r['tau_FRET_bg_error_ns']:.3f}" if ('tau_FRET_bg_error_ns' in r and not np.isnan(r['tau_FRET_bg_error_ns'])) else "..."
       
        fret_irf = f"{r['tau_FRET_irf_ns']:.3f}" if ('tau_FRET_irf_ns' in r and not np.isnan(r['tau_FRET_irf_ns'])) else "..."
        err_irf  = f"{r['tau_FRET_irf_error_ns']:.3f}" if ('tau_FRET_irf_error_ns' in r and not np.isnan(r['tau_FRET_irf_error_ns'])) else "..."
        
        fret_irf_bg = f"{r['tau_FRET_irf_bg_ns']:.3f}" if ('tau_FRET_irf_bg_ns' in r and not np.isnan(r['tau_FRET_irf_bg_ns'])) else "..."
        err_irf_bg  = f"{r['tau_FRET_irf_bg_error_ns']:.3f}" if ('tau_FRET_irf_bg_error_ns' in r and not np.isnan(r['tau_FRET_irf_bg_error_ns'])) else "..."

        print(f"{v_low} & {v_high} & {fret_mono} & {err_mono} & {fret_bg} & {err_bg} & {fret_irf} & {err_irf} & {fret_irf_bg} & {err_irf_bg} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{adjustbox}")
    print(r"\end{table}")

    # ------------------------------------------------------------------
    # TABEL 6: Shortcuts voor FRET Overdrachtstijden (tau_FRET in ns)
    # ------------------------------------------------------------------
    print("\n% " + "="*60)
    print("% TABEL 6: Shortcuts FRET Overdrachtstijden (tau_FRET in ns)")
    print("% " + "="*60)
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\caption{Skelettabel van de berekende FRET-overdrachtstijden ($\tau_{\mathrm{FRET}}$) en bijbehorende standaardfouten ($\sigma$), bepaald met de modelvrije levensduurschatters. Alle tijden zijn uitgedrukt in nanoseconden.}")
    print(r"\label{tab:tau_fret_shortcuts}")
    print(r"\begin{adjustbox}{max width=\textwidth}")
    print(r"\begin{tabular}{cc|cc|cc|cc}")
    print(r"\toprule")
    print(r"$\mathbf{V_{\mathrm{laag}}}$ & $\mathbf{V_{\mathrm{hoog}}}$ & $\mathbf{\tau_{\mathrm{FRET},1/e}}$ & $\mathbf{\sigma_{\tau_{\mathrm{FRET},1/e}}}$ & $\mathbf{\tau_{\mathrm{FRET},\mathrm{RLD}}}$ & $\mathbf{\sigma_{\tau_{\mathrm{FRET},\mathrm{RLD}}}}$ & $\mathbf{\tau_{\mathrm{FRET},\mathrm{cen}}}$ & $\mathbf{\sigma_{\tau_{\mathrm{FRET},\mathrm{cen}}}}$ \\")
    print(r"$(\mu\mathrm{L})$ & $(\mu\mathrm{L})$ & (ns) & (ns) & (ns) & (ns) & (ns) & (ns) \\")
    print(r"\midrule")
   
    for r in results:
        v_low = f"{r.get('v_low', 0):.0f}"
        v_high = f"{r.get('v_high', 0):.0f}"
       
        fret_sc = f"{r['tau_FRET_shortcut_ns']:.3f}" if ('tau_FRET_shortcut_ns' in r and not np.isnan(r['tau_FRET_shortcut_ns'])) else "..."
        err_sc  = f"{r['tau_FRET_shortcut_error_ns']:.3f}" if ('tau_FRET_shortcut_error_ns' in r and not np.isnan(r['tau_FRET_shortcut_error_ns'])) else "..."
       
        fret_rld = f"{r['tau_FRET_rld_ns']:.3f}" if ('tau_FRET_rld_ns' in r and not np.isnan(r['tau_FRET_rld_ns'])) else "..."
        err_rld  = f"{r['tau_FRET_rld_error_ns']:.3f}" if ('tau_FRET_rld_error_ns' in r and not np.isnan(r['tau_FRET_rld_error_ns'])) else "..."
       
        fret_cen = f"{r['tau_FRET_centroid_ns']:.3f}" if ('tau_FRET_centroid_ns' in r and not np.isnan(r['tau_FRET_centroid_ns'])) else "..."
        err_cen  = f"{r['tau_FRET_centroid_error_ns']:.3f}" if ('tau_FRET_centroid_error_ns' in r and not np.isnan(r['tau_FRET_centroid_error_ns'])) else "..."
       
        print(f"{v_low} & {v_high} & {fret_sc} & {err_sc} & {fret_rld} & {err_rld} & {fret_cen} & {err_cen} \\\\")
       
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{adjustbox}")
    print(r"\end{table}")

    # ------------------------------------------------------------------
    # TABEL 7: Fitmodellen voor Donor Referentie (tau_D in ns)
    # ------------------------------------------------------------------
    print("\n% " + "="*60)
    print("% TABEL 7: Fitmodellen Donor Referentie (tau_D in ns)")
    print("% " + "="*60)
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\caption{Skelettabel van de gemeten donorlevensduren ($\tau_D$) en bijbehorende standaardfouten ($\sigma$), bepaald met de exponentiële fitmodellen. Alle levensduren zijn uitgedrukt in nanoseconden.}")
    print(r"\label{tab:tau_d_fitmodels_ns}")
    print(r"\begin{adjustbox}{max width=\textwidth}")
    print(r"\begin{tabular}{cc|cc|cc|cc}")
    print(r"\toprule")
    print(r"$\mathbf{V_{\mathrm{laag}}}$ & $\mathbf{V_{\mathrm{hoog}}}$ & $\mathbf{\tau_{D,\mathrm{mono}}}$ & $\mathbf{\sigma}$ & $\mathbf{\tau_{D,\mathrm{BG}}}$ & $\mathbf{\sigma}$ & $\mathbf{\tau_{D,\mathrm{IRF}}}$ & $\mathbf{\sigma}$ & $\mathbf{\tau_{D,\mathrm{IRF+BG}}}$ & $\mathbf{\sigma}$ \\") # NEW
    print(r"$(\mu\mathrm{L})$ & $(\mu\mathrm{L})$ & (ns) & (ns) & (ns) & (ns) & (ns) & (ns) & (ns) & (ns) \\")
    print(r"\midrule")
   
    t_d_mono = f"{donor_dict['tau_D_ns']:.3f}" if 'tau_D_ns' in donor_dict else "..."
    s_d_mono = f"{donor_dict['tau_D_err_ns']:.3f}" if 'tau_D_err_ns' in donor_dict else "..."
   
    t_d_bg = f"{donor_dict['tau_D_bg_ns']:.3f}" if 'tau_D_bg_ns' in donor_dict else "..."
    s_d_bg = f"{donor_dict['tau_D_bg_err_ns']:.3f}" if 'tau_D_bg_err_ns' in donor_dict else "..."
   
    t_d_irf = f"{donor_dict['tau_D_irf_ns']:.3f}" if 'tau_D_irf_ns' in donor_dict else "..."
    s_d_irf = f"{donor_dict['tau_D_irf_err']:.3f}" if 'tau_D_irf_err' in donor_dict else "..."
   
    t_d_irf_bg = f"{donor_dict['tau_D_irf_bg_ns']:.3f}" if 'tau_D_irf_bg_ns' in donor_dict else "..."
    s_d_irf_bg = f"{donor_dict['tau_D_irf_bg_err']:.3f}" if 'tau_D_irf_bg_err' in donor_dict else "..."

    print(f"0 & 0 & {t_d_mono} & {s_d_mono} & {t_d_bg} & {s_d_bg} & {t_d_irf} & {s_d_irf} & {t_d_irf_bg} & {s_d_irf_bg} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{adjustbox}")
    print(r"\end{table}")

    # ------------------------------------------------------------------
    # TABEL 8: Shortcuts voor Donor Referentie (tau_D in ns)
    # ------------------------------------------------------------------
    print("\n% " + "="*60)
    print("% TABEL 8: Shortcuts Donor Referentie (tau_D in ns)")
    print("% " + "="*60)
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\caption{Skelettabel van de gemeten donorlevensduren ($\tau_D$) en bijbehorende standaardfouten ($\sigma$), bepaald met de modelvrije levensduurschatters. Alle levensduren zijn uitgedrukt in nanoseconden.}")
    print(r"\label{tab:tau_d_shortcuts_ns}")
    print(r"\begin{adjustbox}{max width=\textwidth}")
    print(r"\begin{tabular}{cc|cc|cc|cc}")
    print(r"\toprule")
    print(r"$\mathbf{V_{\mathrm{laag}}}$ & $\mathbf{V_{\mathrm{hoog}}}$ & $\mathbf{\tau_{D,1/e}}$ & $\mathbf{\sigma_{\tau_{D,1/e}}}$ & $\mathbf{\tau_{D,\mathrm{RLD}}}$ & $\mathbf{\sigma_{\tau_{D,\mathrm{RLD}}}}$ & $\mathbf{\tau_{D,\mathrm{cen}}}$ & $\mathbf{\sigma_{\tau_{D,\mathrm{cen}}}}$ \\")
    print(r"$(\mu\mathrm{L})$ & $(\mu\mathrm{L})$ & (ns) & (ns) & (ns) & (ns) & (ns) & (ns) \\")
    print(r"\midrule")
   
    t_d_sc = f"{donor_dict['tau_D_shortcut']:.5f}" if 'tau_D_shortcut' in donor_dict else "..."
    s_d_sc = f"{donor_dict['tau_D_shortcut_err']:.5f}" if 'tau_D_shortcut_err' in donor_dict else "..."
   
    t_d_rld = f"{donor_dict['tau_D_rld']:.5f}" if 'tau_D_rld' in donor_dict else "..."
    s_d_rld = f"{donor_dict['tau_D_rld_err']:.5f}" if 'tau_D_rld_err' in donor_dict else "..."
   
    t_d_cen = f"{donor_dict['tau_D_centroid']:.5f}" if 'tau_D_centroid' in donor_dict else "..."
    s_d_cen = f"{donor_dict['tau_D_centroid_err']:.5f}" if 'tau_D_centroid_err' in donor_dict else "..."
   
    print(f"0 & 0 & {t_d_sc} & {s_d_sc} & {t_d_rld} & {s_d_rld} & {t_d_cen} & {s_d_cen} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{adjustbox}")
    print(r"\end{table}")


def print_latex_ceff_r_table(results, NA=6.02214076e23):
    """
    Prints pre-formatted LaTeX table for effective acceptor concentration (c_eff) and calculated intermolecular distance (r) (Table 9).
    """
    print("\n% " + "="*60)
    print("% TABEL 9: Acceptorconcentratie (c_eff) en Afstand (r)")
    print("% " + "="*60)
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\caption{Skelettabel van de effectieve acceptorconcentratie ($c_{\mathrm{eff}}$), de berekende gemiddelde donor-acceptorafstand ($r$) en de bijbehorende standaardfouten ($\sigma$).}")
    print(r"\label{tab:ceff_r_skeleton}")
    print(r"\begin{adjustbox}{max width=\textwidth}")
    print(r"\begin{tabular}{cc|cc|cc}")
    print(r"\toprule")
    print(r"$\mathbf{V_{\mathrm{laag}}}$ & $\mathbf{V_{\mathrm{hoog}}}$ & $\mathbf{c_{\mathrm{eff}}}$ & $\mathbf{\sigma_{c_{\mathrm{eff}}}}$ & $\mathbf{r}$ & $\mathbf{\sigma_r}$ \\")
    print(r"$(\mu\mathrm{L})$ & $(\mu\mathrm{L})$ & (M) & (M) & (nm) & (nm) \\")
    print(r"\midrule")
   
    for r in results:
        v_low = f"{r.get('v_low', 0):.0f}"
        v_high = f"{r.get('v_high', 0):.0f}"
       
        c = r.get('c_eff', np.nan)
        sigma_c = r.get('c_eff_error', np.nan)
       
        c_str = f"{c:.3e}" if (not np.isnan(c) and c > 0) else "..."
        sigma_c_str = f"{sigma_c:.3e}" if (not np.isnan(sigma_c) and sigma_c > 0) else "..."
       
        if not np.isnan(c) and c > 0:
            r_nm = (c * 1000 * NA)**(-1/3) * 1e9
            sigma_r = (1/3) * (1000 * NA)**(-1/3) * (c**(-4/3)) * sigma_c * 1e9
           
            r_str = f"{r_nm:.2f}"
            sigma_r_str = f"{sigma_r:.2f}"
        else:
            r_str = "..."
            sigma_r_str = "..."
           
        print(f"{v_low} & {v_high} & {c_str} & {sigma_c_str} & {r_str} & {sigma_r_str} \\\\")
       
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{adjustbox}")
    print(r"\end{table}")


def print_latex_intensity_table(results, F_D, F_D_err=None):
    """
    Prints pre-formatted LaTeX table for donor-acceptor integrated intensities (F_DA) and donor reference equation F_D (Table 10 & Eq 1).
    """
    print("\n% " + "="*60)
    print("% TABEL 10: Steady-State Intensiteit (F_DA) & Donor Intensiteit (F_D)")
    print("% " + "="*60)
   
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\begin{adjustbox}{max width=\textwidth}")
    print(r"\begin{tabular}{|cc|cc|}")
    print(r"\toprule")
    print(r"$\mathbf{V_{\mathrm{laag}}}$ & $\mathbf{V_{\mathrm{hoog}}}$ & $\mathbf{F_{DA}}$ & $\mathbf{\sigma_{F_{DA}}}$ \\")
    print(r"$(\mu\mathrm{L})$ & $(\mu\mathrm{L})$ & (a.u.) & (a.u.) \\")
    print(r"\midrule")
   
    for r in results:
        v_low = f"{r.get('v_low', 0):.0f}"
        v_high = f"{r.get('v_high', 0):.0f}"
       
        if 'F_DA' in r:
            fda_val = r['F_DA']
        elif 'E_intensity' in r and not np.isnan(r['E_intensity']):
            fda_val = F_D * (1.0 - r['E_intensity'])
        else:
            fda_val = np.nan
           
        if not np.isnan(fda_val) and fda_val > 0:
            fda_str = f"{fda_val:.5e}"
            sigma_fda_str = f"{np.sqrt(fda_val):.5e}"
        else:
            fda_str = "..."
            sigma_fda_str = "..."
           
        print(f"{v_low} & {v_high} & {fda_str} & {sigma_fda_str} \\\\")
       
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{adjustbox}")
    print(r"\caption{Skelettabel van de gemeten donor-acceptorfluorescentie-intensiteiten ($F_{DA}$) en bijbehorende standaardfouten ($\sigma$), bepaald met de steady-state intensiteitsmethode.}")
    print(r"\label{tab:fda_skeleton}")
    print(r"\end{table}")
   
    if F_D_err is None:
        F_D_err = np.sqrt(F_D)
       
    print("\n% Donor Reference Intensity Equation:")
    print(r"\begin{equation}")
    print(f"F_D = {F_D:.2e} \\pm {F_D_err:.2e} \\,\\mathrm{{a.u.}}")
    print(r"\label{eq:FD}")
    print(r"\end{equation}")


def print_latex_efficiency_tables(results):
    """
    Prints pre-formatted LaTeX tables for FRET efficiencies (E) across all models and methods (Tables 11 & 12).
    """
   
    # ------------------------------------------------------------------
    # TABEL 11: Fitmodellen voor FRET Efficiënties (E)
    # ------------------------------------------------------------------
    print("\n% " + "="*60)
    print("% TABEL 11: Fitmodellen FRET Efficiënties (E)")
    print("% " + "="*60)
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\caption{Skelettabel van de berekende FRET-efficiënties ($E$) en bijbehorende standaardfouten ($\sigma$), bepaald met de exponentiële fitmodellen.}")
    print(r"\label{tab:E_fitmodels}")
    print(r"\begin{adjustbox}{max width=\textwidth}")
    print(r"\begin{tabular}{|cc|cc|cc|cc|}")
    print(r"\toprule")
    print(r"$\mathbf{V_{\mathrm{laag}}}$ & $\mathbf{V_{\mathrm{hoog}}}$ & $\mathbf{E_{\mathrm{mono}}}$ & $\mathbf{\sigma}$ & $\mathbf{E_{\mathrm{BG}}}$ & $\mathbf{\sigma}$ & $\mathbf{E_{\mathrm{IRF}}}$ & $\mathbf{\sigma}$ & $\mathbf{E_{\mathrm{IRF+BG}}}$ & $\mathbf{\sigma}$ \\") # NEW
    print(r"$(\mu\mathrm{L})$ & $(\mu\mathrm{L})$ & (-) & (-) & (-) & (-) & (-) & (-) & (-) & (-) \\")
    print(r"\midrule")
   
    for r in results:
        v_low = f"{r.get('v_low', 0):.0f}"
        v_high = f"{r.get('v_high', 0):.0f}"
       
        e_mono = f"{r['FRET_efficiency']:.3f}" if ('FRET_efficiency' in r and not np.isnan(r['FRET_efficiency'])) else "..."
        err_mono = f"{r['FRET_efficiency_error']:.3f}" if ('FRET_efficiency_error' in r and not np.isnan(r['FRET_efficiency_error'])) else "..."
       
        e_bg = f"{r['FRET_efficiency_bg']:.3f}" if ('FRET_efficiency_bg' in r and not np.isnan(r['FRET_efficiency_bg'])) else "..."
        err_bg = f"{r['FRET_efficiency_bg_error']:.3f}" if ('FRET_efficiency_bg_error' in r and not np.isnan(r['FRET_efficiency_bg_error'])) else "..."
       
        e_irf = f"{r['FRET_efficiency_irf']:.3f}" if ('FRET_efficiency_irf' in r and not np.isnan(r['FRET_efficiency_irf'])) else "..."
        err_irf = f"{r['FRET_efficiency_irf_error']:.3f}" if ('FRET_efficiency_irf_error' in r and not np.isnan(r['FRET_efficiency_irf_error'])) else "..."
        
        e_irf_bg = f"{r['FRET_efficiency_irf_bg']:.3f}" if ('FRET_efficiency_irf_bg' in r and not np.isnan(r['FRET_efficiency_irf_bg'])) else "..."
        err_irf_bg = f"{r['FRET_efficiency_irf_bg_error']:.3f}" if ('FRET_efficiency_irf_bg_error' in r and not np.isnan(r['FRET_efficiency_irf_bg_error'])) else "..."
       
        print(f"{v_low} & {v_high} & {e_mono} & {err_mono} & {e_bg} & {err_bg} & {e_irf} & {err_irf} & {e_irf_bg} & {err_irf_bg} \\\\")
       
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{adjustbox}")
    print(r"\end{table}")

    # ------------------------------------------------------------------
    # TABEL 12: Shortcuts & Intensiteit voor FRET Efficiënties (E)
    # ------------------------------------------------------------------
    print("\n% " + "="*60)
    print("% TABEL 12: Shortcuts & Intensiteit FRET Efficiënties (E)")
    print("% " + "="*60)
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\caption{Skelettabel van de berekende FRET-efficiënties ($E$) en bijbehorende standaardfouten ($\sigma$), bepaald met de modelvrije levensduurschatters en de steady-state intensiteitsmethode.}")
    print(r"\label{tab:E_shortcuts_intensity}")
    print(r"\begin{adjustbox}{max width=\textwidth}")
    print(r"\begin{tabular}{cc|cc|cc|cc|cc}")
    print(r"\toprule")
    print(r"$\mathbf{V_{\mathrm{laag}}}$ & $\mathbf{V_{\mathrm{hoog}}}$ & $\mathbf{E_{1/e}}$ & $\mathbf{\sigma_{E_{1/e}}}$ & $\mathbf{E_{\mathrm{RLD}}}$ & $\mathbf{\sigma_{E_{\mathrm{RLD}}}}$ & $\mathbf{E_{\mathrm{cen}}}$ & $\mathbf{\sigma_{E_{\mathrm{cen}}}}$ & $\mathbf{E_{\mathrm{int}}}$ & $\mathbf{\sigma_{E_{\mathrm{int}}}}$ \\")
    print(r"$(\mu\mathrm{L})$ & $(\mu\mathrm{L})$ & (-) & (-) & (-) & (-) & (-) & (-) & (-) & (-) \\")
    print(r"\midrule")
   
    for r in results:
        v_low = f"{r.get('v_low', 0):.0f}"
        v_high = f"{r.get('v_high', 0):.0f}"
       
        e_sc = f"{r['FRET_efficiency_shortcut']:.3f}" if ('FRET_efficiency_shortcut' in r and not np.isnan(r['FRET_efficiency_shortcut'])) else "..."
        err_sc = f"{r['FRET_efficiency_shortcut_error']:.3f}" if ('FRET_efficiency_shortcut_error' in r and not np.isnan(r['FRET_efficiency_shortcut_error'])) else "..."
       
        e_rld = f"{r['FRET_efficiency_rld']:.3f}" if ('FRET_efficiency_rld' in r and not np.isnan(r['FRET_efficiency_rld'])) else "..."
        err_rld = f"{r['FRET_efficiency_rld_error']:.3f}" if ('FRET_efficiency_rld_error' in r and not np.isnan(r['FRET_efficiency_rld_error'])) else "..."
       
        e_cen = f"{r['FRET_efficiency_centroid']:.3f}" if ('FRET_efficiency_centroid' in r and not np.isnan(r['FRET_efficiency_centroid'])) else "..."
        err_cen = f"{r['FRET_efficiency_centroid_error']:.3f}" if ('FRET_efficiency_centroid_error' in r and not np.isnan(r['FRET_efficiency_centroid_error'])) else "..."
       
        e_int = f"{r['E_intensity']:.5f}" if ('E_intensity' in r and not np.isnan(r['E_intensity'])) else "..."
        err_int = f"{r['E_intensity_error']:.5f}" if ('E_intensity_error' in r and not np.isnan(r['E_intensity_error'])) else "..."
       
        print(f"{v_low} & {v_high} & {e_sc} & {err_sc} & {e_rld} & {err_rld} & {e_cen} & {err_cen} & {e_int} & {err_int} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{adjustbox}")
    print(r"\end{table}")


def print_latex_tau_da_ns_tables(results):
    """
    Gabateewwan LaTeX kan sa'aatii jireenyaa donor-acceptor (tau_DA) naanoosekoondii (ns) keessatti agarsiisu.
    """
   
    # ------------------------------------------------------------------
    # TABEL: Moodeloota Fit Donor-Acceptor (tau_DA in ns)
    # ------------------------------------------------------------------
    print("\n% " + "="*60)
    print("% TABEL: Fitmodellen Donor-Acceptor (tau_DA in ns)")
    print("% " + "="*60)
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\caption{Skelettabel van de gemeten en gefitte donor-acceptorlevensduren ($\tau_{\mathrm{DA}}$) en bijbehorende standaardfouten ($\sigma$), bepaald met de exponentiële fitmodellen. Alle levensduren zijn uitgedrukt in nanoseconden.}")
    print(r"\label{tab:tau_da_fitmodels_ns}")
    print(r"\begin{adjustbox}{max width=\textwidth}")
    
    # 10 kolommen: 2 voor volume + 4 varianten × (waarde + fout)
    print(r"\begin{tabular}{cc|cc|cc|cc|cc}")
    print(r"\toprule")
    print(r"$\mathbf{V_{\mathrm{laag}}}$ & $\mathbf{V_{\mathrm{hoog}}}$ & $\mathbf{\tau_{\mathrm{DA},\mathrm{mono}}}$ & $\mathbf{\sigma_{\tau_{\mathrm{DA},\mathrm{mono}}}}$ & $\mathbf{\tau_{\mathrm{DA},\mathrm{BG}}}$ & $\mathbf{\sigma_{\tau_{\mathrm{DA},\mathrm{BG}}}}$ & $\mathbf{\tau_{\mathrm{DA},\mathrm{IRF}}}$ & $\mathbf{\sigma_{\tau_{\mathrm{DA},\mathrm{IRF}}}}$ & $\mathbf{\tau_{\mathrm{DA},\mathrm{IRF+BG}}}$ & $\mathbf{\sigma_{\tau_{\mathrm{DA},\mathrm{IRF+BG}}}}$ \\")
    print(r"$(\mu\mathrm{L})$ & $(\mu\mathrm{L})$ & (ns) & (ns) & (ns) & (ns) & (ns) & (ns) & (ns) & (ns) \\")
    print(r"\midrule")
   
    for r in results:
        v_low = f"{r.get('v_low', 0):.0f}"
        v_high = f"{r.get('v_high', 0):.0f}"
       
        tau_mono = f"{r['tau_ns']:.5f}" if ('tau_ns' in r and not np.isnan(r['tau_ns'])) else "..."
        err_mono = f"{r['tau_error_ns']:.3f}" if ('tau_error_ns' in r and not np.isnan(r['tau_error_ns'])) else "..."
       
        tau_bg = f"{r['tau_bg_ns']:.5f}" if ('tau_bg_ns' in r and not np.isnan(r['tau_bg_ns'])) else "..."
        err_bg = f"{r['tau_bg_error_ns']:.5f}" if ('tau_bg_error_ns' in r and not np.isnan(r['tau_bg_error_ns'])) else "..."
       
        tau_irf = f"{r['tau_irf_ns']:.5f}" if ('tau_irf_ns' in r and not np.isnan(r['tau_irf_ns'])) else "..."
        err_irf = f"{r['tau_irf_error_ns']:.5f}" if ('tau_irf_error_ns' in r and not np.isnan(r['tau_irf_error_ns'])) else "..."
       
        # 4. IRF + BG (juiste variabelen uit jouw code)
        tau_irf_bg = f"{r['tau_irf_bg_ns']:.5f}" if ('tau_irf_bg_ns' in r and not np.isnan(r['tau_irf_bg_ns'])) else "..."
        err_irf_bg = f"{r['tau_irf_bg_error_ns']:.5f}" if ('tau_irf_bg_error_ns' in r and not np.isnan(r['tau_irf_bg_error_ns'])) else "..."

        print(f"{v_low} & {v_high} & {tau_mono} & {err_mono} & {tau_bg} & {err_bg} & {tau_irf} & {err_irf} & {tau_irf_bg} & {err_irf_bg} \\\\")
        # yes we found the right delta t, we got this man.
        # Good job man, we found the right delta t yeeeees  DT_NS = 2.083507e-2
        # Found the delta t, the solution to the tau_D problem, being inaccurate
        # Fill in, and patch the code where it gets stuck. We are back baby, 17:32
       
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{adjustbox}")
    print(r"\end{table}")

    # ------------------------------------------------------------------
    # TABEL: Muraasota Donor-Acceptor (tau_DA in ns)
    # ------------------------------------------------------------------
    print("\n% " + "="*60)
    print("% TABEL: Shortcuts Donor-Acceptor (tau_DA in ns)")
    print("% " + "="*60)
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\caption{Skelettabel van de gemeten donor-acceptorlevensduren ($\tau_{\mathrm{DA}}$) en bijbehorende standaardfouten ($\sigma$), bepaald met de modelvrije levensduurschatters. Alle levensduren zijn uitgedrukt in nanoseconden.}")
    print(r"\label{tab:tau_da_shortcuts_ns}")
    print(r"\begin{adjustbox}{max width=\textwidth}")
    print(r"\begin{tabular}{cc|cc|cc|cc}")
    print(r"\toprule")
    print(r"$\mathbf{V_{\mathrm{laag}}}$ & $\mathbf{V_{\mathrm{hoog}}}$ & $\mathbf{\tau_{\mathrm{DA},1/e}}$ & $\mathbf{\sigma_{\tau_{\mathrm{DA},1/e}}}$ & $\mathbf{\tau_{\mathrm{DA},\mathrm{RLD}}}$ & $\mathbf{\sigma_{\tau_{\mathrm{DA},\mathrm{RLD}}}}$ & $\mathbf{\tau_{\mathrm{DA},\mathrm{cen}}}$ & $\mathbf{\sigma_{\tau_{\mathrm{DA},\mathrm{cen}}}}$ \\")
    print(r"$(\mu\mathrm{L})$ & $(\mu\mathrm{L})$ & (ns) & (ns) & (ns) & (ns) & (ns) & (ns) \\")
    print(r"\midrule")
   
    for r in results:
        v_low = f"{r.get('v_low', 0):.0f}"
        v_high = f"{r.get('v_high', 0):.0f}"
       
        tau_sc = f"{r['tau_shortcut_ns']:.5f}" if ('tau_shortcut_ns' in r and not np.isnan(r['tau_shortcut_ns'])) else "..."
        err_sc = f"{r['tau_shortcut_err_ns']:.5f}" if ('tau_shortcut_err_ns' in r and not np.isnan(r['tau_shortcut_err_ns'])) else "..."
       
        tau_rld = f"{r['tau_rld_ns']:.5f}" if ('tau_rld_ns' in r and not np.isnan(r['tau_rld_ns'])) else "..."
        err_rld = f"{r['tau_rld_err_ns']:.5f}" if ('tau_rld_err_ns' in r and not np.isnan(r['tau_rld_err_ns'])) else "..."
       
        tau_cen = f"{r['tau_centroid_ns']:.5f}" if ('tau_centroid_ns' in r and not np.isnan(r['tau_centroid_ns'])) else "..."
        err_cen = f"{r['tau_centroid_err_ns']:.5f}" if ('tau_centroid_err_ns' in r and not np.isnan(r['tau_centroid_err_ns'])) else "..."
       
        print(f"{v_low} & {v_high} & {tau_sc} & {err_sc} & {tau_rld} & {err_rld} & {tau_cen} & {err_cen} \\\\")
       
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{adjustbox}")
    print(r"\end{table}")


donor_info = {
    'tau_D_pixels': tau_D_pixels,
    'tau_D_err_pixels': tau_D_err / DT_NS,
    'tau_D_ns': tau_D,
    'tau_D_err_ns': tau_D_err,
    'tau_D_bg_ns': tau_D_bg,
    'tau_D_bg_err_ns': tau_D_bg_err_ns,
    'tau_D_irf_ns': tau_D_irf,
    'tau_D_irf_err': tau_D_irf_err,
    'tau_D_shortcut': tau_D_shortcut,
    'tau_D_shortcut_err': tau_D_shortcut_err,
    'tau_D_rld': tau_D_rld,
    'tau_D_rld_err': tau_D_rld_err,
    'tau_D_centroid': tau_D_centroid,
    'tau_D_centroid_err': tau_D_centroid_err,
    'tau_D_irf_bg_ns': tau_D_irf_bg,
    'tau_D_irf_bg_err': tau_D_irf_bg_err,
}

# Print all 12 LaTeX tables & equations
print_latex_pixel_tables(results, DT_NS, donor_info)
print_latex_ns_tables(results, donor_info)
print_latex_ceff_r_table(results)
print_latex_intensity_table(results, F_D)
print_latex_efficiency_tables(results)
print_latex_tau_da_ns_tables(results)

print(r"\begin{table}[htbp]")
print(r"\centering")
print(r"\footnotesize")
print(r"\caption{Skelettabel van de gereduceerde chi-kwadraatwaarden ($\chi_r^2$) en bijbehorende standaardfouten ($\sigma$) voor de exponentiële fitmodellen.}")
print(r"\label{tab:chi2_reduced}")
print(r"")
print(r"\begin{adjustbox}{max width=\textwidth}")
print(r"\begin{tabular}{cc|cc|cc|cc}")
print(r"\toprule")
print(r"$\mathbf{V_{\mathrm{laag}}}$ & $\mathbf{V_{\mathrm{hoog}}}$ & $\mathbf{\chi_{r,\mathrm{mono}}^2}$ & $\mathbf{\sigma}$ & $\mathbf{\chi_{r,\mathrm{BG}}^2}$ & $\mathbf{\sigma}$ & $\mathbf{\chi_{r,\mathrm{IRF}}^2}$ & $\mathbf{\sigma}$ & $\mathbf{\chi_{r,\mathrm{IRF+BG}}^2}$ & $\mathbf{\sigma}$ \\") # NEW
print(r"$(\mu\mathrm{L})$ & $(\mu\mathrm{L})$ & (-) & (-) & (-) & (-) & (-) & (-) & (-) & (-) \\")
print(r"\midrule")

for r in results:
    v_low = f"{r['v_low']:.0f}"
    v_high = f"{r['v_high']:.0f}"
   
    # Haal de chi2 en sigmas op (vul op met '...' als nan of niet aanwezig)
    c_mono = f"{r['chi2_mono']:.3f}" if ('chi2_mono' in r and not np.isnan(r['chi2_mono'])) else "..."
    s_mono = f"{r['sig_chi2_mono']:.3f}" if ('sig_chi2_mono' in r and not np.isnan(r['sig_chi2_mono'])) else "..."
   
    c_bg   = f"{r['chi2_bg']:.3f}" if ('chi2_bg' in r and not np.isnan(r['chi2_bg'])) else "..."
    s_bg   = f"{r['sig_chi2_bg']:.3f}" if ('sig_chi2_bg' in r and not np.isnan(r['sig_chi2_bg'])) else "..."
   
    c_irf  = f"{r['chi2_irf']:.3f}" if ('chi2_irf' in r and not np.isnan(r['chi2_irf'])) else "..."
    s_irf  = f"{r['sig_chi2_irf']:.3f}" if ('sig_chi2_irf' in r and not np.isnan(r['sig_chi2_irf'])) else "..."
    
    c_irf_bg = f"{r['chi2_irf_bg']:.3f}" if ('chi2_irf_bg' in r and not np.isnan(r['chi2_irf_bg'])) else "..."
    s_irf_bg = f"{r['sig_chi2_irf_bg']:.3f}" if ('sig_chi2_irf_bg' in r and not np.isnan(r['sig_chi2_irf_bg'])) else "..."

    print(f"{v_low} & {v_high} & {c_mono} & {s_mono} & {c_bg} & {s_bg} & {c_irf} & {s_irf} & {c_irf_bg} & {s_irf_bg} \\\\")

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{adjustbox}")
print(r"")
print(r"\end{table}")

# hey jonas, pieter hey

# Donor, reduced chi-squared values and their errors, tabel, just print them
# --------------------------------------------------
# PRINT DE LATEX TABEL VOOR HET DONORSTAAL (D) for reduced chisquared values
# --------------------------------------------------

print(r"\begin{table}[htbp]")
print(r"\centering")
print(r"\footnotesize")
print(r"\begin{adjustbox}{max width=\textwidth}")
print(r"\begin{tabular}{|cc|cc|cc|cc|cc|}")
print(r"\toprule")
print(r"$\mathbf{V_{\mathrm{laag}}}$ &")
print(r"$\mathbf{V_{\mathrm{hoog}}}$ &")
print(r"$\mathbf{\chi_{r,\mathrm{mono}}^2}$ &")
print(r"$\mathbf{\sigma_{\chi_{r,\mathrm{mono}}^2}}$ &")
print(r"$\mathbf{\chi_{r,\mathrm{mono +BG}}^2}$ &")
print(r"$\mathbf{\sigma_{\chi_{r,\mathrm{mono +BG}}^2}}$ &")
print(r"$\mathbf{\chi_{r,\mathrm{mono +IRF}}^2}$ &")
print(r"$\mathbf{\sigma_{\chi_{r,\mathrm{mono +IRF}}^2}}$ &")
print(r"$\mathbf{\chi_{r,\mathrm{mono +IRF+BG}}^2}$ &")
print(r"$\mathbf{\sigma_{\chi_{r,\mathrm{mono +IRF+BG}}^2}}$")
print(r"\\")
print(r"$(\mu\mathrm{L})$ &")
print(r"$(\mu\mathrm{L})$ &")
print(r"(-) & (-) &")
print(r"(-) & (-) &")
print(r"(-) & (-) &")
print(r"(-) & (-)")
print(r"\\")
print(r"\midrule")

print(f"0 & 0 & {chi2_mono_d:.3f} & {sig_mono_d:.3f} & {chi2_bg_d:.3f} & {sig_bg_d:.3f} & {chi2_irf_d:.3f} & {sig_irf_d:.3f} & {chi2_irf_bg_d:.3f} & {sig_irf_bg_d:.3f} \\\\")

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{adjustbox}")
print(r"\caption{Gereduceerde chi-kwadraatwaarden ($\chi_r^2$) en bijbehorende standaardfouten ($\sigma_{\chi_r^2}$) voor de vier fitmodellen voor het donorstaal.}")
print(r"\label{tab:chi2_reduced_four_fit_methodes_pure_donor}")
print(r"\end{table}")
