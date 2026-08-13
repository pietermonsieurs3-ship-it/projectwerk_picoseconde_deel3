import numpy as np

def exponential_decay(t, A, tau):
    return A * np.exp(-t / tau)

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

DT_NS = 2.083507e-2 # ns/pixel
filename = "zuiver_Rh110_horizontal.prf"
# back up 20:58 608

with open(filename, "r", errors="ignore") as f:
    lines = f.readlines()

for i, line in enumerate(lines[:20]):
    print(i, repr(line))
    
print("Total lines:", len(lines))

for i, line in enumerate(lines[-10:]):
    print(i-len(lines), repr(line))

import numpy as np


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


irf = load_hipic_prf(
    "triggering_1ns_vertical.prf"
)

print(irf.shape)
print(irf[:10])

import numpy as np
import matplotlib.pyplot as plt

def load_hipic_prf(filename):
    data = []
    with open(filename, "r", errors="ignore") as f:
        for line in f:
            if line.count(",") == 1:
                try:
                    t, c = line.strip().split(",")
                    data.append([float(t), float(c)])
                except ValueError:
                    continue
    data = np.array(data)
    return data[:, 0], data[:, 1]


# 1. IRF-data inladen
time_irf, counts_irf = load_hipic_prf("triggering_1ns_vertical.prf")

# ---------------------------------------------------------
# 2. EXACTE DETECTIE VAN DE SECUNDAIRE PIEK
# ---------------------------------------------------------

# A. Hoofdpiek
idx_main = np.argmax(counts_irf)
t_main = time_irf[idx_main]
c_main = counts_irf[idx_main]

# B. Secundaire Piek (Schouder)
# We beginnen het zoekvenster wat eerder (+10 i.p.v. +25) om de hoogste schouder mee te pakken
search_start = idx_main + 10  
search_end = idx_main + 100    

# Vind het exacte maximum binnen dit venster
idx_sec = search_start + np.argmax(counts_irf[search_start:search_end])
t_sec = time_irf[idx_sec]
c_sec = counts_irf[idx_sec]


# ---------------------------------------------------------
# 3. PLOTTEN MET ANNOTATIES RECHTS VAN DE PUNTEN
# ---------------------------------------------------------
plt.figure(figsize=(12, 5))

# --- Subplot 1: Lineair ---
plt.subplot(1, 2, 1)
plt.plot(time_irf, counts_irf, color='black', linewidth=1.5, label='IRF')

# Annotatie Hoofdpiek
plt.annotate(
    'Hoofdpiek\n(Excitatiepuls)',
    xy=(t_main, c_main),
    xytext=(t_main + 0.08, c_main * 0.95),  # Tekst netjes rechts van de top
    arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2),
    fontsize=9,
    fontweight='bold'
)

# Annotatie Secundaire Piek
plt.annotate(
    'Secundaire piek\n(Afterpulse / Reflectie)',
    xy=(t_sec, c_sec),
    xytext=(t_sec + 0.08, c_sec + (c_main * 0.08)),  # Tekst netjes rechts van het punt
    arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2),
    fontsize=9,
    fontweight='bold'
)

plt.xlabel('Time (ns)')
plt.ylabel('Counts')
plt.title('Instrument Response Function (Lineair)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')

# --- Subplot 2: Semi-logaritmisch ---
plt.subplot(1, 2, 2)
plt.semilogy(time_irf, counts_irf, color='crimson', linewidth=1.5, label='IRF (log)')
plt.xlabel('Time (ns)')
plt.ylabel('Counts')
plt.title('Instrument Response Function (Semi-log)')
plt.grid(True, which="both", linestyle='--', alpha=0.6)
plt.legend(loc='upper right')

plt.tight_layout()

plt.savefig("IRF_characterization.png", dpi=600, bbox_inches="tight")

plt.show()

import numpy as np
import matplotlib.pyplot as plt

def load_hipic_prf(filename):
    data = []
    with open(filename, "r", errors="ignore") as f:
        for line in f:
            if line.count(",") == 1:
                try:
                    t, c = line.strip().split(",")
                    data.append([float(t), float(c)])
                except ValueError:
                    continue
    data = np.array(data)
    return data[:, 0], data[:, 1]

# 1. IRF-data inladen
time_irf, counts_irf = load_hipic_prf("triggering_1ns_vertical.prf")

# 2. Exacte detectie van de piek en de secundaire zijpiek
idx_main = np.argmax(counts_irf)
t_main = time_irf[idx_main]
c_main = counts_irf[idx_main]

# Zoekvenster voor de secundaire piek (schouder)
search_start = idx_main + 10  
search_end = idx_main + 100    
idx_sec = search_start + np.argmax(counts_irf[search_start:search_end])
t_sec = time_irf[idx_sec]
c_sec = counts_irf[idx_sec]

# 3. Plotten van enkel de lineaire grafiek
plt.figure(figsize=(8, 5))

plt.plot(time_irf, counts_irf, color='black', linewidth=1.5, label='Gemeten IRF')

# Annotatie Hoofdpiek
plt.annotate(
    'Hoofdpiek\n(Excitatiepuls)',
    xy=(t_main, c_main),
    xytext=(t_main + 0.08, c_main * 0.95),
    arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2),
    fontsize=9,
    fontweight='bold'
)

# Annotatie Secundaire Piek (Artefact)
plt.annotate(
    'Secundaire zijpiek\n(Instrumenteel artefact)',
    xy=(t_sec, c_sec),
    xytext=(t_sec + 0.08, c_sec + (c_main * 0.08)),
    arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2),
    fontsize=9,
    fontweight='bold'
)

plt.xlabel('Tijd (ns)', fontsize=10)
plt.ylabel('Intensiteit (Counts)', fontsize=10)
plt.title('Instrument Response Function (IRF) met secundair artefact', fontweight='bold', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')

plt.tight_layout()

# Opslaan als hoge-resolutie afbeelding voor je verslag
plt.savefig("IRF_linear_artifact.png", dpi=600, bbox_inches="tight")

plt.show()

#### Mock below of inset

import matplotlib.pyplot as plt
import numpy as np

# 1. Maak hoofddata
x = np.linspace(1, 10, 100)
y = np.log(x)

# 2. Maak de hoofdfiguur en hoofdas aan
fig, ax = plt.subplots(figsize=(8, 5))

# Plot op de hoofdas
ax.plot(x, y, label='log(x)', color='blue')
ax.set_xlabel('X-as')
ax.set_ylabel('Y-as')
ax.set_title('Hoofdplot met Inset')

# 3. Voeg de INSET toe: [x_pos, y_pos, breedte, hoogte] in fracties van de hoofdas
ax_inset = ax.inset_axes([0.55, 0.15, 0.38, 0.35])

# 4. Plot de data in de inset (bijvoorbeeld inzoomen op x tussen 8 en 10)
ax_inset.plot(x, y, color='blue', marker='o', ms=2)

# Zoom in op de inset door de limieten in te stellen:
ax_inset.set_xlim(8, 10)
ax_inset.set_ylim(np.log(8) - 0.05, np.log(10) + 0.05)

# Opmaak van de inset (optioneel kleiner lettertype en raster)
ax_inset.set_title('Zoom (x=8..10)', fontsize=8)
ax_inset.tick_params(labelsize=7)
ax_inset.grid(True, linestyle=':', alpha=0.6)

# 5. OPTIONEEL: Teken verbindingslijnen van de hoofdplot naar de inset
ax.indicate_inset_zoom(ax_inset, edgecolor="gray", alpha=0.7)

plt.show()

######

# Validation code to check for pile-up

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tifffile as tiff

# ==============================================================================
# 1. EXPLICIETE LINUX PADEN (.tif)
# ==============================================================================
path_normal = Path("/home/vboxuser/Downloads/pw/Files_pp_ZNP/dag3/C307_Rh110_85mul_70mul.tif")
path_dimmer = Path("/home/vboxuser/Downloads/pw/Files_pp_ZNP/dag3/C307_Rh110_85mul_70mul_dimmer.tif")

# TCSPC tijdskalibratie (in nanoseconden per bin, bijv. 0.0125 ns)
time_per_bin = 2.083507e-2

# ==============================================================================
# 2. HULPFUNCTIE MET EXPLICITE AXIS=1 SOMMATIE
# ==============================================================================
def load_tcspc_decay_axis1(file_path):
    """
    Leest de TIFF in en isoleert de tijdsas op axis=1.
    Sommeert over de ruimtelijke assen (0 en 2).
    """
    with tiff.TiffFile(file_path) as tf:
        data = tf.asarray()
   
    print(f"Data shape voor {file_path.name}: {data.shape}")

    # Als de data 3D is [y, t, x] of [z, t, x] met tijdsas op axis=1:
    if data.ndim == 3:
        # Sommeer over as 0 en as 2, zodat alleen as 1 (tijd) overblijft
        decay = np.sum(data, axis=1)
    elif data.ndim == 2:
        # Als het een 2D matrix is [t, x], sommeer over as 1
        decay = np.sum(data, axis=1)
    else:
        # Mocht er een extra dimensie zijn [c, y, t, x]
        sum_axes = tuple(i for i in range(data.ndim) if i != 1)
        decay = np.sum(data, axis=sum_axes)

    return decay.astype(float)

def mono_exp(t, A, tau, bg):
    return A * np.exp(-t / tau) + bg

def exponential_decay(t_pix, A, tau_pix):
    # Pure mono-exponentieel (worst-case zonder achtergrond)
    return A * np.exp(-t_pix / tau_pix)

# 6:52, time to add mono too
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tifffile as tiff

# ==============================================================================
# 1. INSTELLINGEN (100% GELIJK AAN FRET-CODE)
# ==============================================================================
path_normal = Path("/home/vboxuser/Downloads/pw/Files_pp_ZNP/dag3/C307_Rh110_85mul_70mul.tif")
path_dimmer = Path("/home/vboxuser/Downloads/pw/Files_pp_ZNP/dag3/C307_Rh110_85mul_70mul_dimmer.tif")

DT_NS = 2.083507e-2  # ns per pixel
CUTOFF_NS = 477 * DT_NS
CUTOFF_PIXEL = int(CUTOFF_NS / DT_NS)

# Exacte ROI overgenomen uit de FRET-code
ROI_START = 50
ROI_END = 150

# ==============================================================================
# 2. FUNCTIES
# ==============================================================================
def load_tcspc_decay(file_path, r_start, r_end):
    image = tiff.imread(file_path)
    if image.ndim == 3:
        roi = image[:, r_start:r_end]
        decay = np.sum(roi, axis=(0, 2))
    elif image.ndim == 2:
        roi = image[:, r_start:r_end]
        decay = np.sum(roi, axis=1)
    else:
        decay = np.sum(image, axis=1)
    return decay.astype(float)

def exponential_background_decay(t_pix, A, tau_pix, B):
    # Fit werkt stabieler in pixel-eenheden; tau_pix wordt later omgezet naar ns
    return A * np.exp(-t_pix / tau_pix) + B

# ==============================================================================
# 3. DATA INLADEN & FITTEN
# ==============================================================================
decay_norm = load_tcspc_decay(path_normal, ROI_START, ROI_END)
decay_dim  = load_tcspc_decay(path_dimmer, ROI_START, ROI_END)

t = np.arange(len(decay_norm))
time_axis_ns = t * DT_NS

peak_norm = np.argmax(decay_norm)
peak_dim  = np.argmax(decay_dim)

fit_start_norm = peak_norm + 15
fit_start_dim  = peak_dim + 15
fit_end = min(len(decay_norm), CUTOFF_PIXEL)

# --- Fit Normaal (in pixels, om 0.03 ns bug te voorkomen) ---
t_fit_n = t[fit_start_norm:fit_end]
y_fit_n = decay_norm[fit_start_norm:fit_end]
mask_n = y_fit_n > 0
t_fit_n = t_fit_n[mask_n] - t_fit_n[0]  # Start exact op 0 pixels
y_fit_n = y_fit_n[mask_n]
sigma_n = np.sqrt(y_fit_n)
sigma_n[sigma_n == 0] = 1e-5

# Verwachte tau in pixels (bijv. 4.8 ns / 0.0208 ns ≈ 230 pixels)
popt_n, _ = curve_fit(
    exponential_background_decay, t_fit_n, y_fit_n,
    sigma=sigma_n, absolute_sigma=True,
    p0=[np.max(y_fit_n), 200.0, np.percentile(y_fit_n, 5)],
    bounds=([0, 1.0, 0], [np.inf, np.inf, np.inf]), maxfev=50000
)

# --- Fit Gedimd ---
t_fit_d = t[fit_start_dim:fit_end]
y_fit_d = decay_dim[fit_start_dim:fit_end]
mask_d = y_fit_d > 0
t_fit_d = t_fit_d[mask_d] - t_fit_d[0]
y_fit_d = y_fit_d[mask_d]
sigma_d = np.sqrt(y_fit_d)
sigma_d[sigma_d == 0] = 1e-5

popt_d, _ = curve_fit(
    exponential_background_decay, t_fit_d, y_fit_d,
    sigma=sigma_d, absolute_sigma=True,
    p0=[np.max(y_fit_d), 200.0, np.percentile(y_fit_d, 5)],
    bounds=([0, 1.0, 0], [np.inf, np.inf, np.inf]), maxfev=50000
)

# Omrekenen van pixel-tau naar nanoseconden
tau_norm_ns = popt_n[1] * DT_NS
tau_dim_ns  = popt_d[1] * DT_NS
diff_perc = abs(tau_norm_ns - tau_dim_ns) / tau_norm_ns * 100

print(f"\nCORRECTE RESULTATEN:")
print(f"  • Tau Normaal : {tau_norm_ns:.3f} ns")
print(f"  • Tau Gedimd  : {tau_dim_ns:.3f} ns")
print(f"  • Verschil    : {diff_perc:.2f}%\n")

# --- Fit Normaal (Worst-Case: puur mono zonder BG) ---
popt_n_mono, _ = curve_fit(
    exponential_decay, t_fit_n, y_fit_n,
    sigma=sigma_n, absolute_sigma=True,
    p0=[np.max(y_fit_n), 200.0],
    bounds=([0, 1.0], [np.inf, np.inf]), maxfev=50000
)

# --- Fit Gedimd (Worst-Case: puur mono zonder BG) ---
popt_d_mono, _ = curve_fit(
    exponential_decay, t_fit_d, y_fit_d,
    sigma=sigma_d, absolute_sigma=True,
    p0=[np.max(y_fit_d), 200.0],
    bounds=([0, 1.0], [np.inf, np.inf]), maxfev=50000
)

# Omrekenen van pixel-tau naar nanoseconden voor het worst-case model
tau_norm_ns_mono = popt_n_mono[1] * DT_NS
tau_dim_ns_mono  = popt_d_mono[1] * DT_NS
diff_perc_mono = abs(tau_norm_ns_mono - tau_dim_ns_mono) / tau_norm_ns_mono * 100

print(f"WORST-CASE RESULTATEN (Puur Mono):")
print(f"  • Tau Normaal (mono) : {tau_norm_ns_mono:.5f} ns")
print(f"  • Tau Gedimd  (mono) : {tau_dim_ns_mono:.5f} ns")
print(f"  • Verschil    (mono) : {diff_perc_mono:.2f}%\n")

# ==============================================================================
# 4. PLOTTEN (INCLUSIEF WORST-CASE MONO-FITS)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(9, 8),
    sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)

# Bovenste plot: Rauwe data op lineaire schaal
ax1.plot(time_axis_ns, decay_norm, label='Data Normaal', color='blue', alpha=0.5, linewidth=1.2)
ax1.plot(time_axis_ns, decay_dim, label='Data Gedimd', color='green', alpha=0.5, linewidth=1.2)

# 1. Best-Case Fits plotten (mono + BG)
t_plot_n_ns = (t[fit_start_norm:fit_end][mask_n] - t[fit_start_norm]) * DT_NS
t_plot_d_ns = (t[fit_start_dim:fit_end][mask_d] - t[fit_start_dim]) * DT_NS

ax1.plot(t_plot_n_ns + (fit_start_norm * DT_NS), exponential_background_decay(t_fit_n, *popt_n),
         color='red', linestyle='--', linewidth=1.5, label=f'Fit Normaal (mono+BG, $\\tau$ = {tau_norm_ns:.3f} ns)')
ax1.plot(t_plot_d_ns + (fit_start_dim * DT_NS), exponential_background_decay(t_fit_d, *popt_d),
         color='orange', linestyle='--', linewidth=1.5, label=f'Fit Gedimd (mono+BG, $\\tau$ = {tau_dim_ns:.3f} ns)')

# 2. Worst-Case Fits plotten (Puur mono / zonder BG)
ax1.plot(t_plot_n_ns + (fit_start_norm * DT_NS), exponential_decay(t_fit_n, *popt_n_mono),
         color='purple', linestyle=':', linewidth=1.5, label=f'Fit Normaal (mono, $\\tau$ = {tau_norm_ns_mono:.3f} ns)')
ax1.plot(t_plot_d_ns + (fit_start_dim * DT_NS), exponential_decay(t_fit_d, *popt_d_mono),
         color='magenta', linestyle=':', linewidth=1.5, label=f'Fit Gedimd (mono, $\\tau$ = {tau_dim_ns_mono:.3f} ns)')

ax1.set_ylabel("Intensiteit (Counts)", fontsize=10)
ax1.set_title(
    r"Validatie: Vergelijking best-case (mono + BG) en worst-case (mono) fits" + "\n" +
    r"voor sample $V_{\mathrm{laag}}$ = 85 $\mu$L en $V_{\mathrm{hoog}}$ = 70 $\mu$L",
    fontweight='bold', fontsize=11, pad=12
)
ax1.legend(loc='upper right', fontsize=8, frameon=True, facecolor='white')
ax1.grid(True, linestyle=":", alpha=0.5)

# Onderste plot: Het normale verschil tussen data
min_len = min(len(decay_norm), len(decay_dim))
t_diff = time_axis_ns[:min_len]
normal_difference = decay_norm[:min_len] - decay_dim[:min_len]

ax2.plot(t_diff, normal_difference, color='teal', linewidth=1.0, label='Verschil ($I_{norm} - I_{dim}$)')
ax2.axhline(0, color='black', linestyle='--', linewidth=1.0, label='Identiek signaal')
ax2.set_xlabel("Tijd (ns)", fontsize=10)
ax2.set_ylabel("Verschil (Counts)", fontsize=9)
ax2.legend(loc='best', fontsize=8.5, frameon=True, facecolor='white')
ax2.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
plt.show()

#### next val, PCA

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from sklearn.decomposition import PCA

# ==============================================================================
# 1. EXPLICIETE PADEN (.tif)
# ==============================================================================
path_normal = Path("/home/vboxuser/Downloads/pw/Files_pp_ZNP/dag3/C307_Rh110_85mul_70mul.tif")
path_dimmer = Path("/home/vboxuser/Downloads/pw/Files_pp_ZNP/dag3/C307_Rh110_85mul_70mul_dimmer.tif")

# ==============================================================================
# 2. HULPFUNCTIE: OMZETTEN VAN TIFF STACK NAAR PCA MATRIX (PIXELS x TIJDSBINS)
# ==============================================================================
def tiff_to_pixel_matrix(file_path, threshold_ratio=0.05):
    """
    Verwerkt 2D FLIM data van de vorm (480, 640).
    As 0 (480) = Ruimtelijke pixels/profielen
    As 1 (640) = Tijdsbins (features voor PCA)
    """
    with tiff.TiffFile(file_path) as tf:
        data = tf.asarray().astype(float)
       
    print(f"  • {file_path.name} geladen met vorm: {data.shape}")

    # Als de data al 2D is (480 pixels x 640 tijdsbins)
    if data.ndim == 2:
        matrix = data
    else:
        # Mocht er toch een extra dimensie zijn, flatten we alle assen behalve as 1
        matrix = np.transpose(data, (0, 2, 1)).reshape(-1, data.shape[1]) if data.ndim == 3 else data

    # 1. Filter achtergrondruis (neem alleen pixels mee met voldoende fotonen)
    max_per_pixel = np.max(matrix, axis=1)
    cutoff = np.max(max_per_pixel) * threshold_ratio
    valid_pixels = matrix[max_per_pixel > cutoff]
   
    # 2. L1-Normalisatie: Schaal elke rij naar som = 1.0
    # (Hiermee kijken we puur naar de VORM van het verval, onafhankelijk van hoe fel de pixel is)
    row_sums = valid_pixels.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # Voorkom delen door nul
    normalized_pixels = valid_pixels / row_sums

    return normalized_pixels

print("🔄 Data inladen en klaarmaken voor PCA...")
X_norm = tiff_to_pixel_matrix(path_normal)
X_dim = tiff_to_pixel_matrix(path_dimmer)

print(f"  • Actieve pixels Normaal: {X_norm.shape[0]}")
print(f"  • Actieve pixels Gedimd : {X_dim.shape[0]}")

# ==============================================================================
# 3. PCA TRAINEN EN TRANSFORMEREN
# ==============================================================================
# Voeg beide datasetjes samen om één gedeelde PCA-ruimte te bouwen
X_combined = np.vstack([X_norm, X_dim])

# Fit PCA op de gecombineerde data (reduceren naar 2 hoofdcomponenten)
pca = PCA(n_components=2)
pca.fit(X_combined)

# Transformeer beide datasets naar de nieuwe PC1-PC2 assen
X_norm_pca = pca.transform(X_norm)
X_dim_pca = pca.transform(X_dim)

exp_var = pca.explained_variance_ratio_ * 100
print(f"✅ PCA voltooid! Uitgelegde variantie: PC1 = {exp_var[0]:.2f}%, PC2 = {exp_var[1]:.2f}%")

# ==============================================================================
# 4. VISUALISATIE
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse

def add_confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """
    Voegt een covariance/betrouwbaarheidsellips toe aan een Matplotlib 'ax'.
    """
    if x.size != y.size:
        raise ValueError("x en y moeten dezelfde lengte hebben")

    # 1. Bereken de covariantie en Pearson correlatiecoëfficiënt
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
   
    # 2. Bepaal de stralen van de standaard-ellips
    rad_x = np.sqrt(1 + pearson)
    rad_y = np.sqrt(1 - pearson)
   
    ellipse = Ellipse((0, 0), width=rad_x * 2, height=rad_y * 2, **kwargs)
   
    # 3. Schaal en roteer de ellips naar de data-coördinaten
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
   
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
   
    # Gebruik de juiste transformatie uit matplotlib.transforms
    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )
   
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# ==============================================================================
# PLOT 2: PCA SCATTER PLOT MET CENTROÏDEN EN MENGING-ANALYSE
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from matplotlib.patches import Ellipse

# 1. BEREKEN DE CENTROÏDEN (HET MIDDELPUNT VAN ELKE WOLK)
centroid_norm = np.mean(X_norm_pca, axis=0)
centroid_dim  = np.mean(X_dim_pca, axis=0)

# 2. BEREKEN DE EUCLIDISCHE AFSTAND TUSSEN DE CENTROÏDEN
centroid_dist = np.linalg.norm(centroid_norm - centroid_dim)

# 3. BEREKEN DE SILHOUETTE SCORE (Kwantitatieve overlap: ~0.00 = 100% overlap)
# Neem een representatieve sample voor snelheid als de dataset heel groot is
labels = np.array([0] * len(X_norm_pca) + [1] * len(X_dim_pca))
X_all_pca = np.vstack([X_norm_pca, X_dim_pca])

# Bereken op maximaal 3000 punten om de berekening super snel te houden
sample_indices = np.random.choice(len(X_all_pca), size=min(3000, len(X_all_pca)), replace=False)
sil_score = silhouette_score(X_all_pca[sample_indices], labels[sample_indices])

print("\n" + "="*50)
print("PCA VALIDATIE METRIEKEN:")
print(f"  • Centroïde Afstand : {centroid_dist:.6f}")
print(f"  • Silhouette Score  : {sil_score:.4f} (0.000 = Volledige overlap)")
print("="*50 + "\n")

# 4. PLOTTEN
plt.figure(figsize=(8.5, 6))

# A. Scatter plot van de datapunten
plt.scatter(
    X_norm_pca[:, 0], X_norm_pca[:, 1],
    c='blue', alpha=0.25, s=12, label='Data normaal'
)
plt.scatter(
    X_dim_pca[:, 0], X_dim_pca[:, 1],
    c='red', alpha=0.25, s=12, label='Data gedimd'
)

# B. Centroïden markeren met opvallende kruisen
plt.plot(
    centroid_norm[0], centroid_norm[1],
    color='blue', marker='X', markersize=12, markeredgecolor='white', markeredgewidth=1.5,
    linestyle='None', label='Centroïde normaal'
)
plt.plot(
    centroid_dim[0], centroid_dim[1],
    color='red', marker='X', markersize=12, markeredgecolor='white', markeredgewidth=1.5,
    linestyle='None', label='Centroïde gedimd'
)

# ==============================================================================
# BETROUWBAARHEIDSELLIPSEN TOEVOEGEN (95% / 2-sigma)
# ==============================================================================
# Ellips voor Data Normaal (Blauw)
add_confidence_ellipse(
    X_norm_pca[:, 0], X_norm_pca[:, 1],
    plt.gca(), n_std=2.0,
    edgecolor='blue', facecolor='blue', alpha=0.1, linestyle='--', linewidth=1.5,
    label='95% conf. ellips normaal'
)

# Ellips voor Data Gedimd (Rood)
add_confidence_ellipse(
    X_dim_pca[:, 0], X_dim_pca[:, 1],
    plt.gca(), n_std=2.0,
    edgecolor='red', facecolor='red', alpha=0.1, linestyle='--', linewidth=1.5,
    label='95% conf. ellips gedimd'
)

# ==============================================================================
# ANNOTATIE VOOR DE OVERLAPPENDE CENTROÏDEN
# ==============================================================================
# Pijl en tekst die naar de rode/blauwe centroïde op de grafiek wijst
plt.annotate(
    "Centroïden overlappen volledig\n(Blauw kruis ligt direct achter rood)",
    xy=(centroid_dim[0], centroid_dim[1]),          # Wijst naar het rode/blauwe kruis
    xytext=(centroid_dim[0] -0.003, centroid_dim[1] +0.005), # Positie van de tekst
    arrowprops=dict(
        arrowstyle='->',
        color='black',
        linewidth=1.2,
        connectionstyle="arc3,rad=-0.1"
    ),
    fontsize=9,
    fontweight='bold',
    color='black',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', alpha=0.9)
)

# C. Statistische tekstbox toevoegen
stats_pca_text = (
    f"Centroïde afstand: {centroid_dist:.4f}\n"
    f"Silhouette score : {sil_score:.4f}"
)
plt.gca().text(
    0.04, 0.06, stats_pca_text,
    transform=plt.gca().transAxes, fontsize=8.5,
    verticalalignment='bottom', horizontalalignment='left',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray')
)

# D. Titels, Assen & Legenda
plt.title("Machine learning: PCA scatter plot", fontweight='bold', fontsize=12, pad=12)
plt.xlabel(f"Principal Component 1 ({exp_var[0]:.1f}% variantie)", fontsize=10)
plt.ylabel(f"Principal Component 2 ({exp_var[1]:.1f}% variantie)", fontsize=10)

plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
plt.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()

####

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

# ==============================================================================
# COSINE SIMILARITY & CROSS-CORRELATIE BEREKENEN
# ==============================================================================
# Zorg dat beide gecropt/uitgelijnd zijn op de piek (zoals in je vervalcode)
u = decay_norm  # Genormaliseerde array van Normaal
v = decay_dim   # Genormaliseerde array van Gedimd

# 1. Cosine Similarity (Vorm-overeenkomst)
dot_product = np.dot(u, v)
norm_u = np.linalg.norm(u)
norm_v = np.linalg.norm(v)
cosine_sim = dot_product / (norm_u * norm_v)

# 2. Genormaliseerde Cross-Correlatie (over verschillende tijdverschuivingen/lags)
cross_corr = correlate(u, v, mode='full')
cross_corr /= np.max(cross_corr)  # Schaal piek naar 1.0
lags = np.arange(-len(u) + 1, len(u))

print("\n" + "="*50)
print("3e VALIDATIE: SIGNAAL SIMILARITY METRIEK")
print(f"  • Cosine Similarity : {cosine_sim:.6f}  (1.000000 = Identiek)")
print(f"  • Piek Correlatie   : {np.max(cross_corr):.6f} op lag {lags[np.argmax(cross_corr)]}")
print("="*50 + "\n")

# Plot van de Cross-Correlatie Piek
plt.figure(figsize=(7, 3))
plt.plot(lags, cross_corr, color='teal', linewidth=1.5)
plt.axvline(0, color='red', linestyle='--', alpha=0.7, label='Nul-verschuiving (Lag = 0)')
plt.title(f"Genormaliseerde Cross-Correlatie (Cosine Sim = {cosine_sim:.4f})", fontweight='bold')
plt.xlabel("Tijdsverschuiving / Lag (bins)")
plt.ylabel("Correlatie Coëfficiënt")
plt.xlim(-50, 50)  # Zoom in op de piek
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()

print("let's do this ******************")

import numpy as np
import tifffile
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve

# ---------------------------------------------------------------------
# 1. HULPFUNCTIES & MODELLEN
# ---------------------------------------------------------------------

def load_hipic_prf(filename):
    rows = []
    with open(filename, "r", errors="ignore") as f:
        for line in f:
            clean = line.replace(";", "").strip()
            if clean == "":
                continue
            parts = clean.replace(",", " ").split()
            numbers = []
            for p in parts:
                try:
                    numbers.append(float(p))
                except:
                    pass
            if len(numbers) >= 2:
                rows.append(numbers[:2])
    return np.array(rows)

# Laad de IRF in
irf_data_file = load_hipic_prf("triggering_1ns_vertical.prf")
irf_counts = irf_data_file[:, 1]
irf_counts = irf_counts / np.max(irf_counts)  # normaliseren

# Fitfuncties
def exponential_decay(t, A, tau):
    return A * np.exp(-t / tau)

def exponential_background_decay(t, A, tau, B):
    return A * np.exp(-t / tau) + B

def irf_mono_decay_causal(t_arr, A, tau, t0, irf):
    shift_idx = int(np.round(t0))
    t_model = np.arange(len(irf)) - shift_idx
    mono = np.exp(-t_model / tau)
    mono[t_model < 0] = 0
    convoluted = fftconvolve(mono, irf, mode='full')[:len(t_arr)]
    max_val = np.max(convoluted)
    if max_val == 0:
        return np.zeros_like(t_arr)
    return A * convoluted / max_val

def irf_mono_decay_causal_bg(t_arr, A, tau, t0, B, irf):
    shift_idx = int(np.round(t0))
    t_model = np.arange(len(irf)) - shift_idx
    mono = np.exp(-t_model / tau)
    mono[t_model < 0] = 0
    convoluted = fftconvolve(mono, irf, mode='full')[:len(t_arr)]
    max_val = np.max(convoluted)
    if max_val == 0:
        return B * np.ones_like(t_arr)
    return A * (convoluted / max_val) + B


# ---------------------------------------------------------------------
# 2. VALIDATIE FUNCTIE VOOR ÉÉN BESTAND (Met jouw originele FRET-instellingen)
# ---------------------------------------------------------------------
def run_validation_for_file(filepath):
    image = tifffile.imread(filepath)
    ROI_START, ROI_END = 50, 150  
   
    if image.ndim == 3:
        roi = image[:, ROI_START:ROI_END]
        raw_decay = np.sum(roi, axis=(0, 2))
    else:
        roi = image[:, ROI_START:ROI_END]
        raw_decay = np.sum(roi, axis=1)
       
    decay = raw_decay.copy()
    t = np.arange(len(decay))
    peak_index = np.argmax(decay)
   
    # Fit window voor tail (mono & mono+BG)
    fit_start_tail = peak_index + 15
    fit_end = min(len(decay), int((477 * 2.083507e-2) / 2.083507e-2))
   
    t_fit_tail = t[fit_start_tail:fit_end] - t[fit_start_tail]
    y_fit_tail = decay[fit_start_tail:fit_end]
    mask_tail = y_fit_tail > 0
    t_fit_tail, y_fit_tail = t_fit_tail[mask_tail], y_fit_tail[mask_tail]
    sigma_fit_tail = np.sqrt(y_fit_tail)

    # EXACT JOUW ORIGINELE IRF FIT-VENSTER EN GISSINGEN:
    fit_start_irf = max(0, peak_index - 50)
    t_fit_irf = t[fit_start_irf:fit_end] - t[fit_start_irf]
    y_fit_irf = decay[fit_start_irf:fit_end]
    mask_irf = y_fit_irf > 0
    t_fit_irf, y_fit_irf = t_fit_irf[mask_irf], y_fit_irf[mask_irf]
    sigma_fit_irf = np.sqrt(y_fit_irf)

    DT_NS = 2.083507e-2

    # 1. Model: MONO
    popt_mono, _ = curve_fit(exponential_decay, t_fit_tail, y_fit_tail, sigma=sigma_fit_tail, absolute_sigma=True, p0=[np.max(y_fit_tail), 50], maxfev=20000)
    tau_mono = popt_mono[1] * DT_NS

    # 2. Model: MONO + BG
    initial_bg = np.percentile(y_fit_tail, 5)
    popt_bg, _ = curve_fit(exponential_background_decay, t_fit_tail, y_fit_tail, sigma=sigma_fit_tail, absolute_sigma=True, p0=[np.max(y_fit_tail), 50, initial_bg], maxfev=20000)
    tau_bg = popt_bg[1] * DT_NS

    # 3. Model: MONO + IRF (Exact jouw FRET-code instellingen)
    popt_irf, _ = curve_fit(
        lambda t, A, tau, t0: irf_mono_decay_causal(t, A, tau, t0, irf_counts),
        t_fit_irf, y_fit_irf,
        sigma=sigma_fit_irf, absolute_sigma=True,
        p0=[np.max(y_fit_irf), 100, 50],
        bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]),
        maxfev=50000
    )
    tau_irf = popt_irf[1] * DT_NS

    # 4. Model: MONO + IRF + BG (Exact jouw FRET-code instellingen)
    A_guess_sample = np.max(y_fit_irf)
    rel_peak_offset = 50  # of peak_index afhankelijk van hoe je t_fit_irf indexeert
    initial_bg_sample = np.percentile(y_fit_irf, 5)
   
    popt_irf_bg, _ = curve_fit(
        lambda t_arr, A, tau, t0, B_val: irf_mono_decay_causal_bg(t_arr, A, tau, t0, B_val, irf_counts),
        t_fit_irf, y_fit_irf,
        sigma=sigma_fit_irf, absolute_sigma=True,
        p0=[A_guess_sample, 100, rel_peak_offset, initial_bg_sample],
        bounds=([0, 0, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf]),
        maxfev=50000
    )
    tau_irf_bg = popt_irf_bg[1] * DT_NS

    return {'mono': tau_mono, 'bg': tau_bg, 'irf': tau_irf, 'irf_bg': tau_irf_bg}


# ---------------------------------------------------------------------
# 3. UITVOEREN VOOR NORMAAL EN GEDIMD
# ---------------------------------------------------------------------
path_normal = "C307_Rh110_85mul_70mul.tif"                  
path_dimmer = "C307_Rh110_85mul_70mul_dimmer.tif"

print("Bezig met doorrekenen van de validatiemodellen met originele FRET-instellingen...")
res_norm = run_validation_for_file(path_normal)
res_dim  = run_validation_for_file(path_dimmer)

print("\n==================================================")
print(" RESULTATEN VALIDATIE: NORMAAL VS. GEDIMD")
print("==================================================")
for model_name in ['mono', 'bg', 'irf', 'irf_bg']:
    t_n = res_norm[model_name]
    t_d = res_dim[model_name]
    diff_pct = abs(t_n - t_d) / t_n * 100
    print(f"Model [{model_name.upper():<9}] -> Normaal: {t_n:.3f} ns | Gedimd: {t_d:.3f} ns | Verschil: {diff_pct:.2f}%")
print("==================================================")


import numpy as np
import tifffile
import matplotlib.pyplot as plt

# Laad je normale meetbestand
path_normal = "C307_Rh110_85mul_70mul.tif"
image = tifffile.imread(path_normal)

# Tel de intensiteit op over de tijd-as (axis 0 of de juiste as in jouw data)
# Dit geeft een ruimtelijk profiel van je pixels
if image.ndim == 3:
    spatial_profile = np.sum(image, axis=0)
    # Als het een 2D-spatial array is over de pixels, tellen we even handig op:
    if spatial_profile.ndim == 2:
        spatial_profile = np.sum(spatial_profile, axis=1) # of axis=0 afhankelijk van je oriëntatie
else:
    spatial_profile = np.sum(image, axis=0)

# Plot het profiel om te zien waar je monster zit
plt.figure(figsize=(8, 4))
plt.plot(spatial_profile, label="Intensiteit per pixel", color='purple')

# Teken jouw gekozen ROI lijnen erin als verticale lijnen
plt.axvline(x=430, color='red', linestyle='--', label='ROI Start (50)')
plt.axvline(x=590, color='blue', linestyle='--', label='ROI End (150)')

plt.xlabel("Pixel Index")
plt.ylabel("Totale Intensiteit")
plt.title("Check je ROI: Valt dit precies over je piek?")
plt.legend()
plt.grid(True)
plt.show()
