# This code is meant for the analysis of part 3 of picoseconde project
# The simple inspection

from pathlib import Path
import tifffile
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Paths
# --------------------------------------------------

# Files in the same folder
# as script
DATA_DIR = Path(".")

# Debug
print("Looking in:", DATA_DIR.resolve())

# --------------------------------------------------
# Find all tif files
# --------------------------------------------------

tif_files = sorted(DATA_DIR.glob("*.tif"))

print("\nFound files:\n")

for file in tif_files:
    print(file.name)

# --------------------------------------------------
# Load first file as example
# --------------------------------------------------

example_file = tif_files[0]

image = tifffile.imread(example_file)

print("\nExample file:")
print(example_file.name)

print("\nShape:")
print(image.shape)

print("\nDatatype:")
print(image.dtype)

# --------------------------------------------------
# Show raw streak image
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.imshow(image, aspect='auto', origin='lower')

plt.title(example_file.name)
plt.xlabel("Time pixel")
plt.ylabel("Spatial pixel")

plt.colorbar(label="Intensity")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Preliminary decay extraction
# --------------------------------------------------

# Sum over spatial axis
# Ignore noisy edge pixels
# This is where the interest lies
roi = image[20:460, :]

# Sum over spatial direction
decay_curve = np.sum(roi, axis=0)

# Normalize
decay_curve = decay_curve / np.max(decay_curve)

# Plot decay
plt.figure(figsize=(7, 5))

plt.plot(decay_curve)

plt.yscale("log")

plt.xlabel("Time pixel")
plt.ylabel("Normalized intensity")

plt.title("Preliminary fluorescence decay")

plt.tight_layout()
plt.show()


import tifffile
import numpy as np
import matplotlib.pyplot as plt

files = [
    "C307_Rh110_5mul.tif",
    "C307_Rh110_10mul.tif",
    "C307_Rh110_85mul_70mul_dimmer.tif",
    "zuiver_Rh110.tif",
    "zuiver_C307.tif"
]

for f in files:
    img = tifffile.imread(f)
    roi = img[20:460, :]
    trace = np.sum(roi, axis=0)
    trace = trace / np.max(trace)

    plt.plot(trace, label=f)

plt.yscale("log")
plt.legend()
plt.title("Checking for pseudo-IRF candidates")
plt.show()

files = [
    "zuiver_C307.tif",
    "zuiver_Rh110.tif",
    "C307_Rh110_5mul.tif",
    "C307_Rh110_85mul_70mul_dimmer.tif"
]

for f in files:
    img = tifffile.imread(f)
    roi = img[20:460, :]
    trace = np.sum(roi, axis=0)
    trace = trace / np.max(trace)

    plt.plot(trace, label=f)

plt.yscale("log")
plt.legend()
plt.title("IRF candidate check")
plt.show()

from pathlib import Path

file_path = Path("triggering_1ns_vertical.prf")

with open(file_path, "r", errors="ignore") as f:
    for i, line in enumerate(f):
        print(line.rstrip())
        if i > 200:
            break
