import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1. Fluorescence decay (lifetime model)
# =========================================================

t = np.linspace(0, 10, 500)

tau_D = 3.0
tau_DA = 1.5

I_D = np.exp(-t / tau_D)
I_DA = np.exp(-t / tau_DA)

plt.figure(figsize=(7, 4.5))
plt.plot(t, I_D, label=r'Donor only ($\tau_D$)', linewidth=2)
plt.plot(t, I_DA, label=r'With acceptor ($\tau_{DA}$)', linewidth=2)

plt.xlabel("Time (t)")
plt.ylabel("Fluorescence intensity I(t)")
plt.title("Schematic fluorescence decay: $I(t) \\propto e^{-t/\\tau}$")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fluorescence_decay_schematic.png", dpi=300)
plt.show()


# =========================================================
# 2. FRET efficiency vs distance
# =========================================================

R = np.linspace(0.1, 3, 500)
R0 = 1.0

E = 1 / (1 + (R / R0)**6)

plt.figure(figsize=(7, 4.5))
plt.plot(R, E, linewidth=2)

plt.xlabel("Distance R (normalized to R0)")
plt.ylabel("FRET efficiency E")
plt.title("FRET efficiency vs distance")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fret_efficiency_vs_distance.png", dpi=300)
plt.show()


# =========================================================
# 3. FRET rate vs distance
# =========================================================

R = np.linspace(0.2, 3, 500)

k = 1 / R**6

plt.figure(figsize=(7, 4.5))
plt.plot(R, k, linewidth=2)

plt.xlabel("Distance R (normalized units)")
plt.ylabel(r"$k_{FRET} \propto 1/R^6$")
plt.title("FRET rate dependence on distance")

plt.yscale("log")

plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("fret_rate_vs_distance.png", dpi=300)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# ratio tau_DA / tau_D
ratio = np.linspace(0, 1, 500)

# FRET efficiency
E = 1 - ratio

plt.figure(figsize=(7, 4.5))
plt.plot(ratio, E, linewidth=2)

plt.xlabel(r"Lifetime ratio $\tau_{DA} / \tau_D$")
plt.ylabel("FRET efficiency E")
plt.title(r"FRET efficiency from lifetime: $E = 1 - \tau_{DA}/\tau_D$")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fret_efficiency_lifetime_ratio.png", dpi=300)
plt.show()
