import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# === User-defined parameters ===
I_ref = 5.0    # Reference drain current per JFET in mA
e_ref = 0.56   # Measured total noise for N_ref parallel JFETs at I_ref (nV/√Hz)
N_ref = 4      # Number of JFETs in reference measurement

# Compute equivalent single-JFET reference noise
e_single_ref = e_ref * np.sqrt(N_ref)

# === Range of drain currents for simulation ===
ID = np.logspace(np.log10(0.5 * I_ref), np.log10(2 * I_ref), 200)

# === List of parallel counts to compare ===
N_values = [1, 2, 4, 8, 10, 16]

# === Plot setup ===
fig, ax = plt.subplots()
for N in N_values:
    e_total = (e_single_ref * (I_ref / ID) ** 0.25) / np.sqrt(N)
    ax.loglog(ID, e_total, label=f'N={N}')

# Highlight the reference measurement point
ax.scatter([I_ref], [e_ref], marker='o', label=f'Measured (N={N_ref})')

# Set x-axis limits to match the calculation range
ax.set_xlim(ID.min(), ID.max())

# Compute ticks at min, reference, and max currents
xticks = [ID.min(), I_ref, ID.max()]
ax.set_xticks(xticks)
ax.set_xticklabels([f'{val:g}' for val in xticks])

# Format axis
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.ticklabel_format(axis='x', style='plain')

ax.set_xlabel('Drain current per JFET (mA)')
ax.set_ylabel('Total noise density (nV/√Hz)')
ax.set_title('Noise vs Drain Current for Various Parallel Counts')
ax.grid(True, which='both', linestyle='--')
ax.legend()
fig.tight_layout()
plt.show()
