import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 14,
    'axes.linewidth': 1.8,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 7,
    'ytick.major.size': 7,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

# Data from the table
viscosity = [0.02, 0.038, 0.2]

# 3.5 µL droplet
G1_3p5 = [0.101, 0.072, 0.02]
G2_3p5 = [0.105, 0.0715, 0.02]
G3_3p5 = [0.1, 0.07, 0.0195]

# 10 µL droplet
G1_10 = [0.0815, 0.0515, 0.01]
G2_10 = [0.08, 0.05, 0.01]
G3_10 = [0.081, 0.0495, 0.01]

fig, ax = plt.subplots(figsize=(7.5, 5.5))

ms = 10
lw = 2.2

# 3.5 µL
ax.plot(viscosity, G1_3p5, 'o-', color='#E41A1C', markersize=ms, linewidth=lw,
        markerfacecolor='#E41A1C', markeredgecolor='#E41A1C',
        label='Grid#1 (3.5 µL)')
ax.plot(viscosity, G2_3p5, 's-', color='#0000FF', markersize=ms, linewidth=lw,
        markerfacecolor='#0000FF', markeredgecolor='#0000FF',
        label='Grid#2 (3.5 µL)')
ax.plot(viscosity, G3_3p5, '>-', color='#FF00FF', markersize=ms, linewidth=lw,
        markerfacecolor='#FF00FF', markeredgecolor='#FF00FF',
        label='Grid#3 (3.5 µL)')

# 10 µL
ax.plot(viscosity, G1_10, 'D-', color='#00AA00', markersize=ms, linewidth=lw,
        markerfacecolor='#00AA00', markeredgecolor='#00AA00',
        label='Grid#1 (10 µL)')
ax.plot(viscosity, G2_10, '<-', color='#000080', markersize=ms, linewidth=lw,
        markerfacecolor='#000080', markeredgecolor='#000080',
        label='Grid#2 (10 µL)')
ax.plot(viscosity, G3_10, 'p-', color='#8B4513', markersize=ms, linewidth=lw,
        markerfacecolor='#8B4513', markeredgecolor='#8B4513',
        label='Grid#3 (10 µL)')

# Axis labels
ax.set_xlabel('Viscosity, $\\mu$ (kg/m$\\cdot$s)', fontsize=16, fontweight='bold')
ax.set_ylabel('Critical splitting velocity (m/s)', fontsize=16, fontweight='bold')

# Keep all four spines visible (boxed look)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)

# Ticks only on bottom and left — NO ticks on top and right
ax.tick_params(axis='both', which='major', labelsize=13,
               top=False, right=False, bottom=True, left=True)

# Axis limits
ax.set_xlim(0.0, 0.22)
ax.set_ylim(0.0, 0.12)

# Custom x-ticks
ax.set_xticks([0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22])

# Legend
legend = ax.legend(loc='upper right', fontsize=10.5, ncol=1,
                   borderpad=0.6, handlelength=2.5, handletextpad=0.5,
                   frameon=True, edgecolor='black', fancybox=False)
legend.get_frame().set_linewidth(1.2)

plt.tight_layout()

# Save to same directory as the script
save_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(save_dir, 'grid_independence_splitting.png'), dpi=600,
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(os.path.join(save_dir, 'grid_independence_splitting.pdf'), dpi=600,
            bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Saved to: {save_dir}")
plt.show()