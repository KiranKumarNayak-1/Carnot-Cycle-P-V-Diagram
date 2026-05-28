"""
╔══════════════════════════════════════════════════════════════════════╗
║         CARNOT CYCLE — P-V DIAGRAM VISUALIZER                       ║
║         Author  : Kiran Kumar Nayak                                  ║
║         College : PMEC, Berhampur (BPUT)                            ║
║         Branch  : Mechanical Engineering                             ║
║         Purpose : Physics / Thermodynamics Visualization             ║
╚══════════════════════════════════════════════════════════════════════╝

Description:
    Generates a publication-quality Pressure–Volume (P-V) diagram for
    the ideal Carnot cycle, consisting of:
        1. Isothermal Expansion   (A → B)  at T_H (Hot Source)
        2. Adiabatic Expansion    (B → C)
        3. Isothermal Compression (C → D)  at T_L (Cold Sink)
        4. Adiabatic Compression  (D → A)

    Also computes and displays:
        • Carnot Efficiency (η)
        • Heat Absorbed (Q_H)
        • Heat Rejected (Q_L)
        • Net Work Done (W_net)

Usage:
    python carnot_pv_diagram.py

Dependencies:
    matplotlib, numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────
# 1. THERMODYNAMIC PARAMETERS (Easily Adjustable)
# ─────────────────────────────────────────────────
T_H   = 800.0    # Hot reservoir temperature  [K]
T_L   = 300.0    # Cold reservoir temperature [K]
V_A   = 1.0      # Volume at state A          [m³] (start of isothermal expansion)
V_B   = 3.0      # Volume at state B          [m³] (end   of isothermal expansion)
n     = 1.0      # Moles of ideal gas         [mol]
R     = 8.314    # Universal gas constant     [J / mol·K]
gamma = 1.4      # Heat capacity ratio (Cp/Cv) for diatomic gas

# ─────────────────────────────────────────────────
# 2. DERIVED QUANTITIES
# ─────────────────────────────────────────────────
P_A = (n * R * T_H) / V_A               # Pressure at A
P_B = (n * R * T_H) / V_B               # Pressure at B

# Adiabatic: P·V^γ = const
# From B → C  (T_H → T_L):  V_C = V_B * (T_H / T_L)^(1/(γ-1))
V_C = V_B * (T_H / T_L) ** (1 / (gamma - 1))
P_C = (n * R * T_L) / V_C

# From D → A  (T_L → T_H):  V_D = V_A * (T_H / T_L)^(1/(γ-1))
V_D = V_A * (T_H / T_L) ** (1 / (gamma - 1))
P_D = (n * R * T_L) / V_D

# ─────────────────────────────────────────────────
# 3. PROCESS CURVES
# ─────────────────────────────────────────────────
def isothermal(V_start, V_end, T, n=n, R=R, num=400):
    """P = nRT / V"""
    V = np.linspace(V_start, V_end, num)
    P = (n * R * T) / V
    return V, P

def adiabatic(V_start, V_end, P_start, gamma=gamma, num=400):
    """P·V^γ = const → P = P_start * (V_start/V)^γ"""
    V = np.linspace(V_start, V_end, num)
    P = P_start * (V_start / V) ** gamma
    return V, P

# Process 1 — Isothermal Expansion   A → B
V1, P1 = isothermal(V_A, V_B, T_H)

# Process 2 — Adiabatic Expansion    B → C
V2, P2 = adiabatic(V_B, V_C, P_B)

# Process 3 — Isothermal Compression C → D
V3, P3 = isothermal(V_C, V_D, T_L)

# Process 4 — Adiabatic Compression  D → A
V4, P4 = adiabatic(V_D, V_A, P_D)

# ─────────────────────────────────────────────────
# 4. THERMODYNAMIC CALCULATIONS
# ─────────────────────────────────────────────────
Q_H    = n * R * T_H * np.log(V_B / V_A)     # Heat absorbed   [J]
Q_L    = n * R * T_L * np.log(V_C / V_D)     # Heat rejected   [J] (positive magnitude)
W_net  = Q_H - Q_L                            # Net work done   [J]
eta    = 1 - (T_L / T_H)                      # Carnot efficiency

# ─────────────────────────────────────────────────
# 5. FIGURE SETUP
# ─────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "serif",
    "mathtext.fontset" : "cm",
    "axes.linewidth"   : 1.4,
    "xtick.direction"  : "in",
    "ytick.direction"  : "in",
    "xtick.major.size" : 6,
    "ytick.major.size" : 6,
    "xtick.minor.size" : 3,
    "ytick.minor.size" : 3,
})

DARK_BG   = "#0D1117"
PANEL_BG  = "#161B22"
GRID_COL  = "#30363D"
TEXT_COL  = "#E6EDF3"
ACCENT    = "#58A6FF"
HOT_COL   = "#FF6B6B"
COLD_COL  = "#4ECDC4"
ADIAB_COL = "#FFD166"
FILL_COL  = "#1F6FEB"

fig = plt.figure(figsize=(16, 9), facecolor=DARK_BG)
fig.suptitle(
    "CARNOT CYCLE  ·  Pressure–Volume (P–V) Diagram",
    fontsize=20, fontweight="bold", color=TEXT_COL,
    y=0.97
)

gs = GridSpec(1, 2, figure=fig, width_ratios=[2.2, 1],
              left=0.06, right=0.97, bottom=0.10, top=0.91, wspace=0.35)

ax    = fig.add_subplot(gs[0])          # Main P-V plot
ax_info = fig.add_subplot(gs[1])       # Info / results panel

# ─────────────────────────────────────────────────
# 6. SHADED ENCLOSED AREA (Net Work)
# ─────────────────────────────────────────────────
ax.set_facecolor(PANEL_BG)

V_fill = np.concatenate([V1, V2, V3[::-1], V4[::-1]])
P_fill = np.concatenate([P1, P2, P3[::-1], P4[::-1]])
ax.fill(V_fill, P_fill, color=FILL_COL, alpha=0.18, zorder=1, label="_nolegend_")

# ─────────────────────────────────────────────────
# 7. PROCESS CURVES
# ─────────────────────────────────────────────────
lw = 2.6

ax.plot(V1, P1, color=HOT_COL,   lw=lw, zorder=3,
        label=r"(1) Isothermal Expansion  $T_H$")
ax.plot(V2, P2, color=ADIAB_COL, lw=lw, zorder=3,
        label=r"(2) Adiabatic Expansion")
ax.plot(V3, P3, color=COLD_COL,  lw=lw, zorder=3,
        label=r"(3) Isothermal Compression  $T_L$")
ax.plot(V4, P4, color=ADIAB_COL, lw=lw, zorder=3, linestyle="--",
        label=r"(4) Adiabatic Compression")

# ─────────────────────────────────────────────────
# 8. STATE POINTS  A, B, C, D
# ─────────────────────────────────────────────────
states = {
    "A": (V_A, P_A),
    "B": (V_B, P_B),
    "C": (V_C, P_C),
    "D": (V_D, P_D),
}
offsets = {
    "A": (-0.25, +0.04),
    "B": (+0.08, +0.04),
    "C": (+0.08, -0.08),
    "D": (-0.30, -0.08),
}

for label, (vx, px) in states.items():
    ax.scatter(vx, px, s=90, color=ACCENT, zorder=6, edgecolors="white", linewidths=0.8)
    dx, dy = offsets[label]
    ax.text(vx + dx, px + dy, label,
            fontsize=14, fontweight="bold", color=ACCENT,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc=PANEL_BG, ec=ACCENT, lw=0.8))

# ─────────────────────────────────────────────────
# 9. DIRECTIONAL ARROWS ALONG CURVES
# ─────────────────────────────────────────────────
def mid_arrow(ax, V_arr, P_arr, color, frac=0.5):
    idx = int(len(V_arr) * frac)
    dx  = V_arr[idx+1] - V_arr[idx-1]
    dy  = P_arr[idx+1] - P_arr[idx-1]
    ax.annotate("",
        xy    =(V_arr[idx] + dx*0.1, P_arr[idx] + dy*0.1),
        xytext=(V_arr[idx] - dx*0.1, P_arr[idx] - dy*0.1),
        arrowprops=dict(arrowstyle="->", color=color, lw=2.0),
        annotation_clip=False, zorder=7)

mid_arrow(ax, V1, P1, HOT_COL)
mid_arrow(ax, V2, P2, ADIAB_COL)
mid_arrow(ax, V3[::-1], P3[::-1], COLD_COL)
mid_arrow(ax, V4[::-1], P4[::-1], ADIAB_COL)

# ─────────────────────────────────────────────────
# 10. AXES FORMATTING
# ─────────────────────────────────────────────────
ax.set_xlabel("Volume  V  (m³)", fontsize=13, color=TEXT_COL, labelpad=8)
ax.set_ylabel("Pressure  P  (Pa)", fontsize=13, color=TEXT_COL, labelpad=8)
ax.tick_params(colors=TEXT_COL, labelsize=10)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID_COL)
ax.grid(True, color=GRID_COL, linestyle="--", linewidth=0.6, alpha=0.7)
ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
from matplotlib.ticker import AutoMinorLocator
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# Legend
leg = ax.legend(
    loc="upper right", fontsize=10,
    facecolor=PANEL_BG, edgecolor=GRID_COL,
    labelcolor=TEXT_COL, framealpha=0.95,
    borderpad=0.9, handlelength=2.5
)

# Central "W_net" label inside the loop
V_mid = (V_A + V_C) / 2
P_mid = (P_B + P_D) / 2 * 0.75
ax.text(V_mid, P_mid,
        f"$W_{{net}}$\n{W_net:,.1f} J",
        fontsize=11, color=TEXT_COL, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.5", fc=FILL_COL, alpha=0.35, ec=ACCENT, lw=0.8))

# ─────────────────────────────────────────────────
# 11. INFO PANEL (Right Side)
# ─────────────────────────────────────────────────
ax_info.set_facecolor(PANEL_BG)
ax_info.set_xlim(0, 1)
ax_info.set_ylim(0, 1)
ax_info.axis("off")
for spine in ax_info.spines.values():
    spine.set_visible(False)

# Title
ax_info.text(0.5, 0.97, "THERMODYNAMIC RESULTS",
             fontsize=11, fontweight="bold", color=ACCENT,
             ha="center", va="top", transform=ax_info.transAxes,
             fontfamily="monospace")

ax_info.plot([0, 1], [0.93, 0.93], color=ACCENT, lw=0.8)

# Parameter table
params = [
    ("Hot Temp  T_H",   f"{T_H:.0f} K"),
    ("Cold Temp  T_L",  f"{T_L:.0f} K"),
    ("",                ""),
    ("Volume A  V_A",   f"{V_A:.2f} m³"),
    ("Volume B  V_B",   f"{V_B:.2f} m³"),
    ("Volume C  V_C",   f"{V_C:.2f} m³"),
    ("Volume D  V_D",   f"{V_D:.2f} m³"),
    ("",                ""),
    ("Pressure A",      f"{P_A:,.1f} Pa"),
    ("Pressure B",      f"{P_B:,.1f} Pa"),
    ("Pressure C",      f"{P_C:,.1f} Pa"),
    ("Pressure D",      f"{P_D:,.1f} Pa"),
]

y_start = 0.88
dy      = 0.048

for key, val in params:
    if key == "":
        y_start -= dy * 0.4
        ax_info.plot([0.05, 0.95], [y_start + dy*0.2, y_start + dy*0.2], color=GRID_COL, lw=0.5)
        continue
    ax_info.text(0.05, y_start, key, fontsize=9.5, color="#8B949E",
                 transform=ax_info.transAxes, fontfamily="monospace")
    ax_info.text(0.97, y_start, val, fontsize=9.5, color=TEXT_COL,
                 transform=ax_info.transAxes, ha="right", fontfamily="monospace",
                 fontweight="bold")
    y_start -= dy

# Key Results Box
box_y = y_start - 0.02
results = [
    ("Heat Absorbed  Q_H", f"{Q_H:,.1f} J",  HOT_COL),
    ("Heat Rejected  Q_L", f"{Q_L:,.1f} J",  COLD_COL),
    ("Net Work  W_net",    f"{W_net:,.1f} J", ADIAB_COL),
    ("Efficiency  η",      f"{eta*100:.2f} %", ACCENT),
]

ax_info.text(0.5, box_y, "KEY RESULTS",
             fontsize=10, fontweight="bold", color=TEXT_COL,
             ha="center", transform=ax_info.transAxes, fontfamily="monospace")
box_y -= dy * 0.8
ax_info.plot([0.05, 0.95], [box_y + dy*0.4, box_y + dy*0.4], color=ACCENT, lw=0.6)

for label, val, col in results:
    box_y -= dy
    ax_info.text(0.05, box_y, label, fontsize=9.5, color="#8B949E",
                 transform=ax_info.transAxes, fontfamily="monospace")
    ax_info.text(0.97, box_y, val, fontsize=10.5, color=col,
                 transform=ax_info.transAxes, ha="right", fontfamily="monospace",
                 fontweight="bold")

# Efficiency formula
box_y -= dy * 1.6
ax_info.text(0.5, box_y,
             r"$\eta = 1 - \dfrac{T_L}{T_H}$",
             fontsize=14, color=ACCENT, ha="center",
             transform=ax_info.transAxes)

box_y -= dy * 1.5
ax_info.text(0.5, box_y,
             f"= 1 − ({T_L:.0f}/{T_H:.0f})\n= {eta:.4f}  ≈  {eta*100:.2f}%",
             fontsize=9.5, color=TEXT_COL, ha="center",
             transform=ax_info.transAxes, fontfamily="monospace",
             linespacing=1.7)

# Footer
ax_info.plot([0, 1], [0.04, 0.04], color=GRID_COL, lw=0.8)
ax_info.text(0.5, 0.02,
             "Kiran Kumar Nayak  ·  PMEC Berhampur  ·  Mech Engg",
             fontsize=7.5, color="#484F58", ha="center",
             transform=ax_info.transAxes, style="italic")

# ─────────────────────────────────────────────────
# 12. SAVE & SHOW
# ─────────────────────────────────────────────────
output_path = "carnot_pv_diagram.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight",
            facecolor=DARK_BG, edgecolor="none")
print(f"\n✅  Saved: {output_path}")
print(f"\n{'═'*52}")
print(f"  CARNOT CYCLE RESULTS")
print(f"{'═'*52}")
print(f"  Hot Reservoir Temp    T_H  = {T_H:.0f} K")
print(f"  Cold Reservoir Temp   T_L  = {T_L:.0f} K")
print(f"  Heat Absorbed         Q_H  = {Q_H:,.2f} J")
print(f"  Heat Rejected         Q_L  = {Q_L:,.2f} J")
print(f"  Net Work Done         W    = {W_net:,.2f} J")
print(f"  Carnot Efficiency     η    = {eta:.4f}  ({eta*100:.2f}%)")
print(f"{'═'*52}\n")

plt.show()
