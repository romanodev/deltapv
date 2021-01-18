from jaxpv import scales, objects, util
from jax import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

PVDesign = objects.PVDesign
Potentials = objects.Potentials
Array = util.Array

COLORS = ["darkorange", "yellow", "limegreen", "cyan", "indigo"]


def plot_bars(design: PVDesign) -> None:

    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_zorder(1)
    ax1.patch.set_visible(False)

    Ec = -scales.energy * design.Chi
    Ev = -scales.energy * (design.Chi + design.Eg)
    dim_grid = scales.length * design.grid * 1e4

    idx = np.concatenate(
        [np.array([0]),
         np.argwhere(Ec[:-1] != Ec[1:]).flatten() + 1])

    uc = Ec[idx]
    uv = Ev[idx]
    startx = dim_grid[idx]
    starty = uv
    height = uc - uv
    width = np.diff(np.append(startx, dim_grid[-1]))

    for i in range(startx.size):
        x, y, w, h = startx[i], starty[i], width[i], height[i]
        rect = Rectangle((x, y),
                         w,
                         h,
                         color=COLORS[i % len(COLORS)],
                         linewidth=0, alpha=.2)
        ax1.add_patch(rect)
        ax1.text(x + w / 2,
                 y + h + .1,
                 round(y + h, 2),
                 ha="center",
                 va="bottom")
        ax1.text(x + w / 2, y - .1, round(y, 2), ha="center", va="top")

    ax1.set_xlim(0, dim_grid[-1])
    ax1.set_ylim(np.min(uv) * 1.2, 0)
    ax1.set_xlabel("position / um")
    ax1.set_ylabel("energy / eV")

    ax2.set_yscale("log")
    dim_Ndop = scales.density * design.Ndop

    ax2.plot(dim_grid,
             dim_Ndop,
             label="donor",
             color="lightcoral")

    ax2.plot(dim_grid,
             -dim_Ndop,
             label="acceptor",
             color="cornflowerblue")

    posline = np.argwhere(dim_Ndop[:-1] != dim_Ndop[1:]).flatten()

    for x in posline:
        ax2.axvline(dim_grid[x], color="white", linewidth=2)
        ax2.axvline(dim_grid[x + 1], color="white", linewidth=2)
        ax2.axvline(dim_grid[x], color="lightgray", linewidth=1, linestyle="dashed")

    ax2.margins(y=.5)
    ax2.legend()
    ax2.set_ylabel("doping / cm^(-3)")

    plt.show()


def plot_band_diagram(design: PVDesign, pot: Potentials, eq=False) -> None:

    Ec = -scales.energy * (design.Chi + pot.phi)
    Ev = -scales.energy * (design.Chi + design.Eg + pot.phi)
    x = scales.length * design.grid * 1e4

    plt.plot(x,
             Ec,
             color="lightcoral",
             label="conduction band",
             linestyle="dashed")
    plt.plot(x,
             Ev,
             color="cornflowerblue",
             label="valence band",
             linestyle="dashed")

    if not eq:
        plt.plot(x,
                 scales.energy * pot.phi_n,
                 color="lightcoral",
                 label="e- quasi-Fermi energy")
        plt.plot(x,
                 scales.energy * pot.phi_p,
                 color="cornflowerblue",
                 label="hole quasi-Fermi energy")
    else:
        plt.plot(x,
                 scales.energy * pot.phi_p,
                 color="lightgray",
                 label="Fermi level")

    plt.xlabel("position / um")
    plt.ylabel("energy / eV")
    plt.legend()
    plt.show()


def plot_iv_curve(voltages: Array, currents: Array) -> None:

    plt.plot(voltages, currents)
    plt.xlabel("bias / V")
    plt.ylabel("current density / A/cm^2")
    plt.ylim(bottom=0)
    plt.show()
