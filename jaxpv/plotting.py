from jaxpv import scales, objects
from jax import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

PVDesign = objects.PVDesign
Potentials = objects.Potentials


def plot_bars(design: PVDesign):

    Ec = -scales.energy * design.Chi
    Ev = -scales.energy * (design.Chi + design.Eg)

    _, idx = np.unique(Ec, return_index=True)
    idx = np.sort(idx)

    uc = Ec[idx]
    uv = Ev[idx]
    startx = scales.length * design.grid[idx]
    starty = uv
    height = uc - uv
    width = np.diff(np.append(startx, scales.length * design.grid[-1]))

    for i in range(startx.size):
        x, y, w, h = startx[i], starty[i], width[i], height[i]
        rect = Rectangle((x, y),
                         w,
                         h,
                         facecolor="cornflowerblue",
                         edgecolor="white",
                         linewidth=10)
        plt.text(x + w / 2,
                 y + h + .1,
                 round(y + h, 2),
                 ha="center",
                 va="center")
        plt.text(x + w / 2, y - .1, round(y, 2), ha="center", va="center")
        plt.gca().add_patch(rect)

    plt.xlim(0, scales.length * design.grid[-1])
    plt.ylim(np.min(uv) * 1.2, 0)

    plt.xlabel("position / cm")
    plt.ylabel("energy / eV")

    plt.show()


def plot_band_diagram(design: PVDesign, pot: Potentials, eq=False):

    Ec = -scales.energy * (design.Chi + pot.phi)
    Ev = -scales.energy * (design.Chi + design.Eg + pot.phi)
    x = scales.length * design.grid

    plt.plot(x, Ec, color="red", label="conduction band", linestyle="dashed")
    plt.plot(x, Ev, color="blue", label="valence band", linestyle="dashed")

    if not eq:
        plt.plot(x, scales.energy * pot.phi_n, color="red", label="e- quasi-Fermi energy")
        plt.plot(x, scales.energy * pot.phi_p, color="blue", label="hole quasi-Fermi energy")
    else:
        plt.plot(x, scales.energy * pot.phi_p, color="grey", label="Fermi level")

    plt.xlabel("position / cm")
    plt.ylabel("energy / eV")
    plt.legend()
    plt.show()
