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
    ax2.margins(x=.2, y=.5)
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
                         linewidth=0,
                         alpha=.2)
        ax1.add_patch(rect)
        ax1.text(x + w / 2,
                 y + h + .1,
                 round(y + h, 2),
                 ha="center",
                 va="bottom")
        ax1.text(x + w / 2, y - .1, round(y, 2), ha="center", va="top")

    ax1.set_ylim(np.min(uv) * 1.2, 0)
    ax1.set_xlabel("position / $\mu m$")
    ax1.set_ylabel("energy / $eV$")

    ax2.set_yscale("log")
    dim_Ndop = scales.density * design.Ndop

    ax2.plot(dim_grid, dim_Ndop, label="donor", color="lightcoral")

    ax2.plot(dim_grid, -dim_Ndop, label="acceptor", color="cornflowerblue")

    posline = np.argwhere(dim_Ndop[:-1] != dim_Ndop[1:]).flatten()

    for idx in posline:
        ax2.axvline(dim_grid[idx], color="white", linewidth=2)
        ax2.axvline(dim_grid[idx + 1], color="white", linewidth=2)
        ax2.axvline(dim_grid[idx],
                    color="lightgray",
                    linewidth=1,
                    linestyle="dashed")

    for idx in [0, -1]:
        ax2.axvline(dim_grid[idx], color="white", linewidth=2)
        ax2.axvline(dim_grid[idx],
                    color="lightgray",
                    linewidth=1,
                    linestyle="dashed")

    if design.PhiM0 > 0:
        phim0 = -design.PhiM0 * scales.energy
        xstart, _ = ax1.get_xlim()
        width = -xstart
        height = .2
        ystart = phim0 - height / 2
        rect = Rectangle((xstart, ystart),
                         width,
                         height,
                         color="black",
                         linewidth=0,
                         alpha=.2)
        ax1.add_patch(rect)
        ax1.text(xstart + width / 2,
                 ystart + height + 0.1,
                 round(phim0, 2),
                 ha="center",
                 va="bottom")
        ax1.text(xstart + width / 2,
                 ystart - 0.1,
                 "contact",
                 ha="center",
                 va="top")

    if design.PhiML > 0:
        phiml = -design.PhiML * scales.energy
        xstart = dim_grid[-1]
        _, xend = ax1.get_xlim()
        width = xend - xstart
        height = .2
        ystart = phiml - height / 2
        rect = Rectangle((xstart, ystart),
                         width,
                         height,
                         color="black",
                         linewidth=0,
                         alpha=.2)
        ax1.add_patch(rect)
        ax1.text(xstart + width / 2,
                 ystart + height + 0.1,
                 round(phiml, 2),
                 ha="center",
                 va="bottom")
        ax1.text(xstart + width / 2,
                 ystart - 0.1,
                 "contact",
                 ha="center",
                 va="top")

    ax2.legend()
    ax2.set_ylabel("doping / $cm^{-3}$")

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

    plt.xlabel("position / $\mu m$")
    plt.ylabel("energy / $eV$")
    plt.legend()
    plt.show()


def plot_iv_curve(voltages: Array, currents: Array) -> None:

    m = (currents[-1] - currents[-2]) / (voltages[-1] - voltages[-2])
    voc = voltages[-1] + currents[-1] / m
    vint = np.linspace(0, voc, 500)
    jint = np.interp(vint, voltages, currents)
    idx = np.argmax(vint * jint)
    vmax = vint[idx]
    jmax = jint[idx]
    pmax = vmax * jmax
    p0 = np.sum(np.diff(vint) * (jint[:-1] + jint[1:]) / 2)
    FF = pmax / p0 * 100

    rect = Rectangle((0, 0),
                     vmax,
                     1e3 * jmax,
                     fill=False,
                     edgecolor="lightgray",
                     hatch="/",
                     linestyle="--")
    plt.text(vmax / 2,
             1e3 * jmax / 2,
             f"$FF = {round(FF, 2)}\%$\n$MPP = {round(pmax * 1e4, 2)} W / m^2$",
             ha="center",
             va="center")
    plt.gca().add_patch(rect)
    plt.plot(vint, 1e3 * jint, color="black")
    plt.xlabel("bias / $V$")
    plt.ylabel("current density / $mA/cm^2$")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.show()
