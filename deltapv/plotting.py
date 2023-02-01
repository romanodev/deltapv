from deltapv import scales, physics, objects, spline, util
from jax import numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rcParams
from matplotlib.patches import Rectangle
import os

PVDesign = objects.PVDesign
Potentials = objects.Potentials
Array = util.Array

COLORS = ["darkorange", "yellow", "limegreen", "cyan", "indigo"]

font_dir = os.path.join(os.path.dirname(__file__), "fonts")
font_files = fm.findSystemFonts(fontpaths=font_dir)
for font_file in font_files:
    fm.fontManager.addfont(font_file)
rcParams["font.family"] = "CMU Serif"
rcParams["font.size"] = 15
rcParams["mathtext.fontset"] = "custom"
rcParams["mathtext.rm"] = "CMU Serif"
rcParams["mathtext.it"] = "CMU Serif:italic"
rcParams["mathtext.bf"] = "CMU Serif:bold"
rcParams["lines.linewidth"] = 2
rcParams["patch.linewidth"] = 2
rcParams["hatch.linewidth"] = 2
rcParams["figure.figsize"] = (8.0, 5.0)
rcParams["figure.dpi"] = 80
rcParams["savefig.dpi"] = 300


def plot_bars(design: PVDesign, filename=None) -> None:

    _, ax1 = plt.subplots()
    ax1.set_zorder(1)
    ax1.patch.set_visible(False)

    Ec = scales.energy * physics.Ec(design)
    Ev = scales.energy * physics.Ev(design)
    EF = scales.energy * physics.EF(design)
    dim_grid = scales.length * design.grid * 1e4
    if design.PhiM0 > 0 and design.PhiML > 0:
        ax1.margins(x=.2, y=.5)
    else:
        ax1.margins(y=.5)
        ax1.set_xlim(0, dim_grid[-1])

    idx = jnp.concatenate(
        [jnp.array([0]),
         jnp.argwhere(Ec[:-1] != Ec[1:]).flatten() + 1])

    uc = Ec[idx]
    uv = Ev[idx]
    startx = dim_grid[idx]
    starty = uv
    height = uc - uv
    width = jnp.diff(jnp.append(startx, dim_grid[-1]))

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

    ax1.plot(dim_grid, EF, linestyle="--", color="black", label="$E_{F}$")

    if design.PhiM0 > 0:
        phim0 = -design.PhiM0 * scales.energy
        xstart, _ = ax1.get_xlim()
        width = -xstart
        height = .2
        ystart = phim0 - height / 2
        rect = Rectangle((xstart, ystart),
                         width,
                         height,
                         color="red",
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
        ax1.axhline(y=phim0,
                    xmin=0,
                    xmax=1 / 7,
                    linestyle="--",
                    color="black",
                    linewidth=2)
        ax1.axvline(dim_grid[0], color="white", linewidth=2)
        ax1.axvline(dim_grid[0],
                    color="lightgray",
                    linewidth=2,
                    linestyle="dashed")

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
                         color="blue",
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
        ax1.axhline(y=phiml,
                    xmin=6 / 7,
                    xmax=1,
                    linestyle="--",
                    color="black",
                    linewidth=2)
        ax1.axvline(dim_grid[-1], color="white", linewidth=2)
        ax1.axvline(dim_grid[-1],
                    color="lightgray",
                    linewidth=2,
                    linestyle="dashed")

    posline = jnp.argwhere(design.Ndop[:-1] != design.Ndop[1:]).flatten()

    for idx in posline:
        vpos = (dim_grid[idx] + dim_grid[idx + 1]) / 2
        ax1.axvline(vpos, color="white", linewidth=4)
        ax1.axvline(vpos, color="lightgray", linewidth=2, linestyle="dashed")

    ax1.set_ylim(jnp.min(uv) * 1.5, 0)
    ax1.set_xlabel("position / μm")
    ax1.set_ylabel("energy / eV")
    ax1.legend()

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def plot_band_diagram(design: PVDesign,
                      pot: Potentials,
                      eq=False,
                      filename=None) -> None:

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

    plt.xlabel("position / μm")
    plt.ylabel("energy / eV")
    plt.legend()
    plt.xlim(0, x[-1])
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def plot_iv_curve(voltages: Array, currents: Array, flip=False, filename=None) -> None:

    if flip:
        voltages = -voltages
        currents = -currents
    currents = 1e3 * currents  # A / cm^2 -> mA / cm^2
    coef = spline.qspline(voltages, currents)

    a, b, c = coef
    al, bl, cl = a[-1], b[-1], c[-1]
    discr = jnp.sqrt(bl**2 - 4 * al * cl)
    x0 = (-bl - discr) / (2 * al)
    voc = jnp.clip(x0, voltages[-2], voltages[-1])

    vint = jnp.linspace(0, voc, 500)
    jint = spline.predict(vint, voltages, coef)
    idx = jnp.argmax(vint * jint)
    vmax = vint[idx]
    jmax = jint[idx]
    pmax = vmax * jmax
    p0 = jnp.sum(jnp.diff(vint) * (jint[:-1] + jint[1:]) / 2)
    FF = pmax / p0 * 100  # %

    rect = Rectangle((0, 0),
                     vmax,
                     jmax,
                     fill=False,
                     edgecolor="lightgray",
                     hatch="/",
                     linestyle="--")
    plt.text(vmax / 2,
             jmax / 2,
             f"$FF = {round(FF, 2)}\%$\n$MPP = {round(pmax * 10, 2)}$ W/m$^2$",
             ha="center",
             va="center")
    plt.gca().add_patch(rect)
    plt.plot(vint, jint, color="black")
    plt.scatter(voltages, currents, color="black", marker=".")
    plt.xlabel("bias / V")
    plt.ylabel("current density / mA/cm$^2$")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def plot_charge(design: PVDesign, pot: Potentials, filename=None):

    n = scales.density * physics.n(design, pot)
    p = scales.density * physics.p(design, pot)

    x = scales.length * design.grid * 1e4

    plt.plot(x, n, label="electron", color="lightcoral")
    plt.plot(x, p, label="hole", color="cornflowerblue")

    plt.yscale("log")
    plt.xlabel("position / μm")
    plt.ylabel("density / cm$^{-3}$")
    plt.legend()
    plt.xlim(0, x[-1])
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()
