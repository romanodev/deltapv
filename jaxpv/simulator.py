from jaxpv import objects, scales, optical, sun, iv, materials, solver, bcond, util
from jax import numpy as np, ops, lax, vmap
from typing import Callable, Tuple
import matplotlib.pyplot as plt

PVCell = objects.PVCell
PVDesign = objects.PVDesign
LightSource = objects.LightSource
Potentials = objects.Potentials
Array = util.Array
f64 = util.f64


def create_design(dim_grid: Array) -> PVDesign:

    n = dim_grid.size
    grid = dim_grid / scales.units["grid"]
    init_params = {key: np.zeros(n) for key in PVDesign.__dataclass_fields__}
    init_params.update({"grid": grid})
    init_params.update({key: f64(0) for key in {"Snl", "Snr", "Spl", "Spr"}})

    return PVDesign(**init_params)


def add_material(cell: PVDesign, mat: materials.Material,
                 f: Callable[[f64], bool]) -> PVDesign:

    vf = vmap(f)
    idx = vf(cell.grid * scales.units["grid"])

    return objects.update(
        cell, **{
            param: np.where(idx, value / scales.units[param],
                            getattr(cell, param))
            for param, value in mat
        })


def contacts(cell: PVDesign, Snl: f64, Snr: f64, Spl: f64,
             Spr: f64) -> PVDesign:

    return objects.update(cell,
                          Snl=f64(Snl / scales.units["Snl"]),
                          Snr=f64(Snr / scales.units["Snr"]),
                          Spl=f64(Spl / scales.units["Spl"]),
                          Spr=f64(Spr / scales.units["Spr"]))


def single_pn_junction(cell: PVDesign, Nleft: f64, Nright: f64,
                       x: f64) -> PVDesign:

    idx = cell.grid * scales.units["grid"] <= x
    doping = np.where(idx, Nleft, Nright) / scales.units["Ndop"]

    return objects.update(cell, Ndop=doping)


def doping(cell: PVDesign, N: f64, f: Callable[[f64], bool]) -> PVDesign:

    vf = vmap(f)
    idx = vf(cell.grid * scales.units["grid"])
    doping = np.where(idx, N / scales.units["Ndop"], cell.Ndop)

    return objects.update(cell, Ndop=doping)


def incident_light(kind: str = "sun",
                   Lambda: Array = None,
                   P_in: Array = None) -> LightSource:

    if kind == "sun":
        return LightSource(Lambda=sun.wavelength, P_in=sun.power)

    if kind == "white":
        return LightSource(Lambda=np.linspace(4e2, 8e2, 5),
                           P_in=2e2 * np.ones(5, dtype=np.float64))

    if kind == "monochromatic":
        if Lambda is None:
            Lambda = np.array([4e2])
        return LightSource(Lambda=Lambda, P_in=np.array([1e3]))

    if kind == "user":
        assert Lambda is not None
        assert P_in is not None
        P_in = 1e3 * P_in / np.sum(P_in)
        return LightSource(Lambda=Lambda, P_in=P_in)


def init_cell(design: PVDesign, ls: LightSource) -> PVCell:

    G = optical.compute_G(design, ls)
    dgrid = np.diff(design.grid)
    params = design.__dict__.copy()
    params["dgrid"] = dgrid
    params.pop("grid")
    params["G"] = G

    return PVCell(**params)


def iv_curve(design: PVDesign, ls: LightSource) -> Tuple[Array, Array]:

    cell = init_cell(design, ls)
    currents, voltages = iv.calc_iv(cell)
    dim_currents = scales.J * currents
    dim_voltages = scales.E * voltages

    return dim_voltages, dim_currents


def efficiency(design: PVDesign, ls: LightSource) -> f64:

    cell = init_cell(design, ls)
    currents, voltages = iv.calc_iv(cell)
    pmax = np.max(scales.E * voltages * scales.J * currents) * 1e4  # W/m2
    eff = pmax / 1e3

    return eff
