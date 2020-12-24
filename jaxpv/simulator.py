from jaxpv import objects, scales, optical, sun, IV, materials, solver, util
from jax import numpy as np, ops, lax, vmap
from typing import Callable, Tuple
import matplotlib.pyplot as plt

PVCell = objects.PVCell
LightSource = objects.LightSource
Array = util.Array
f64 = util.f64


def create_cell(dim_grid: Array) -> PVCell:

    n = dim_grid.size
    grid = dim_grid / scales.units["grid"]
    dgrid = np.diff(grid)
    init_params = {key: np.zeros(n) for key in PVCell.__dataclass_fields__}
    init_params.update({"grid": grid, "dgrid": dgrid})
    init_params.update({key: f64(0) for key in {"Snl", "Snr", "Spl", "Spr"}})
    init_params.update(G=-np.ones(n))

    return PVCell(**init_params)


def update(cell: PVCell, **kwargs) -> PVCell:

    return PVCell(
        **{
            key: kwargs[key] if key in kwargs else value
            for key, value in cell.__dict__.items()
        })


def add_material(cell: PVCell, mat: materials.Material,
                 f: Callable[[f64], bool]) -> PVCell:

    vf = vmap(f)
    idx = vf(cell.grid * scales.units["grid"])

    return update(
        cell, **{
            param: np.where(idx, value / scales.units[param],
                            getattr(cell, param))
            for param, value in mat
        })


def contacts(cell: PVCell, Snl: f64, Snr: f64, Spl: f64, Spr: f64) -> PVCell:

    return update(cell,
                  Snl=f64(Snl / scales.units["Snl"]),
                  Snr=f64(Snr / scales.units["Snr"]),
                  Spl=f64(Spl / scales.units["Spl"]),
                  Spr=f64(Spr / scales.units["Spr"]))


def single_pn_junction(cell: PVCell, Nleft: f64, Nright: f64,
                       x: f64) -> PVCell:

    idx = cell.grid * scales.units["grid"] <= x
    doping = np.where(idx, Nleft, Nright) / scales.units["Ndop"]

    return update(cell, Ndop=doping)


def doping(cell: PVCell, N: f64, f: Callable[[f64], bool]) -> PVCell:

    vf = vmap(f)
    idx = vf(cell.grid * scales.units["grid"])
    doping = np.where(idx, N / scales.units["Ndop"], cell.Ndop)

    return update(cell, Ndop=doping)


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


def custom_generation(cell: PVCell, dim_G: Array) -> PVCell:

    G = dim_G / scales.units["G"]

    return update(cell, G=G)


def get_generation(cell: PVCell, ls: LightSource):

    G = util.switch(np.any(cell.G < 0), optical.compute_G(cell, ls), cell.G)

    return update(cell, G=G)


def IV_curve(
    cell: PVCell, ls: LightSource = LightSource()) -> Tuple[Array, Array]:

    cell = get_generation(cell, ls)
    Vincr = IV.Vincrement(cell)
    currents, voltages = IV.calc_IV(cell, Vincr)
    dim_currents = scales.J * currents
    dim_voltages = scales.E * voltages

    return dim_voltages, dim_currents


def efficiency(cell: PVCell, ls: LightSource = LightSource()) -> f64:

    cell = get_generation(cell, ls)
    Vincr = IV.Vincrement(cell)
    currents, voltages = IV.calc_IV(cell, Vincrement)
    Pmax = np.max(scales.E * voltages * scales.J * currents) * 1e4  # W/m2
    eff = Pmax / 1e3

    return eff


def solve_equilibrium(cell: PVCell, ls: LightSource = LightSource()) -> Array:

    cell = get_generation(cell, ls)
    Vincr = IV.Vincrement(cell)
    phi_ini = IV.eq_init_phi(cell)
    phi_eq = solver.solve_eq(cell, phi_ini)

    return phi_eq
