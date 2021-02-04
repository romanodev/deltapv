from deltapv import objects, scales, optical, sun, materials, solver, bcond, current, spline, util, adjoint, plotting
from jax import numpy as np, ops, lax, vmap
from typing import Callable, Tuple
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger("deltapv")

PVCell = objects.PVCell
PVDesign = objects.PVDesign
LightSource = objects.LightSource
Potentials = objects.Potentials
Array = util.Array
f64 = util.f64
i64 = util.i64
DIM_V_INIT = 0.01


def create_design(dim_grid: Array) -> PVDesign:

    n = dim_grid.size
    grid = dim_grid / scales.units["grid"]
    init_params = {
        key: np.zeros(n)
        for key in {
            "eps", "Chi", "Eg", "Nc", "Nv", "mn", "mp", "tn", "tp", "Et", "Br",
            "Cn", "Cp", "A", "Ndop"
        }
    }
    init_params.update({
        key: f64(0)
        for key in {"Snl", "Snr", "Spl", "Spr", "PhiM0", "PhiML"}
    })
    init_params.update({"grid": grid, "alpha": np.zeros((100, n))})

    return PVDesign(**init_params)


def add_material(cell: PVDesign, mat: materials.Material,
                 f: Callable[[f64], bool]) -> PVDesign:

    vf = vmap(f)
    idx = vf(cell.grid * scales.units["grid"])

    updatedict = {}
    for param, value in mat:
        if param == "alpha":
            updatedict[param] = np.where(
                idx,
                value.reshape(-1, 1) / scales.units[param],
                getattr(cell, param))
        else:
            updatedict[param] = np.where(idx, value / scales.units[param],
                                         getattr(cell, param))

    return objects.update(cell, **updatedict)


def contacts(cell: PVDesign,
             Snl: f64,
             Snr: f64,
             Spl: f64,
             Spr: f64,
             PhiM0: f64 = -1,
             PhiML: f64 = -1) -> PVDesign:

    return objects.update(cell,
                          Snl=f64(Snl / scales.units["Snl"]),
                          Snr=f64(Snr / scales.units["Snr"]),
                          Spl=f64(Spl / scales.units["Spl"]),
                          Spr=f64(Spr / scales.units["Spr"]),
                          PhiM0=f64(PhiM0 / scales.units["PhiM0"]),
                          PhiML=f64(PhiML / scales.units["PhiML"]))


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
        return LightSource(Lambda=sun.Lambda_eff, P_in=sun.P_in_eff)

    if kind == "white":
        return LightSource(Lambda=np.linspace(4e2, 8e2, 100),
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


def init_cell(design: PVDesign,
              ls: LightSource,
              optics: bool = True) -> PVCell:

    G = optical.compute_G(design, ls, optics=optics)
    dgrid = np.diff(design.grid)
    params = design.__dict__.copy()
    params["dgrid"] = dgrid
    params.pop("grid")
    params.pop("A")
    params.pop("alpha")
    params["G"] = G

    return PVCell(**params)


def equilibrium(design: PVDesign, ls: LightSource) -> Potentials:

    cell = init_cell(design, ls)
    logger.info("Solving equilibrium...")
    bound_eq = bcond.boundary_eq(cell)
    pot_ini = solver.eq_guess(cell, bound_eq)
    pot = solver.solve_eq(cell, bound_eq, pot_ini)

    return pot


def simulate(design: PVDesign, ls: LightSource, optics: bool = True) -> Array:

    pot_eq = equilibrium(design, ls)

    cell = init_cell(design, ls, optics=optics)
    currents = np.array([], dtype=f64)
    voltages = np.array([], dtype=f64)
    dv = solver.vincr(cell)
    vstep = 0

    while vstep < 100:

        v = dv * vstep
        scaled_v = v * scales.energy
        logger.info(f"Solving for {scaled_v} V (Step {vstep})...")

        if vstep == 0:
            # Just use a rough guess from equilibrium
            guess = solver.ooe_guess(cell, pot_eq)
            total_j, pot = adjoint.solve_pdd(cell, v, guess)
        elif vstep == 1:
            # Solve for a voltage close to zero for linear guess
            potl = pot
            logger.info(f"Solving for {DIM_V_INIT} V for convergence...")
            vinit = DIM_V_INIT / scales.energy
            _, potinit = adjoint.solve_pdd(cell, vinit, pot)
            # Generate linear guess
            logger.info(f"Continuing...")
            guess = solver.genlinguess(potinit, pot, vinit, dv - vinit)
            total_j, pot = adjoint.solve_pdd(cell, v, guess)
        elif vstep == 2:
            # Generate linear guess from first two steps
            potll = potl
            guess = solver.linguess(pot, potl)
            total_j, new = adjoint.solve_pdd(cell, v, guess)
            potl = pot
            pot = new
        else:
            # Generate quadratic guess from last three steps
            guess = solver.quadguess(pot, potl, potll)
            total_j, new = adjoint.solve_pdd(cell, v, guess)
            potll = potl
            potl = pot
            pot = new

        currents = np.append(currents, total_j)
        voltages = np.append(voltages, dv * vstep)
        vstep += 1

        if currents.size > 2:
            if (currents[-2] * currents[-1]) <= 0:
                break
    
    dim_currents = scales.current * currents
    dim_voltages = scales.energy * voltages

    pmax = spline.calcPmax(dim_voltages, dim_currents * 1e4)  # A/cm^2 -> A/m2

    eff = pmax / np.sum(ls.P_in)
    eff_print = np.round(eff * 100, 2)

    logger.info(f"Finished simulation with efficiency {eff_print}%")

    results = {
        "cell": cell,
        "eq": pot_eq,
        "Voc": pot,
        "mpp": pmax,
        "eff": eff,
        "iv": (dim_voltages, dim_currents)
    }

    return results
