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
    """Create an empty PVDesign

    Args:
        dim_grid (Array): Discretized coordinates in cm

    Returns:
        PVDesign: An empty PVDesign
    """
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
    """Add a material to a cell on a defined region

    Args:
        cell (PVDesign): A cell
        mat (materials.Material): Material to be added
        f (Callable[[f64], bool]): Function that returns true when material should be added to the input position

    Returns:
        PVDesign: Cell with material added
    """
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
    """Define contact recombination velocities and workfunctions

    Args:
        cell (PVDesign): A cell
        Snl (f64): Electron recombination velocity at front contact
        Snr (f64): Electron recombination velocity at back contact
        Spl (f64): Hole recombination velocity at front contact
        Spr (f64): Hole recombination velocity at back contact
        PhiM0 (f64, optional): Workfunction of front contact. Defaults to -1.
        PhiML (f64, optional): Workfunction of back contact. Defaults to -1.

    Returns:
        PVDesign: Cell with recombination velocities and workfunctions defined
    """
    return objects.update(cell,
                          Snl=f64(Snl / scales.units["Snl"]),
                          Snr=f64(Snr / scales.units["Snr"]),
                          Spl=f64(Spl / scales.units["Spl"]),
                          Spr=f64(Spr / scales.units["Spr"]),
                          PhiM0=f64(PhiM0 / scales.units["PhiM0"]),
                          PhiML=f64(PhiML / scales.units["PhiML"]))


def single_pn_junction(cell: PVDesign, Nleft: f64, Nright: f64,
                       x: f64) -> PVDesign:
    """Helper function for quickly defining a pn junction

    Args:
        cell (PVDesign): A cell
        Nleft (f64): Doping concentration in 1/cm^3 on the left of the junction. Positive and negative values refer to donor and acceptor concentrations respectively
        Nright (f64): Doping concentration in 1/cm^3 on the right of the junction
        x (f64): Position of junction from front of cell, in cm

    Returns:
        PVDesign: Cell with pn junction defined
    """
    idx = cell.grid * scales.units["grid"] <= x
    doping = np.where(idx, Nleft, Nright) / scales.units["Ndop"]

    return objects.update(cell, Ndop=doping)


def doping(cell: PVDesign, N: f64, f: Callable[[f64], bool]) -> PVDesign:
    """Define a doping profile on a region of a cell

    Args:
        cell (PVDesign): A cell
        N (f64): Doping concentration in 1/cm^3
        f (Callable[[f64], bool]): Function that returns true when material should be doped at the input position

    Returns:
        PVDesign: Doped cell
    """
    vf = vmap(f)
    idx = vf(cell.grid * scales.units["grid"])
    doping = np.where(idx, N / scales.units["Ndop"], cell.Ndop)

    return objects.update(cell, Ndop=doping)


def incident_light(kind: str = "sun",
                   Lambda: Array = None,
                   P_in: Array = None) -> LightSource:
    """Define a light source object for simulation

    Args:
        kind (str, optional): One of "sun", "white", "monochromatic", and "user". Defaults to "sun".
        Lambda (Array, optional): Spectrum wavelengths in nm for "monochromatic" and "user" cases. Defaults to None.
        P_in (Array, optional): Power for specified wavelengths for "monochromatic" and "user" cases. Defaults to None.

    Returns:
        LightSource: A light source object for simulation
    """
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
    """Initialize a cell by calculating generation density with optical model

    Args:
        design (PVDesign): A cell
        ls (LightSource): A light source
        optics (bool, optional): Whether to use optical model to calculate the absorption coefficients. If False, model uses input absorption coefficients as specified in the PVDesign object to calculate generation density. Defaults to True.

    Returns:
        PVCell: An initialized cell ready for simulation
    """
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
    """Solve equilibrium system for a cell

    Args:
        design (PVDesign): A cell
        ls (LightSource): A light source

    Returns:
        Potentials: Equilibrium potential and quasi-Fermi energies
    """
    cell = init_cell(design, ls)
    logger.info("Solving equilibrium...")
    bound_eq = bcond.boundary_eq(cell)
    pot_ini = solver.eq_guess(cell, bound_eq)
    pot = solver.solve_eq(cell, bound_eq, pot_ini)

    return pot


def simulate(design: PVDesign, ls: LightSource, optics: bool = True, n_steps: i64 = None) -> dict:
    """Solve equilibrium and out-of-equilibrium systems for a cell.

    Args:
        design (PVDesign): A cell
        ls (LightSource): A light source
        optics (bool, optional): Whether to use optical model to calculate the absorption coefficients. If False, model uses input absorption coefficients as specified in the PVDesign object to calculate generation density. Defaults to True.
        n_steps (i64, optional): How many voltage steps to solve for. May be useful when an IV curve of a specific range is needed, but unnecessary in other cases. Defaults to None.

    Returns:
        dict: Dictionary of results: "cell" is the initialized cell, "eq" is the equilibrium solution, "Voc" is the final solution beyond the open circuit voltage, "mpp" is the maximum power found in W, "eff" is the power conversion efficiency, "iv" is a tuple (v, i) of the IV curve
    """
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

        if n_steps is not None:
            if vstep == n_steps:
                break

        if currents.size > 2 and n_steps is None:
            ll, l = currents[-2], currents[-1]
            if (ll * l <= 0) or l < 0:
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
