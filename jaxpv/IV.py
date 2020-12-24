from jaxpv import objects, solver, scales, current, physics, util
from jax import numpy as np, ops

PVCell = objects.PVCell
LightSource = objects.LightSource
Potentials = objects.Potentials
Boundary = objects.Boundary
Array = util.Array
f64 = util.f64


def Vincrement(cell: PVCell, num_vals: int = 50) -> f64:

    phi_ini_left = util.switch(
        cell.Ndop[0] > 0, -cell.Chi[0] + np.log(cell.Ndop[0] / cell.Nc[0]),
        -cell.Chi[0] - cell.Eg[0] - np.log(np.abs(cell.Ndop[0]) / cell.Nv[0]))

    phi_ini_right = util.switch(
        cell.Ndop[-1] > 0, -cell.Chi[-1] + np.log(cell.Ndop[-1] / cell.Nc[-1]),
        -cell.Chi[-1] - cell.Eg[-1] -
        np.log(np.abs(cell.Ndop[-1]) / cell.Nv[-1]))

    incr_step = np.abs(phi_ini_right - phi_ini_left) / num_vals
    incr_sign = (-1)**(phi_ini_right > phi_ini_left)

    return incr_sign * incr_step


def eq_init_phi(cell: PVCell) -> Array:

    phi_ini_left = util.switch(
        cell.Ndop[0] > 0, -cell.Chi[0] + np.log(cell.Ndop[0] / cell.Nc[0]),
        -cell.Chi[0] - cell.Eg[0] - np.log(-cell.Ndop[0] / cell.Nv[0]))

    phi_ini_right = util.switch(
        cell.Ndop[-1] > 0, -cell.Chi[-1] + np.log(cell.Ndop[-1] / cell.Nc[-1]),
        -cell.Chi[-1] - cell.Eg[-1] - np.log(-cell.Ndop[-1] / cell.Nv[-1]))

    return np.linspace(phi_ini_left, phi_ini_right, cell.grid.size)


def calc_IV(cell: PVCell, Vincrement: f64) -> Array:

    N = cell.grid.size

    phi_ini = eq_init_phi(cell)
    pot_ini = Potentials(phi_ini, np.zeros(N), np.zeros(N))
    pot_eq = solver.solve_eq(cell, pot_ini)

    neq0 = cell.Nc[0] * np.exp(cell.Chi[0] + pot_eq.phi[0])
    neqL = cell.Nc[-1] * np.exp(cell.Chi[-1] + pot_eq.phi[-1])
    peq0 = cell.Nv[0] * np.exp(-cell.Chi[0] - cell.Eg[0] - pot_eq.phi[0])
    peqL = cell.Nv[-1] * np.exp(-cell.Chi[-1] - cell.Eg[-1] - pot_eq.phi[-1])
    bound = Boundary(neq0, neqL, peq0, peqL)

    pot = pot_eq
    jcurve = np.array([], dtype=f64)
    voltages = np.array([], dtype=f64)
    v = 0
    max_iter = 100
    niter = 0
    terminate = False

    while not terminate and niter < max_iter:

        scaled_V = v * scales.E
        print(f"Solving for V = {scaled_V}")

        pot_sol = solver.solve(cell, bound, pot)
        total_j, _ = current.total_current(cell, pot_sol)

        jcurve = np.append(jcurve, total_j)
        voltages = np.append(voltages, v)

        niter += 1
        v += Vincrement
        pot = Potentials(
            ops.index_update(pot_sol.phi, ops.index[-1], pot_eq.phi[-1] + v),
            pot_sol.phi_n, pot_sol.phi_p)

        if jcurve.size > 2:
            terminate = (jcurve[-2] * jcurve[-1]) <= 0

    return jcurve, voltages
