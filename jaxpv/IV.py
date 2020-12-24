from jaxpv import objects, solver, scales, current, physics, util
from jax import numpy as np, ops

PVCell = objects.PVCell
LightSource = objects.LightSource
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
    phi_eq = solver.solve_eq(cell, phi_ini)

    neq_0 = cell.Nc[0] * np.exp(cell.Chi[0] + phi_eq[0])
    neq_L = cell.Nc[-1] * np.exp(cell.Chi[-1] + phi_eq[-1])
    peq_0 = cell.Nv[0] * np.exp(-cell.Chi[0] - cell.Eg[0] - phi_eq[0])
    peq_L = cell.Nv[-1] * np.exp(-cell.Chi[-1] - cell.Eg[-1] - phi_eq[-1])
    phis = np.concatenate([np.zeros(2 * N), phi_eq], axis=0)

    jcurve = np.array([], dtype=f64)
    voltages = np.array([], dtype=f64)
    max_iter = 100
    niter = 0
    v = 0
    terminate = False

    while not terminate and niter < max_iter:

        scaled_V = v * scales.E
        print(f"Solving for V = {scaled_V}")
        
        sol = solver.solve(cell, neq_0, neq_L, peq_0, peq_L, phis)
        total_j, _ = current.total_current(cell, sol[0:N], sol[N:2 * N],
                                           sol[2 * N:])

        jcurve = np.concatenate([jcurve, np.array([total_j])])
        voltages = np.concatenate([voltages, np.array([v])])

        niter += 1
        v += Vincrement
        phis = ops.index_update(sol, -1, phi_eq[-1] + v)

        if jcurve.size > 2:
            terminate = (jcurve[-2] * jcurve[-1] <= 0)

    return jcurve, voltages
