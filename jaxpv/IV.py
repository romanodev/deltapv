from jaxpv import objects, initial_guess, solver, scales, current, util
from jax import numpy as np, ops

PVCell = objects.PVCell
LightSource = objects.LightSource
Array = util.Array
f64 = util.f64


def calc_IV(cell: PVCell, Vincrement: f64) -> Array:

    N = cell.grid.size

    phi_ini = initial_guess.eq_init_phi(cell)
    phi_eq = solver.solve_eq(cell, phi_ini)

    neq_0 = cell.Nc[0] * np.exp(cell.Chi[0] + phi_eq[0])
    neq_L = cell.Nc[-1] * np.exp(cell.Chi[-1] + phi_eq[-1])
    peq_0 = cell.Nv[0] * np.exp(-cell.Chi[0] - cell.Eg[0] - phi_eq[0])
    peq_L = cell.Nv[-1] * np.exp(-cell.Chi[-1] - cell.Eg[-1] - phi_eq[-1])
    phis = np.concatenate([np.zeros(2 * N), phi_eq], axis=0)

    jcurve = np.array([], dtype=np.float64)
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

        v += Vincrement
        phis = ops.index_update(sol, -1, phi_eq[-1] + v)
        niter += 1

        if jcurve.size > 2:
            terminate = (jcurve[-2] * jcurve[-1] <= 0)

    return jcurve
