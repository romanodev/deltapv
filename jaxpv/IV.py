from . import initial_guess
from . import solver
from . import scaling
from . import current

import jax.numpy as np
from jax import ops

scale = scaling.scales()


def calc_IV(data, Vincrement):

    dgrid = data["dgrid"]
    Chi = data["Chi"]
    Eg = data["Eg"]
    Nc = data["Nc"]
    Nv = data["Nv"]
    Ndop = data["Ndop"]
    N = Chi.size

    phi_ini = initial_guess.eq_init_phi(data)
    phi_eq = solver.solve_eq(data, phi_ini)

    neq_0 = Nc[0] * np.exp(Chi[0] + phi_eq[0])
    neq_L = Nc[-1] * np.exp(Chi[-1] + phi_eq[-1])
    peq_0 = Nv[0] * np.exp(-Chi[0] - Eg[0] - phi_eq[0])
    peq_L = Nv[-1] * np.exp(-Chi[-1] - Eg[-1] - phi_eq[-1])
    phis = np.concatenate([np.zeros(2 * N), phi_eq], axis=0)

    jcurve = np.array([], dtype=np.float64)
    max_iter = 100
    niter = 0
    v = 0
    terminate = False

    while not terminate and niter < max_iter:

        scaled_V = v * scale["E"]
        print(f"Solving for V = {scaled_V}")
        sol = solver.solve(data, neq_0, neq_L, peq_0, peq_L, phis)
        total_j, _ = current.total_current(data, sol[0:N], sol[N:2 * N],
                                           sol[2 * N:])

        jcurve = np.concatenate([jcurve, np.array([total_j])])

        v = v + Vincrement
        phis = ops.index_update(sol, -1, phi_eq[-1] + v)
        niter += 1

        if jcurve.size > 2:
            terminate = (jcurve[-2] * jcurve[-1] <= 0)

    return jcurve
