from deltapv import objects, solver, bcond, current, residual, linalg, scales, util
from jax import numpy as np, custom_jvp, jvp, jacfwd, jacrev, jit
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger("deltapv")

PVCell = objects.PVCell
Potentials = objects.Potentials
Array = util.Array
f64 = util.f64


def solve_pdd(cell: PVCell, v: f64, pot_ini: Potentials):

    # Determine boundary conditions
    bound = bcond.boundary(cell, v)

    # Solve system
    pot = solver.solve(cell, bound, pot_ini)

    # Compute total current
    flux = current.total_current(cell, pot)

    return flux, pot


@custom_jvp
def solve_pdd_adjoint(cell: PVCell, v: f64, pot_ini: Potentials):

    # Determine boundary conditions
    bound = bcond.boundary(cell, v)

    # Solve system
    pot = solver.solve(cell, bound, pot_ini)

    # Compute total current
    flux = current.total_current(cell, pot)

    return flux, pot


def F_wb(cell, v, pot):
    bound = bcond.boundary(cell, v)
    F = residual.comp_F(cell, bound, pot)
    return F


@solve_pdd_adjoint.defjvp
def solve_pdd_adjoint_jvp(primals, tangents):

    cell, v, pot_ini = primals
    dcell, _, _ = tangents

    # Solve forward problem
    bound = bcond.boundary(cell, v)
    pot = solver.solve(cell, bound, pot_ini)
    flux = current.total_current(cell, pot)

    # Compute gradients with adjoint method
    n = cell.Eg.size
    zp = objects.zero_pot(n)

    _, delF = jvp(F_wb, (cell, v, pot), (dcell, 0., zp))  # Fp @ dp
    _, delg = jvp(current.total_current, (cell, pot), (dcell, zp))  # gp @ dp

    gx_pot = jacrev(current.total_current, argnums=1)(cell, pot)
    gx = solver.pot2vec(gx_pot)  # vector form

    spFx = residual.comp_F_deriv(cell, bound, pot)
    FxT = linalg.sparse2dense(spFx).T

    lam = np.linalg.solve(FxT, gx)

    dg = delg - np.dot(lam, delF)  # total derivative

    primals_out = flux, pot
    tangents_out = dg, solver.vec2pot(np.zeros_like(gx))

    return primals_out, tangents_out
