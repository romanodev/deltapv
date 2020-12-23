from jaxpv import objects, residual, splinalg, util
from jax import numpy as np, scipy, jit
from jax.scipy.sparse.linalg import gmres
from typing import Tuple

PVCell = objects.PVCell
LightSource = objects.LightSource
Array = util.Array
f64 = util.f64


@jit
def damp(dx):

    damped = np.where(
        np.abs(dx) > 1,
        np.log(1 + np.abs(dx) * 1.72) * np.sign(dx), dx)

    return damped


@jit
def linsol(spmat: Array, vec: Array) -> Array:
    
    mvp = lambda x: splinalg.spmatvec(spmat, x)
    precond = splinalg.invjvp(spmat)
    
    sol, _ = scipy.sparse.linalg.gmres(mvp, vec, M=precond,
                                       tol=1e-12, atol=0.,
                                       maxiter=100)
    
    return sol


@jit
def step(cell: PVCell, neq0: f64, neqL: f64, peq0: f64, peqL: f64,
         phis: Array) -> Tuple[Array, f64]:

    N = cell.grid.size

    F = residual.F(cell, neq0, neqL, peq0, peqL, phis[0:N], phis[N:2 * N],
                   phis[2 * N:])
    spgradF = residual.F_deriv(cell, neq0, neqL, peq0, peqL, phis[0:N],
                               phis[N:2 * N], phis[2 * N:])
    
    move = linsol(spgradF, -F)
    error = np.max(np.abs(move))
    damp_move = damp(move)
    
    return np.concatenate(
        (phis[0:N] + damp_move[0:3 * N:3], phis[N:2 * N] +
         damp_move[1:3 * N:3], phis[2 * N:] + damp_move[2:3 * N:3]),
        axis=0), error


def solve(cell: PVCell, neq0: f64, neqL: f64, peq0: f64, peqL: f64,
          phis_ini: Array) -> Array:

    N = cell.grid.size

    phis = phis_ini
    error = 1
    niter = 0

    while error > 1e-6 and niter < 100:

        phis, error = step(cell, neq0, neqL, peq0, peqL, phis)
        niter += 1
        print(f"\t iteration: {str(niter).ljust(10)} error: {error}")

    return phis


@jit
def step_eq(cell: PVCell, phi: Array) -> Tuple[Array, f64]:

    N = cell.grid.size

    Feq = residual.F_eq(cell, np.zeros(N), np.zeros(N), phi)
    spgradFeq = residual.F_eq_deriv(cell, np.zeros(N), np.zeros(N), phi)
    
    move = linsol(spgradFeq, -Feq)
    error = np.max(np.abs(move))
    damp_move = damp(move)

    return phi + damp_move, error


def solve_eq(cell: PVCell, phi_ini: Array) -> Array:

    print("Solving equilibrium...")
    N = cell.grid.size

    error = 1
    niter = 0
    phi = phi_ini

    while error > 1e-6 and niter < 100:

        phi, error = step_eq(cell, phi)
        niter += 1
        print(f"\t iteration: {str(niter).ljust(10)} error: {error}")

    return phi
