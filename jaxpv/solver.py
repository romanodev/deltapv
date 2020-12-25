from jaxpv import objects, residual, linalg, util
from jax import numpy as np, jit
from typing import Tuple, Callable
from functools import partial

PVCell = objects.PVCell
LightSource = objects.LightSource
Potentials = objects.Potentials
Boundary = objects.Boundary
Array = util.Array
f64 = util.f64


@jit
def damp(move: Array) -> Array:

    damped = np.where(
        np.abs(move) > 1,
        np.log(1 + np.abs(move) * 1.72) * np.sign(move), move)

    return damped


@jit
def step(cell: PVCell, bound: Boundary,
         pot: Potentials) -> Tuple[Potentials, f64]:

    N = cell.grid.size

    F = residual.comp_F(cell, bound, pot)
    spgradF = residual.comp_F_deriv(cell, bound, pot)

    move = linalg.linsol(spgradF, -F)
    error = np.max(np.abs(move))
    damp_move = damp(move)
    phi_new = pot.phi + damp_move[2:3 * N:3]
    phi_n_new = pot.phi_n + damp_move[:3 * N:3]
    phi_p_new = pot.phi_p + damp_move[1:3 * N:3]

    pot_new = Potentials(phi_new, phi_n_new, phi_p_new)

    return pot_new, error


@jit
def step_eq(cell: PVCell, bound: Boundary, pot: Potentials) -> Tuple[Potentials, f64]:

    N = cell.grid.size

    Feq = residual.comp_F_eq(cell, bound, pot)
    spgradFeq = residual.comp_F_eq_deriv(cell, bound, pot)

    move = linalg.linsol(spgradFeq, -Feq)
    error = np.max(np.abs(move))
    damp_move = damp(move)

    pot_new = Potentials(pot.phi + damp_move, pot.phi_n, pot.phi_p)

    return pot_new, error


def _solve(f: Callable[[Tuple[PVCell, Boundary, Potentials]], Tuple[Potentials, f64]],
           cell: PVCell, bound: Boundary, pot_ini: Potentials) -> Potentials:
    
    pot = pot_ini
    error = 1
    niter = 0
    
    while error > 1e-6 and niter < 100:

        pot, error = f(cell, bound, pot)
        niter += 1
        print(f"\t iteration: {str(niter).ljust(10)} error: {error}")

    return pot


solve = partial(_solve, step)
solve_eq = partial(_solve, step_eq)