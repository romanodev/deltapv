from jaxpv import objects, residual, linalg, util
from jax import numpy as np, jit, ops, custom_jvp, jvp
from typing import Tuple, Callable
from functools import partial
import logging

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
def pot2vec(pot: Potentials) -> Array:
    
    n = pot.phi.size
    vec = np.zeros(3 * n)
    vec = ops.index_update(vec, ops.index[0::3], pot.phi_n)
    vec = ops.index_update(vec, ops.index[1::3], pot.phi_p)
    vec = ops.index_update(vec, ops.index[2::3], pot.phi)
    
    return vec


@jit
def step(cell: PVCell, bound: Boundary,
         pot: Potentials) -> Tuple[Potentials, f64]:

    N = cell.Eg.size

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
def step_eq(cell: PVCell, bound: Boundary,
            pot: Potentials) -> Tuple[Potentials, f64]:

    N = cell.Eg.size

    Feq = residual.comp_F_eq(cell, bound, pot)
    spgradFeq = residual.comp_F_eq_deriv(cell, bound, pot)

    move = linalg.linsol(spgradFeq, -Feq)
    error = np.max(np.abs(move))
    damp_move = damp(move)

    pot_new = Potentials(pot.phi + damp_move, pot.phi_n, pot.phi_p)

    return pot_new, error


@custom_jvp
def solve(cell: PVCell, bound: Boundary, pot_ini: Potentials) -> Potentials:

    pot = pot_ini
    error = 1
    niter = 0

    while error > 1e-6 and niter < 100:

        pot, error = step(cell, bound, pot)
        niter += 1
        logging.info(f"\t iteration: {str(niter).ljust(10)} error: {error}")

    return pot


@custom_jvp
def solve_eq(cell: PVCell, bound: Boundary, pot_ini: Potentials) -> Potentials:

    pot = pot_ini
    error = 1
    niter = 0

    while error > 1e-6 and niter < 100:

        pot, error = step_eq(cell, bound, pot)
        niter += 1
        logging.info(f"\t iteration: {str(niter).ljust(10)} error: {error}")

    return pot


@solve.defjvp
def solve_jvp(primals, tangents):
    
    cell, bound, pot_ini = primals
    dcell, dbound, _ = tangents
    sol = solve(cell, bound, pot_ini)
    
    zerodpot = Potentials(np.zeros_like(sol.phi),
                          np.zeros_like(sol.phi_n),
                          np.zeros_like(sol.phi_p))
    
    _, rhs = jvp(residual.comp_F,
                 (cell, bound, sol),
                 (dcell, dbound, zerodpot))
    
    spF_pot = residual.comp_F_deriv(cell, bound, sol)
    F_pot = linalg.sparse2dense(spF_pot)
    dF = np.linalg.solve(F_pot, -rhs)
    
    primal_out = sol
    tangent_out = Potentials(dF[2::3], dF[0::3], dF[1::3])
    
    return primal_out, tangent_out


@solve_eq.defjvp
def solve_eq_jvp(primals, tangents):
    
    cell, bound, pot_ini = primals
    dcell, dbound, _ = tangents # pot_ini is an initial guess; irrelevant
    sol = solve_eq(cell, bound, pot_ini)
    
    # solve_eq returns pot that satisfies F_eq(cell, bound, pot(cell, bound)) = 0
    
    # dF_eq/dcell = F_eq_cell + F_eq_pot @ pot_cell = 0
    # pot_cell = - inv(F_eq_pot) @ F_eq_cell
    # jvp_cell = - inv(F_eq_pot) @ F_eq_cell @ dcell
    # First compute F_eq_cell @ dcell, then linear solve
    
    # dF_eq/dbound = F_eq_bound + F_eq_pot @ pot_bound = 0
    # pot_bound = - inv(F_eq_pot) @ F_eq_bound
    # jvp_bound = - inv(F_eq_pot) @ F_eq_bound @ dbound
    # First compute F_eq_bound @ dbound, then linear solve
    
    # jvp_total = - inv(F_eq_pot) @ (F_eq_cell @ dcell + F_eq_bound @ dbound)
    # (F_eq_cell @ dcell + F_eq_bound @ dbound) = jvp(F_eq, (cell, bound, sol), (dcell, dbound, 0))
    
    zerodpot = Potentials(np.zeros_like(sol.phi),
                          np.zeros_like(sol.phi_n),
                          np.zeros_like(sol.phi_p))
    
    _, rhs = jvp(residual.comp_F_eq,
                 (cell, bound, sol),
                 (dcell, dbound, zerodpot))
    
    spF_eq_pot = residual.comp_F_eq_deriv(cell, bound, sol)
    F_eq_pot = linalg.sparse2dense(spF_eq_pot)
    dF_eq = np.linalg.solve(F_eq_pot, -rhs)
    
    primal_out = sol
    tangent_out = Potentials(dF_eq,
                             np.zeros_like(sol.phi_n),
                             np.zeros_like(sol.phi_p))
    
    return primal_out, tangent_out
