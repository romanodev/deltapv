from jaxpv import objects, residual, linalg, util
from jax import numpy as np, jit, ops, custom_jvp, jvp, jacfwd, vmap
from typing import Tuple, Callable
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger("jaxpv")

PVCell = objects.PVCell
LightSource = objects.LightSource
Potentials = objects.Potentials
Boundary = objects.Boundary
Array = util.Array
f64 = util.f64
NLS = 500


@jit
def logdamp(move: Array) -> Array:

    damped = np.where(
        np.abs(move) > 1,
        np.log(1 + np.abs(move) * 1.72) * np.sign(move), move)

    return damped


@jit
def scaledamp(move: Array) -> Array:

    big = np.max(np.abs(move))
    gamma = np.maximum(big, 50)
    damped = 50 * move / gamma

    return damped


@jit
def pot2vec(pot: Potentials) -> Array:

    n = pot.phi.size
    vec = np.zeros(3 * n)
    vec = vec.at[0::3].set(pot.phi_n)
    vec = vec.at[1::3].set(pot.phi_p)
    vec = vec.at[2::3].set(pot.phi)

    return vec


@jit
def modify(pot: Potentials, move: Array) -> Potentials:

    phi_new = pot.phi + move[2::3]
    phi_n_new = pot.phi_n + move[::3]
    phi_p_new = pot.phi_p + move[1::3]
    pot_new = Potentials(phi_new, phi_n_new, phi_p_new)

    return pot_new


@jit
def residnorm(cell, bound, pot, move, alpha):

    pot_new = modify(pot, alpha * move)
    F = residual.comp_F(cell, bound, pot_new)
    Fnorm = np.linalg.norm(F)

    return Fnorm


@jit
def step(cell: PVCell, bound: Boundary,
         pot: Potentials) -> Tuple[Potentials, dict]:

    F = residual.comp_F(cell, bound, pot)
    spJ = residual.comp_F_deriv(cell, bound, pot)
    p = linalg.linsol(spJ, -F)

    error = np.max(np.abs(p))
    resid = np.linalg.norm(F)
    dx = logdamp(p)

    """alphas = np.linspace(0, 2, NLS)
    R = vmap(residnorm, (None, None, None, None, 0))(cell, bound, pot,
                                                     dx, alphas)
    alpha_best = alphas[NLS // 10:][np.argmin(R[NLS // 10:])]"""

    alpha_best = 1
    pot_new = modify(pot, alpha_best * dx)

    stats = {"error": error, "resid": resid}

    return pot_new, stats


@jit
def step_eq(cell: PVCell, bound: Boundary,
            pot: Potentials) -> Tuple[Potentials, f64]:

    Feq = residual.comp_F_eq(cell, bound, pot)
    spJeq = residual.comp_F_eq_deriv(cell, bound, pot)
    p = linalg.linsol(spJeq, -Feq)

    error = np.max(np.abs(p))
    resid = np.linalg.norm(Feq)
    dx = logdamp(p)

    pot_new = Potentials(pot.phi + dx, pot.phi_n, pot.phi_p)

    stats = {"error": error, "resid": resid}

    return pot_new, stats


@custom_jvp
def solve(cell: PVCell, bound: Boundary, pot_ini: Potentials) -> Potentials:

    pot = pot_ini
    error = 1
    niter = 0

    while error > 1e-6 and niter < 100:

        pot, stats = step(cell, bound, pot)
        error = stats["error"]
        resid = stats["resid"]
        niter += 1
        logger.info(
            f"\t iteration: {str(niter).ljust(5)} |p|: {str(error).ljust(25)} |F|: {str(resid)}"
        )

    return pot


@custom_jvp
def solve_eq(cell: PVCell, bound: Boundary, pot_ini: Potentials) -> Potentials:

    pot = pot_ini
    error = 1
    niter = 0

    while error > 1e-6 and niter < 100:

        pot, stats = step_eq(cell, bound, pot)
        error = stats["error"]
        resid = stats["resid"]
        niter += 1
        logger.info(
            f"\t iteration: {str(niter).ljust(5)} |p|: {str(error).ljust(25)} |F|: {str(resid)}"
        )

    return pot


@solve.defjvp
def solve_jvp(primals, tangents):

    cell, bound, pot_ini = primals
    dcell, dbound, _ = tangents
    sol = solve(cell, bound, pot_ini)

    zerodpot = Potentials(np.zeros_like(sol.phi), np.zeros_like(sol.phi_n),
                          np.zeros_like(sol.phi_p))

    _, rhs = jvp(residual.comp_F, (cell, bound, sol),
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
    dcell, dbound, _ = tangents
    sol = solve_eq(cell, bound, pot_ini)

    zerodpot = Potentials(np.zeros_like(sol.phi), np.zeros_like(sol.phi_n),
                          np.zeros_like(sol.phi_p))

    _, rhs = jvp(residual.comp_F_eq, (cell, bound, sol),
                 (dcell, dbound, zerodpot))

    spF_eq_pot = residual.comp_F_eq_deriv(cell, bound, sol)
    F_eq_pot = linalg.sparse2dense(spF_eq_pot)
    dF_eq = np.linalg.solve(F_eq_pot, -rhs)

    primal_out = sol
    tangent_out = Potentials(dF_eq, np.zeros_like(sol.phi_n),
                             np.zeros_like(sol.phi_p))

    return primal_out, tangent_out
