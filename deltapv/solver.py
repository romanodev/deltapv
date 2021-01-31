from deltapv import objects, residual, linalg, util
from jax import numpy as np, jit, ops, custom_jvp, jvp, jacfwd, vmap
from typing import Tuple, Callable
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger("deltapv")

PVCell = objects.PVCell
LightSource = objects.LightSource
Potentials = objects.Potentials
Boundary = objects.Boundary
Array = util.Array
f64 = util.f64
n_lnsrch = 500


@jit
def logdamp(move: Array) -> Array:

    damped = np.where(
        np.abs(move) > 1,
        np.log(1 + np.abs(move) * 1.72) * np.sign(move), move)

    return damped


@jit
def scaledamp(move: Array, threshold: f64 = 50) -> Array:

    big = np.max(np.abs(move))
    gamma = np.maximum(big, threshold)
    damped = threshold * move / gamma

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
def linesearch(cell: PVCell, bound: Boundary, pot: Potentials,
               p: Array) -> Array:

    alphas = np.linspace(0, 2, n_lnsrch)
    R = vmap(residnorm, (None, None, None, None, 0))(cell, bound, pot, p,
                                                     alphas)
    alpha_best = alphas[n_lnsrch // 10:][np.argmin(R[n_lnsrch // 10:])]

    return alpha_best


def linguess(pot: Potentials, potl: Potentials):

    return Potentials(2 * pot.phi - potl.phi, 2 * pot.phi_n - potl.phi_n,
                      2 * pot.phi_p - potl.phi_p)


def genlinguess(pot: Potentials, potl: Potentials, dx1: f64, dx2: f64):

    return Potentials(pot.phi + (pot.phi - potl.phi) * dx2 / dx1,
                      pot.phi_n + (pot.phi_n - potl.phi_n) * dx2 / dx1,
                      pot.phi_p + (pot.phi_p - potl.phi_p) * dx2 / dx1)


def quadguess(pot: Potentials, potl: Potentials, potll: Potentials):

    f, fn, fp = pot.phi, pot.phi_n, pot.phi_p
    fl, fnl, fpl = potl.phi, potl.phi_n, potl.phi_p
    fll, fnll, fpll = potll.phi, potll.phi_n, potll.phi_p

    return Potentials(3 * f - 3 * fl + fll, 3 * fn - 3 * fnl + fnll,
                      3 * fp - 3 * fpl + fpll)


@jit
def step_eq_dense(cell: PVCell, bound: Boundary,
                  pot: Potentials) -> Tuple[Potentials, f64]:

    Feq = residual.comp_F_eq(cell, bound, pot)
    spJeq = residual.comp_F_eq_deriv(cell, bound, pot)
    Jeq = linalg.sparse2dense(spJeq)
    p = np.linalg.solve(Jeq, -Feq)

    error = np.max(np.abs(p))
    resid = np.linalg.norm(Feq)
    dx = logdamp(p)

    pot_new = Potentials(pot.phi + dx, pot.phi_n, pot.phi_p)

    stats = {"error": error, "resid": resid}

    return pot_new, stats


@jit
def step_eq(cell: PVCell, bound: Boundary,
            pot: Potentials) -> Tuple[Potentials, f64]:

    Feq = residual.comp_F_eq(cell, bound, pot)
    spJeq = residual.comp_F_eq_deriv(cell, bound, pot)
    p = linalg.linsol(spJeq, -Feq, tol=1e-6)

    error = np.max(np.abs(p))
    resid = np.linalg.norm(Feq)
    dx = logdamp(p)

    pot_new = Potentials(pot.phi + dx, pot.phi_n, pot.phi_p)

    stats = {"error": error, "resid": resid}

    return pot_new, stats


def solve_eq_dense(cell: PVCell, bound: Boundary,
                   pot_ini: Potentials) -> Potentials:

    pot = pot_ini
    error = 1
    niter = 0

    while niter < 100 and error > 1e-6:

        pot, stats = step_eq_dense(cell, bound, pot)
        error = stats["error"]
        resid = stats["resid"]
        niter += 1
        logger.info(
            f"\titeration: {str(niter).ljust(5)} |p|: {str(error).ljust(25)} |F|: {str(resid)}"
        )

        if np.isnan(error) or error == 0:
            logger.critical("\tDense solver failed! It's all over.")
            raise SystemExit

    return pot


@custom_jvp
def solve_eq(cell: PVCell, bound: Boundary, pot_ini: Potentials) -> Potentials:

    pot = pot_ini
    error = 1
    niter = 0

    while niter < 100 and error > 1e-6:

        pot, stats = step_eq(cell, bound, pot)
        error = stats["error"]
        resid = stats["resid"]
        niter += 1
        logger.info(
            f"\titeration: {str(niter).ljust(5)} |p|: {str(error).ljust(25)} |F|: {str(resid)}"
        )

        if np.isnan(error) or error == 0:
            logger.error("\tSparse solver failed! Switching to dense.")
            return solve_eq_dense(cell, bound, pot_ini)

    return pot


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


@jit
def step_dense(cell: PVCell, bound: Boundary,
               pot: Potentials) -> Tuple[Potentials, dict]:

    F = residual.comp_F(cell, bound, pot)
    spJ = residual.comp_F_deriv(cell, bound, pot)
    J = linalg.sparse2dense(spJ)
    p = np.linalg.solve(J, -F)

    error = np.max(np.abs(p))
    resid = np.linalg.norm(F)
    dx = logdamp(p)

    pot_new = modify(pot, dx)

    stats = {"error": error, "resid": resid}

    return pot_new, stats


@jit
def step(cell: PVCell, bound: Boundary,
         pot: Potentials) -> Tuple[Potentials, dict]:

    F = residual.comp_F(cell, bound, pot)
    spJ = residual.comp_F_deriv(cell, bound, pot)
    p = linalg.linsol(spJ, -F, tol=1e-6)

    error = np.max(np.abs(p))
    resid = np.linalg.norm(F)
    dx = logdamp(p)

    pot_new = modify(pot, dx)

    stats = {"error": error, "resid": resid}

    return pot_new, stats


def solve_dense(cell: PVCell, bound: Boundary,
                pot_ini: Potentials) -> Potentials:

    pot = pot_ini
    error = 1
    niter = 0

    while niter < 100 and error > 1e-6:

        pot, stats = step_dense(cell, bound, pot)
        error = stats["error"]
        resid = stats["resid"]
        niter += 1
        logger.info(
            f"\titeration: {str(niter).ljust(5)} |p|: {str(error).ljust(25)} |F|: {str(resid)}"
        )

        if np.isnan(error) or error == 0:
            logger.critical("\tDense solver failed! It's all over.")
            raise SystemExit

    return pot


@custom_jvp
def solve(cell: PVCell, bound: Boundary, pot_ini: Potentials) -> Potentials:

    pot = pot_ini
    error = 1
    niter = 0

    while niter < 100 and error > 1e-6:

        pot, stats = step(cell, bound, pot)
        error = stats["error"]
        resid = stats["resid"]
        niter += 1
        logger.info(
            f"\titeration: {str(niter).ljust(5)} |p|: {str(error).ljust(25)} |F|: {str(resid)}"
        )

        if np.isnan(error) or error == 0:
            logger.error("\tSparse solver failed! Switching to dense.")
            return solve_dense(cell, bound, pot_ini)

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
