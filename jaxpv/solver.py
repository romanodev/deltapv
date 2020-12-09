from . import residual
from . import splinalg

import jax.numpy as np
from jax.scipy.sparse.linalg import gmres

import matplotlib.pyplot as plt


def damp(move):

    tmp = 1e10
    approx_sign = np.tanh(tmp * move)
    approx_abs = approx_sign * move
    approx_H = 1 - (1 + tmp * np.exp(-(move**2 - 1)))**(-1)
    return np.log(1 + approx_abs) * approx_sign + approx_H * (
        move - np.log(1 + approx_abs) * approx_sign)


def step(data, neq0, neqL, peq0, peqL, phis):

    dgrid = data["dgrid"]
    N = dgrid.size + 1

    F = residual.F(data, neq0, neqL, peq0, peqL, phis[0:N], phis[N:2 * N],
                   phis[2 * N:])
    values, indices, indptr = residual.F_deriv(data, neq0, neqL, peq0, peqL,
                                               phis[0:N], phis[N:2 * N],
                                               phis[2 * N:])

    # gradF_jvp = lambda x: splinalg.spdot(values, indices, indptr, x)
    # precond_jvp = splinalg.spilu(values, indices, indptr)

    # move, conv_info = gmres(gradF_jvp, -F, tol=1e-10, maxiter=5, M=precond_jvp)

    jacobian = splinalg.dense(values, indices, indptr)
    move = np.linalg.solve(jacobian, -F)

    error = np.linalg.norm(move)
    damp_move = damp(move)

    return np.concatenate(
        (phis[0:N] + damp_move[0:3 * N:3], phis[N:2 * N] +
         damp_move[1:3 * N:3], phis[2 * N:] + damp_move[2:3 * N:3]),
        axis=0), error


def solve(data, neq0, neqL, peq0, peqL, phis_ini):

    dgrid = data["dgrid"]
    N = dgrid.size + 1

    phis = phis_ini
    error = 1
    niter = 0

    while error > 1e-6 and niter < 100:

        phis, error = step(data, neq0, neqL, peq0, peqL, phis)
        niter += 1
        print(f"\t iteration: {niter}    error: {error}")

    return phis


def step_eq(data, phi):

    dgrid = data["dgrid"]
    N = dgrid.size + 1

    Feq = residual.F_eq(data, np.zeros(phi.size), np.zeros(phi.size), phi)
    values, indices, indptr = residual.F_eq_deriv(data, np.zeros(phi.size),
                                                  np.zeros(phi.size), phi)

    # gradFeq_jvp = lambda x: splinalg.spdot(values, indices, indptr, x)
    # precond_jvp = splinalg.spilu(values, indices, indptr)
    # move, conv_info = gmres(gradFeq_jvp,
    #                         -Feq,
    #                         tol=1e-10,
    #                         maxiter=5,
    #                         M=precond_jvp)

    jacobian = splinalg.dense(values, indices, indptr)
    move = np.linalg.solve(jacobian, -Feq)

    error = np.linalg.norm(move)
    damp_move = damp(move)

    return phi + damp_move, error


def solve_eq(data, phi_ini):

    print("Solving equilibrium...")
    dgrid = data["dgrid"]
    N = dgrid.size + 1

    error = 1
    niter = 0
    phi = phi_ini

    while error > 1e-6 and niter < 100:

        phi, error = step_eq(data, phi)
        niter += 1
        print(f"\t iteration: {niter}    error: {error}")

    return phi
