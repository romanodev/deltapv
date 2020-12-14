from . import residual
from . import splinalg

import jax.numpy as np
from jax.scipy.sparse.linalg import gmres

import matplotlib.pyplot as plt


def damp(dx):
    
    damped = np.where(np.abs(dx) > 1,
                      np.log(1 + np.abs(dx) * 1.72) * np.sign(dx),
                      dx)
    
    return damped


def step(data, neq0, neqL, peq0, peqL, phis):

    dgrid = data["dgrid"]
    N = dgrid.size + 1

    F = residual.F(data, neq0, neqL, peq0, peqL, phis[0:N], phis[N:2 * N],
                   phis[2 * N:])
    spgradF = residual.F_deriv(data, neq0, neqL, peq0, peqL, phis[0:N],
                               phis[N:2 * N], phis[2 * N:])

    gradF_jvp = lambda x: splinalg.spmatvec(spgradF, x)
    precond_jvp = splinalg.invjvp(spgradF)

    move, conv_info = gmres(gradF_jvp, -F,
                            M=precond_jvp,
                            tol=1e-12,
                            atol=0.)

    error = np.max(np.abs(move))
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
        print(f"\t iteration: {str(niter).ljust(10)} error: {error}")

    return phis


def step_eq(data, phi):

    dgrid = data["dgrid"]
    N = dgrid.size + 1

    Feq = residual.F_eq(data, np.zeros(phi.size), np.zeros(phi.size), phi)
    spgradFeq = residual.F_eq_deriv(data, np.zeros(phi.size),
                                    np.zeros(phi.size), phi)

    gradFeq_jvp = lambda x: splinalg.spmatvec(spgradFeq, x)
    precond_jvp = splinalg.invjvp(spgradFeq)
    move, conv_info = gmres(gradFeq_jvp, -Feq,
                            M=precond_jvp,
                            tol=1e-12,
                            atol=0.)

    error = np.max(np.abs(move))
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
        print(f"\t iteration: {str(niter).ljust(10)} error: {error}")

    return phi
