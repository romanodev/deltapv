from . import scales
from . import physics

import jax.numpy as np
from jax import vmap


def photonflux(data):

    Lambda, P_in = data["Lambda"], data["P_in"]

    return P_in / (scales.hc / Lambda)


def alpha(data, lambdax):

    Eg, A = data["Eg"], data["A"]
    alpha = np.where(scales.hc / lambdax / (scales.KB * scales.T) > Eg,
                     A * np.sqrt(scales.hc / lambdax / (scales.KB * scales.T) - Eg), 0)
    
    return alpha


def alpha_deriv(data, lambdax):

    Eg, A = data["Eg"], data["A"]
    dalpha_dEg = np.where(scales.hc / lambdax / (scales.KB * scales.T) > Eg,
                          -1 / (2 * np.sqrt(scales.hc / lambdax / (scales.KB * scales.T) - Eg)), 0)
    dalpha_dA = np.where(scales.hc / lambdax / (scales.KB * scales.T) > Eg,
                         np.sqrt(scales.hc / lambdax / (scales.KB * scales.T) - Eg), 0)

    return dalpha_dEg, dalpha_dA


def generation_lambda(data, phi_0, alpha):

    dgrid = data["dgrid"]
    phi = phi_0 * np.exp(-np.cumsum(
        np.concatenate([np.zeros(1), alpha[:-1] * dgrid])))
    g = phi * alpha

    return g


def compute_G(data):

    dgrid = data["dgrid"]
    Lambda = data["Lambda"]
    phis = photonflux(data)
    valpha = vmap(alpha, (None, 0))
    alphas = valpha(data, data["Lambda"])
    vgenlambda = vmap(generation_lambda, (None, 0, 0))
    all_generations = vgenlambda(data, phis, alphas)
    tot_generation = np.sum(all_generations, axis=0)

    return tot_generation / scales.U


def deriv_G(data):
    
    # There are issues with this function
    dgrid = data["dgrid"]
    Lambda = data["Lambda"]
    phi_0 = photonflux(data)
    G = 0
    dG_dEg, dG_dA = np.zeros((Eg.size, Eg.size))
    for i in range(Lambda.size):
        G_at_lambda = generation_lambda(data, phi_0[i], alpha(data, Lambda[i]))
        G += G_at_lambda
        dalpha_dEg, dalpha_dA = alpha_deriv(data, Lambda[i])
        dG_dEg = -G_at_lambda * np.cumsum(dalpha_dEg, dgrid)
        dG_dA = -G_at_lambda * np.cumsum(dalpha_dA, dgrid)

    G /= scales.U

    return dG_dEg / scales.U, dG_dA / scales.U
