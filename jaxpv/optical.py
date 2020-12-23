from jaxpv import objects, scales, physics, util
from jax import numpy as np, vmap

PVCell = objects.PVCell
LightSource = objects.LightSource
Array = util.Array
f64 = util.f64


def photonflux(ls: LightSource) -> Array:

    return ls.P_in / (scales.hc / ls.Lambda)


def alpha(cell: PVCell, lambdax: f64) -> Array:

    alpha = np.where(
        scales.hc / lambdax / (scales.KB * scales.T) > cell.Eg,
        cell.A *
        np.sqrt(np.abs(scales.hc / lambdax /
                       (scales.KB * scales.T) - cell.Eg)), 0)

    return alpha


def generation_lambda(cell: PVCell, phi_0: f64, alpha: Array) -> Array:

    phi = phi_0 * np.exp(
        -np.cumsum(np.concatenate([np.zeros(1), alpha[:-1] * cell.dgrid])))
    g = phi * alpha

    return g


def compute_G(cell: PVCell, ls: LightSource) -> Array:

    phis = photonflux(ls)
    valpha = vmap(alpha, (None, 0))
    alphas = valpha(cell, ls.Lambda)
    vgenlambda = vmap(generation_lambda, (None, 0, 0))
    all_generations = vgenlambda(cell, phis, alphas)
    tot_generation = np.sum(all_generations, axis=0)

    return tot_generation / scales.U
