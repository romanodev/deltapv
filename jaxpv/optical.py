from jaxpv import objects, scales, physics, util
from jax import numpy as np, vmap

PVDesign = objects.PVDesign
LightSource = objects.LightSource
Array = util.Array
f64 = util.f64


def photonflux(ls: LightSource) -> Array:
    
    return ls.P_in / (scales.hc / ls.Lambda)


def alpha(design: PVDesign, lambdax: f64) -> Array:

    alpha = design.A * np.sqrt(
        np.clip(
            scales.hc / lambdax /
            (scales.kB * scales.temperature) - design.Eg, 0))

    return alpha


def generation_lambda(design: PVDesign, phi_0: f64, alpha: Array) -> Array:

    phi = phi_0 * np.exp(-np.cumsum(
        np.concatenate([np.zeros(1), alpha[:-1] * np.diff(design.grid)])))
    g = phi * alpha

    return g


def compute_G(design: PVDesign, ls: LightSource, optics: bool = True) -> Array:

    phis = photonflux(ls)
    valpha = vmap(alpha, (None, 0))

    if optics:
        alphas = valpha(design, ls.Lambda)
    else:
        alphas = vmap(np.interp, (None, None, 1),
                      1)(ls.Lambda, np.linspace(200, 1000, 100), design.alpha)

    vgenlambda = vmap(generation_lambda, (None, 0, 0))
    all_generations = vgenlambda(design, phis, alphas)
    tot_generation = np.sum(all_generations, axis=0)

    return tot_generation / scales.gratedens
