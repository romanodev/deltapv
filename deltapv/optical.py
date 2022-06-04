from deltapv import objects, scales, util
from jax import numpy as jnp, vmap

PVDesign = objects.PVDesign
LightSource = objects.LightSource
Array = util.Array
f64 = util.f64


def photonflux(ls: LightSource) -> Array:

    I = ls.P_in  # W / m^2  noqa
    lamb = ls.Lambda * scales.nm  # m
    phi0 = I / (scales.hc / lamb)  # 1 / (m^2 s)

    return phi0


def alpha(design: PVDesign, lambdax: f64) -> Array:

    A_si = design.A / scales.cm / jnp.sqrt(scales.eV)  # 1 / (m J^(1/2))
    Eg_si = design.Eg * scales.energy * scales.eV  # J
    lamb_si = lambdax * scales.nm  # m

    alpha = jnp.where(scales.hc / lamb_si - Eg_si > 0,
                      A_si * jnp.sqrt(jnp.abs(scales.hc / lamb_si - Eg_si)),
                      0)  # 1 / m

    return alpha


def generation_lambda(design: PVDesign, phi_0: f64, alpha: Array) -> Array:

    # phi_0, alpha expected to be in SI units

    x = design.grid * scales.length * scales.cm  # m
    dx = jnp.diff(x)  # m

    phi = phi_0 * jnp.exp(-jnp.cumsum(
        jnp.concatenate([jnp.zeros(1), alpha[:-1] * dx])))  # 1 / (m^2 s)
    g = phi * alpha  # 1 / (m^3 s)

    return g


def compute_G(design: PVDesign, ls: LightSource, optics: bool = True) -> Array:

    phis = photonflux(ls)
    valpha = vmap(alpha, (None, 0))

    if optics:
        alphas = valpha(design, ls.Lambda)  # 1 / m
    else:
        alphas = vmap(jnp.interp, (None, None, 1),
                      1)(ls.Lambda, jnp.linspace(200, 1000, 100), design.alpha)
        alphas = alphas / scales.cm  # 1 / m

    vgenlambda = vmap(generation_lambda, (None, 0, 0))
    all_generations = vgenlambda(design, phis, alphas)
    tot_generation = jnp.sum(all_generations, axis=0)  # 1 / (m^3 s)
    G_dim = tot_generation / 1e6 / scales.gratedens

    return G_dim
