from deltapv import objects, ddiff, bcond, poisson, linalg, util
from jax import numpy as jnp, ops, jit, jacfwd

PVCell = objects.PVCell
Potentials = objects.Potentials
Boundary = objects.Boundary
Array = util.Array
f64 = util.f64


@jit
def comp_F(cell: PVCell, bound: Boundary, pot: Potentials) -> Array:

    ddn = ddiff.ddn(cell, pot)
    ddp = ddiff.ddp(cell, pot)
    pois = poisson.pois(cell, pot)

    ctct_0_phin, ctct_L_phin = bcond.contact_phin(cell, bound, pot)
    ctct_0_phip, ctct_L_phip = bcond.contact_phip(cell, bound, pot)
    ctct_0_phi, ctct_L_phi = bcond.contact_phi(cell, bound, pot)

    lenF = 3 + 3 * len(pois) + 3
    result = jnp.zeros(lenF, dtype=jnp.float64)
    result = result.at[:3].set(jnp.array([ctct_0_phin, ctct_0_phip, ctct_0_phi]))
    result = result.at[3:lenF - 5:3].set(ddn)
    result = result.at[4:lenF - 4:3].set(ddp)
    result = result.at[5:lenF - 3:3].set(pois)
    result = result.at[-3:].set(jnp.array([ctct_L_phin, ctct_L_phip, ctct_L_phi]))

    return result


@jit
def comp_F_deriv(cell: PVCell, bound: Boundary, pot: Potentials) -> Array:

    dde_phin_, dde_phin__, dde_phin___, dde_phip__, dde_phi_, dde_phi__, dde_phi___ = ddiff.ddn_deriv(
        cell, pot)
    ddp_phin__, ddp_phip_, ddp_phip__, ddp_phip___, ddp_phi_, ddp_phi__, ddp_phi___ = ddiff.ddp_deriv(
        cell, pot)
    dpois_phi_, dpois_phi__, dpois_phi___, dpois_dphin__, dpois_dphip__ = poisson.pois_deriv(
        cell, pot)
    dctct_phin = bcond.contact_phin_deriv(cell, pot)
    dctct_phip = bcond.contact_phip_deriv(cell, pot)

    N = cell.Eg.size

    row = jnp.concatenate([
        jnp.zeros(4),
        jnp.ones(4),
        jnp.array([2]),
        jnp.tile(3 * (N - 1), 4),
        jnp.tile(3 * (N - 1) + 1, 4),
        jnp.array([3 * (N - 1) + 2]),
        jnp.tile(jnp.arange(3, 3 * (N - 1), 3), 7),
        jnp.tile(jnp.arange(4, 3 * (N - 1) + 1, 3), 7),
        jnp.tile(jnp.arange(5, 3 * (N - 1) + 2, 3), 5)
    ]).astype(jnp.int32)

    col = jnp.concatenate([
        jnp.array([0, 2, 3, 5]),
        jnp.array([1, 2, 4, 5]),
        jnp.array([2]),
        jnp.array([3 * (N - 2), 3 * (N - 2) + 2, 3 * (N - 1), 3 * (N - 1) + 2]),
        jnp.array([
            3 * (N - 2) + 1, 3 * (N - 2) + 2, 3 * (N - 1) + 1, 3 * (N - 1) + 2
        ]),
        jnp.array([3 * (N - 1) + 2]),
        jnp.arange(0, 3 * (N - 2), 3),
        jnp.arange(3, 3 * (N - 1), 3),
        jnp.arange(6, 3 * N, 3),
        jnp.arange(4, 3 * (N - 1) + 1, 3),
        jnp.arange(2, 3 * (N - 2) + 2, 3),
        jnp.arange(5, 3 * (N - 1) + 2, 3),
        jnp.arange(8, 3 * N + 2, 3),
        jnp.arange(3, 3 * (N - 1), 3),
        jnp.arange(1, 3 * (N - 2) + 1, 3),
        jnp.arange(4, 3 * (N - 1) + 1, 3),
        jnp.arange(7, 3 * N + 1, 3),
        jnp.arange(2, 3 * (N - 2) + 2, 3),
        jnp.arange(5, 3 * (N - 1) + 2, 3),
        jnp.arange(8, 3 * N + 2, 3),
        jnp.arange(2, 3 * (N - 2) + 2, 3),
        jnp.arange(5, 3 * (N - 1) + 2, 3),
        jnp.arange(8, 3 * N + 2, 3),
        jnp.arange(3, 3 * (N - 1), 3),
        jnp.arange(4, 3 * (N - 1) + 1, 3)
    ]).astype(jnp.int32)

    dF = jnp.concatenate([
        jnp.array([dctct_phin[0], dctct_phin[2], dctct_phin[1], dctct_phin[3]]),
        jnp.array([dctct_phip[0], dctct_phip[2], dctct_phip[1], dctct_phip[3]]),
        jnp.ones(1),
        jnp.array([dctct_phin[4], dctct_phin[6], dctct_phin[5], dctct_phin[7]]),
        jnp.array([dctct_phip[4], dctct_phip[6], dctct_phip[5], dctct_phip[7]]),
        jnp.ones(1),
        dde_phin_,
        dde_phin__,
        dde_phin___,
        dde_phip__,
        dde_phi_,
        dde_phi__,
        dde_phi___,
        ddp_phin__,
        ddp_phip_,
        ddp_phip__,
        ddp_phip___,
        ddp_phi_,
        ddp_phi__,
        ddp_phi___,
        dpois_phi_,
        dpois_phi__,
        dpois_phi___,
        dpois_dphin__,
        dpois_dphip__,
    ])

    spF = linalg.coo2sparse(row, col, dF, 3 * N)

    return spF


@jit
def comp_F_eq(cell: PVCell, bound: Boundary, pot: Potentials) -> Array:

    pois = poisson.pois(cell, pot)
    ctct_0_phi, ctct_L_phi = bcond.contact_phi(cell, bound, pot)
    resid = jnp.concatenate(
        [jnp.array([ctct_0_phi]), pois,
         jnp.array([ctct_L_phi])])

    return resid


@jit
def comp_F_eq_deriv(cell: PVCell, bound: Boundary, pot: Potentials) -> Array:

    N = cell.Eg.size
    dpois_phi_, dpois_phi__, dpois_phi___ = poisson.pois_deriv_eq(cell, pot)

    row = jnp.concatenate([
        jnp.array([0]),
        jnp.arange(1, N - 1),
        jnp.arange(1, N - 1),
        jnp.arange(1, N - 1),
        jnp.array([N - 1])
    ]).astype(jnp.int32)

    col = jnp.concatenate([
        jnp.array([0]),
        jnp.arange(0, N - 2),
        jnp.arange(1, N - 1),
        jnp.arange(2, N),
        jnp.array([N - 1])
    ]).astype(jnp.int32)

    dFeq = jnp.concatenate(
        [jnp.array([1]), dpois_phi_, dpois_phi__, dpois_phi___,
         jnp.array([1])])

    spFeq = linalg.coo2sparse(row, col, dFeq, N)

    return spFeq
