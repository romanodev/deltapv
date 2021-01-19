from jaxpv import objects, ddiff, bcond, poisson, linalg, util
from jax import numpy as np, ops, jit

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
    result = np.zeros(lenF, dtype=np.float64)
    result = result.at[:3].set(np.array([ctct_0_phin, ctct_0_phip, ctct_0_phi]))
    result = result.at[3:lenF - 5:3].set(ddn)
    result = result.at[4:lenF - 4:3].set(ddp)
    result = result.at[5:lenF - 3:3].set(pois)
    result = result.at[-3:].set(np.array([ctct_L_phin, ctct_L_phip, ctct_L_phi]))

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

    row = np.concatenate([
        np.zeros(4),
        np.ones(4),
        np.array([2]),
        np.tile(3 * (N - 1), 4),
        np.tile(3 * (N - 1) + 1, 4),
        np.array([3 * (N - 1) + 2]),
        np.tile(np.arange(3, 3 * (N - 1), 3), 7),
        np.tile(np.arange(4, 3 * (N - 1) + 1, 3), 7),
        np.tile(np.arange(5, 3 * (N - 1) + 2, 3), 5)
    ]).astype(np.int32)

    col = np.concatenate([
        np.array([0, 2, 3, 5]),
        np.array([1, 2, 4, 5]),
        np.array([2]),
        np.array([3 * (N - 2), 3 * (N - 2) + 2, 3 * (N - 1), 3 * (N - 1) + 2]),
        np.array([
            3 * (N - 2) + 1, 3 * (N - 2) + 2, 3 * (N - 1) + 1, 3 * (N - 1) + 2
        ]),
        np.array([3 * (N - 1) + 2]),
        np.arange(0, 3 * (N - 2), 3),
        np.arange(3, 3 * (N - 1), 3),
        np.arange(6, 3 * N, 3),
        np.arange(4, 3 * (N - 1) + 1, 3),
        np.arange(2, 3 * (N - 2) + 2, 3),
        np.arange(5, 3 * (N - 1) + 2, 3),
        np.arange(8, 3 * N + 2, 3),
        np.arange(3, 3 * (N - 1), 3),
        np.arange(1, 3 * (N - 2) + 1, 3),
        np.arange(4, 3 * (N - 1) + 1, 3),
        np.arange(7, 3 * N + 1, 3),
        np.arange(2, 3 * (N - 2) + 2, 3),
        np.arange(5, 3 * (N - 1) + 2, 3),
        np.arange(8, 3 * N + 2, 3),
        np.arange(2, 3 * (N - 2) + 2, 3),
        np.arange(5, 3 * (N - 1) + 2, 3),
        np.arange(8, 3 * N + 2, 3),
        np.arange(3, 3 * (N - 1), 3),
        np.arange(4, 3 * (N - 1) + 1, 3)
    ]).astype(np.int32)

    dF = np.concatenate([
        np.array([dctct_phin[0], dctct_phin[2], dctct_phin[1], dctct_phin[3]]),
        np.array([dctct_phip[0], dctct_phip[2], dctct_phip[1], dctct_phip[3]]),
        np.ones(1),
        np.array([dctct_phin[4], dctct_phin[6], dctct_phin[5], dctct_phin[7]]),
        np.array([dctct_phip[4], dctct_phip[6], dctct_phip[5], dctct_phip[7]]),
        np.ones(1),
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
    resid = np.concatenate(
        [np.array([ctct_0_phi]), pois,
         np.array([ctct_L_phi])])

    return resid


@jit
def comp_F_eq_deriv(cell: PVCell, bound: Boundary, pot: Potentials) -> Array:

    N = cell.Eg.size
    dpois_phi_, dpois_phi__, dpois_phi___ = poisson.pois_deriv_eq(cell, pot)

    row = np.concatenate([
        np.array([0]),
        np.arange(1, N - 1),
        np.arange(1, N - 1),
        np.arange(1, N - 1),
        np.array([N - 1])
    ]).astype(np.int32)

    col = np.concatenate([
        np.array([0]),
        np.arange(0, N - 2),
        np.arange(1, N - 1),
        np.arange(2, N),
        np.array([N - 1])
    ]).astype(np.int32)

    dFeq = np.concatenate(
        [np.array([1]), dpois_phi_, dpois_phi__, dpois_phi___,
         np.array([1])])

    spFeq = linalg.coo2sparse(row, col, dFeq, N)

    return spFeq
