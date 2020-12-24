from jaxpv import objects, drift_diffusion as dd, boundary_conditions as bc, poisson, splinalg, util
from jax import numpy as np, ops, jit

PVCell = objects.PVCell
Potentials = objects.Potentials
Boundary = objects.Boundary
Array = util.Array
f64 = util.f64


@jit
def F(cell: PVCell, bound: Boundary, pot: Potentials) -> Array:

    ddn = dd.ddn(cell, pot)
    ddp = dd.ddp(cell, pot)
    pois = poisson.pois(cell, pot)

    ctct_0_phin, ctct_L_phin = bc.contact_phin(cell, bound, pot)
    ctct_0_phip, ctct_L_phip = bc.contact_phip(cell, bound, pot)

    lenF = 3 + 3 * len(pois) + 3
    result = np.zeros(lenF, dtype=np.float64)
    result = ops.index_update(result, ops.index[:3],
                              np.array([ctct_0_phin, ctct_0_phip, 0]))
    result = ops.index_update(result, ops.index[3:lenF - 5:3], ddn)
    result = ops.index_update(result, ops.index[4:lenF - 4:3], ddp)
    result = ops.index_update(result, ops.index[5:lenF - 3:3], pois)
    result = ops.index_update(result, ops.index[-3:],
                              np.array([ctct_L_phin, ctct_L_phip, 0]))

    return result


@jit
def F_deriv(cell: PVCell, bound: Boundary, pot: Potentials) -> Array:

    dde_phin_, dde_phin__, dde_phin___, dde_phip__, dde_phi_, dde_phi__, dde_phi___ = dd.ddn_deriv(
        cell, pot)
    ddp_phin__, ddp_phip_, ddp_phip__, ddp_phip___, ddp_phi_, ddp_phi__, ddp_phi___ = dd.ddp_deriv(
        cell, pot)
    dpois_phi_, dpois_phi__, dpois_phi___, dpois_dphin__, dpois_dphip__ = poisson.pois_deriv(
        cell, pot)
    dctct_phin = bc.contact_phin_deriv(cell, pot)
    dctct_phip = bc.contact_phip_deriv(cell, pot)

    N = cell.grid.size

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

    spF = splinalg.coo2sparse(row, col, dF, 3 * N)

    return spF


@jit
def F_eq(cell: PVCell, pot: Potentials) -> Array:

    pois = poisson.pois(cell, pot)
    return np.pad(pois, pad_width=1)


@jit
def F_eq_deriv(cell: PVCell, pot: Potentials) -> Array:

    N = cell.grid.size
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

    spFeq = splinalg.coo2sparse(row, col, dFeq, N)

    return spFeq
