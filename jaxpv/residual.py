from . import e_drift_diffusion as edd
from . import h_drift_diffusion as hdd
from . import boundary_conditions as bc
from . import poisson
from . import splinalg

import jax.numpy as np
from jax import ops, jit


@jit
def F(data, neq_0, neq_L, peq_0, peq_L, phi_n, phi_p, phi):

    ddn = edd.ddn(data, phi_n, phi_p, phi)
    ddp = hdd.ddp(data, phi_n, phi_p, phi)
    pois = poisson.pois(data, phi_n, phi_p, phi)

    ctct_0_phin, ctct_L_phin = bc.contact_phin(data, neq_0, neq_L, phi_n, phi)
    ctct_0_phip, ctct_L_phip = bc.contact_phip(data, peq_0, peq_L, phi_p, phi)

    lenF = 3 + 3 * len(pois) + 3
    result = np.zeros(lenF, dtype=np.float64)
    result = ops.index_update(result, ops.index[0:3],
                              np.array([ctct_0_phin, ctct_0_phip, 0.0]))
    result = ops.index_update(result, ops.index[3:lenF - 5:3], ddn)
    result = ops.index_update(result, ops.index[4:lenF - 4:3], ddp)
    result = ops.index_update(result, ops.index[5:lenF - 3:3], pois)
    result = ops.index_update(result, ops.index[-3:],
                              np.array([ctct_L_phin, ctct_L_phip, 0.0]))

    return result


@jit
def F_deriv(data, neq_0, neq_L, peq_0, peq_L, phi_n, phi_p, phi):

    dde_phin_, dde_phin__, dde_phin___, dde_phip__, dde_phi_, dde_phi__, dde_phi___ = edd.ddn_deriv(
        data, phi_n, phi_p, phi)
    ddp_phin__, ddp_phip_, ddp_phip__, ddp_phip___, ddp_phi_, ddp_phi__, ddp_phi___ = hdd.ddp_deriv(
        data, phi_n, phi_p, phi)
    dpois_phi_, dpois_phi__, dpois_phi___, dpois_dphin__, dpois_dphip__ = poisson.pois_deriv(
        data, phi_n, phi_p, phi)
    dctct_phin = bc.contact_phin_deriv(data, phi_n, phi)
    dctct_phip = bc.contact_phip_deriv(data, phi_p, phi)

    N = phi.size

    row = np.array([0, 0, 0, 0])
    col = np.array([0, 2, 3, 5])
    dF = np.array([dctct_phin[0], dctct_phin[2], dctct_phin[1], dctct_phin[3]])

    row = np.concatenate((row, np.array([1, 1, 1, 1])))
    col = np.concatenate((col, np.array([1, 2, 4, 5])))
    dF = np.concatenate(
        (dF,
         np.array([dctct_phip[0], dctct_phip[2], dctct_phip[1],
                   dctct_phip[3]])))

    row = np.concatenate((row, np.array([2])))
    col = np.concatenate((col, np.array([2])))
    dF = np.concatenate((dF, np.array([1.0])))

    row = np.concatenate(
        (row, np.array([3 * (N - 1), 3 * (N - 1), 3 * (N - 1), 3 * (N - 1)])))
    col = np.concatenate(
        (col,
         np.array([3 * (N - 2), 3 * (N - 2) + 2, 3 * (N - 1),
                   3 * (N - 1) + 2])))
    dF = np.concatenate(
        (dF,
         np.array([dctct_phin[4], dctct_phin[6], dctct_phin[5],
                   dctct_phin[7]])))

    row = np.concatenate((row,
                          np.array([
                              3 * (N - 1) + 1, 3 * (N - 1) + 1,
                              3 * (N - 1) + 1, 3 * (N - 1) + 1
                          ])))
    col = np.concatenate((col,
                          np.array([
                              3 * (N - 2) + 1, 3 * (N - 2) + 2,
                              3 * (N - 1) + 1, 3 * (N - 1) + 2
                          ])))
    dF = np.concatenate(
        (dF,
         np.array([dctct_phip[4], dctct_phip[6], dctct_phip[5],
                   dctct_phip[7]])))

    row = np.concatenate((row, np.array([3 * (N - 1) + 2])))
    col = np.concatenate((col, np.array([3 * (N - 1) + 2])))
    dF = np.concatenate((dF, np.array([1.0])))

    row = np.concatenate((row, np.arange(3, 3 * (N - 1), 3)))
    col = np.concatenate((col, np.arange(0, 3 * (N - 2), 3)))
    dF = np.concatenate((dF, dde_phin_))

    row = np.concatenate((row, np.arange(3, 3 * (N - 1), 3)))
    col = np.concatenate((col, np.arange(3, 3 * (N - 1), 3)))
    dF = np.concatenate((dF, dde_phin__))

    row = np.concatenate((row, np.arange(3, 3 * (N - 1), 3)))
    col = np.concatenate((col, np.arange(6, 3 * N, 3)))
    dF = np.concatenate((dF, dde_phin___))

    row = np.concatenate((row, np.arange(3, 3 * (N - 1), 3)))
    col = np.concatenate((col, np.arange(4, 3 * (N - 1) + 1, 3)))
    dF = np.concatenate((dF, dde_phip__))

    row = np.concatenate((row, np.arange(3, 3 * (N - 1), 3)))
    col = np.concatenate((col, np.arange(2, 3 * (N - 2) + 2, 3)))
    dF = np.concatenate((dF, dde_phi_))

    row = np.concatenate((row, np.arange(3, 3 * (N - 1), 3)))
    col = np.concatenate((col, np.arange(5, 3 * (N - 1) + 2, 3)))
    dF = np.concatenate((dF, dde_phi__))

    row = np.concatenate((row, np.arange(3, 3 * (N - 1), 3)))
    col = np.concatenate((col, np.arange(8, 3 * N + 2, 3)))
    dF = np.concatenate((dF, dde_phi___))

    row = np.concatenate((row, np.arange(4, 3 * (N - 1) + 1, 3)))
    col = np.concatenate((col, np.arange(3, 3 * (N - 1), 3)))
    dF = np.concatenate((dF, ddp_phin__))

    row = np.concatenate((row, np.arange(4, 3 * (N - 1) + 1, 3)))
    col = np.concatenate((col, np.arange(1, 3 * (N - 2) + 1, 3)))
    dF = np.concatenate((dF, ddp_phip_))

    row = np.concatenate((row, np.arange(4, 3 * (N - 1) + 1, 3)))
    col = np.concatenate((col, np.arange(4, 3 * (N - 1) + 1, 3)))
    dF = np.concatenate((dF, ddp_phip__))

    row = np.concatenate((row, np.arange(4, 3 * (N - 1) + 1, 3)))
    col = np.concatenate((col, np.arange(7, 3 * N + 1, 3)))
    dF = np.concatenate((dF, ddp_phip___))

    row = np.concatenate((row, np.arange(4, 3 * (N - 1) + 1, 3)))
    col = np.concatenate((col, np.arange(2, 3 * (N - 2) + 2, 3)))
    dF = np.concatenate((dF, ddp_phi_))

    row = np.concatenate((row, np.arange(4, 3 * (N - 1) + 1, 3)))
    col = np.concatenate((col, np.arange(5, 3 * (N - 1) + 2, 3)))
    dF = np.concatenate((dF, ddp_phi__))

    row = np.concatenate((row, np.arange(4, 3 * (N - 1) + 1, 3)))
    col = np.concatenate((col, np.arange(8, 3 * N + 2, 3)))
    dF = np.concatenate((dF, ddp_phi___))

    row = np.concatenate((row, np.arange(5, 3 * (N - 1) + 2, 3)))
    col = np.concatenate((col, np.arange(2, 3 * (N - 2) + 2, 3)))
    dF = np.concatenate((dF, dpois_phi_))

    row = np.concatenate((row, np.arange(5, 3 * (N - 1) + 2, 3)))
    col = np.concatenate((col, np.arange(5, 3 * (N - 1) + 2, 3)))
    dF = np.concatenate((dF, dpois_phi__))

    row = np.concatenate((row, np.arange(5, 3 * (N - 1) + 2, 3)))
    col = np.concatenate((col, np.arange(8, 3 * N + 2, 3)))
    dF = np.concatenate((dF, dpois_phi___))

    row = np.concatenate((row, np.arange(5, 3 * (N - 1) + 2, 3)))
    col = np.concatenate((col, np.arange(3, 3 * (N - 1), 3)))
    dF = np.concatenate((dF, dpois_dphin__))

    row = np.concatenate((row, np.arange(5, 3 * (N - 1) + 2, 3)))
    col = np.concatenate((col, np.arange(4, 3 * (N - 1) + 1, 3)))
    dF = np.concatenate((dF, dpois_dphip__))

    spF = splinalg.coo2sparse(row, col, dF, 3 * N)  # make sure it is 3N

    return spF


@jit
def F_eq(data, phi_n, phi_p, phi):

    pois = poisson.pois(data, phi_n, phi_p, phi)
    return np.concatenate((np.array([0.]), pois, np.array([0.])))


@jit
def F_eq_deriv(data, phi_n, phi_p, phi):

    N = phi.size
    dpois_phi_, dpois_phi__, dpois_phi___ = poisson.pois_deriv_eq(
        data, phi_n, phi_p, phi)

    row = np.array([0])
    col = np.array([0])
    dFeq = np.array([1.0])

    row = np.concatenate((row, np.arange(1, N - 1, 1)))
    col = np.concatenate((col, np.arange(0, N - 2, 1)))
    dFeq = np.concatenate((dFeq, dpois_phi_))

    row = np.concatenate((row, np.arange(1, N - 1, 1)))
    col = np.concatenate((col, np.arange(1, N - 1, 1)))
    dFeq = np.concatenate((dFeq, dpois_phi__))

    row = np.concatenate((row, np.arange(1, N - 1, 1)))
    col = np.concatenate((col, np.arange(2, N, 1)))
    dFeq = np.concatenate((dFeq, dpois_phi___))

    row = np.concatenate((row, np.array([N - 1])))
    col = np.concatenate((col, np.array([N - 1])))
    dFeq = np.concatenate((dFeq, np.array([1.0])))

    spFeq = splinalg.coo2sparse(row, col, dFeq, N)

    return spFeq
