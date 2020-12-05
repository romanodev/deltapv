from . import e_drift_diffusion
from . import h_drift_diffusion
from . import poisson
from . import boundary_conditions


def F(dgrid, neq_0, neq_L, peq_0, peq_L, phi_n, phi_p, phi, eps, Chi, Eg, Nc,
      Nv, Ndop, mn, mp, Et, tn, tp, Br, Cn, Cp, Snl, Spl, Snr, Spr, G):

    _ddn = ddn(dgrid, phi_n, phi_p, phi, Chi, Eg, Nc, Nv, mn, Et, tn, tp, Br,
               Cn, Cp, G)
    _ddp = ddp(dgrid, phi_n, phi_p, phi, Chi, Eg, Nc, Nv, mp, Et, tn, tp, Br,
               Cn, Cp, G)
    _pois = pois(dgrid, phi_n, phi_p, phi, eps, Chi, Eg, Nc, Nv, Ndop)
    ctct_0_phin, ctct_L_phin = contact_phin(dgrid, neq_0, neq_L, phi_n, phi,
                                            Chi, Nc, mn, Snl, Snr)
    ctct_0_phip, ctct_L_phip = contact_phip(dgrid, peq_0, peq_L, phi_p, phi,
                                            Chi, Eg, Nv, mp, Spl, Spr)

    result = [ctct_0_phin, ctct_0_phip, 0.0]
    for i in range(len(_pois)):
        result.append(_ddn[i])
        result.append(_ddp[i])
        result.append(_pois[i])
    result = result + [ctct_L_phin, ctct_L_phip, 0.0]
    return np.array(result)


def F_deriv(dgrid, neq_0, neq_L, peq_0, peq_L, phi_n, phi_p, phi, eps, Chi, Eg,
            Nc, Nv, Ndop, mn, mp, Et, tn, tp, Br, Cn, Cp, Snl, Spl, Snr, Spr,
            G):
    
    dde_phin_, dde_phin__, dde_phin___, dde_phip__, dde_phi_, dde_phi__, dde_phi___ = ddn_deriv(
        dgrid, phi_n, phi_p, phi, Chi, Eg, Nc, Nv, mn, Et, tn, tp, Br, Cn, Cp,
        G)
    ddp_phin__, ddp_phip_, ddp_phip__, ddp_phip___, ddp_phi_, ddp_phi__, ddp_phi___ = ddp_deriv(
        dgrid, phi_n, phi_p, phi, Chi, Eg, Nc, Nv, mp, Et, tn, tp, Br, Cn, Cp,
        G)
    dpois_phi_, dpois_phi__, dpois_phi___, dpois_dphin__, dpois_dphip__ = pois_deriv(
        dgrid, phi_n, phi_p, phi, eps, Chi, Eg, Nc, Nv)
    dctct_phin = contact_phin_deriv(dgrid, phi_n, phi, Chi, Nc, mn, Snl, Snr)
    dctct_phip = contact_phip_deriv(dgrid, phi_p, phi, Chi, Eg, Nv, mp, Spl,
                                    Spr)

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

    # remove zero elements
    nonzero_idx = dF != 0
    row, col, dF = row[nonzero_idx], col[nonzero_idx], dF[nonzero_idx]

    # sort col elements
    sortcol_idx = np.argsort(col, kind="stable")
    row, col, dF = row[sortcol_idx], col[sortcol_idx], dF[sortcol_idx]

    # sort row elements
    sortrow_idx = np.argsort(row, kind="stable")
    row, col, dF = row[sortrow_idx], col[sortrow_idx], dF[sortrow_idx]

    # create "indptr" for csr format. "data" is "dF", "indices" is "col"
    indptr = np.nonzero(np.diff(np.concatenate([[-1], row, [3 * N]])))[0]

    return dF, col, indptr


def F_eq(dgrid, phi_n, phi_p, phi, eps, Chi, Eg, Nc, Nv, Ndop):
    
    _pois = pois(dgrid, phi_n, phi_p, phi, eps, Chi, Eg, Nc, Nv, Ndop)
    return np.concatenate((np.array([0.0]), _pois, np.array([0.0])))


def F_eq_deriv(dgrid, phi_n, phi_p, phi, eps, Chi, Eg, Nc, Nv):
    
    N = phi.size
    dpois_phi_, dpois_phi__, dpois_phi___ = pois_deriv_eq(
        dgrid, phi_n, phi_p, phi, eps, Chi, Eg, Nc, Nv)

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

    # remove zero elements
    nonzero_idx = dFeq != 0
    row, col, dFeq = row[nonzero_idx], col[nonzero_idx], dFeq[nonzero_idx]

    # sort col elements
    sortcol_idx = np.argsort(col, kind="stable")
    row, col, dFeq = row[sortcol_idx], col[sortcol_idx], dFeq[sortcol_idx]

    # sort row elements
    sortrow_idx = np.argsort(row, kind="stable")
    row, col, dFeq = row[sortrow_idx], col[sortrow_idx], dFeq[sortrow_idx]

    # create "indptr" for csr format. "data" is "dFeq", "indices" is "col"
    indptr = np.nonzero(np.diff(np.concatenate([np.array([-1]), row, np.array([N])])))[0]
    
    return dFeq, col, indptr
