from deltapv import objects, util
from jax import numpy as np
from typing import Tuple

PVCell = objects.PVCell
Potentials = objects.Potentials
Array = util.Array
f64 = util.f64


def Jn(cell: PVCell, pot: Potentials) -> Array:

    phi = pot.phi
    phi_n = pot.phi_n
    phi_p = pot.phi_p

    psi_n = cell.Chi + np.log(cell.Nc) + phi
    Dpsin = -np.diff(psi_n)

    around_zero = np.abs(Dpsin) < 1e-5

    fm = np.diff(np.exp(phi_n))

    numerator = (1 - around_zero) * Dpsin + around_zero
    denominator = (1 - around_zero) * (np.exp(Dpsin) - 1) + around_zero * (
        1 + 0.5 * Dpsin + Dpsin**2 / 6.)
    Dpsin_Dexppsin = np.exp(psi_n[:-1]) * numerator / denominator

    return cell.mn[:-1] * Dpsin_Dexppsin * fm / cell.dgrid


def Jn_deriv(cell: PVCell,
             pot: Potentials) -> Tuple[Array, Array, Array, Array]:

    phi = pot.phi
    phi_n = pot.phi_n
    phi_p = pot.phi_p

    psi_n = cell.Chi + np.log(cell.Nc) + phi
    Dpsin = -np.diff(psi_n)

    around_zero = np.abs(Dpsin) < 1e-5

    fm = np.diff(np.exp(phi_n))

    numerator = (1 - around_zero) * Dpsin + around_zero
    denominator = (1 - around_zero) * (np.exp(Dpsin) - 1) + around_zero * (
        1 + 0.5 * Dpsin + Dpsin**2 / 6.)
    Dpsin_Dexppsin = np.exp(psi_n[:-1]) * numerator / denominator

    fm_deriv_maindiag = -np.exp(phi_n[:-1])
    fm_deriv_upperdiag = np.exp(phi_n[1:])

    numerator2 = (1 - around_zero) * (
        -Dpsin + np.exp(Dpsin) - 1) + around_zero * (
            -3 + psi_n[:-1] + psi_n[1:] + 2 * psi_n[:-1] * psi_n[1:] -
            psi_n[:-1]**2 - psi_n[1:]**2)
    denominator2 = (1 - around_zero) * (np.exp(Dpsin) - 1)**2 + around_zero * (
        1 + 0.5 * psi_n[:-1] - 0.5 * psi_n[1:] - 1 / 3. * psi_n[:-1] *
        psi_n[1:] + 1 / 6. * psi_n[:-1]**2 + 1 / 6. * psi_n[1:]**2)**2
    numerator3 = (1 - around_zero) * (-np.exp(Dpsin) + 1 +
                                      Dpsin * np.exp(Dpsin)) + around_zero * (
                                          -3 - 2 * psi_n[:-1] + 2 * psi_n[1:])
    denominator3 = (1 - around_zero) * (np.exp(Dpsin) - 1)**2 + around_zero * (
        1 + 0.5 * psi_n[:-1] - 0.5 * psi_n[1:] - 1 / 3. * psi_n[:-1] *
        psi_n[1:] + 1 / 6. * psi_n[:-1]**2 + 1 / 6. * psi_n[1:]**2)**2

    Dpsin_Dexppsin_deriv_maindiag = np.exp(
        psi_n[:-1]) * numerator2 / denominator2
    Dpsin_Dexppsin_deriv_upperdiag = np.exp(
        psi_n[:-1]) * numerator3 / denominator3

    dJn_phin__ = cell.mn[:-1] * Dpsin_Dexppsin / cell.dgrid * fm_deriv_maindiag
    dJn_phin___ = cell.mn[:-1] * Dpsin_Dexppsin / cell.dgrid * fm_deriv_upperdiag

    dJn_phi__ = cell.mn[:-1] * fm / cell.dgrid * Dpsin_Dexppsin_deriv_maindiag
    dJn_phi___ = cell.mn[:-1] * fm / cell.dgrid * Dpsin_Dexppsin_deriv_upperdiag

    return dJn_phin__, dJn_phin___, dJn_phi__, dJn_phi___


def Jp(cell: PVCell, pot: Potentials) -> Array:

    phi = pot.phi
    phi_n = pot.phi_n
    phi_p = pot.phi_p

    psi_p = cell.Chi + cell.Eg - np.log(cell.Nv) + phi
    Dpsip = -np.diff(psi_p)

    around_zero = np.abs(Dpsip) < 1e-5

    fm = np.diff(np.exp(-phi_p))

    numerator = (1 - around_zero) * Dpsip + around_zero * 1
    denominator = (1 - around_zero) * (np.exp(-Dpsip) - 1) + around_zero * (
        -1 + 0.5 * Dpsip - Dpsip**2 / 6.)
    Dpsip_Dexppsip = np.exp(-psi_p[:-1]) * numerator / denominator

    return cell.mp[:-1] * Dpsip_Dexppsip * fm / cell.dgrid


def Jp_deriv(cell: PVCell,
             pot: Potentials) -> Tuple[Array, Array, Array, Array]:

    phi = pot.phi
    phi_n = pot.phi_n
    phi_p = pot.phi_p

    psi_p = cell.Chi + cell.Eg - np.log(cell.Nv) + phi
    Dpsip = -np.diff(psi_p)

    around_zero = np.abs(Dpsip) < 1e-5

    fm = np.diff(np.exp(-phi_p))

    numerator = (1 - around_zero) * Dpsip + around_zero * 1
    denominator = (1 - around_zero) * (np.exp(-Dpsip) - 1) + around_zero * (
        -1 + 0.5 * Dpsip - Dpsip**2 / 6.)
    Dpsip_Dexppsip = np.exp(-psi_p[:-1]) * numerator / denominator

    fm_deriv_maindiag = np.exp(-phi_p[:-1])
    fm_deriv_upperdiag = -np.exp(-phi_p[1:])

    numerator2 = (1 - around_zero) * (
        Dpsip + np.exp(-Dpsip) - 1) + around_zero * (
            -3 + psi_p[:-1] - psi_p[1:] + 2 * psi_p[:-1] * psi_p[1:] -
            psi_p[:-1]**2 - psi_p[1:]**2)
    denominator2 = (1 - around_zero) * (
        np.exp(-Dpsip) - 1)**2 + around_zero * (
            1 - 0.5 * psi_p[:-1] + 0.5 * psi_p[1:] - 1 / 3. * psi_p[:-1] *
            psi_p[1:] + 1 / 6. * psi_p[:-1]**2 + 1 / 6. * psi_p[1:]**2)**2
    numerator3 = (1 - around_zero) * (-np.exp(-Dpsip) + 1 -
                                      Dpsip * np.exp(-Dpsip)) + around_zero * (
                                          -3 + 2 * psi_p[:-1] - 2 * psi_p[1:])
    denominator3 = (1 - around_zero) * (
        np.exp(-Dpsip) - 1)**2 + around_zero * (
            1 - 0.5 * psi_p[:-1] + 0.5 * psi_p[1:] - 1 / 3. * psi_p[:-1] *
            psi_p[1:] + 1 / 6. * psi_p[:-1]**2 + 1 / 6. * psi_p[1:]**2)**2

    Dpsip_Dexppsip_deriv_maindiag = np.exp(
        -psi_p[:-1]) * numerator2 / denominator2
    Dpsip_Dexppsip_deriv_upperdiag = np.exp(
        -psi_p[:-1]) * numerator3 / denominator3

    dJp_phip__ = cell.mp[:-1] * Dpsip_Dexppsip / cell.dgrid * fm_deriv_maindiag
    dJp_phip___ = cell.mp[:-1] * Dpsip_Dexppsip / cell.dgrid * fm_deriv_upperdiag

    dJp_phi__ = cell.mp[:-1] * fm / cell.dgrid * Dpsip_Dexppsip_deriv_maindiag
    dJp_phi___ = cell.mp[:-1] * fm / cell.dgrid * Dpsip_Dexppsip_deriv_upperdiag

    return dJp_phip__, dJp_phip___, dJp_phi__, dJp_phi___


def total_current(cell: PVCell, pot: Potentials) -> Array:

    phi = pot.phi
    phi_n = pot.phi_n
    phi_p = pot.phi_p

    psin0 = cell.Chi[0] + np.log(cell.Nc[0]) + phi[0]
    psin1 = cell.Chi[1] + np.log(cell.Nc[1]) + phi[1]
    psip0 = cell.Chi[0] + cell.Eg[0] - np.log(cell.Nv[0]) + phi[0]
    psip1 = cell.Chi[1] + cell.Eg[1] - np.log(cell.Nv[1]) + phi[1]
    Dpsin = psin0 - psin1
    Dpsip = psip0 - psip1

    around_zero_n = np.abs(Dpsin) < 1e-5
    around_zero_p = np.abs(Dpsip) < 1e-5
    
    fmn = np.exp(phi_n[1]) - np.exp(phi_n[0])
    numerator = (1 - around_zero_n) * Dpsin + around_zero_n * 1
    denominator = (1 - around_zero_n) * (np.exp(Dpsin) - 1) + around_zero_n * (
        1 + 0.5 * Dpsin + 1 / 6. * Dpsin**2)
    Dpsin_Dexppsin = np.exp(psin0) * numerator / denominator
    dfmn_dphin0 = -np.exp(phi_n[0])
    dfmn_dphin1 = np.exp(phi_n[1])
    numerator2 = (1 - around_zero_n) * (
        -Dpsin + np.exp(Dpsin) - 1) + around_zero_n * (
            -3 + psin0 + psin1 + 2 * psin0 * psin1 - psin0**2 - psin1**2)
    denominator2 = (1 - around_zero_n) * (
        np.exp(Dpsin) - 1)**2 + around_zero_n * (
            1 + 0.5 * psin0 - 0.5 * psin1 - 1 / 3. * psin0 * psin1 +
            1 / 6. * psin0**2 + 1 / 6. * psin1**2)**2
    numerator3 = (1 - around_zero_n) * (
        -np.exp(Dpsin) + 1 +
        Dpsin * np.exp(Dpsin)) + around_zero_n * (-3 - 2 * psin0 + 2 * psin1)
    denominator3 = (1 - around_zero_n) * (
        np.exp(Dpsin) - 1)**2 + around_zero_n * (
            1 + 0.5 * psin0 - 0.5 * psin1 - 1 / 3. * psin0 * psin1 +
            1 / 6. * psin0**2 + 1 / 6. * psin1**2)**2
    Dpsin_Dexppsin_dpsin0 = np.exp(psin0) * numerator2 / denominator2
    Dpsin_Dexppsin_dpsin1 = np.exp(psin0) * numerator3 / denominator3

    fmp = np.exp(-phi_p[1]) - np.exp(-phi_p[0])
    _numerator = (1 - around_zero_p) * Dpsip + around_zero_p * 1
    _denominator = (1 - around_zero_p) * (np.exp(
        -Dpsip) - 1) + around_zero_p * (-1 + 0.5 * Dpsip - 1 / 6. * Dpsip**2)
    Dpsip_Dexppsip = np.exp(-psip0) * _numerator / _denominator
    dfmp_dphip0 = np.exp(-phi_p[0])
    dfmp_dphip1 = -np.exp(-phi_p[1])
    _numerator2 = (1 - around_zero_p) * (
        Dpsip + np.exp(-Dpsip) - 1) + around_zero_p * (
            -3 + psip0 - psip1 + 2 * psip0 * psip1 - psip0**2 - psip1**2)
    _denominator2 = (1 - around_zero_p) * (
        np.exp(-Dpsip) - 1)**2 + around_zero_p * (
            1 - 0.5 * psip0 + 0.5 * psip1 - 1 / 3. * psip0 * psip1 +
            1 / 6. * psip0**2 + 1 / 6. * psip1**2)**2
    _numerator3 = (1 - around_zero_p) * (
        -np.exp(-Dpsip) + 1 -
        Dpsip * np.exp(-Dpsip)) + around_zero_p * (-3 + 2 * psip0 - 2 * psip1)
    _denominator3 = (1 - around_zero_p) * (
        np.exp(-Dpsip) - 1)**2 + around_zero_p * (
            1 - 0.5 * psip0 + 0.5 * psip1 - 1 / 3. * psip0 * psip1 +
            1 / 6. * psip0**2 + 1 / 6. * psip1**2)**2
    Dpsip_Dexppsip_dpsip0 = np.exp(-psip0) * _numerator2 / _denominator2
    Dpsip_Dexppsip_dpsip1 = np.exp(-psip0) * _numerator3 / _denominator3

    Fcurrent = cell.mn[0] * Dpsin_Dexppsin * fmn / cell.dgrid[0] + cell.mp[
        0] * Dpsip_Dexppsip * fmp / cell.dgrid[0]

    return Fcurrent


def total_current_deriv(cell: PVCell, pot: Potentials) -> dict:

    phi = pot.phi
    phi_n = pot.phi_n
    phi_p = pot.phi_p

    psin0 = cell.Chi[0] + np.log(cell.Nc[0]) + phi[0]
    psin1 = cell.Chi[1] + np.log(cell.Nc[1]) + phi[1]
    psip0 = cell.Chi[0] + cell.Eg[0] - np.log(cell.Nv[0]) + phi[0]
    psip1 = cell.Chi[1] + cell.Eg[1] - np.log(cell.Nv[1]) + phi[1]
    Dpsin = psin0 - psin1
    Dpsip = psip0 - psip1

    around_zero_n = np.abs(Dpsin) < 1e-5
    around_zero_p = np.abs(Dpsip) < 1e-5

    fmn = np.exp(phi_n[1]) - np.exp(phi_n[0])
    numerator = (1 - around_zero_n) * Dpsin + around_zero_n * 1
    denominator = (1 - around_zero_n) * (np.exp(Dpsin) - 1) + around_zero_n * (
        1 + 0.5 * Dpsin + 1 / 6. * Dpsin**2)
    Dpsin_Dexppsin = np.exp(psin0) * numerator / denominator
    dfmn_dphin0 = -np.exp(phi_n[0])
    dfmn_dphin1 = np.exp(phi_n[1])
    numerator2 = (1 - around_zero_n) * (
        -Dpsin + np.exp(Dpsin) - 1) + around_zero_n * (
            -3 + psin0 + psin1 + 2 * psin0 * psin1 - psin0**2 - psin1**2)
    denominator2 = (1 - around_zero_n) * (
        np.exp(Dpsin) - 1)**2 + around_zero_n * (
            1 + 0.5 * psin0 - 0.5 * psin1 - 1 / 3. * psin0 * psin1 +
            1 / 6. * psin0**2 + 1 / 6. * psin1**2)**2
    numerator3 = (1 - around_zero_n) * (
        -np.exp(Dpsin) + 1 +
        Dpsin * np.exp(Dpsin)) + around_zero_n * (-3 - 2 * psin0 + 2 * psin1)
    denominator3 = (1 - around_zero_n) * (
        np.exp(Dpsin) - 1)**2 + around_zero_n * (
            1 + 0.5 * psin0 - 0.5 * psin1 - 1 / 3. * psin0 * psin1 +
            1 / 6. * psin0**2 + 1 / 6. * psin1**2)**2
    Dpsin_Dexppsin_dpsin0 = np.exp(psin0) * numerator2 / denominator2
    Dpsin_Dexppsin_dpsin1 = np.exp(psin0) * numerator3 / denominator3

    fmp = np.exp(-phi_p[1]) - np.exp(-phi_p[0])
    _numerator = (1 - around_zero_p) * Dpsip + around_zero_p * 1
    _denominator = (1 - around_zero_p) * (np.exp(
        -Dpsip) - 1) + around_zero_p * (-1 + 0.5 * Dpsip - 1 / 6. * Dpsip**2)
    Dpsip_Dexppsip = np.exp(-psip0) * _numerator / _denominator
    dfmp_dphip0 = np.exp(-phi_p[0])
    dfmp_dphip1 = -np.exp(-phi_p[1])
    _numerator2 = (1 - around_zero_p) * (
        Dpsip + np.exp(-Dpsip) - 1) + around_zero_p * (
            -3 + psip0 - psip1 + 2 * psip0 * psip1 - psip0**2 - psip1**2)
    _denominator2 = (1 - around_zero_p) * (
        np.exp(-Dpsip) - 1)**2 + around_zero_p * (
            1 - 0.5 * psip0 + 0.5 * psip1 - 1 / 3. * psip0 * psip1 +
            1 / 6. * psip0**2 + 1 / 6. * psip1**2)**2
    _numerator3 = (1 - around_zero_p) * (
        -np.exp(-Dpsip) + 1 -
        Dpsip * np.exp(-Dpsip)) + around_zero_p * (-3 + 2 * psip0 - 2 * psip1)
    _denominator3 = (1 - around_zero_p) * (
        np.exp(-Dpsip) - 1)**2 + around_zero_p * (
            1 - 0.5 * psip0 + 0.5 * psip1 - 1 / 3. * psip0 * psip1 +
            1 / 6. * psip0**2 + 1 / 6. * psip1**2)**2
    Dpsip_Dexppsip_dpsip0 = np.exp(-psip0) * _numerator2 / _denominator2
    Dpsip_Dexppsip_dpsip1 = np.exp(-psip0) * _numerator3 / _denominator3

    Fcurrent = cell.mn[0] * Dpsin_Dexppsin * fmn / cell.dgrid[0] + cell.mp[
        0] * Dpsip_Dexppsip * fmp / cell.dgrid[0]

    deriv = {}

    deriv["dChi0"] = cell.mn[0] * fmn / cell.dgrid[
        0] * Dpsin_Dexppsin_dpsin0 + cell.mp[0] * fmp / cell.dgrid[
            0] * Dpsip_Dexppsip_dpsip0
    deriv["dChi1"] = cell.mn[0] * fmn / cell.dgrid[
        0] * Dpsin_Dexppsin_dpsin1 + cell.mp[0] * fmp / cell.dgrid[
            0] * Dpsip_Dexppsip_dpsip1
    deriv["dEg0"] = cell.mp[0] * fmp / cell.dgrid[0] * Dpsip_Dexppsip_dpsip0
    deriv["dEg1"] = cell.mp[0] * fmp / cell.dgrid[0] * Dpsip_Dexppsip_dpsip1
    deriv["dNc0"] = 1 / cell.Nc[0] * cell.mn[0] * fmn / cell.dgrid[
        0] * Dpsin_Dexppsin_dpsin0
    deriv["dNc1"] = 1 / cell.Nc[1] * cell.mn[0] * fmn / cell.dgrid[
        0] * Dpsin_Dexppsin_dpsin1
    deriv["dNv0"] = -1 / cell.Nv[0] * cell.mp[0] * fmp / cell.dgrid[
        0] * Dpsip_Dexppsip_dpsip0
    deriv["dNv1"] = -1 / cell.Nv[1] * cell.mp[0] * fmp / cell.dgrid[
        0] * Dpsip_Dexppsip_dpsip1
    deriv["dmn0"] = Dpsin_Dexppsin * fmn / cell.dgrid[0]
    deriv["dmp0"] = Dpsip_Dexppsip * fmp / cell.dgrid[0]

    deriv["dphin0"] = cell.mn[0] * Dpsin_Dexppsin / cell.dgrid[0] * dfmn_dphin0
    deriv["dphin1"] = cell.mn[0] * Dpsin_Dexppsin / cell.dgrid[0] * dfmn_dphin1
    deriv["dphip0"] = cell.mp[0] * Dpsip_Dexppsip / cell.dgrid[0] * dfmp_dphip0
    deriv["dphip1"] = cell.mp[0] * Dpsip_Dexppsip / cell.dgrid[0] * dfmp_dphip1
    deriv["dphi0"] = cell.mn[0] * fmn / cell.dgrid[
        0] * Dpsin_Dexppsin_dpsin0 + cell.mp[0] * fmp / cell.dgrid[
            0] * Dpsip_Dexppsip_dpsip0
    deriv["dphi1"] = cell.mn[0] * fmn / cell.dgrid[
        0] * Dpsin_Dexppsin_dpsin1 + cell.mp[0] * fmp / cell.dgrid[
            0] * Dpsip_Dexppsip_dpsip1

    return deriv
