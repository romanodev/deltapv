from deltapv import objects, util
from jax import numpy as jnp
from typing import Tuple

PVCell = objects.PVCell
Potentials = objects.Potentials
Array = util.Array
f64 = util.f64


def Jn(cell: PVCell, pot: Potentials) -> Array:

    phi = pot.phi
    phi_n = pot.phi_n
    phi_n0 = phi_n[:-1]
    phi_n1 = phi_n[1:]
    fm = jnp.exp(phi_n1) - jnp.exp(phi_n0)
    mn0 = cell.mn[:-1]

    psi_n = cell.Chi + jnp.log(cell.Nc) + phi
    psi_n0 = psi_n[:-1]
    psi_n1 = psi_n[1:]
    Dpsin = psi_n0 - psi_n1

    # AVOID NANS
    Dpsin_norm = jnp.where(jnp.abs(Dpsin) < 1e-5, 1e-5, Dpsin)
    Dpsin_taylor = jnp.clip(Dpsin, -1e-5, 1e-5)

    Dpsin_Dexppsin = jnp.where(
        jnp.abs(Dpsin) < 1e-5,
        jnp.exp(psi_n0) / (1 + Dpsin_taylor / 2 + Dpsin_taylor**2 / 6),
        jnp.exp(psi_n0) * Dpsin_norm / (jnp.exp(Dpsin_norm) - 1))

    return mn0 * Dpsin_Dexppsin * fm / cell.dgrid


def Jp(cell: PVCell, pot: Potentials) -> Array:

    phi = pot.phi
    phi_p = pot.phi_p
    phi_p0 = phi_p[:-1]
    phi_p1 = phi_p[1:]
    fm = jnp.exp(-phi_p1) - jnp.exp(-phi_p0)
    mp0 = cell.mp[:-1]

    psi_p = cell.Chi + cell.Eg - jnp.log(cell.Nv) + phi
    psi_p0 = psi_p[:-1]
    psi_p1 = psi_p[1:]
    Dpsip = psi_p0 - psi_p1

    # AVOID NANS
    Dpsip_norm = jnp.where(jnp.abs(Dpsip) < 1e-5, 1e-5, Dpsip)
    Dpsip_taylor = jnp.clip(Dpsip, -1e-5, 1e-5)

    Dpsip_Dexppsip = jnp.where(
        jnp.abs(Dpsip) < 1e-5,
        jnp.exp(-psi_p0) / (-1 + Dpsip_taylor / 2 - Dpsip_taylor**2 / 6),
        jnp.exp(-psi_p0) * Dpsip_norm / (jnp.exp(-Dpsip_norm) - 1))

    return mp0 * Dpsip_Dexppsip * fm / cell.dgrid


def Jn_deriv(cell: PVCell, pot: Potentials) -> Tuple[Array, Array,
                                                     Array, Array]:

    phi = pot.phi
    phi_n = pot.phi_n
    phi_n0 = phi_n[:-1]
    phi_n1 = phi_n[1:]
    fm = jnp.exp(phi_n1) - jnp.exp(phi_n0)
    mn0 = cell.mn[:-1]

    psi_n = cell.Chi + jnp.log(cell.Nc) + phi
    psi_n0 = psi_n[:-1]
    psi_n1 = psi_n[1:]
    Dpsin = psi_n0 - psi_n1

    # AVOID NANS
    Dpsin_norm = jnp.where(jnp.abs(Dpsin) < 1e-5, 1e-5, Dpsin)
    Dpsin_taylor = jnp.clip(Dpsin, -1e-5, 1e-5)

    expDpsin = jnp.exp(Dpsin)
    exppsi_n0 = jnp.exp(psi_n0)

    Q = jnp.where(
        jnp.abs(Dpsin) < 1e-5,
        jnp.exp(psi_n0) / (1 + Dpsin_taylor / 2 + Dpsin_taylor**2 / 6),
        jnp.exp(psi_n0) * Dpsin_norm / (jnp.exp(Dpsin_norm) - 1))

    DQDphi0_norm = exppsi_n0 / (expDpsin - 1) * (Dpsin + 1 - Dpsin * expDpsin /
                                                 (expDpsin - 1))
    DQDphi1_norm = exppsi_n0 / (expDpsin - 1) * (-1 + Dpsin * expDpsin /
                                                 (expDpsin - 1))

    DQDphi0_taylor = 6 * exppsi_n0 * (
        3 + psi_n0 + psi_n0**2 - psi_n1 - 2 * psi_n0 * psi_n1 +
        psi_n1**2) / (6 + 3 * psi_n0 + psi_n0**2 - 3 * psi_n1 -
                      2 * psi_n0 * psi_n1 + psi_n1**2)**2
    DQDphi1_taylor = -exppsi_n0 * (-1 / 2 - Dpsin / 3) / (1 + Dpsin / 2 +
                                                          Dpsin**2 / 6)**2

    DfmDphi_n0 = -jnp.exp(phi_n0)
    DfmDphi_n1 = jnp.exp(phi_n1)

    DJnDphi0 = mn0 * fm / cell.dgrid * jnp.where(
        jnp.abs(Dpsin) < 1e-5, DQDphi0_taylor, DQDphi0_norm)

    DJnDphi1 = mn0 * fm / cell.dgrid * jnp.where(
        jnp.abs(Dpsin) < 1e-5, DQDphi1_taylor, DQDphi1_norm)

    DJnDphi_n0 = mn0 * Q / cell.dgrid * DfmDphi_n0
    DJnDphi_n1 = mn0 * Q / cell.dgrid * DfmDphi_n1

    return DJnDphi_n0, DJnDphi_n1, DJnDphi0, DJnDphi1


def Jp_deriv(cell: PVCell, pot: Potentials) -> Tuple[Array, Array,
                                                     Array, Array]:

    phi = pot.phi
    phi_p = pot.phi_p
    phi_p0 = phi_p[:-1]
    phi_p1 = phi_p[1:]
    fm = jnp.exp(-phi_p1) - jnp.exp(-phi_p0)
    mp0 = cell.mp[:-1]

    psi_p = cell.Chi + cell.Eg - jnp.log(cell.Nv) + phi
    psi_p0 = psi_p[:-1]
    psi_p1 = psi_p[1:]
    Dpsip = psi_p0 - psi_p1

    # AVOID NANS
    Dpsip_norm = jnp.where(jnp.abs(Dpsip) < 1e-5, 1e-5, Dpsip)
    Dpsip_taylor = jnp.clip(Dpsip, -1e-5, 1e-5)

    _expDpsip = jnp.exp(Dpsip)
    expmpsi_p0 = jnp.exp(-psi_p0)

    Q = jnp.where(
        jnp.abs(Dpsip) < 1e-5,
        expmpsi_p0 / (-1 + Dpsip_taylor / 2 - Dpsip_taylor**2 / 6),
        expmpsi_p0 * Dpsip_norm / (jnp.exp(-Dpsip_norm) - 1))

    DQDphi0_norm = (jnp.exp(psi_p1) - jnp.exp(psi_p0) *
                    (1 + psi_p1 - psi_p0)) / (jnp.exp(psi_p0) -
                                              jnp.exp(psi_p1))**2
    DQDphi1_norm = (jnp.exp(psi_p0) - jnp.exp(psi_p1) *
                    (1 + psi_p0 - psi_p1)) / (jnp.exp(psi_p0) -
                                              jnp.exp(psi_p1))**2

    DQDphi0_taylor = -expmpsi_p0 / (
        -1 + Dpsip / 2 - Dpsip**2 / 6) - expmpsi_p0 * (1 / 2 - Dpsip / 3) / (
            -1 + Dpsip / 2 - Dpsip**2 / 6)**2
    DQDphi1_taylor = -expmpsi_p0 * (-1 / 2 + Dpsip / 3) / (-1 + Dpsip / 2 -
                                                           Dpsip**2 / 6)**2

    DfmDphi_p0 = jnp.exp(-phi_p0)
    DfmDphi_p1 = -jnp.exp(-phi_p1)

    DJpDphi0 = mp0 * fm / cell.dgrid * jnp.where(
        jnp.abs(Dpsip) < 1e-5, DQDphi0_taylor, DQDphi0_norm)

    DJpDphi1 = mp0 * fm / cell.dgrid * jnp.where(
        jnp.abs(Dpsip) < 1e-5, DQDphi1_taylor, DQDphi1_norm)

    DJpDphi_p0 = mp0 * Q / cell.dgrid * DfmDphi_p0
    DJpDphi_p1 = mp0 * Q / cell.dgrid * DfmDphi_p1

    return DJpDphi_p0, DJpDphi_p1, DJpDphi0, DJpDphi1


def total_current(cell: PVCell, pot: Potentials) -> f64:

    Jtotal = Jn(cell, pot) + Jp(cell, pot)
    curr = jnp.mean(Jtotal)

    return curr


def total_current_old(cell: PVCell, pot: Potentials) -> f64:

    phi = pot.phi
    phi_n = pot.phi_n
    phi_p = pot.phi_p

    psin0 = cell.Chi[0] + jnp.log(cell.Nc[0]) + phi[0]
    psin1 = cell.Chi[1] + jnp.log(cell.Nc[1]) + phi[1]
    psip0 = cell.Chi[0] + cell.Eg[0] - jnp.log(cell.Nv[0]) + phi[0]
    psip1 = cell.Chi[1] + cell.Eg[1] - jnp.log(cell.Nv[1]) + phi[1]
    Dpsin = psin0 - psin1
    Dpsip = psip0 - psip1

    around_zero_n = jnp.abs(Dpsin) < 1e-5
    around_zero_p = jnp.abs(Dpsip) < 1e-5

    fmn = jnp.exp(phi_n[1]) - jnp.exp(phi_n[0])
    numerator = (1 - around_zero_n) * Dpsin + around_zero_n * 1
    denominator = (1 - around_zero_n) * (jnp.exp(Dpsin) - 1) + around_zero_n\
        * (1 + 0.5 * Dpsin + 1 / 6. * Dpsin**2)
    Dpsin_Dexppsin = jnp.exp(psin0) * numerator / denominator
    _dfmn_dphin0 = -jnp.exp(phi_n[0])
    _dfmn_dphin1 = jnp.exp(phi_n[1])
    numerator2 = (1 - around_zero_n) * (
        -Dpsin + jnp.exp(Dpsin) - 1) + around_zero_n * (
            -3 + psin0 + psin1 + 2 * psin0 * psin1 - psin0**2 - psin1**2)
    denominator2 = (1 - around_zero_n) * (
        jnp.exp(Dpsin) - 1)**2 + around_zero_n * (
            1 + 0.5 * psin0 - 0.5 * psin1 - 1 / 3. * psin0 * psin1 +
            1 / 6. * psin0**2 + 1 / 6. * psin1**2)**2
    numerator3 = (1 - around_zero_n) * (
        -jnp.exp(Dpsin) + 1 +
        Dpsin * jnp.exp(Dpsin)) + around_zero_n * (-3 - 2 * psin0 + 2 * psin1)
    denominator3 = (1 - around_zero_n) * (
        jnp.exp(Dpsin) - 1)**2 + around_zero_n * (
            1 + 0.5 * psin0 - 0.5 * psin1 - 1 / 3. * psin0 * psin1 +
            1 / 6. * psin0**2 + 1 / 6. * psin1**2)**2
    _Dpsin_Dexppsin_dpsin0 = jnp.exp(psin0) * numerator2 / denominator2
    _Dpsin_Dexppsin_dpsin1 = jnp.exp(psin0) * numerator3 / denominator3

    fmp = jnp.exp(-phi_p[1]) - jnp.exp(-phi_p[0])
    _numerator = (1 - around_zero_p) * Dpsip + around_zero_p * 1
    _denominator = (1 - around_zero_p) * (jnp.exp(
        -Dpsip) - 1) + around_zero_p * (-1 + 0.5 * Dpsip - 1 / 6. * Dpsip**2)
    Dpsip_Dexppsip = jnp.exp(-psip0) * _numerator / _denominator
    _dfmp_dphip0 = jnp.exp(-phi_p[0])
    _dfmp_dphip1 = -jnp.exp(-phi_p[1])
    _numerator2 = (1 - around_zero_p) * (
        Dpsip + jnp.exp(-Dpsip) - 1) + around_zero_p * (
            -3 + psip0 - psip1 + 2 * psip0 * psip1 - psip0**2 - psip1**2)
    _denominator2 = (1 - around_zero_p) * (
        jnp.exp(-Dpsip) - 1)**2 + around_zero_p * (
            1 - 0.5 * psip0 + 0.5 * psip1 - 1 / 3. * psip0 * psip1 +
            1 / 6. * psip0**2 + 1 / 6. * psip1**2)**2
    _numerator3 = (1 - around_zero_p) * (
        -jnp.exp(-Dpsip) + 1 -
        Dpsip * jnp.exp(-Dpsip)) + around_zero_p * (-3 + 2 * psip0 - 2 * psip1)
    _denominator3 = (1 - around_zero_p) * (
        jnp.exp(-Dpsip) - 1)**2 + around_zero_p * (
            1 - 0.5 * psip0 + 0.5 * psip1 - 1 / 3. * psip0 * psip1 +
            1 / 6. * psip0**2 + 1 / 6. * psip1**2)**2
    _Dpsip_Dexppsip_dpsip0 = jnp.exp(-psip0) * _numerator2 / _denominator2
    _Dpsip_Dexppsip_dpsip1 = jnp.exp(-psip0) * _numerator3 / _denominator3

    Fcurrent = cell.mn[0] * Dpsin_Dexppsin * fmn / cell.dgrid[0] + cell.mp[
        0] * Dpsip_Dexppsip * fmp / cell.dgrid[0]

    return Fcurrent


def total_current_deriv(cell: PVCell, pot: Potentials) -> dict:

    phi = pot.phi
    phi_n = pot.phi_n
    phi_p = pot.phi_p

    psin0 = cell.Chi[0] + jnp.log(cell.Nc[0]) + phi[0]
    psin1 = cell.Chi[1] + jnp.log(cell.Nc[1]) + phi[1]
    psip0 = cell.Chi[0] + cell.Eg[0] - jnp.log(cell.Nv[0]) + phi[0]
    psip1 = cell.Chi[1] + cell.Eg[1] - jnp.log(cell.Nv[1]) + phi[1]
    Dpsin = psin0 - psin1
    Dpsip = psip0 - psip1

    around_zero_n = jnp.abs(Dpsin) < 1e-5
    around_zero_p = jnp.abs(Dpsip) < 1e-5

    fmn = jnp.exp(phi_n[1]) - jnp.exp(phi_n[0])
    numerator = (1 - around_zero_n) * Dpsin + around_zero_n * 1
    denominator = (1 - around_zero_n) * (jnp.exp(Dpsin) - 1) + around_zero_n\
        * (1 + 0.5 * Dpsin + 1 / 6. * Dpsin**2)
    Dpsin_Dexppsin = jnp.exp(psin0) * numerator / denominator
    dfmn_dphin0 = -jnp.exp(phi_n[0])
    dfmn_dphin1 = jnp.exp(phi_n[1])
    numerator2 = (1 - around_zero_n) * (
        -Dpsin + jnp.exp(Dpsin) - 1) + around_zero_n * (
            -3 + psin0 + psin1 + 2 * psin0 * psin1 - psin0**2 - psin1**2)
    denominator2 = (1 - around_zero_n) * (
        jnp.exp(Dpsin) - 1)**2 + around_zero_n * (
            1 + 0.5 * psin0 - 0.5 * psin1 - 1 / 3. * psin0 * psin1 +
            1 / 6. * psin0**2 + 1 / 6. * psin1**2)**2
    numerator3 = (1 - around_zero_n) * (
        -jnp.exp(Dpsin) + 1 +
        Dpsin * jnp.exp(Dpsin)) + around_zero_n * (-3 - 2 * psin0 + 2 * psin1)
    denominator3 = (1 - around_zero_n) * (
        jnp.exp(Dpsin) - 1)**2 + around_zero_n * (
            1 + 0.5 * psin0 - 0.5 * psin1 - 1 / 3. * psin0 * psin1 +
            1 / 6. * psin0**2 + 1 / 6. * psin1**2)**2
    Dpsin_Dexppsin_dpsin0 = jnp.exp(psin0) * numerator2 / denominator2
    Dpsin_Dexppsin_dpsin1 = jnp.exp(psin0) * numerator3 / denominator3

    fmp = jnp.exp(-phi_p[1]) - jnp.exp(-phi_p[0])
    _numerator = (1 - around_zero_p) * Dpsip + around_zero_p * 1
    _denominator = (1 - around_zero_p) * (jnp.exp(
        -Dpsip) - 1) + around_zero_p * (-1 + 0.5 * Dpsip - 1 / 6. * Dpsip**2)
    Dpsip_Dexppsip = jnp.exp(-psip0) * _numerator / _denominator
    dfmp_dphip0 = jnp.exp(-phi_p[0])
    dfmp_dphip1 = -jnp.exp(-phi_p[1])
    _numerator2 = (1 - around_zero_p) * (
        Dpsip + jnp.exp(-Dpsip) - 1) + around_zero_p * (
            -3 + psip0 - psip1 + 2 * psip0 * psip1 - psip0**2 - psip1**2)
    _denominator2 = (1 - around_zero_p) * (
        jnp.exp(-Dpsip) - 1)**2 + around_zero_p * (
            1 - 0.5 * psip0 + 0.5 * psip1 - 1 / 3. * psip0 * psip1 +
            1 / 6. * psip0**2 + 1 / 6. * psip1**2)**2
    _numerator3 = (1 - around_zero_p) * (
        -jnp.exp(-Dpsip) + 1 -
        Dpsip * jnp.exp(-Dpsip)) + around_zero_p * (-3 + 2 * psip0 - 2 * psip1)
    _denominator3 = (1 - around_zero_p) * (
        jnp.exp(-Dpsip) - 1)**2 + around_zero_p * (
            1 - 0.5 * psip0 + 0.5 * psip1 - 1 / 3. * psip0 * psip1 +
            1 / 6. * psip0**2 + 1 / 6. * psip1**2)**2
    Dpsip_Dexppsip_dpsip0 = jnp.exp(-psip0) * _numerator2 / _denominator2
    Dpsip_Dexppsip_dpsip1 = jnp.exp(-psip0) * _numerator3 / _denominator3

    _Fcurrent = cell.mn[0] * Dpsin_Dexppsin * fmn / cell.dgrid[0] + cell.mp[
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
