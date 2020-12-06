def Jn(data, phi_n, phi):

    dgrid = data["dgrid"]
    Chi = data["Chi"]
    Nc = data["Nc"]
    mn = data["mn"]
    psi_n = Chi + np.log(Nc) + phi
    Dpsin = -np.diff(psi_n)
    thr = 1e-5
    around_zero = 0.5 * (np.tanh(1e50 *
                                 (Dpsin + thr)) - np.tanh(1e50 *
                                                          (Dpsin - thr)))

    fm = np.diff(np.exp(phi_n))

    numerator = (1 - around_zero) * Dpsin + around_zero
    denominator = (1 - around_zero) * (np.exp(Dpsin) - 1) + around_zero * (
        1 + 0.5 * Dpsin + Dpsin**2 / 6.0)
    Dpsin_Dexppsin = np.exp(psi_n[:-1]) * numerator / denominator

    return mn[:-1] * Dpsin_Dexppsin * fm / dgrid


def Jn_deriv(data, phi_n, phi):

    dgrid = data["dgrid"]
    Chi = data["Chi"]
    Nc = data["Nc"]
    mn = data["mn"]
    psi_n = Chi + np.log(Nc) + phi
    Dpsin = -np.diff(psi_n)
    thr = 1e-5
    around_zero = 0.5 * (np.tanh(1e50 *
                                 (Dpsin + thr)) - np.tanh(1e50 *
                                                          (Dpsin - thr)))

    fm = np.diff(np.exp(phi_n))

    numerator = (1 - around_zero) * Dpsin + around_zero
    denominator = (1 - around_zero) * (np.exp(Dpsin) - 1) + around_zero * (
        1 + 0.5 * Dpsin + Dpsin**2 / 6.0)
    Dpsin_Dexppsin = np.exp(psi_n[:-1]) * numerator / denominator

    fm_deriv_maindiag = -np.exp(phi_n[:-1])
    fm_deriv_upperdiag = np.exp(phi_n[1:])

    numerator2 = (1 - around_zero) * (
        -Dpsin + np.exp(Dpsin) - 1) + around_zero * (
            -3 + psi_n[:-1] + psi_n[1:] + 2 * psi_n[:-1] * psi_n[1:] -
            psi_n[:-1]**2 - psi_n[1:]**2)
    denominator2 = (1 - around_zero) * (np.exp(Dpsin) - 1)**2 + around_zero * (
        1 + 0.5 * psi_n[:-1] - 0.5 * psi_n[1:] - 1 / 3.0 * psi_n[:-1] *
        psi_n[1:] + 1 / 6.0 * psi_n[:-1]**2 + 1 / 6.0 * psi_n[1:]**2)**2
    numerator3 = (1 - around_zero) * (-np.exp(Dpsin) + 1 +
                                      Dpsin * np.exp(Dpsin)) + around_zero * (
                                          -3 - 2 * psi_n[:-1] + 2 * psi_n[1:])
    denominator3 = (1 - around_zero) * (np.exp(Dpsin) - 1)**2 + around_zero * (
        1 + 0.5 * psi_n[:-1] - 0.5 * psi_n[1:] - 1 / 3.0 * psi_n[:-1] *
        psi_n[1:] + 1 / 6.0 * psi_n[:-1]**2 + 1 / 6.0 * psi_n[1:]**2)**2

    Dpsin_Dexppsin_deriv_maindiag = np.exp(
        psi_n[:-1]) * numerator2 / denominator2
    Dpsin_Dexppsin_deriv_upperdiag = np.exp(
        psi_n[:-1]) * numerator3 / denominator3

    dJn_phin__ = mn[:-1] * Dpsin_Dexppsin / dgrid * fm_deriv_maindiag
    dJn_phin___ = mn[:-1] * Dpsin_Dexppsin / dgrid * fm_deriv_upperdiag

    dJn_phi__ = mn[:-1] * fm / dgrid * Dpsin_Dexppsin_deriv_maindiag
    dJn_phi___ = mn[:-1] * fm / dgrid * Dpsin_Dexppsin_deriv_upperdiag

    return dJn_phin__, dJn_phin___, dJn_phi__, dJn_phi___


def Jp(data, phi_p, phi):

    dgrid = data["dgrid"]
    Chi = data["Chi"]
    Eg = data["Eg"]
    Nv = data["Nv"]
    mp = data["mp"]
    psi_p = Chi + Eg - np.log(Nv) + phi
    Dpsip = -np.diff(psi_p)
    thr = 1e-5
    around_zero = 0.5 * (np.tanh(1e50 *
                                 (Dpsip + thr)) - np.tanh(1e50 *
                                                          (Dpsip - thr)))

    fm = np.diff(np.exp(-phi_p))

    numerator = (1 - around_zero) * Dpsip + around_zero * 1
    denominator = (1 - around_zero) * (np.exp(-Dpsip) - 1) + around_zero * (
        -1 + 0.5 * Dpsip - Dpsip**2 / 6.0)
    Dpsip_Dexppsip = np.exp(-psi_p[:-1]) * numerator / denominator

    return mp[:-1] * Dpsip_Dexppsip * fm / dgrid


def Jp_deriv(data, phi_p, phi):

    dgrid = data["dgrid"]
    Chi = data["Chi"]
    Eg = data["Eg"]
    Nv = data["Nv"]
    mp = data["mp"]
    psi_p = Chi + Eg - np.log(Nv) + phi
    Dpsip = -np.diff(psi_p)
    thr = 1e-5
    around_zero = 0.5 * (np.tanh(1e50 *
                                 (Dpsip + thr)) - np.tanh(1e50 *
                                                          (Dpsip - thr)))

    fm = np.diff(np.exp(-phi_p))

    numerator = (1 - around_zero) * Dpsip + around_zero * 1
    denominator = (1 - around_zero) * (np.exp(-Dpsip) - 1) + around_zero * (
        -1 + 0.5 * Dpsip - Dpsip**2 / 6.0)
    Dpsip_Dexppsip = np.exp(-psi_p[:-1]) * numerator / denominator

    fm_deriv_maindiag = np.exp(-phi_p[:-1])
    fm_deriv_upperdiag = -np.exp(-phi_p[1:])

    numerator2 = (1 - around_zero) * (
        Dpsip + np.exp(-Dpsip) - 1) + around_zero * (
            -3 + psi_p[:-1] - psi_p[1:] + 2 * psi_p[:-1] * psi_p[1:] -
            psi_p[:-1]**2 - psi_p[1:]**2)
    denominator2 = (1 - around_zero) * (
        np.exp(-Dpsip) - 1)**2 + around_zero * (
            1 - 0.5 * psi_p[:-1] + 0.5 * psi_p[1:] - 1 / 3.0 * psi_p[:-1] *
            psi_p[1:] + 1 / 6.0 * psi_p[:-1]**2 + 1 / 6.0 * psi_p[1:]**2)**2
    numerator3 = (1 - around_zero) * (-np.exp(-Dpsip) + 1 -
                                      Dpsip * np.exp(-Dpsip)) + around_zero * (
                                          -3 + 2 * psi_p[:-1] - 2 * psi_p[1:])
    denominator3 = (1 - around_zero) * (
        np.exp(-Dpsip) - 1)**2 + around_zero * (
            1 - 0.5 * psi_p[:-1] + 0.5 * psi_p[1:] - 1 / 3.0 * psi_p[:-1] *
            psi_p[1:] + 1 / 6.0 * psi_p[:-1]**2 + 1 / 6.0 * psi_p[1:]**2)**2

    Dpsip_Dexppsip_deriv_maindiag = np.exp(
        -psi_p[:-1]) * numerator2 / denominator2
    Dpsip_Dexppsip_deriv_upperdiag = np.exp(
        -psi_p[:-1]) * numerator3 / denominator3

    dJp_phip__ = mp[:-1] * Dpsip_Dexppsip / dgrid * fm_deriv_maindiag
    dJp_phip___ = mp[:-1] * Dpsip_Dexppsip / dgrid * fm_deriv_upperdiag

    dJp_phi__ = mp[:-1] * fm / dgrid * Dpsip_Dexppsip_deriv_maindiag
    dJp_phi___ = mp[:-1] * fm / dgrid * Dpsip_Dexppsip_deriv_upperdiag

    return dJp_phip__, dJp_phip___, dJp_phi__, dJp_phi___


def total_current(data, phi_n, phi_p, phi):

    dgrid = data["dgrid"]
    Chi = data["Chi"]
    Eg = data["Eg"]
    Nc = data["Nc"]
    Nv = data["Nv"]
    mn = data["mn"]
    mp = data["mp"]
    psin0 = Chi[0] + np.log(Nc[0]) + phi[0]
    psin1 = Chi[1] + np.log(Nc[1]) + phi[1]
    psip0 = Chi[0] + Eg[0] - np.log(Nv[0]) + phi[0]
    psip1 = Chi[1] + Eg[1] - np.log(Nv[1]) + phi[1]
    Dpsin = psin0 - psin1
    Dpsip = psip0 - psip1
    thr = 1e-5
    around_zero_n = 0.5 * (np.tanh(1e50 *
                                   (Dpsin + thr)) - np.tanh(1e50 *
                                                            (Dpsin - thr)))
    around_zero_p = 0.5 * (np.tanh(1e50 *
                                   (Dpsip + thr)) - np.tanh(1e50 *
                                                            (Dpsip - thr)))

    fmn = np.exp(phi_n[1]) - np.exp(phi_n[0])
    numerator = (1 - around_zero_n) * Dpsin + around_zero_n * 1
    denominator = (1 - around_zero_n) * (np.exp(Dpsin) - 1) + around_zero_n * (
        1 + 0.5 * Dpsin + 1 / 6.0 * Dpsin**2)
    Dpsin_Dexppsin = np.exp(psin0) * numerator / denominator
    dfmn_dphin0 = -np.exp(phi_n[0])
    dfmn_dphin1 = np.exp(phi_n[1])
    numerator2 = (1 - around_zero_n) * (
        -Dpsin + np.exp(Dpsin) - 1) + around_zero_n * (
            -3 + psin0 + psin1 + 2 * psin0 * psin1 - psin0**2 - psin1**2)
    denominator2 = (1 - around_zero_n) * (
        np.exp(Dpsin) - 1)**2 + around_zero_n * (
            1 + 0.5 * psin0 - 0.5 * psin1 - 1 / 3.0 * psin0 * psin1 +
            1 / 6.0 * psin0**2 + 1 / 6.0 * psin1**2)**2
    numerator3 = (1 - around_zero_n) * (
        -np.exp(Dpsin) + 1 +
        Dpsin * np.exp(Dpsin)) + around_zero_n * (-3 - 2 * psin0 + 2 * psin1)
    denominator3 = (1 - around_zero_n) * (
        np.exp(Dpsin) - 1)**2 + around_zero_n * (
            1 + 0.5 * psin0 - 0.5 * psin1 - 1 / 3.0 * psin0 * psin1 +
            1 / 6.0 * psin0**2 + 1 / 6.0 * psin1**2)**2
    Dpsin_Dexppsin_dpsin0 = np.exp(psin0) * numerator2 / denominator2
    Dpsin_Dexppsin_dpsin1 = np.exp(psin0) * numerator3 / denominator3

    fmp = np.exp(-phi_p[1]) - np.exp(-phi_p[0])
    _numerator = (1 - around_zero_p) * Dpsip + around_zero_p * 1
    _denominator = (1 - around_zero_p) * (np.exp(
        -Dpsip) - 1) + around_zero_p * (-1 + 0.5 * Dpsip - 1 / 6.0 * Dpsip**2)
    Dpsip_Dexppsip = np.exp(-psip0) * _numerator / _denominator
    dfmp_dphip0 = np.exp(-phi_p[0])
    dfmp_dphip1 = -np.exp(-phi_p[1])
    _numerator2 = (1 - around_zero_p) * (
        Dpsip + np.exp(-Dpsip) - 1) + around_zero_p * (
            -3 + psip0 - psip1 + 2 * psip0 * psip1 - psip0**2 - psip1**2)
    _denominator2 = (1 - around_zero_p) * (
        np.exp(-Dpsip) - 1)**2 + around_zero_p * (
            1 - 0.5 * psip0 + 0.5 * psip1 - 1 / 3.0 * psip0 * psip1 +
            1 / 6.0 * psip0**2 + 1 / 6.0 * psip1**2)**2
    _numerator3 = (1 - around_zero_p) * (
        -np.exp(-Dpsip) + 1 -
        Dpsip * np.exp(-Dpsip)) + around_zero_p * (-3 + 2 * psip0 - 2 * psip1)
    _denominator3 = (1 - around_zero_p) * (
        np.exp(-Dpsip) - 1)**2 + around_zero_p * (
            1 - 0.5 * psip0 + 0.5 * psip1 - 1 / 3.0 * psip0 * psip1 +
            1 / 6.0 * psip0**2 + 1 / 6.0 * psip1**2)**2
    Dpsip_Dexppsip_dpsip0 = np.exp(-psip0) * _numerator2 / _denominator2
    Dpsip_Dexppsip_dpsip1 = np.exp(-psip0) * _numerator3 / _denominator3

    Fcurrent = mn[0] * Dpsin_Dexppsin * fmn / dgrid[0] + mp[
        0] * Dpsip_Dexppsip * fmp / dgrid[0]

    deriv = {}

    deriv["dChi0"] = mn[0] * fmn / dgrid[0] * Dpsin_Dexppsin_dpsin0 + mp[
        0] * fmp / dgrid[0] * Dpsip_Dexppsip_dpsip0
    deriv["dChi1"] = mn[0] * fmn / dgrid[0] * Dpsin_Dexppsin_dpsin1 + mp[
        0] * fmp / dgrid[0] * Dpsip_Dexppsip_dpsip1
    deriv["dEg0"] = mp[0] * fmp / dgrid[0] * Dpsip_Dexppsip_dpsip0
    deriv["dEg1"] = mp[0] * fmp / dgrid[0] * Dpsip_Dexppsip_dpsip1
    deriv["dNc0"] = 1 / Nc[0] * mn[0] * fmn / dgrid[0] * Dpsin_Dexppsin_dpsin0
    deriv["dNc1"] = 1 / Nc[1] * mn[0] * fmn / dgrid[0] * Dpsin_Dexppsin_dpsin1
    deriv["dNv0"] = -1 / Nv[0] * mp[0] * fmp / dgrid[0] * Dpsip_Dexppsip_dpsip0
    deriv["dNv1"] = -1 / Nv[1] * mp[0] * fmp / dgrid[0] * Dpsip_Dexppsip_dpsip1
    deriv["dmn0"] = Dpsin_Dexppsin * fmn / dgrid[0]
    deriv["dmp0"] = Dpsip_Dexppsip * fmp / dgrid[0]

    deriv["dphin0"] = mn[0] * Dpsin_Dexppsin / dgrid[0] * dfmn_dphin0
    deriv["dphin1"] = mn[0] * Dpsin_Dexppsin / dgrid[0] * dfmn_dphin1
    deriv["dphip0"] = mp[0] * Dpsip_Dexppsip / dgrid[0] * dfmp_dphip0
    deriv["dphip1"] = mp[0] * Dpsip_Dexppsip / dgrid[0] * dfmp_dphip1
    deriv["dphi0"] = mn[0] * fmn / dgrid[0] * Dpsin_Dexppsin_dpsin0 + mp[
        0] * fmp / dgrid[0] * Dpsip_Dexppsip_dpsip0
    deriv["dphi1"] = mn[0] * fmn / dgrid[0] * Dpsin_Dexppsin_dpsin1 + mp[
        0] * fmp / dgrid[0] * Dpsip_Dexppsip_dpsip1

    return Fcurrent, deriv
