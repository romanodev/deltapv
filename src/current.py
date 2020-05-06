from .scales import *
if USE_JAX:
    from jax.config import config
    config.update("jax_enable_x64", True)
    import jax.numpy as np
else:
    import numpy as np

def Jn( dgrid , phi_n , phi , Chi , Nc , mn ):
    """
    Computes the e- current.

    Parameters
    ----------
        dgrid    : numpy array , shape = ( N - 1 )
            array of distances between consecutive grid points
        phi_n    : numpy array , shape = ( N )
            e- quasi-Fermi energy
        phi      : numpy array , shape = ( N )
            electrostatic potential
        Chi      : numpy array , shape = ( N )
            electron affinity
        Nc       : numpy array , shape = ( N )
            e- density of states
        mn       : numpy array , shape = ( N )
            e- mobility

    Returns
    -------
        numpy array , shape = ( N - 1 )
            e- current

    """

    phi_n = np.linspace( 0 , phi_n.size , num = phi_n.size )

    psi_n = Chi + np.log( Nc ) + phi
    Dpsin = psi_n[:-1] - psi_n[1:]
    thr = 1e-5
    around_zero = 0.5 * ( np.tanh( 500 * ( Dpsin + thr ) ) - np.tanh( 500 * ( Dpsin - thr ) ) )

    fm = np.exp( phi_n[1:] ) - np.exp( phi_n[:-1] )

#    Dpsin_Dexppsin = np.exp( psi_n[:-1] ) * Dpsin * ( np.exp( Dpsin ) - 1 )**(-1)
    numerator = ( 1 - around_zero ) * Dpsin + around_zero * 1
    denominator = ( 1 - around_zero ) * ( np.exp( Dpsin ) - 1 ) + around_zero * ( 1 + 0.5*Dpsin + 1/6.0*Dpsin**2 )
    Dpsin_Dexppsin = np.exp( psi_n[:-1] ) * numerator / denominator

    print( mn[:-1] * Dpsin_Dexppsin * fm / dgrid )
    quit()

    return mn[:-1] * Dpsin_Dexppsin * fm / dgrid





def Jn_deriv( dgrid , phi_n , phi , Chi , Nc , mn ):
    """
    Computes the derivatives of the e- current with respect to the e- quasi-Fermi energy and electrostatic potential.

    Parameters
    ----------
        dgrid    : numpy array , shape = ( N - 1 )
            array of distances between consecutive grid points
        phi_n       : numpy array , shape = ( N )
            e- quasi-Fermi energy
        phi         : numpy array , shape = ( N )
            electrostatic potential
        Chi         : numpy array , shape = ( N )
            electron affinity
        Nc          : numpy array , shape = ( N )
            e- density of states
        mn          : numpy array , shape = ( N )
            e- mobility

    Returns
    -------
        dJn_phin__  : numpy array , shape = ( N - 1 )
            derivative of e- current at point i w.r.t. phi_n[i]
        dJn_phin___ : numpy array , shape = ( N - 1 )
            derivative of e- current at point i w.r.t. phi_n[i+1]
        dJn_phi__   : numpy array , shape = ( N - 1 )
            derivative of e- current at point i w.r.t. phi[i]
        dJn_phi___  : numpy array , shape = ( N - 1 )
            derivative of e- current at point i w.r.t. phi[i+1]

    """
    psi_n = Chi + np.log( Nc ) + phi
    Dpsin = psi_n[:-1] - psi_n[1:]
    thr = 1e-5
    around_zero = 0.5 * ( np.tanh( 500 * ( Dpsin + thr ) ) - np.tanh( 500 * ( Dpsin - thr ) ) )

    fm = np.exp( phi_n[1:] ) - np.exp( phi_n[:-1] )

#    Dpsin_Dexppsin = np.exp( psi_n[:-1] ) * Dpsin * ( np.exp( Dpsin ) - 1 )**(-1)
    numerator = ( 1 - around_zero ) * Dpsin + around_zero * 1
    denominator = ( 1 - around_zero ) * ( np.exp( Dpsin ) - 1 ) + around_zero * ( 1 + 0.5*Dpsin + 1/6.0*Dpsin**2 )
    Dpsin_Dexppsin = np.exp( psi_n[:-1] ) * numerator / denominator

    fm_deriv_maindiag = - np.exp( phi_n[:-1] )
    fm_deriv_upperdiag = np.exp( phi_n[1:] )

#    Dpsin_Dexppsin_deriv_maindiag = np.exp( psi_n[:-1] ) * ( - Dpsin + np.exp( Dpsin ) - 1 ) * ( np.exp( Dpsin ) - 1 )**(-2)
#    Dpsin_Dexppsin_deriv_upperdiag = np.exp( psi_n[:-1] ) * ( - np.exp( Dpsin ) + 1 + Dpsin * np.exp( Dpsin ) ) * ( np.exp( Dpsin ) - 1 )**(-2)
    numerator2 = ( 1 - around_zero ) * ( - Dpsin + np.exp( Dpsin ) - 1 ) + around_zero * ( 3 + psi_n[:-1] - psi_n[1:] - 2*psi_n[:-1]*psi_n[1:] + psi_n[:-1]**2 + psi_n[1:]**2 )
    denominator2 = ( 1 - around_zero ) * ( np.exp( Dpsin ) - 1 )**2 + around_zero * ( 1 + 0.5*psi_n[:-1] - 0.5*psi_n[1:] - 1/3.0*psi_n[:-1]*psi_n[1:] + 1/6.0*psi_n[:-1]**2 + 1/6.0*psi_n[1:]**2 )**2
    numerator3 = ( 1 - around_zero ) * ( - np.exp( Dpsin ) + 1 + Dpsin * np.exp( Dpsin ) ) + around_zero * ( 3 + 2*psi_n[:-1] - 2*psi_n[1:] )
    denominator3 = ( 1 - around_zero ) * ( np.exp( Dpsin ) - 1 )**2 + around_zero * ( 1 + 0.5*psi_n[:-1] - 0.5*psi_n[1:] - 1/3.0*psi_n[:-1]*psi_n[1:] + 1/6.0*psi_n[:-1]**2 + 1/6.0*psi_n[1:]**2 )**2

    Dpsin_Dexppsin_deriv_maindiag = np.exp( psi_n[:-1] ) * numerator2 / denominator2
    Dpsin_Dexppsin_deriv_upperdiag = np.exp( psi_n[:-1] ) * numerator3 / denominator3

    dJn_phin__ = mn[:-1] * Dpsin_Dexppsin / dgrid * fm_deriv_maindiag
    dJn_phin___ = mn[:-1] * Dpsin_Dexppsin / dgrid * fm_deriv_upperdiag

    dJn_phi__ = mn[:-1] * fm / dgrid * Dpsin_Dexppsin_deriv_maindiag
    dJn_phi___ = mn[:-1] * fm / dgrid * Dpsin_Dexppsin_deriv_upperdiag

    return dJn_phin__ , dJn_phin___ , dJn_phi__ , dJn_phi___





def Jp( dgrid , phi_p , phi , Chi , Eg , Nv , mp ):
    """
    Computes the hole current.

    Parameters
    ----------
        dgrid    : numpy array , shape = ( N - 1 )
            array of distances between consecutive grid points
        phi_p    : numpy array , shape = ( N )
            hole quasi-Fermi energy
        phi      : numpy array , shape = ( N )
            electrostatic potential
        Chi      : numpy array , shape = ( N )
            electron affinity
        Eg       : numpy array , shape = ( N )
            band gap
        Nv       : numpy array , shape = ( N )
            hole density of states
        mp       : numpy array , shape = ( N )
            hole mobility

    Returns
    -------
        numpy array , shape = ( N - 1 )
            hole current

    """
    psi_p = Chi + Eg - np.log( Nv ) + phi
    Dpsip = psi_p[:-1] - psi_p[1:]
    thr = 1e-5
    around_zero = 0.5 * ( np.tanh( 500 * ( Dpsip + thr ) ) - np.tanh( 500 * ( Dpsip - thr ) ) )

    fm = np.exp( - phi_p[1:] ) - np.exp( - phi_p[:-1] )

#    Dpsip_Dexppsip = np.exp( - psi_p[:-1] ) * Dpsip * ( np.exp( - Dpsip ) - 1 )**(-1)
    numerator = ( 1 - around_zero ) * Dpsip + around_zero * 1
    denominator = ( 1 - around_zero ) * ( np.exp( - Dpsip ) - 1 ) + around_zero * ( 1 + 0.5*Dpsip - 1/6.0*Dpsip**2 )
    Dpsip_Dexppsip = np.exp( - psi_p[:-1] ) * numerator / denominator

    return mp[:-1] * Dpsip_Dexppsip * fm / dgrid





def Jp_deriv( dgrid , phi_p , phi , Chi , Eg , Nv , mp ):
    """
    Computes the derivatives of the hole current with respect to the hole quasi-Fermi energy and electrostatic potential.

    Parameters
    ----------
        dgrid    : numpy array , shape = ( N - 1 )
            array of distances between consecutive grid points
        phi_p       : numpy array , shape = ( N )
            hole quasi-Fermi energy
        phi         : numpy array , shape = ( N )
            electrostatic potential
        Chi         : numpy array , shape = ( N )
            electron affinity
        Eg          : numpy array , shape = ( N )
            band gap
        Nv          : numpy array , shape = ( N )
            hole density of states
        mp          : numpy array , shape = ( N )
            hole mobility

    Returns
    -------
        dJp_phip__  : numpy array , shape = ( N - 1 )
            derivative of hole current at point i w.r.t. phi_p[i]
        dJp_phip___ : numpy array , shape = ( N - 1 )
            derivative of hole current at point i w.r.t. phi_p[i+1]
        dJp_phi__   : numpy array , shape = ( N - 1 )
            derivative of hole current at point i w.r.t. phi[i]
        dJp_phi___  : numpy array , shape = ( N - 1 )
            derivative of hole current at point i w.r.t. phi[i+1]

    """
    psi_p = Chi + Eg - np.log( Nv ) + phi
    Dpsip = psi_p[:-1] - psi_p[1:]
    thr = 1e-5
    around_zero = 0.5 * ( np.tanh( 500 * ( Dpsip + thr ) ) - np.tanh( 500 * ( Dpsip - thr ) ) )

    fm = np.exp( - phi_p[1:] ) - np.exp( - phi_p[:-1] )

#    Dpsip_Dexppsip = np.exp( - psi_p[:-1] ) * Dpsip * ( np.exp( - Dpsip ) - 1 )**(-1)
    numerator = ( 1 - around_zero ) * Dpsip + around_zero * 1
    denominator = ( 1 - around_zero ) * ( np.exp( - Dpsip ) - 1 ) + around_zero * ( 1 + 0.5*Dpsip - 1/6.0*Dpsip**2 )
    Dpsip_Dexppsip = np.exp( - psi_p[:-1] ) * numerator / denominator

    fm_deriv_maindiag = np.exp( - phi_p[:-1] )
    fm_deriv_upperdiag = - np.exp( - phi_p[1:] )

#    Dpsip_Dexppsip_deriv_maindiag = np.exp( - psi_p[:-1] ) * ( Dpsip + np.exp( - Dpsip ) - 1 ) * ( np.exp( - Dpsip ) - 1 )**(-2)
#    Dpsip_Dexppsip_deriv_upperdiag = np.exp( - psi_p[:-1] ) * ( - np.exp( - Dpsip ) + 1 - Dpsip * np.exp( - Dpsip ) ) * ( np.exp( - Dpsip ) - 1 )**(-2)
    numerator2 = ( 1 - around_zero ) * ( Dpsip + np.exp( - Dpsip ) - 1 ) + around_zero * ( 3 - psi_p[:-1] + psi_p[1:] - 2*psi_p[:-1]*psi_p[1:] + psi_p[:-1]**2 + psi_p[1:]**2 )
    denominator2 = ( 1 - around_zero ) * ( np.exp( - Dpsip ) - 1 )**2 + around_zero * ( 1 - 0.5*psi_p[:-1] + 0.5*psi_p[1:] - 1/3.0*psi_p[:-1]*psi_p[1:] + 1/6.0*psi_p[:-1]**2 + 1/6.0*psi_p[1:]**2 )**2
    numerator3 = ( 1 - around_zero ) * ( - np.exp( - Dpsip ) + 1 - Dpsip * np.exp( - Dpsip ) ) + around_zero * ( -3 + 2*psi_p[:-1] - 2*psi_p[1:] )
    denominator3 = ( 1 - around_zero ) * ( np.exp( - Dpsip ) - 1 )**2 + around_zero * ( 1 - 0.5*psi_p[:-1] + 0.5*psi_p[1:] - 1/3.0*psi_p[:-1]*psi_p[1:] + 1/6.0*psi_p[:-1]**2 + 1/6.0*psi_p[1:]**2 )**2

    Dpsip_Dexppsip_deriv_maindiag = np.exp( - psi_p[:-1] ) * numerator2 / denominator2
    Dpsip_Dexppsip_deriv_upperdiag = np.exp( - psi_p[:-1] ) * numerator3 / denominator3


    dJp_phip__ = mp[:-1] * Dpsip_Dexppsip / dgrid * fm_deriv_maindiag
    dJp_phip___ = mp[:-1] * Dpsip_Dexppsip / dgrid * fm_deriv_upperdiag

    dJp_phi__ = mp[:-1] * fm / dgrid * Dpsip_Dexppsip_deriv_maindiag
    dJp_phi___ = mp[:-1] * fm / dgrid * Dpsip_Dexppsip_deriv_upperdiag

    return dJp_phip__ , dJp_phip___ , dJp_phi__ , dJp_phi___





def total_current( dgrid , phi_n , phi_p , phi , Chi , Eg , Nc , Nv , mn , mp ):
    """
    Computes the total current and its derivatives.

    Parameters
    ----------
        dgrid    : numpy array , shape = ( N - 1 )
            array of distances between consecutive grid points
        phi_n    : numpy array , shape = ( N )
            e- quasi-Fermi energy
        phi_p    : numpy array , shape = ( N )
            hole quasi-Fermi energy
        phi      : numpy array , shape = ( N )
            electrostatic potential
        Chi      : numpy array , shape = ( N )
            electron affinity
        Eg       : numpy array , shape = ( N )
            band gap
        Nc       : numpy array , shape = ( N )
            e- density of states
        Nv       : numpy array , shape = ( N )
            hole density of states
        mn       : numpy array , shape = ( N )
            e- mobility
        mp       : numpy array , shape = ( N )
            hole mobility

    Returns
    -------
        Fcurrent : float
            total current
        deriv    : dictionnary of floats ( 16 keys )
            total current derivatives
            'dChi0'   -> derivative with respect to Chi[ 0 ]
            'dChi1'   -> derivative with respect to Chi[ 1 ]
            'dEg0'    -> derivative with respect to Eg[ 0 ]
            'dEg1'    -> derivative with respect to Eg[ 1 ]
            'dNc0'    -> derivative with respect to Nc[ 0 ]
            'dNc1'    -> derivative with respect to Nc[ 1 ]
            'dNv0'    -> derivative with respect to Nv[ 0 ]
            'dNv1'    -> derivative with respect to Nv[ 1 ]
            'dmn0'    -> derivative with respect to mn[ 0 ]
            'dmp0'    -> derivative with respect to mp[ 0 ]
            'dphin0'  -> derivative with respect to phi_n[ 0 ]
            'dphin1'  -> derivative with respect to phi_n[ 1 ]
            'dphip0'  -> derivative with respect to phi_p[ 0 ]
            'dphip1'  -> derivative with respect to phi_p[ 1 ]
            'dphi0'   -> derivative with respect to phi[ 0 ]
            'dphi1'   -> derivative with respect to phi[ 1 ]

    """
    psin0 = Chi[0] + np.log( Nc[0] ) + phi[0]
    psin1 = Chi[1] + np.log( Nc[1] ) + phi[1]
    psip0 = Chi[0] + Eg[0] - np.log( Nv[0] ) + phi[0]
    psip1 = Chi[1] + Eg[1] - np.log( Nv[1] ) + phi[1]
    Dpsin = psin0 - psin1
    Dpsip = psip0 - psip1
    around_zero_n = np.exp( - 500 * Dpsin**2 )
    around_zero_p = np.exp( - 500 * Dpsip**2 )

    fmn = np.exp( phi_n[1] ) - np.exp( phi_n[0] )
    numerator = ( 1 - around_zero ) * Dpsin + around_zero * 1
    denominator = ( 1 - around_zero ) * ( np.exp( Dpsin ) - 1 ) + around_zero * ( 1 + 0.5*Dpsin + 1/6.0*Dpsin**2 )
    Dpsin_Dexppsin = np.exp( psin0 ) * numerator / denominator
    dfmn_dphin0 = - np.exp( phi_n[0] )
    dfmn_dphin1 = np.exp( phi_n[1] )
    numerator2 = ( 1 - around_zero ) * ( - Dpsin + np.exp( Dpsin ) - 1 ) + around_zero * ( 3 + psin0 - psin1 - 2*psin0*psin1 + psin0**2 + psin1**2 )
    denominator2 = ( 1 - around_zero ) * ( np.exp( Dpsin ) - 1 )**2 + around_zero * ( 1 + 0.5*psin0 - 0.5*psin1 - 1/3.0*psin0*psin1 + 1/6.0*psin0**2 + 1/6.0*psin1**2 )**2
    numerator3 = ( 1 - around_zero ) * ( - np.exp( Dpsin ) + 1 + Dpsin * np.exp( Dpsin ) ) + around_zero * ( 3 + 2*psin0 - 2*psin1 )
    denominator3 = ( 1 - around_zero ) * ( np.exp( Dpsin ) - 1 )**2 + around_zero * ( 1 + 0.5*psin0 - 0.5*psin1 - 1/3.0*psi_n[:-1]*psi_n[1:] + 1/6.0*psin0**2 + 1/6.0*psin1**2 )**2
    Dpsin_Dexppsin_dpsin0 = np.exp( psin0 ) * _numerator2 / _denominator2
    Dpsin_Dexppsin_dpsin1 = np.exp( psin0 ) * _numerator3 / _denominator3

    fmp = np.exp( - phi_p[1] ) - np.exp( - phi_p[0] )
    _numerator = ( 1 - around_zero ) * Dpsip + around_zero * 1
    _denominator = ( 1 - around_zero ) * ( np.exp( - Dpsip ) - 1 ) + around_zero * ( 1 + 0.5*Dpsip - 1/6.0*Dpsip**2 )
    Dpsip_Dexppsip = np.exp( - psip0 ) * _numerator / _denominator
    dfmp_dphip0 = np.exp( - phi_p[0] )
    dfmp_dphip1 = - np.exp( - phi_p[1] )
    numerator2 = ( 1 - around_zero ) * ( Dpsip + np.exp( - Dpsip ) - 1 ) + around_zero * ( 3 - psip0 + psip1 - 2*psip0*psip1 + psip0**2 + psip1**2 )
    denominator2 = ( 1 - around_zero ) * ( np.exp( - Dpsip ) - 1 )**2 + around_zero * ( 1 - 0.5*psip0 + 0.5*psip1 - 1/3.0*psip0*psip1 + 1/6.0*psip0**2 + 1/6.0*psip1**2 )**2
    numerator3 = ( 1 - around_zero ) * ( - np.exp( - Dpsip ) + 1 - Dpsip * np.exp( - Dpsip ) ) + around_zero * ( -3 + 2*psip0 - 2*psip1 )
    denominator3 = ( 1 - around_zero ) * ( np.exp( - Dpsip ) - 1 )**2 + around_zero * ( 1 - 0.5*psip0 + 0.5*psip1 - 1/3.0*psip0*psip1 + 1/6.0*psip0**2 + 1/6.0*psip1**2 )**2
    Dpsip_Dexppsip_dpsip0 = np.exp( - psip0 ) * _numerator2 / _denominator2
    Dpsip_Dexppsip_dpsip1 = np.exp( - psip0 ) * _numerator3 / _denominator3

    Fcurrent = mn[0] * Dpsin_Dexppsin * fmn / dgrid[0] + mp[0] * Dpsip_Dexppsip * fmp / dgrid[0]

    deriv = {}

    deriv['dChi0'] = mn[0] * fmn / dgrid[0] * Dpsin_Dexppsin_dpsin0 + mp[0] * fmp / dgrid[0] * Dpsip_Dexppsip_dpsip0
    deriv['dChi1'] = mn[0] * fmn / dgrid[0] * Dpsin_Dexppsin_dpsin1 + mp[0] * fmp / dgrid[0] * Dpsip_Dexppsip_dpsip1
    deriv['dEg0'] = mp[0] * fmp / dgrid[0] * Dpsip_Dexppsip_dpsip0
    deriv['dEg1'] = mp[0] * fmp / dgrid[0] * Dpsip_Dexppsip_dpsip1
    deriv['dNc0'] = 1 / Nc[0] * mn[0] * fmn / dgrid[0] * Dpsin_Dexppsin_dpsin0
    deriv['dNc1'] = 1 / Nc[1] * mn[0] * fmn / dgrid[0] * Dpsin_Dexppsin_dpsin1
    deriv['dNv0'] = - 1 / Nv[0] * mp[0] * fmp / dgrid[0] * Dpsip_Dexppsip_dpsip0
    deriv['dNv1'] = - 1 / Nv[1] * mp[0] * fmp / dgrid[0] * Dpsip_Dexppsip_dpsip1
    deriv['dmn0'] = Dpsin_Dexppsin * fmn / dgrid[0]
    deriv['dmp0'] = Dpsip_Dexppsip * fmp / dgrid[0]

    deriv['dphin0'] = mn[0] * Dpsin_Dexppsin / dgrid[0] * dfmn_dphin0
    deriv['dphin1'] = mn[0] * Dpsin_Dexppsin / dgrid[0] * dfmn_dphin1
    deriv['dphip0'] = mp[0] * Dpsip_Dexppsip / dgrid[0] * dfmp_dphip0
    deriv['dphip1'] = mp[0] * Dpsip_Dexppsip / dgrid[0] * dfmp_dphip1
    deriv['dphi0'] = mn[0] * fmn / dgrid[0] * Dpsin_Dexppsin_dpsin0 + mp[0] * fmp / dgrid[0] * Dpsip_Dexppsip_dpsip0
    deriv['dphi1'] = mn[0] * fmn / dgrid[0] * Dpsin_Dexppsin_dpsin1 + mp[0] * fmp / dgrid[0] * Dpsip_Dexppsip_dpsip1

    return Fcurrent , deriv
