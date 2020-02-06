from .scales import *
if USE_JAX:
    import jax.numpy as np
else:
    import numpy as np

def Jn( phi_n , phi , dgrid , Chi , Nc , mn ):

    psi_n = Chi + np.log( Nc ) + phi
    Dpsin = psi_n[:-1] - psi_n[1:]

    fm = np.exp( phi_n[1:] ) - np.exp( phi_n[:-1] )
    Dpsin_Dexppsin = np.exp( psi_n[:-1] ) * Dpsin * ( np.exp( Dpsin ) - 1 )**(-1)

    return mn[:-1] * Dpsin_Dexppsin * fm / dgrid

def Jn_deriv( phi_n , phi , dgrid , Chi , Nc , mn ):
    psi_n = Chi + np.log( Nc ) + phi
    Dpsin = psi_n[:-1] - psi_n[1:]

    fm = np.exp( phi_n[1:] ) - np.exp( phi_n[:-1] )
    Dpsin_Dexppsin = np.exp( psi_n[:-1] ) * Dpsin * ( np.exp( Dpsin ) - 1 )**(-1)

    fm_deriv_maindiag = - np.exp( phi_n[:-1] )
    fm_deriv_upperdiag = np.exp( phi_n[1:] )

    Dpsin_Dexppsin_deriv_maindiag = - np.exp( psi_n[:-1] ) * ( 1 + Dpsin - np.exp( Dpsin ) ) * ( 1 - np.exp( Dpsin ) )**(-2)
    Dpsin_Dexppsin_deriv_upperdiag = - np.exp( psi_n[1:] ) * ( 1 - Dpsin - np.exp( - Dpsin ) ) * ( 1 - np.exp( Dpsin ) )**(-2)

    dJn_phin_maindiag = mn[:-1] * Dpsin_Dexppsin / dgrid * fm_deriv_maindiag
    dJn_phin_upperdiag = mn[:-1] * Dpsin_Dexppsin / dgrid * fm_deriv_upperdiag

    dJn_phi_maindiag = mn[:-1] * fm / dgrid * Dpsin_Dexppsin_deriv_maindiag
    dJn_phi_upperdiag = mn[:-1] * fm / dgrid * Dpsin_Dexppsin_deriv_upperdiag

    return dJn_phin_maindiag , dJn_phin_upperdiag , dJn_phi_maindiag , dJn_phi_upperdiag

def Jp( phi_p , phi , dgrid , Chi , Eg , Nv , mp ):
    psi_p = Chi + Eg - np.log( Nv ) + phi
    Dpsip = psi_p[:-1] - psi_p[1:]

    fm = np.exp( - phi_p[1:] ) - np.exp( - phi_p[:-1] )
    Dpsip_Dexppsin = np.exp( - psi_p[:-1] ) * Dpsip * ( np.exp( - Dpsip ) - 1 )**(-1)

    return mp[:-1] * Dpsip_Dexppsin * fm / dgrid

def Jp_deriv( phi_p , phi , dgrid , Chi , Eg , Nv , mp ):
    psi_p = Chi + Eg - np.log( Nv ) + phi
    Dpsip = psi_p[:-1] - psi_p[1:]

    fm = np.exp( - phi_p[1:] ) - np.exp( - phi_p[:-1] )
    Dpsip_Dexppsin = np.exp( - psi_p[:-1] ) * Dpsip * ( np.exp( - Dpsip ) - 1 )**(-1)

    fm_deriv_maindiag = np.exp( - phi_p[:-1] )
    fm_deriv_upperdiag = - np.exp( - phi_p[1:] )

    Dpsip_Dexppsin_deriv_maindiag = - np.exp( - psi_p[:-1] ) * ( 1 - Dpsip - np.exp( - Dpsip ) ) * ( 1 - np.exp( - Dpsip ) )**(-2)
    Dpsip_Dexppsin_deriv_upperdiag = np.exp( - psi_p[:-1] ) * ( 1 - Dpsip * np.exp( - Dpsip ) - np.exp( - Dpsip ) ) * ( 1 - np.exp( - Dpsip ) )**(-2)

    dJp_phip_maindiag = mp[:-1] * Dpsip_Dexppsin / dgrid * fm_deriv_maindiag
    dJp_phip_upperdiag = mp[:-1] * Dpsip_Dexppsin / dgrid * fm_deriv_upperdiag

    dJp_phi_maindiag = mp[:-1] * fm / dgrid * Dpsip_Dexppsin_deriv_maindiag
    dJp_phi_upperdiag = mp[:-1] * fm / dgrid * Dpsip_Dexppsin_deriv_upperdiag

    return dJp_phip_maindiag , dJp_phip_upperdiag , dJp_phi_maindiag , dJp_phi_upperdiag
