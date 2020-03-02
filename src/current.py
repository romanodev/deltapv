from .scales import *
if USE_JAX:
    from jax.config import config
    config.update("jax_enable_x64", True)
    import jax.numpy as np
else:
    import numpy as np

### Compute the e- current
## Inputs :
#      phi_n (array:N) -> e- quasi-Fermi energy
#      phi (array:N) -> electrostatic potential
#      dgrid (array:N-1) -> array of distances between consecutive grid points
#      Chi (array:N) -> electron affinity
#      Nc (array:N) -> e- density of states
#      mn (array:N) -> e- mobility
## Outputs :
#      1 (array:N-1) -> e- current

def Jn( phi_n , phi , dgrid , Chi , Nc , mn ):

    psi_n = Chi + np.log( Nc ) + phi
    Dpsin = psi_n[:-1] - psi_n[1:]

    fm = np.exp( phi_n[1:] ) - np.exp( phi_n[:-1] )
    Dpsin_Dexppsin = np.exp( psi_n[:-1] ) * Dpsin * ( np.exp( Dpsin ) - 1 )**(-1)

    return mn[:-1] * Dpsin_Dexppsin * fm / dgrid





### Compute the derivatives of e- current w.r.t. the e- quasi-Fermi energy and electrostatic potential
## Inputs :
#      phi_n (array:N) -> e- quasi-Fermi energy
#      phi (array:N) -> electrostatic potential
#      dgrid (array:N-1) -> array of distances between consecutive grid points
#      Chi (array:N) -> electron affinity
#      Nc (array:N) -> e- density of states
#      mn (array:N) -> e- mobility
## Outputs :
#      1 (array:N-1) -> derivative of e- current at point i w.r.t. phi_n[i]
#      2 (array:N-1) -> derivative of e- current at point i w.r.t. phi_n[i+1]
#      3 (array:N-1) -> derivative of e- current at point i w.r.t. phi[i]
#      4 (array:N-1) -> derivative of e- current at point i w.r.t. phi[i+1]

def Jn_deriv( phi_n , phi , dgrid , Chi , Nc , mn ):
    psi_n = Chi + np.log( Nc ) + phi
    Dpsin = psi_n[:-1] - psi_n[1:]

    fm = np.exp( phi_n[1:] ) - np.exp( phi_n[:-1] )
    Dpsin_Dexppsin = np.exp( psi_n[:-1] ) * Dpsin * ( np.exp( Dpsin ) - 1 )**(-1)

    fm_deriv_maindiag = - np.exp( phi_n[:-1] )
    fm_deriv_upperdiag = np.exp( phi_n[1:] )

    Dpsin_Dexppsin_deriv_maindiag = np.exp( psi_n[:-1] ) * ( - Dpsin + np.exp( Dpsin ) - 1 ) * ( np.exp( Dpsin ) - 1 )**(-2)
    Dpsin_Dexppsin_deriv_upperdiag = np.exp( psi_n[:-1] ) * ( - np.exp( Dpsin ) + 1 + Dpsin * np.exp( Dpsin ) ) * ( np.exp( Dpsin ) - 1 )**(-2)

    dJn_phin_maindiag = mn[:-1] * Dpsin_Dexppsin / dgrid * fm_deriv_maindiag
    dJn_phin_upperdiag = mn[:-1] * Dpsin_Dexppsin / dgrid * fm_deriv_upperdiag

    dJn_phi_maindiag = mn[:-1] * fm / dgrid * Dpsin_Dexppsin_deriv_maindiag
    dJn_phi_upperdiag = mn[:-1] * fm / dgrid * Dpsin_Dexppsin_deriv_upperdiag

    return dJn_phin_maindiag , dJn_phin_upperdiag , dJn_phi_maindiag , dJn_phi_upperdiag





### Compute the hole current
## Inputs :
#      phi_p (array:N) -> hole quasi-Fermi energy
#      phi (array:N) -> electrostatic potential
#      dgrid (array:N-1) -> array of distances between consecutive grid points
#      Chi (array:N) -> electron affinity
#      Eg (array:N) -> band gap
#      Nv (array:N) -> hole density of states
#      mp (array:N) -> hole mobility
## Outputs :
#      1 (array:N-1) -> hole current

def Jp( phi_p , phi , dgrid , Chi , Eg , Nv , mp ):
    psi_p = Chi + Eg - np.log( Nv ) + phi
    Dpsip = psi_p[:-1] - psi_p[1:]

    fm = np.exp( - phi_p[1:] ) - np.exp( - phi_p[:-1] )
    Dpsip_Dexppsin = np.exp( - psi_p[:-1] ) * Dpsip * ( np.exp( - Dpsip ) - 1 )**(-1)

    return mp[:-1] * Dpsip_Dexppsin * fm / dgrid





### Compute the derivatives of hole current w.r.t. the hole quasi-Fermi energy and electrostatic potential
## Inputs :
#      phi_p (array:N) -> hole quasi-Fermi energy
#      phi (array:N) -> electrostatic potential
#      dgrid (array:N-1) -> array of distances between consecutive grid points
#      Chi (array:N) -> electron affinity
#      Eg (array:N) -> band gap
#      Nv (array:N) -> hole density of states
#      mp (array:N) -> hole mobility
## Outputs :
#      1 (array:N-1) -> derivative of hole current at point i w.r.t. phi_p[i]
#      2 (array:N-1) -> derivative of hole current at point i w.r.t. phi_p[i+1]
#      3 (array:N-1) -> derivative of hole current at point i w.r.t. phi[i]
#      4 (array:N-1) -> derivative of hole current at point i w.r.t. phi[i+1]

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





def total_current( phi_n , phi_p , phi , dgrid , Chi , Eg , Nc , Nv , mn , mp ):
    psin0 = Chi[0] + np.log( Nc[0] ) + phi[0]
    psin1 = Chi[1] + np.log( Nc[1] ) + phi[1]
    psip0 = Chi[0] + Eg[0] - np.log( Nv[0] ) + phi[0]
    psip1 = Chi[1] + Eg[1] - np.log( Nv[1] ) + phi[1]
    Dpsin = psin0 - psin1
    Dpsip = psip0 - psip1

    fmn = np.exp( phi_n[1] ) - np.exp( phi_n[0] )
    Dpsin_Dexppsin = np.exp( psin0 ) * Dpsin * ( np.exp( Dpsin ) - 1 )**(-1)

    Dpsin_Dexppsin_dpsin0 = - np.exp( psin0 ) * ( 1 + Dpsin - np.exp( Dpsin ) ) * ( 1 - np.exp( Dpsin ) )**(-2)
    Dpsin_Dexppsin_dpsin1 = - np.exp( Chi[1] + np.log( Nc[1] ) + phi[1] ) * ( 1 - Dpsin - np.exp( - Dpsin ) ) * ( 1 - np.exp( Dpsin ) )**(-2)

    fmp = np.exp( - phi_p[1] ) - np.exp( - phi_p[0] )
    Dpsip_Dexppsin = np.exp( - psip0 ) * Dpsip * ( np.exp( - Dpsip ) - 1 )**(-1)
    Dpsip_Dexppsin_dpsip0 = - np.exp( - ( Chi[0] + Eg[0] - np.log( Nv[0] ) + phi[0] ) ) * ( 1 - Dpsip - np.exp( - Dpsip ) ) * ( 1 - np.exp( - Dpsip ) )**(-2)
    Dpsip_Dexppsin_dpsip1 = np.exp( - ( Chi[1] + Eg[1] - np.log( Nv[1] ) + phi[1] ) ) * ( 1 - Dpsip * np.exp( - Dpsip ) - np.exp( - Dpsip ) ) * ( 1 - np.exp( - Dpsip ) )**(-2)

    Fcurrent = mp[0] * Dpsip_Dexppsin * fmp / dgrid + mn[0] * Dpsin_Dexppsin * fmn / dgrid

#    dFcurr_dChi0 = mp[0] * fmp / dgrid * Dpsin_Dexppsin_dpsin0 *  + mn[0] * fmn / dgrid *
    dFcurr_dmn0 = Dpsin_Dexppsin * fmn / dgrid
    dFcurr_dmp0 = Dpsip_Dexppsin * fmp / dgrid
