from .SHR import *
from .auger import *
from .radiative import *
from .current import *

def ddn( dgrid , phi_n , phi_p , phi , Chi , Eg , Nc , Nv , mn , Et , tn , tp , Br , Cn , Cp , G ):
    """
    Computes the left side term of the drift-diffusion equation for e-.

    Parameters
    ----------
        phi_n    : numpy array , shape = ( N )
            e- quasi-Fermi energy
        phi_p    : numpy array , shape = ( N )
            hole quasi-Fermi energy
        phi      : numpy array , shape = ( N )
            electrostatic potential
        dgrid    : numpy array , shape = ( N - 1 )
            array of distances between consecutive grid points
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
        Et       : numpy array , shape = ( N )
            SHR trap state energy level
        tn       : numpy array , shape = ( N )
            SHR e- lifetime
        tp       : numpy array , shape = ( N )
            SHR hole lifetime
        Br       : numpy array , shape = ( N )
            radiative recombination coefficient
        Cn       : numpy array , shape = ( N )
            electron Auger coefficient
        Cp       : numpy array , shape = ( N )
            hole Auger coefficient
        G        : numpy array , shape = ( N )
            e-/hole pair generation rate density

    Returns
    -------
        numpy array , shape = ( N - 2 )
            left side term of the drift-diffusion equation for e-

    """
    R = SHR( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Et , tn , tp ) + rad( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Br ) + auger( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Cn , Cp )

    _Jn = Jn( dgrid , phi_n , phi , Chi , Nc , mn )

    ave_dgrid = ( dgrid[:-1] + dgrid[1:] ) / 2.0

    print(R)
    print(G)
    print(_Jn[1:] - _Jn[:-1])
    quit()

    return ( _Jn[1:] - _Jn[:-1] ) / ave_dgrid - R[1:-1] + G[1:-1]





def ddn_deriv( dgrid , phi_n , phi_p , phi , Chi , Eg , Nc , Nv , mn , Et , tn , tp , Br , Cn , Cp , G ):
    """
    Compute the derivatives of the drift-diffusion equation for e- with respect to the potentials.

    Parameters
    ----------
        dgrid       : numpy array , shape = ( N - 1 )
            array of distances between consecutive grid points
        phi_n       : numpy array , shape = ( N )
            e- quasi-Fermi energy
        phi_p       : numpy array , shape = ( N )
            hole quasi-Fermi energy
        phi         : numpy array , shape = ( N )
            electrostatic potential
        Chi         : numpy array , shape = ( N )
            electron affinity
        Eg          : numpy array , shape = ( N )
            band gap
        Nc          : numpy array , shape = ( N )
            e- density of states
        Nv          : numpy array , shape = ( N )
            hole density of states
        mn          : numpy array , shape = ( N )
            e- mobility
        Et          : numpy array , shape = ( N )
            SHR trap state energy level
        tn          : numpy array , shape = ( N )
            SHR e- lifetime
        tp          : numpy array , shape = ( N )
            SHR hole lifetime
        Br          : numpy array , shape = ( N )
            radiative recombination coefficient
        Cn          : numpy array , shape = ( N )
            electron Auger coefficient
        Cp          : numpy array , shape = ( N )
            hole Auger coefficient
        G           : numpy array , shape = ( N )
            e-/hole pair generation rate density

    Returns
    -------
        dde_phin_   : numpy array , shape = ( N - 2 )
            derivative of drift-diffusion equation for e- at point i w.r.t. phi_n[i-1]
        dde_phin__  : numpy array , shape = ( N - 2 )
            derivative of drift-diffusion equation for e- at point i i w.r.t. phi_n[i]
        dde_phin___ : numpy array , shape = ( N - 2 )
            derivative of drift-diffusion equation for e- at point i i w.r.t. phi_n[i+1]
        dde_phip__  : numpy array , shape = ( N - 2 )
            derivative of drift-diffusion equation for e- at point i i w.r.t. phi_p[i]
        dde_phi_    : numpy array , shape = ( N - 2 )
            derivative of drift-diffusion equation for e- at point i i w.r.t. phi[i-1]
        dde_phi__   : numpy array , shape = ( N - 2 )
            derivative of drift-diffusion equation for e- at point i i w.r.t. phi[i]
        dde_phi___  : numpy array , shape = ( N - 2 )
            derivative of drift-diffusion equation for e- at point i i w.r.t. phi[i+1]

    """
    DR_SHR_phin , DR_SHR_phip , DR_SHR_phi = SHR_deriv( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Et , tn , tp )
    DR_rad_phin , DR_rad_phip , DR_rad_phi = rad_deriv( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Br )
    DR_auger_phin , DR_auger_phip , DR_auger_phi = auger_deriv( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Cn , Cp )
    DR_phin = DR_SHR_phin + DR_rad_phin + DR_auger_phin
    DR_phip = DR_SHR_phip + DR_rad_phip + DR_auger_phip
    DR_phi = DR_SHR_phi + DR_rad_phi + DR_auger_phi

    dJn_phin_maindiag , dJn_phin_upperdiag , dJn_phi_maindiag , dJn_phi_upperdiag = Jn_deriv( dgrid , phi_n , phi , Chi , Nc , mn )

    ave_dgrid = ( dgrid[:-1] + dgrid[1:] ) / 2.0

    dde_phin_ = - dJn_phin_maindiag[:-1] / ave_dgrid
    dde_phin__ = ( - dJn_phin_upperdiag[:-1] + dJn_phin_maindiag[1:] ) / ave_dgrid - DR_phin[1:-1]
    dde_phin___ = dJn_phin_upperdiag[1:] / ave_dgrid

    dde_phi_ = - dJn_phi_maindiag[:-1] / ave_dgrid
    dde_phi__ = ( - dJn_phi_upperdiag[:-1] + dJn_phi_maindiag[1:] ) / ave_dgrid - DR_phi[1:-1]
    dde_phi___ = dJn_phi_upperdiag[1:] / ave_dgrid

    dde_phip__ = - DR_phip[1:-1]

    return dde_phin_ , dde_phin__ , dde_phin___ , dde_phip__ , dde_phi_ , dde_phi__ , dde_phi___
