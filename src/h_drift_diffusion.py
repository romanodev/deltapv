from .SHR import *
from .current import *

def ddp( dgrid , phi_n , phi_p , phi , Chi , Eg , Nc , Nv , mp , Et , tn , tp , Br , Cn , Cp , G ):
    """
    Computes the left side term of the drift-diffusion equation for holes.

    Parameters
    ----------
        dgrid       : numpy array , shape = ( N - 1 )
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
        mp       : numpy array , shape = ( N )
            hole mobility
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
            left side term of the drift-diffusion equation for holes

    """
    R = SHR( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Et , tn , tp ) + rad( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Br ) + auger( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Cn , Cp )
    _Jp = Jp( dgrid , phi_p , phi , Chi , Eg , Nv , mp )
    ave_dgrid = ( dgrid[:-1] + dgrid[1:] ) / 2.0
    return ( _Jp[1:] - _Jp[:-1] ) / ave_dgrid + R[1:-1] - G[1:-1]





def ddp_deriv( dgrid , phi_n , phi_p , phi , Chi , Eg , Nc , Nv , mp , Et , tn , tp , Br , Cn , Cp , G ):
    """
    Compute the derivatives of the drift-diffusion equation for holes with respect to the potentials.

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
        mp          : numpy array , shape = ( N )
            hole mobility
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
        ddp_phin__  : numpy array , shape = ( N - 2 )
            derivative of drift-diffusion equation for holes at point i i w.r.t. phi_n[i]
        ddp_phip_   : numpy array , shape = ( N - 2 )
            derivative of drift-diffusion equation for holes at point i i w.r.t. phi_p[i-1]
        ddp_phip__  : numpy array , shape = ( N - 2 )
            derivative of drift-diffusion equation for holes at point i i w.r.t. phi_p[i]
        ddp_phip___ : numpy array , shape = ( N - 2 )
            derivative of drift-diffusion equation for holes at point i i w.r.t. phi_p[i+1]
        ddp_phi_    : numpy array , shape = ( N - 2 )
            derivative of drift-diffusion equation for holes at point i i w.r.t. phi[i-1]
        ddp_phi__   : numpy array , shape = ( N - 2 )
            derivative of drift-diffusion equation for holes at point i i w.r.t. phi[i]
        ddp_phi___  : numpy array , shape = ( N - 2 )
            derivative of drift-diffusion equation for holes at point i i w.r.t. phi[i+1]

    """
    DR_SHR_phin , DR_SHR_phip , DR_SHR_phi = SHR_deriv( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Et , tn , tp )
    DR_rad_phin , DR_rad_phip , DR_rad_phi = rad_deriv( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Br )
    DR_auger_phin , DR_auger_phip , DR_auger_phi = auger_deriv( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Cn , Cp )
    DR_phin = DR_SHR_phin + DR_rad_phin + DR_auger_phin
    DR_phip = DR_SHR_phip + DR_rad_phip + DR_auger_phip
    DR_phi = DR_SHR_phi + DR_rad_phi + DR_auger_phi

    dJp_phip_maindiag , dJp_phip_upperdiag , dJp_phi_maindiag , dJp_phi_upperdiag = Jp_deriv( dgrid , phi_p , phi , Chi , Eg , Nv , mp )

    ave_dgrid = ( dgrid[:-1] + dgrid[1:] ) / 2.0

    ddp_phip_ = - dJp_phip_maindiag[:-1] / ave_dgrid
    ddp_phip__ = ( - dJp_phip_upperdiag[:-1] + dJp_phip_maindiag[1:] ) / ave_dgrid + DR_phip[1:-1]
    ddp_phip___ = dJp_phip_upperdiag[1:] / ave_dgrid

    ddp_phi_ = - dJp_phi_maindiag[:-1] / ave_dgrid
    ddp_phi__ = ( - dJp_phi_upperdiag[:-1] + dJp_phi_maindiag[1:] ) / ave_dgrid + DR_phi[1:-1]
    ddp_phi___ = dJp_phi_upperdiag[1:] / ave_dgrid

    ddp_phin__ = DR_phin[1:-1]

    return ddp_phin__ , ddp_phip_ , ddp_phip__ , ddp_phip___ , ddp_phi_ , ddp_phi__ , ddp_phi___
