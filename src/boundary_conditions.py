from .physics import *
from .current import *

def contact_phin( dgrid , neq0 , neqL , phi_n , phi , Chi , Nc , mn , Snl , Snr ):
    """
    Computes the left side term of the Dirichlet out of equilibrium boundary condition for e- quasi-Fermi energy.

    Parameters
    ----------
        dgrid    : numpy array , shape = ( N - 1 )
            array of distances between consecutive grid points
        neq0     : float
            e- equilibrium density at left boundary
        neqL     : float
            e- equilibrium density at right boundary
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
        Snl      : float
            e- surface recombination velocity at left boundary
        Snr      : float
            e- surface recombination velocity at right boundary

    Returns
    -------
        float
            left side term of left boundary condition
        float
            left side term of right boundary condition

    """
    _n = n( phi_n , phi , Chi , Nc )
    _Jn = Jn( dgrid , phi_n , phi , Chi , Nc , mn )
    return _Jn[0] - Snl * ( _n[0] - neq0 ) , _Jn[-1] + Snr * ( _n[-1] - neqL )





def contact_phin_deriv( dgrid , phi_n , phi , Chi , Nc , mn , Snl , Snr ):
    """
    Compute the derivatives of the Dirichlet out of equilibrium boundary condition for e- quasi-Fermi energy.

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
        Snl      : float
            e- surface recombination velocity at left boundary
        Snr      : float
            e- surface recombination velocity at right boundary

    Returns
    -------
        float
            derivative of left boundary condition with respect to phi_n[0]
        float
            derivative of left boundary condition with respect to phi_n[1]
        float
            derivative of right boundary condition with respect to phi_n[0]
        float
            derivative of right boundary condition with respect to phi_n[1]
        float
            derivative of left boundary condition with respect to phi[0]
        float
            derivative of left boundary condition with respect to phi[1]
        float
            derivative of right boundary condition with respect to phi[0]
        float
            derivative of right boundary condition with respect to phi[1]

    """
    _n = n( phi_n , phi , Chi , Nc )
    dJn_phin_maindiag , dJn_phin_upperdiag , dJn_phi_maindiag , dJn_phi_upperdiag = Jn_deriv( dgrid , phi_n , phi , Chi , Nc , mn )
    return dJn_phin_maindiag[0] - Snl * _n[0] , dJn_phin_upperdiag[0] , \
    dJn_phin_maindiag[-1] + Snr * _n[-1] , dJn_phin_upperdiag[-1] , \
    dJn_phi_maindiag[0] - Snl * _n[0] , dJn_phi_upperdiag[0] , \
    dJn_phi_maindiag[-1] + Snr * _n[-1] , dJn_phi_upperdiag[-1]





def contact_phip( dgrid , peq0 , peqL , phi_p , phi , Chi , Eg , Nv , mp , Spl , Spr ):
    """
    Computes left side term of the Dirichlet out of equilibrium boundary condition for hole quasi-Fermi energy.

    Parameters
    ----------
        dgrid    : numpy array , shape = ( N - 1 )
            array of distances between consecutive grid points
        peq0     : float
            hole equilibrium density at left boundary
        peqL     : float
            hole equilibrium density at right boundary
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
        Spl      : float
            hole surface recombination velocity at left boundary
        Spr      : float
            hole surface recombination velocity at right boundary

    Returns
    -------
        float
            left side term of left boundary condition
        float
            left side term of right boundary condition

    """
    _p = p( phi_p , phi , Chi , Eg , Nv )
    _Jp = Jp( dgrid , phi_p , phi , Chi , Eg , Nv , mp )
    return _Jp[0] + Spl * ( _p[0] - peq0 ) , _Jp[-1] - Spr * ( _p[-1] - peqL )





def contact_phip_deriv( dgrid , phi_p , phi , Chi , Eg , Nv , mp , Spl , Spr ):
    """
    Computes the derivatives of the Dirichlet out of equilibrium boundary condition for hole quasi-Fermi energy.

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
        Spl      : float
            hole surface recombination velocity at left boundary
        Spr      : float
            hole surface recombination velocity at right boundary

    Returns
    -------
        float
            derivative of left boundary condition with respect to phi_p[0]
        float
            derivative of left boundary condition with respect to phi_p[1]
        float
            derivative of right boundary condition with respect to phi_p[0]
        float
            derivative of right boundary condition with respect to phi_p[1]
        float
            derivative of left boundary condition with respect to phi[0]
        float
            derivative of left boundary condition with respect to phi[1]
        float
            derivative of right boundary condition with respect to phi[0]
        float
            derivative of right boundary condition with respect to phi[1]

    """
    _p = p( phi_p , phi , Chi , Eg , Nv )
    dJp_phip_maindiag , dJp_phip_upperdiag , dJp_phi_maindiag , dJp_phi_upperdiag = Jp_deriv( dgrid , phi_p , phi , Chi , Eg , Nv , mp )

    return dJp_phip_maindiag[0] + Spl * ( - _p[0] ) , dJp_phip_upperdiag[0] , \
    dJp_phip_maindiag[-1] , dJp_phip_upperdiag[-1]  - Spr * ( - _p[-1] ) , \
    dJp_phi_maindiag[0] + Spl * ( - _p[0] ) , dJp_phi_upperdiag[0] , \
    dJp_phi_maindiag[-1] , dJp_phi_upperdiag[-1]  - Spr * ( - _p[-1] )
