from .physics import *

def pois( dgrid , phi_n , phi_p , phi , eps , Chi , Eg , Nc , Nv , Ndop ):
    """
    Computes the left side term of the Poisson equation.

    This function computes the left side of the Poissson equation,
    i.e. it returns the array : d/dx( eps d/dx phi ) - charge.

    Parameters
    ----------
        dgrid : numpy array , shape = ( N - 1 )
            array of distances between consecutive grid points
        phi_n : numpy array , shape = ( N )
            e- quasi-Fermi energy
        phi_p : numpy array , shape = ( N )
            hole quasi-Fermi energy
        phi   : numpy array , shape = ( N )
            electrostatic potential
        eps   : numpy array , shape = ( N )
            relative dieclectric constant
        Chi   : numpy array , shape = ( N )
            electron affinity
        Eg    : numpy array , shape = ( N )
            band gap
        Nc    : numpy array , shape = ( N )
            e- density of states
        Nv    : numpy array , shape = ( N )
            hole density of states
        Ndop  : numpy array , shape = ( N - 2 )
            dopant density ( positive for donors , negative for acceptors )

    Returns
    -------
        numpy array , shape = ( N - 2 )
            d/dx( eps d/dx ( phi ) ) - charge

    """
    ave_dgrid = ( dgrid[:-1] + dgrid[1:] ) / 2.0
    ave_eps = 0.5 * ( eps[1:] + eps[:-1] )
    pois = ( ave_eps[:-1] * ( phi[1:-1] - phi[:-2] ) / dgrid[:-1] - ave_eps[1:] * ( phi[2:] - phi[1:-1] ) / dgrid[1:] ) / ave_dgrid - charge( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Ndop )[1:-1]

    return pois





def pois_deriv_eq( dgrid , phi_n , phi_p , phi , eps , Chi , Eg , Nc , Nv ):
    """
    Computes the derivatives of the left side term of the Poisson equation at equilibirum.

    This function computes the Jacobian matrix of the left side of the Poissson equation,
    with respect to the electrostatic potential at equilibrium.
    It returns only the non-zero values of the matrix.

    Parameters
    ----------
        dgrid : numpy array , shape = ( N - 1 )
            array of distances between consecutive grid points
        phi_n : numpy array , shape = ( N )
            e- quasi-Fermi energy
        phi_p : numpy array , shape = ( N )
            hole quasi-Fermi energy
        phi   : numpy array , shape = ( N )
            electrostatic potential
        eps   : numpy array , shape = ( N )
            relative dieclectric constant
        Chi   : numpy array , shape = ( N )
            electron affinity
        Eg    : numpy array , shape = ( N )
            band gap
        Nc    : numpy array , shape = ( N )
            e- density of states
        Nv    : numpy array , shape = ( N )
            hole density of states

    Returns
    -------
        dpois_phi_ : numpy array , shape = ( N - 2 )
            derivative of the i-th Poisson term with respect to phi[i-1]
        dpois_phi__ : numpy array , shape = ( N - 2 )
            derivative of the i-th Poisson term with respect to phi[i]
        dpois_phi___ : numpy array , shape = ( N - 2 )
            derivative of the i-th Poisson term with respect to phi[i+1]

    """
    ave_dgrid = ( dgrid[:-1] + dgrid[1:] ) / 2.0
    ave_eps = 0.5 * ( eps[1:] + eps[:-1] )
    _n = n( phi_n , phi , Chi , Nc )
    _p = p( phi_p , phi , Chi , Eg , Nv )

    dchg_phi = - _n - _p

    dpois_phi_ =  - ave_eps[:-1] / dgrid[:-1] / ave_dgrid
    dpois_phi__ =  ( ave_eps[:-1] / dgrid[:-1] + ave_eps[1:] / dgrid[1:] ) / ave_dgrid - dchg_phi[1:-1]
    dpois_phi___ =  - ave_eps[1:] / dgrid[1:] / ave_dgrid

    return dpois_phi_ , dpois_phi__ , dpois_phi___





def pois_deriv( dgrid , phi_n , phi_p , phi , eps , Chi , Eg , Nc , Nv ):
    """
    Computes the derivatives of the left side term of the Poisson equation.

    This function computes the Jacobian matrix of the left side of the Poissson equation,
    with respect to the potentials (i.e. e- and hole quasi-Fermi energy and electrostatic potential).
    It returns only the non-zero values of the matrix.

    Parameters
    ----------
        phi_n : numpy array , shape = ( N )
            e- quasi-Fermi energy
        phi_p : numpy array , shape = ( N )
            hole quasi-Fermi energy
        phi   : numpy array , shape = ( N )
            electrostatic potential
        dgrid : numpy array , shape = ( N - 1 )
            array of distances between consecutive grid points
        eps   : numpy array , shape = ( N )
            relative dieclectric constant
        Chi   : numpy array , shape = ( N )
            electron affinity
        Eg    : numpy array , shape = ( N )
            band gap
        Nc    : numpy array , shape = ( N )
            e- density of states
        Nv    : numpy array , shape = ( N )
            hole density of states

    Returns
    -------
        dpois_phi_ : numpy array , shape = ( N - 2 )
            derivative of the i-th Poisson term with respect to phi[i-1]
        dpois_phi__ : numpy array , shape = ( N - 2 )
            derivative of the i-th Poisson term with respect to phi[i]
        dpois_phi___ : numpy array , shape = ( N - 2 )
            derivative of the i-th Poisson term with respect to phi[i+1]
        dpois_dphin__ : numpy array , shape = ( N - 2 )
            derivative of the i-th Poisson term with respect to phi_n[i]
        dpois_dphip__ : numpy array , shape = ( N - 2 )
            derivative of the i-th Poisson term with respect to phi_p[i]

    """
    ave_dgrid = ( dgrid[:-1] + dgrid[1:] ) / 2.0
    ave_eps = 0.5 * ( eps[1:] + eps[:-1] )
    _n = n( phi_n , phi , Chi , Nc )
    _p = p( phi_p , phi , Chi , Eg , Nv )

    dchg_phi_n = - _n
    dchg_phi_p = - _p
    dchg_phi = - _n - _p

    dpois_phi_ =  - ave_eps[:-1] / dgrid[:-1] / ave_dgrid
    dpois_phi__ =  ( ave_eps[:-1] / dgrid[:-1] + ave_eps[1:] / dgrid[1:] ) / ave_dgrid - dchg_phi[1:-1]
    dpois_phi___ =  - ave_eps[1:] / dgrid[1:] / ave_dgrid

    dpois_dphin__ = - dchg_phi_n[1:-1]
    dpois_dphip__ = - dchg_phi_p[1:-1]

    return dpois_phi_ , dpois_phi__ , dpois_phi___ , dpois_dphin__ , dpois_dphip__
