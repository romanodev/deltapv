from .physics import *

def rad( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Br ):
    """
    Computes the radiative bulk recombination rate density.

    Parameters
    ----------
        phi_n : numpy array , shape = ( N )
            e- quasi-Fermi energy
        phi_p : numpy array , shape = ( N )
            hole quasi-Fermi energy
        phi   : numpy array , shape = ( N )
            electrostatic potential
        Chi   : numpy array , shape = ( N )
            electron affinity
        Eg    : numpy array , shape = ( N )
            band gap
        Nc    : numpy array , shape = ( N )
            e- density of states
        Nv    : numpy array , shape = ( N )
            hole density of states
        Br    : numpy array , shape = ( N )
            radiative recombination coefficient

    Returns
    -------
        numpy array , shape = ( N )
            Auger recombination rate density

    """
    _ni = ni( Eg , Nc , Nv )
    _n = n( phi_n , phi , Chi , Nc )
    _p = p( phi_p , phi , Chi , Eg , Nv )
    return Br * ( _n * _p - _ni**2 )





def rad_deriv( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Br ):
    """
    Computes the derivatives of the Auger bulk recombination rate density.

    This function returns the derivative of the Auger recombination with respect to the different potientials
    ( e- and hole quasi-Fermi energy and electrostatic potential ).

    Parameters
    ----------
        phi_n   : numpy array , shape = ( N )
            e- quasi-Fermi energy
        phi_p   : numpy array , shape = ( N )
            hole quasi-Fermi energy
        phi     : numpy array , shape = ( N )
            electrostatic potential
        Chi     : numpy array , shape = ( N )
            electron affinity
        Eg      : numpy array , shape = ( N )
            band gap
        Nc      : numpy array , shape = ( N )
            e- density of states
        Nv      : numpy array , shape = ( N )
            hole density of states
        Br      : numpy array , shape = ( N )
            radiative recombination coefficient

    Returns
    -------
        DR_phin : numpy array , shape = ( N )
            derivative of Auger recombination at point i w.r.t phi_n[i]
        DR_phip : numpy array , shape = ( N )
            derivative of Auger recombination at point i w.r.t phi_p[i]
        DR_phi  : numpy array , shape = ( N )
            derivative of Auger recombination at point i w.r.t phi[i]

    """
    _n = n( phi_n , phi , Chi , Nc )
    _p = p( phi_p , phi , Chi , Eg , Nv )

    DR_phin = Br * ( n * p )
    DR_phip = Br * ( - n * p )
    DR_phi = np.zeros( phi.size )

    return DR_phin , DR_phip , DR_phi
