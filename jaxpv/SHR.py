from .physics import *

def SHR( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Et , tn , tp ):
    """
    Computes the Schokley-Read-Hall bulk recombination rate density.

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
        Et    : numpy array , shape = ( N )
            SHR trap state energy level
        tn    : numpy array , shape = ( N )
            SHR e- lifetime
        tp    : numpy array , shape = ( N )
            SHR hole lifetime

    Returns
    -------
        numpy array , shape = ( N )
            SHR recombination rate density

    """
    _ni = ni( Eg , Nc , Nv )
    _n = n( phi_n , phi , Chi , Nc )
    _p = p( phi_p , phi , Chi , Eg , Nv )
    nR = _ni * np.exp( Et ) + _n
    pR = _ni * np.exp( - Et ) + _p
    return ( _n * _p - _ni**2 ) / ( tp * nR + tn * pR )





def SHR_deriv( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Et , tn , tp ):
    """
    Computes the derivatives of the Schokley-Read-Hall bulk recombination rate density.

    This function returns the derivative of the SHR recombination with respect to the different potientials
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
        Et      : numpy array , shape = ( N )
            SHR trap state energy level
        tn      : numpy array , shape = ( N )
            SHR e- lifetime
        tp      : numpy array , shape = ( N )
            SHR hole lifetime

    Returns
    -------
        DR_phin : numpy array , shape = ( N )
            derivative of SHR recombination at point i w.r.t phi_n[i]
        DR_phip : numpy array , shape = ( N )
            derivative of SHR recombination at point i w.r.t phi_p[i]
        DR_phi  : numpy array , shape = ( N )
            derivative of SHR recombination at point i w.r.t phi[i]

    """
    _ni = ni( Eg , Nc , Nv )
    _n = n( phi_n , phi , Chi , Nc )
    _p = p( phi_p , phi , Chi , Eg , Nv )
    nR = _ni * np.exp( Et ) + _n
    pR = _ni * np.exp( - Et ) + _p
    num = _n * _p - _ni**2
    denom = ( tp * nR + tn * pR )

    DR_phin = ( ( _n * _p ) * denom - num * ( tp * _n ) ) * denom**-2
    DR_phip = ( ( - _n * _p ) * denom - num * ( - tn * _p ) ) * denom**-2
    DR_phi = ( - num * ( tp * _n - tn * _p ) ) * denom**-2

    return DR_phin , DR_phip , DR_phi
