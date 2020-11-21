from .physics import *


def auger(phi_n, phi_p, phi, Chi, Eg, Nc, Nv, Cn, Cp):
    """
    Computes the Auger bulk recombination rate density.

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
        Cn     : numpy array , shape = ( N )
            electron Auger coefficient
        Cp     : numpy array , shape = ( N )
            hole Auger coefficient

    Returns
    -------
        numpy array , shape = ( N )
            Auger recombination rate density

    """
    _ni = ni(Eg, Nc, Nv)
    _n = n(phi_n, phi, Chi, Nc)
    _p = p(phi_p, phi, Chi, Eg, Nv)
    return (Cn * _n + Cp * _p) * (_n * _p - _ni**2)


def auger_deriv(phi_n, phi_p, phi, Chi, Eg, Nc, Nv, Cn, Cp):
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
        Cn     : numpy array , shape = ( N )
            electron Auger coefficient
        Cp     : numpy array , shape = ( N )
            hole Auger coefficient

    Returns
    -------
        DR_phin : numpy array , shape = ( N )
            derivative of Auger recombination at point i w.r.t phi_n[i]
        DR_phip : numpy array , shape = ( N )
            derivative of Auger recombination at point i w.r.t phi_p[i]
        DR_phi  : numpy array , shape = ( N )
            derivative of Auger recombination at point i w.r.t phi[i]

    """
    _ni = ni(Eg, Nc, Nv)
    _n = n(phi_n, phi, Chi, Nc)
    _p = p(phi_p, phi, Chi, Eg, Nv)

    DR_phin = (Cn * _n) * (_n * _p - _ni**2) + (Cn * _n + Cp * _p) * (_n * _p)
    DR_phip = (-Cp * _p) * (_n * _p - _ni**2) + (Cn * _n + Cp * _p) * (-_n *
                                                                       _p)
    DR_phi = (Cn * _n - Cp * _p) * (_n * _p - _ni**2)

    return DR_phin, DR_phip, DR_phi
