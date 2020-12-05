def n(phi_n, phi, Chi, Nc):
    """
    Computes the e- density.

    Parameters
    ----------
        phi_n : numpy array , shape = ( N )
            e- quasi-Fermi energy
        phi   : numpy array , shape = ( N )
            electrostatic potential
        Chi   : numpy array , shape = ( N )
            electron affinity
        Nc    : numpy array , shape = ( N )
            e- density of states

    Returns
    -------
        numpy array , shape = ( N )
            electron density

    """
    return Nc * np.exp(Chi + phi_n + phi)


def p(phi_p, phi, Chi, Eg, Nv):
    """
    Computes the hole density.

    Parameters
    ----------
        phi_p : numpy array , shape = ( N )
            hole quasi-Fermi energy
        phi   : numpy array , shape = ( N )
            electrostatic potential
        Chi   : numpy array , shape = ( N )
            electron affinity
        Eg    : numpy array , shape = ( N )
            band gap
        Nv    : numpy array , shape = ( N )
            hole density of states

    Returns
    -------
        numpy array , shape = ( N )
            hole density

    """
    return Nv * np.exp(-Chi - Eg - phi_p - phi)


def charge(phi_n, phi_p, phi, Chi, Eg, Nc, Nv, Ndop):
    """
    Computes the charge density.

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
        Ndop  : numpy array , shape = ( N )
            dopant density ( positive for donors , negative for acceptors )

    Returns
    -------
        numpy array , shape = ( N )
            charge density

    """
    _n = n(phi_n, phi, Chi, Nc)
    _p = p(phi_p, phi, Chi, Eg, Nv)
    return -_n + _p + Ndop


def ni(Eg, Nc, Nv):
    """
    Computes the intrinsic carrier density.

    Parameters
    ----------
        Eg    : numpy array , shape = ( N )
            band gap
        Nc    : numpy array , shape = ( N )
            e- density of states
        Nv    : numpy array , shape = ( N )
            hole density of states

    Returns
    -------
        numpy array , shape = ( N )
            intrinsic carrier density

    """
    return np.sqrt(Nc * Nv) * np.exp(-Eg / 2)
