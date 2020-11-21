from .poisson import *
from .utils import *


def F_eq(dgrid, phi_n, phi_p, phi, eps, Chi, Eg, Nc, Nv, Ndop):
    """
    Computes the system of equations to solve for the equilibrium electrostatic potential (i.e. the poisson equation).

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
        Ndop  : numpy array , shape = ( N )
            dopant density ( positive for donors , negative for acceptors )

    Returns
    -------
        numpy array , shape = ( N )
            equilibrium equation system at current value of potentials

    """
    _pois = pois(dgrid, phi_n, phi_p, phi, eps, Chi, Eg, Nc, Nv, Ndop)
    return np.concatenate((np.array([0.0]), _pois, np.array([0.0])))


def F_eq_deriv(dgrid, phi_n, phi_p, phi, eps, Chi, Eg, Nc, Nv):
    """
    Computes the Jacobian of the system of equations to solve for the equilibrium electrostatic potential.

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
        Ndop  : numpy array , shape = ( N )
            dopant density ( positive for donors , negative for acceptors )

    Returns
    -------
        numpy array , shape = ( N x N )
            Jacobian matrix of equilibrium equation system at current value of potentials

    """
    N = phi.size
    dpois_phi_, dpois_phi__, dpois_phi___ = pois_deriv_eq(
        dgrid, phi_n, phi_p, phi, eps, Chi, Eg, Nc, Nv)

    row = np.array([0])
    col = np.array([0])
    dFeq = np.array([1.0])

    row = np.concatenate((row, np.arange(1, N - 1, 1)))
    col = np.concatenate((col, np.arange(0, N - 2, 1)))
    dFeq = np.concatenate((dFeq, dpois_phi_))

    row = np.concatenate((row, np.arange(1, N - 1, 1)))
    col = np.concatenate((col, np.arange(1, N - 1, 1)))
    dFeq = np.concatenate((dFeq, dpois_phi__))

    row = np.concatenate((row, np.arange(1, N - 1, 1)))
    col = np.concatenate((col, np.arange(2, N, 1)))
    dFeq = np.concatenate((dFeq, dpois_phi___))

    row = np.concatenate((row, np.array([N - 1])))
    col = np.concatenate((col, np.array([N - 1])))
    dFeq = np.concatenate((dFeq, np.array([1.0])))

    result = np.zeros((N, N))
    if os.environ['JAX'] == 'YES':
        return ops.index_update(result, (row, col), dFeq)
    else:
        for i in range(len(row)):
            result[row[i], col[i]] = dFeq[i]
        return result
