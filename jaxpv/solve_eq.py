from .F_eq import *
from .utils import *
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import gmres, spilu, LinearOperator


def damp(move):
    """
    Computes a damped displacement of electrostatic potential from the Newton method displacement.

    Parameters
    ----------
        move : numpy array , shape = ( N )
            displacement in electrostatic potential computed using the Newton method formula

    Returns
    -------
        numpy array , shape = ( N )
            damped displacement in electrostatic potential

    """
    tmp = 1e10
    approx_sign = np.tanh(tmp * move)
    approx_abs = approx_sign * move
    approx_H = 1 - (1 + tmp * np.exp(-(move**2 - 1)))**(-1)
    return np.log(1 + approx_abs) * approx_sign + approx_H * (
        move - np.log(1 + approx_abs) * approx_sign)


#@jit
def step_eq(dgrid, phi, eps, Chi, Eg, Nc, Nv, Ndop):
    """
    Computes the next electrostatic potential in the Newton method iterative scheme.

    Parameters
    ----------
        dgrid : numpy array , shape = ( N - 1 )
            array of distances between consecutive grid points
        phi   : numpy array , shape = ( N )
            current electrostatic potential
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
        error : float
            norm of the displacement in the electrostatic potential space ( used to estimate the error )
        float
            norm of the value of the polynomial function which defines the equilibrium of the system;
            equilibrium is reached when the zero of the system of equations is found ( printed for user )
        numpy array , shape = ( N )
            next electrostatic potential

    """

    Feq = F_eq(dgrid, np.zeros(phi.size), np.zeros(phi.size), phi, eps, Chi,
               Eg, Nc, Nv, Ndop)
    gradFeq = F_eq_deriv(dgrid, np.zeros(phi.size), np.zeros(phi.size), phi,
                         eps, Chi, Eg, Nc, Nv)

    spgradFeq = csr_matrix(gradFeq)

    lugradFeq = spilu(spgradFeq)
    precond = LinearOperator(gradFeq.shape, lambda x: lugradFeq.solve(x))

    move, conv_info = gmres(spgradFeq,
                            -Feq,
                            tol=1e-9,
                            maxiter=10000,
                            M=precond)

    if conv_info > 0:
        print(f"Early termination of GMRES at {conv_info} iterations")

    error = np.linalg.norm(move)

    damp_move = damp(move)

    return error, np.linalg.norm(Feq), phi + damp_move


#@jit
def step_eq_forgrad(dgrid, phi, eps, Chi, Eg, Nc, Nv, Ndop):
    """
    Computes the next electrostatic potential in the Newton method iterative scheme.

    This function is to be used for gradient calculations with JAX. It doesn't return prints.

    Parameters
    ----------
        dgrid : numpy array , shape = ( N - 1 )
            array of distances between consecutive grid points
        phi   : numpy array , shape = ( N )
            current electrostatic potential
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
            next electrostatic potential

    """
    Feq = F_eq(dgrid, np.zeros(phi.size), np.zeros(phi.size), phi, eps, Chi,
               Eg, Nc, Nv, Ndop)
    gradFeq = F_eq_deriv(dgrid, np.zeros(phi.size), np.zeros(phi.size), phi,
                         eps, Chi, Eg, Nc, Nv)
    move = np.linalg.solve(gradFeq, -Feq)
    damp_move = damp(move)

    return phi + damp_move


def solve_eq(dgrid, phi_ini, eps, Chi, Eg, Nc, Nv, Ndop):
    """
    Solves for the equilibrium electrostatic potential using the Newton method.

    Parameters
    ----------
        dgrid   : numpy array , shape = ( N - 1 )
            array of distances between consecutive grid points
        phi_ini : numpy array , shape = ( N )
            initial guess for the electrostatic potential
        eps     : numpy array , shape = ( N )
            relative dieclectric constant
        Chi     : numpy array , shape = ( N )
            electron affinity
        Eg      : numpy array , shape = ( N )
            band gap
        Nc      : numpy array , shape = ( N )
            e- density of states
        Nv      : numpy array , shape = ( N )
            hole density of states
        Ndop    : numpy array , shape = ( N )
            dopant density ( positive for donors , negative for acceptors )

    Returns
    -------
        phi     : numpy array , shape = ( N )
            equilibrium electrostatic potential

    """

    error = 1
    iter = 0
    print(' ')
    print('Solving equilibrium...')
    print(' ')
    print(' Iteration       |F(x)|                Residual     ')
    print(
        ' -------------------------------------------------------------------')

    phi = phi_ini
    while (error > 1e-9):
        if iter > 100:
            print("Maximum steps exceeded! Ending iteration")
            break
        error_dx, error_F, next_phi = step_eq(dgrid, phi, eps, Chi, Eg, Nc, Nv,
                                              Ndop)
        phi = next_phi
        error = error_dx

        iter += 1
        print('    {0:02d}          {1:.5E}          {2:.5E}'.format(
            iter, error_F.astype(float), error_dx.astype(float)))

    print(
        ' -------------------------------------------------------------------')
    print(' ')
    print('Solving equilibrium... done.')
    print(' ')

    return phi


def solve_eq_forgrad(dgrid, phi_ini, eps, Chi, Eg, Nc, Nv, Ndop):
    """
    Solves for the equilibrium electrostatic potential using the Newton method and computes derivatives.

    This function is to be used for gradient calculations with JAX. It doesn't print variables.

    Parameters
    ----------
        dgrid   : numpy array , shape = ( N - 1 )
            array of distances between consecutive grid points
        phi_ini : numpy array , shape = ( N )
            initial guess for the electrostatic potential
        eps     : numpy array , shape = ( N )
            relative dieclectric constant
        Chi     : numpy array , shape = ( N )
            electron affinity
        Eg      : numpy array , shape = ( N )
            band gap
        Nc      : numpy array , shape = ( N )
            e- density of states
        Nv      : numpy array , shape = ( N )
            hole density of states
        Ndop    : numpy array , shape = ( N )
            dopant density ( positive for donors , negative for acceptors )

    Returns
    -------
        phi     : numpy array , shape = ( N )
            equilibrium electrostatic potential

    """
    N = dgrid.size + 1
    error = 1
    iter = 0
    print(
        'Equilibrium     Iteration       |F(x)|                Residual     ')
    print(
        '-------------------------------------------------------------------')
    grad_step = jit(jacfwd(step_eq_forgrad, (1, 2, 3, 4, 5, 6, 7)))

    phi = phi_ini

    dphi_dphiini = np.eye(N)
    dphi_deps = np.zeros((N, N))
    dphi_dChi = np.zeros((N, N))
    dphi_dEg = np.zeros((N, N))
    dphi_dNc = np.zeros((N, N))
    dphi_dNv = np.zeros((N, N))
    dphi_dNdop = np.zeros((N, N))

    while (error > 1e-6):
        error_dx, error_F, next_phi = step_eq(dgrid, phi, eps, Chi, Eg, Nc, Nv,
                                              Ndop)
        gradstep = grad_step(dgrid, phi, eps, Chi, Eg, Nc, Nv, Ndop)
        phi = next_phi

        dphi_dphiini = np.dot(gradstep[0], dphi_dphiini)
        dphi_deps = gradstep[1] + np.dot(gradstep[0], dphi_deps)
        dphi_dChi = gradstep[2] + np.dot(gradstep[0], dphi_dChi)
        dphi_dEg = gradstep[3] + np.dot(gradstep[0], dphi_dEg)
        dphi_dNc = gradstep[4] + np.dot(gradstep[0], dphi_dNc)
        dphi_dNv = gradstep[5] + np.dot(gradstep[0], dphi_dNv)
        dphi_dNdop = gradstep[6] + np.dot(gradstep[0], dphi_dNdop)

        error = error_dx
        iter += 1
        print('                {0:02d}              {1:.9f}           {2:.9f}'.
              format(iter, float(error_F), float(error_dx)))

    grad_phi = {}
    grad_phi['phi_ini'] = dphi_dphiini
    grad_phi['eps'] = dphi_deps
    grad_phi['Chi'] = dphi_dChi
    grad_phi['Eg'] = dphi_dEg
    grad_phi['Nc'] = dphi_dNc
    grad_phi['Nv'] = dphi_dNv
    grad_phi['Ndop'] = dphi_dNdop

    return phi, grad_phi
