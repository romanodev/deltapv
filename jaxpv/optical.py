from . import scaling
from . import physics

import scipy.constants as const

scale = scaling.scales()


def photonflux(data):
    
    Lambda, P_in = data["Lambda"], data["P_in"]
    hc = const.c * const.h * 1e9  # J.nm
    return P_in / (hc / Lambda)


def alpha(data, lambdax):
    
    Eg, A = data["Eg"], data["A"]
    T = 300.
    KB = const.k
    hc = const.c * const.h * 1e9  # J.nm

    alpha = np.where(hc / lambdax / (KB * T) > Eg,
                     A * np.sqrt(hc / lambdax / (KB * T) - Eg), 0)
    
    return alpha


def alpha_deriv(data, lambdax):
    
    Eg, A = data["Eg"], data["A"]
    T = 300.
    KB = const.k
    hc = const.c * const.h * 1e9  # J.nm

    dalpha_dEg = np.where(hc / lambdax / (KB * T) > Eg,
                          -1 / (2 * np.sqrt(hc / lambdax / (KB * T) - Eg)), 0)
    
    dalpha_dA = np.where(hc / lambdax / (KB * T) > Eg,
                         np.sqrt(hc / lambdax / (KB * T) - Eg), 0)
    
    return dalpha_dEg, dalpha_dA


def generation_lambda(data, phi_0, alpha):
    
    dgrid = data["dgrid"]
    phi = phi_0 * np.exp(-np.cumsum(
        np.concatenate([np.zeros(1, dtpye=np.float64), alpha[:-1] * dgrid])))
    g = phi * alpha
    
    return g


def compute_G(data):
    
    dgrid = data["dgrid"]
    Lambda = data["Lambda"]
    
    phi_0 = photonflux(data)
    
    valpha = np.vectorize(alpha, excluded=[0])
    vgenlambda = np.vectorize(generation_lambda,
                              excluded=[0])
    
    alphas = valpha(data, Lambda)
    all_generations = vgenlambda(data, phi_0, alphas)
    
    tot_generation = np.sum(all_generations, axis=0)

    return tot_generation / scale['U']


def deriv_G(data):
    
    dgrid = data["dgrid"]
    Lambda = data["Lambda"]
    
    """
    Computes the derivative of total e-/hole pair generation rate density with respect to the material parameters.

    Parameters
    ----------
        dgrid  : numpy array , shape = ( N - 1 )
            array of distances between consecutive grid points
        Eg     : numpy array , shape = ( N )
            array of band gaps
        Lambda : numpy array , shape = ( M )
            array of light wavelengths
        P_in   : numpy array , shape = ( M )
            array of incident power for every wavelength
        A      : numpy array , shape = ( N )
            array of coefficients for direct band gap absorption coefficient model

    Returns
    -------
        dG_dEg : numpy array , shape = ( N x N )
            Jacobian matrix of the derivatives of G with respect to the band gap
        dG_dA : numpy array , shape = ( N x N )
            Jacobian matrix of the derivatives of G with respect to the coefficient of direct band gap absorption

    """
    
    phi_0 = photonflux(data)
    G = 0
    dG_dEg, dG_dA = np.zeros((Eg.size, Eg.size))
    for i in range(Lambda.size):
        G_at_lambda = generation_lambda(data,
                                        phi_0[i],
                                        alpha(data, Lambda[i]))
        G += G_at_lambda
        dalpha_dEg, dalpha_dA = alpha_deriv(data, Lambda[i])
        dG_dEg = -G_at_lambda * np.cumsum(dalpha_dEg, dgrid)
        dG_dA = -G_at_lambda * np.cumsum(dalpha_dA, dgrid)

    G /= scale['U']

    return dG_dEg / scale['U'], dG_dA / scale['U']
