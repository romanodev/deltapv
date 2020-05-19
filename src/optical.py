from .scales import *
import scipy.constants as const
from .physics import *

def photonflux( Lambda , P_in ):
    """
    Computes the incident photon flux.

    Parameters
    ----------
        Lambda : numpy array , shape = ( M )
            array of light wavelengths
        P_in   : numpy array , shape = ( M )
            array of incident power for every wavelength

    Returns
    -------
        numpy array , shape = ( M )
            array of photon flux for every wavelength

    """
    hc = const.c * const.h * 1e9 # J.nm
    return P_in / ( hc / Lambda )





def alpha( lambdax , Eg , A ):
    """
    Computes the absorption coefficient for a specific wavelength across the system.

    The absorption coefficient is computed based on material properties.
    Currently only the simplest direct band gap semiconductor is implemented.

    Parameters
    ----------
        lambdax : float
            light wavelength
        Eg      : numpy array , shape = ( N )
            array of band gaps
        A       : numpy array , shape = ( N )
            array of coefficients for direct band gap absorption coefficient model

    Returns
    -------
        numpy array , shape = ( N )
            array of absorption coefficient for wavelength lambdax

    """
    T = 300
    KB = const.k
    hc = const.c * const.h * 1e9 # J.nm
    alpha = [ 0.0 for i in range( Eg.size ) ]
    for i in range( len( alpha ) ):
        if ( hc / lambdax / ( KB * T ) > Eg[ i ] ):
            alpha[ i ] = A[ i ] * np.sqrt( hc / lambdax / ( KB * T ) - Eg[ i ] )
    return np.array( alpha )





def alpha_deriv( lambdax , Eg , A ):
    """
    Computes the derivatives of the absorption coefficient for a specific wavelength across the system.

    Currently only the simplest direct band gap semiconductor is implemented.

    Parameters
    ----------
        lambdax    : float
            light wavelength
        Eg         : numpy array , shape = ( N )
            array of band gaps
        A          : numpy array , shape = ( N )
            array of coefficients for direct band gap absorption coefficient model

    Returns
    -------
        dalpha_dEg : numpy array , shape = ( N )
            derivative of alpha[i] with respect to Eg[i]
        dalpha_dA  : numpy array , shape = ( N )
            derivative of alpha[i] with respect to A[i]

    """
    T = 300
    KB = const.k
    hc = const.c * const.h * 1e9 # J.nm
    dalpha_dEg = [ 0.0 for i in range( Eg.size ) ]
    dalpha_dA = [ 0.0 for i in range( Eg.size ) ]
    for i in range( len(dalpha_dEg) ):
        if ( hc / lambdax / ( KB * T ) > Eg[ i ] ):
            dalpha_dEg[ i ] =  - 1 / ( 2 * np.sqrt( hc / lambdax / ( KB * T ) - Eg[ i ] ) )
            dalpha_dA[ i ] = np.sqrt( hc / lambdax / ( KB * T ) - Eg[ i ] )
    return np.array( dalpha_dEg ) , np.array( dalpha_dA )





def generation_lambda( dgrid , alpha , phi_0 ):
    """
    Computes the e-/hole pair generation rate density for a specific wavelength across the system.

    The function takes the array of absorption coefficients linked to that specific wavelength.

    Parameters
    ----------
        dgrid : numpy array , shape = ( N - 1 )
            array of distances between consecutive grid points
        alpha : numpy array , shape = ( N )
            array of absorption coefficient
        phi_0 : float
            incoming photon flux

    Returns
    -------
        numpy array , shape = ( N )
            array of generation rate density across the system

    """
    phi = phi_0 * np.exp( - np.cumsum( np.insert( alpha[:-1] * dgrid, 0, 0 ) ) )
    return phi





def compute_G( dgrid , Eg , Lambda , P_in , A ):
    """
    Computes the total e-/hole pair generation rate density across the system.

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
        numpy array , shape = ( N )
            array of total generation rate density across the system

    """
    scale = scales()
    phi_0 = photonflux( Lambda , P_in )
    tot_generation = 0
    for i in range( Lambda.size ):
        tot_generation += generation_lambda( dgrid , alpha( Lambda[ i ] , Eg , A ) , phi_0[ i ] )
    return 1 / scale['U'] * tot_generation





def compute_G( dgrid , Eg , Lambda , P_in , A ):
    """
    Computes the total e-/hole pair generation rate density across the system.

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
        numpy array , shape = ( N )
            array of total generation rate density across the system

    """
    scale = scales()
    phi_0 = photonflux( Lambda , P_in )
    tot_generation = 0
    for i in range( Lambda.size ):
        tot_generation += generation_lambda( dgrid , alpha( Lambda[ i ] , Eg , A ) , phi_0[ i ] )
    return 1 / scale['U'] * tot_generation





def deriv_G( dgrid , Eg , Lambda , P_in , A ):
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
    scale = scales()
    phi_0 = photonflux( Lambda , P_in )
    G = 0
    dG_dEg , dG_dA = np.zeros( ( Eg.size , Eg.size ) )
    for i in range( Lambda.size ):
        G_at_lambda = generation_lambda( dgrid , alpha( Lambda[ i ] , Eg , A ) , phi_0[ i ] )
        G += G_at_lambda
        dalpha_dEg , dalpha_dA = alpha_deriv( Lambda[ i ] , Eg , A )
        dG_dEg = - G_at_lambda * np.cumsum( dalpha_dEg , dgrid )
        dG_dA = - G_at_lambda * np.cumsum( dalpha_dA , dgrid )

    G *= 1 / scale['U']
    dG_dEg *= 1 / scale['U']
    dG_dA *= 1 / scale['U']
    return dG_dEg , dG_dA
