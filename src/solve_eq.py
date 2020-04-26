from .F_eq import *
if USE_JAX:
    from jax import jit

def damp( move ):
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
    approx_sign = np.tanh( move )
    approx_abs = approx_sign * move
    approx_H = 1 - ( 1 + np.exp( - 500 * ( move**2 - 1 ) ) )**(-1)
    return np.log( 1 + approx_abs ) * approx_sign + approx_H * ( move - np.log( 1 + approx_abs ) * approx_sign )





@jit
def step_eq( dgrid , phi , eps , Chi , Eg , Nc , Nv , Ndop ):
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
    Feq = F_eq( dgrid , np.zeros( phi.size ) , np.zeros( phi.size ) , phi , eps , Chi , Eg , Nc , Nv , Ndop )
    gradFeq = F_eq_deriv( dgrid , np.zeros( phi.size ) , np.zeros( phi.size ) , phi , eps , Chi , Eg , Nc , Nv )
    move = np.linalg.solve( gradFeq , - Feq )
    error = np.linalg.norm( move )
    damp_move = damp(move)

    return error , np.linalg.norm( Feq ) , phi + damp_move





@jit
def step_eq_forgrad( dgrid , phi , eps , Chi , Eg , Nc , Nv , Ndop ):
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
    Feq = F_eq( dgrid , np.zeros( phi.size ) , np.zeros( phi.size ) , phi , eps , Chi , Eg , Nc , Nv , Ndop )
    gradFeq = F_eq_deriv( dgrid , np.zeros( phi.size ) , np.zeros( phi.size ) , phi , eps , Chi , Eg , Nc , Nv )
    move = np.linalg.solve( gradFeq , - Feq )
    damp_move = damp(move)

    return phi + damp_move





def solve_eq( dgrid , phi_ini , eps , Chi , Eg , Nc , Nv , Ndop ):
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
    print( 'Equilibrium     Iteration       |F(x)|                Residual     ' )
    print( '-------------------------------------------------------------------' )

    phi = phi_ini
    while (error > 1e-6):
        error_dx , error_F , next_phi = step_eq( dgrid , phi , eps , Chi , Eg , Nc , Nv , Ndop )
        print( error_dx )
        print( error_F )
        print( iter )
        phi = next_phi
        error = error_dx
        iter += 1
        print( '                {0:02d}              {1:.9f}           {2:.9f}'.format( str( iter ) , str( error_F ) , str( error_dx ) ) )

    return phi





@jit
def solve_eq_forgrad( dgrid , phi_ini , eps , Chi , Eg , Nc , Nv , Ndop ):
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
    step_nums = 10

    phi = phi_ini
    for i in range( step_nums ):
        next_phi = step_eq_forgrad( dgrid , phi , eps , Chi , Eg , Nc , Nv , Ndop )
        phi = next_phi

    return phi
