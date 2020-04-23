from .F import *
if USE_JAX:
    from jax import jit

def damp( move ):
    """
    Computes a damped move of potentials from the Newton method displacement.

    Parameters
    ----------
        move : numpy array , shape = ( 3N )
            displacement in e-/hole quasi-Fermi energy and electrostatic potential computed from the Newton method

    Returns
    -------
        numpy array , shape = ( 3N )
            damped displacement in potentials

    """
    approx_sign = np.tanh( move )
    approx_abs = approx_sign * move
    approx_H = 1 - ( 1 + np.exp( - 500 * ( move**2 - 1 ) ) )**(-1)
    return np.log( 1 + approx_abs ) * approx_sign + approx_H * ( move - np.log( 1 + approx_abs ) * approx_sign )





@jit
def step( dgrid , neq0 , neqL , peq0 , peqL , phis , eps , Chi , Eg , Nc , Nv , Ndop , mn , mp , Et , tn , tp , Br , Cn , Cp , Snl , Spl , Snr , Spr , G ):
    """
    Computes the next potentials in the Newton method iterative scheme.

    Parameters
    ----------
        dgrid    : numpy array , shape = ( N - 1 )
            array of distances between consecutive grid points
        neq0     : float
            e- equilibrium density at left boundary
        neqL     : float
            e- equilibrium density at right boundary
        peq0     : float
            hole equilibrium density at left boundary
        peqL     : float
            hole equilibrium density at right boundary
        phis     : numpy array , shape = ( 3N )
            current potentials ( e- quasi-Fermi energy / hole quasi-Fermi energy / electrostatic potential )
        eps      : numpy array , shape = ( N )
            relative dieclectric constant
        Chi      : numpy array , shape = ( N )
            electron affinity
        Eg       : numpy array , shape = ( N )
            band gap
        Nc       : numpy array , shape = ( N )
            e- density of states
        Nv       : numpy array , shape = ( N )
            hole density of states
        Ndop     : numpy array , shape = ( N )
            dopant density ( positive for donors , negative for acceptors )
        mn       : numpy array , shape = ( N )
            e- mobility
        mp       : numpy array , shape = ( N )
            hole mobility
        Et       : numpy array , shape = ( N )
            SHR trap state energy level
        tn       : numpy array , shape = ( N )
            SHR e- lifetime
        tp       : numpy array , shape = ( N )
            SHR hole lifetime
        Br       : numpy array , shape = ( N )
            radiative recombination coefficient
        Cn       : numpy array , shape = ( N )
            electron Auger coefficient
        Cp       : numpy array , shape = ( N )
            hole Auger coefficient
        Snl      : float
            e- surface recombination velocity at left boundary
        Spl      : float
            hole surface recombination velocity at left boundary
        Snr      : float
            e- surface recombination velocity at right boundary
        Spr      : float
            hole surface recombination velocity at right boundary
        G        : numpy array , shape = ( N )
            e-/hole pair generation rate density

    Returns
    -------
        error : float
            norm of the displacement in the potentials space ( used to estimate the error )
        float
            norm of the value of the polynomial function which defines the out-of-equilibrium solution for the system;
            the solution is reached when the zero of the system of equations is found ( printed for user )
        numpy array , shape = ( 3N )
            next potentials ( e- quasi-Fermi energy / hole quasi-Fermi energy / electrostatic potential )

    """
    N = dgrid.size + 1
    _F = F( dgrid , neq0 , neqL , peq0 , peqL , phis[0:N] , phis[N:2*N] , phis[2*N:] , eps , Chi , Eg , Nc , Nv , Ndop , mn , mp , Et , tn , tp , Br , Cn , Cp , Snl , Spl , Snr , Spr , G )
    gradF = F_deriv( dgrid , neq0 , neqL , peq0 , peqL , phis[0:N] , phis[N:2*N] , phis[2*N:] , eps , Chi , Eg , Nc , Nv , Ndop , mn , mp , Et , tn , tp , Br , Cn , Cp , Snl , Spl , Snr , Spr , G )

    move = np.linalg.solve( gradF , - _F )
    error = np.linalg.norm( move )
    damp_move = damp( move )

    return error , np.linalg.norm(_F) , np.concatenate( ( phis[0:N] + damp_move[0:3*N:3] , phis[N:2*N] + damp_move[1:3*N:3] , phis[2*N:]+ damp_move[2:3*N:3] ) , axis = 0 )





@jit
def step_forgrad( dgrid , neq0 , neqL , peq0 , peqL , phis , eps , Chi , Eg , Nc , Nv , Ndop , mn , mp , Et , tn , tp , Br , Cn , Cp , Snl , Spl , Snr , Spr , G ):
    """
    Computes the next potentials in the Newton method iterative scheme.

    This function is to be used for gradient calculations with JAX. It doesn't print variables.

    Parameters
    ----------
        dgrid    : numpy array , shape = ( N - 1 )
            array of distances between consecutive grid points
        neq0     : float
            e- equilibrium density at left boundary
        neqL     : float
            e- equilibrium density at right boundary
        peq0     : float
            hole equilibrium density at left boundary
        peqL     : float
            hole equilibrium density at right boundary
        phis     : numpy array , shape = ( 3N )
            current potentials ( e- quasi-Fermi energy / hole quasi-Fermi energy / electrostatic potential )
        eps      : numpy array , shape = ( N )
            relative dieclectric constant
        Chi      : numpy array , shape = ( N )
            electron affinity
        Eg       : numpy array , shape = ( N )
            band gap
        Nc       : numpy array , shape = ( N )
            e- density of states
        Nv       : numpy array , shape = ( N )
            hole density of states
        Ndop     : numpy array , shape = ( N )
            dopant density ( positive for donors , negative for acceptors )
        mn       : numpy array , shape = ( N )
            e- mobility
        mp       : numpy array , shape = ( N )
            hole mobility
        Et       : numpy array , shape = ( N )
            SHR trap state energy level
        tn       : numpy array , shape = ( N )
            SHR e- lifetime
        tp       : numpy array , shape = ( N )
            SHR hole lifetime
        Br       : numpy array , shape = ( N )
            radiative recombination coefficient
        Cn       : numpy array , shape = ( N )
            electron Auger coefficient
        Cp       : numpy array , shape = ( N )
            hole Auger coefficient
        Snl      : float
            e- surface recombination velocity at left boundary
        Spl      : float
            hole surface recombination velocity at left boundary
        Snr      : float
            e- surface recombination velocity at right boundary
        Spr      : float
            hole surface recombination velocity at right boundary
        G        : numpy array , shape = ( N )
            e-/hole pair generation rate density

    Returns
    -------
        numpy array , shape = ( 3N )
            next potentials ( e- quasi-Fermi energy / hole quasi-Fermi energy / electrostatic potential )

    """
    N = dgrid.size + 1
    _F = F( dgrid , neq0 , neqL , peq0 , peqL , phis[0:N] , phis[N:2*N] , phis[2*N:] , eps , Chi , Eg , Nc , Nv , Ndop , mn , mp , Et , tn , tp , Br , Cn , Cp , Snl , Spl , Snr , Spr , G )
    gradF = F_deriv( dgrid , neq0 , neqL , peq0 , peqL , phis[0:N] , phis[N:2*N] , phis[2*N:] , eps , Chi , Eg , Nc , Nv , Ndop , mn , mp , Et , tn , tp , Br , Cn , Cp , Snl , Spl , Snr , Spr , G )

    move = np.linalg.solve( gradF , - _F )
    damp_move = damp( move )

    return np.concatenate( ( phis[0:N] + damp_move[0:3*N:3] , phis[N:2*N] + damp_move[1:3*N:3] , phis[2*N:]+ damp_move[2:3*N:3] ) , axis = 0 )





def solve( dgrid , neq0 , neqL , peq0 , peqL , phis_ini , eps , Chi , Eg , Nc , Nv , Ndop , mn , mp , Et , tn , tp , Br , Cn , Cp , Snl , Spl , Snr , Spr , G ):
    """
    Solves for the e-/hole quasi-Fermi energies and electrostatic potential using the Newton method.

    Parameters
    ----------
        dgrid    : numpy array , shape = ( N - 1 )
            array of distances between consecutive grid points
        neq0     : float
            e- equilibrium density at left boundary
        neqL     : float
            e- equilibrium density at right boundary
        peq0     : float
            hole equilibrium density at left boundary
        peqL     : float
            hole equilibrium density at right boundary
        phis_ini : numpy array , shape = ( 3N )
            current potentials ( e- quasi-Fermi energy / hole quasi-Fermi energy / electrostatic potential )
        eps      : numpy array , shape = ( N )
            relative dieclectric constant
        Chi      : numpy array , shape = ( N )
            electron affinity
        Eg       : numpy array , shape = ( N )
            band gap
        Nc       : numpy array , shape = ( N )
            e- density of states
        Nv       : numpy array , shape = ( N )
            hole density of states
        Ndop     : numpy array , shape = ( N )
            dopant density ( positive for donors , negative for acceptors )
        mn       : numpy array , shape = ( N )
            e- mobility
        mp       : numpy array , shape = ( N )
            hole mobility
        Et       : numpy array , shape = ( N )
            SHR trap state energy level
        tn       : numpy array , shape = ( N )
            SHR e- lifetime
        tp       : numpy array , shape = ( N )
            SHR hole lifetime
        Br       : numpy array , shape = ( N )
            radiative recombination coefficient
        Cn       : numpy array , shape = ( N )
            electron Auger coefficient
        Cp       : numpy array , shape = ( N )
            hole Auger coefficient
        Snl      : float
            e- surface recombination velocity at left boundary
        Spl      : float
            hole surface recombination velocity at left boundary
        Snr      : float
            e- surface recombination velocity at right boundary
        Spr      : float
            hole surface recombination velocity at right boundary
        G        : numpy array , shape = ( N )
            e-/hole pair generation rate density

    Returns
    -------
        phis     : numpy array , shape = ( 3N )
            solution for the e- quasi-Fermi energy / hole quasi-Fermi energy / electrostatic potential

    """
    error = 1
    iter = 0

    phis = phis_ini
    while (error > 1e-6):
        error_dx , error_F , next_phis = step( dgrid , neq0 , neqL , peq0 , peqL , phis , eps , Chi , Eg , Nc , Nv , Ndop , mn , mp , Et , tn , tp , Br , Cn , Cp , Snl , Spl , Snr , Spr , G )
        phis = next_phis
        error = error_dx
        iter += 1
        print( '                {0:02d}              {1:.9f}           {2:.9f}'.format( iter , error_F , error_dx ) )

    return phis





@jit
def solve_forgrad( dgrid , neq0 , neqL , peq0 , peqL , phis_ini , eps , Chi , Eg , Nc , Nv , Ndop , mn , mp , Et , tn , tp , Br , Cn , Cp , Snl , Spl , Snr , Spr , G ):
    """
    Solves for the e-/hole quasi-Fermi energies and electrostatic potential using the Newton method and computes derivatives.

    This function is to be used for gradient calculations with JAX. It doesn't print variables.

    Parameters
    ----------
        dgrid    : numpy array , shape = ( N - 1 )
            array of distances between consecutive grid points
        neq0     : float
            e- equilibrium density at left boundary
        neqL     : float
            e- equilibrium density at right boundary
        peq0     : float
            hole equilibrium density at left boundary
        peqL     : float
            hole equilibrium density at right boundary
        phis_ini : numpy array , shape = ( 3N )
            current potentials ( e- quasi-Fermi energy / hole quasi-Fermi energy / electrostatic potential )
        eps      : numpy array , shape = ( N )
            relative dieclectric constant
        Chi      : numpy array , shape = ( N )
            electron affinity
        Eg       : numpy array , shape = ( N )
            band gap
        Nc       : numpy array , shape = ( N )
            e- density of states
        Nv       : numpy array , shape = ( N )
            hole density of states
        Ndop     : numpy array , shape = ( N )
            dopant density ( positive for donors , negative for acceptors )
        mn       : numpy array , shape = ( N )
            e- mobility
        mp       : numpy array , shape = ( N )
            hole mobility
        Et       : numpy array , shape = ( N )
            SHR trap state energy level
        tn       : numpy array , shape = ( N )
            SHR e- lifetime
        tp       : numpy array , shape = ( N )
            SHR hole lifetime
        Br       : numpy array , shape = ( N )
            radiative recombination coefficient
        Cn       : numpy array , shape = ( N )
            electron Auger coefficient
        Cp       : numpy array , shape = ( N )
            hole Auger coefficient
        Snl      : float
            e- surface recombination velocity at left boundary
        Spl      : float
            hole surface recombination velocity at left boundary
        Snr      : float
            e- surface recombination velocity at right boundary
        Spr      : float
            hole surface recombination velocity at right boundary
        G        : numpy array , shape = ( N )
            e-/hole pair generation rate density

    Returns
    -------
        phis     : numpy array , shape = ( 3N )
            solution for the e- quasi-Fermi energy / hole quasi-Fermi energy / electrostatic potential

    """
    step_nums = 10

    phis = phis_ini
    for i in range( step_nums ):
        next_phis = step_forgrad( dgrid , neq0 , neqL , peq0 , peqL , phis , eps , Chi , Eg , Nc , Nv , Ndop , mn , mp , Et , tn , tp , Br , Cn , Cp , Snl , Spl , Snr , Spr , G )
        phis = next_phis

    return phis
