from .F import *
if USE_JAX:
    from jax import jit , jacfwd

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
        print( '                {0:02d}              {1:.9f}           {2:.9f}'.format( iter , float( error_F ) , float( error_dx ) ) )

    return phis





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
    N = dgrid.size + 1
    error = 1
    iter = 0
    grad_step = jit( jacfwd( step_forgrad , ( 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11, 12 , 13 , 14 , 15 , 16 , 17 , 18 , 19 , 20 , 21 , 22 , 23 , 24 ) ) )

    phis = phis_ini

    dphis_dphiini = np.eye( 3 * N )
    dphis_dneq0 = np.zeros( ( 3 * N , 1 ) )
    dphis_dneqL = np.zeros( ( 3 * N , 1 ) )
    dphis_dpeq0 = np.zeros( ( 3 * N , 1 ) )
    dphis_dpeqL = np.zeros( ( 3 * N , 1 ) )
    dphis_dSnl = np.zeros( ( 3 * N , 1 ) )
    dphis_dSpl = np.zeros( ( 3 * N , 1 ) )
    dphis_dSnr = np.zeros( ( 3 * N , 1 ) )
    dphis_dSpr = np.zeros( ( 3 * N , 1 ) )
    dphis_deps = np.zeros( ( 3 * N , N ) )
    dphis_dChi = np.zeros( ( 3 * N , N ) )
    dphis_dEg = np.zeros( ( 3 * N , N ) )
    dphis_dNc = np.zeros( ( 3 * N , N ) )
    dphis_dNv = np.zeros( ( 3 * N , N ) )
    dphis_dNdop = np.zeros( ( 3 * N , N ) )
    dphis_dmn = np.zeros( ( 3 * N , N ) )
    dphis_dmp = np.zeros( ( 3 * N , N ) )
    dphis_dEt = np.zeros( ( 3 * N , N ) )
    dphis_dtn = np.zeros( ( 3 * N , N ) )
    dphis_dtp = np.zeros( ( 3 * N , N ) )
    dphis_dBr = np.zeros( ( 3 * N , N ) )
    dphis_dCn = np.zeros( ( 3 * N , N ) )
    dphis_dCp = np.zeros( ( 3 * N , N ) )
    dphis_dG = np.zeros( ( 3 * N , N ) )

    while (error > 1e-6):
        error_dx , error_F , next_phis = step( dgrid , neq0 , neqL , peq0 , peqL , phis , eps , Chi , Eg , Nc , Nv , Ndop , mn , mp , Et , tn , tp , Br , Cn , Cp , Snl , Spl , Snr , Spr , G )
        gradstep = grad_step( dgrid , neq0 , neqL , peq0 , peqL , phis , eps , Chi , Eg , Nc , Nv , Ndop , mn , mp , Et , tn , tp , Br , Cn , Cp , Snl , Spl , Snr , Spr , G )
        phis = next_phis

        print( dphis_dneq0.shape )
        print( gradstep[0].shape )
        print( gradstep[4].shape )
        dphis_dneq0 = gradstep[0] + np.dot( gradstep[4] , dphis_dneq0 )
        print( dphis_dneq0.shape )
        quit()
        dphis_dneqL = gradstep[1] + np.dot( gradstep[4] , dphis_dneqL )
        dphis_dpeq0 = gradstep[2] + np.dot( gradstep[4] , dphis_dpeq0 )
        dphis_dpeqL = gradstep[3] + np.dot( gradstep[4] , dphis_dpeqL )
        dphis_dSnl = gradstep[19] + np.dot( gradstep[4] , dphis_dSnl )
        dphis_dSpl = gradstep[20] + np.dot( gradstep[4] , dphis_dSpl )
        dphis_dSnr = gradstep[21] + np.dot( gradstep[4] , dphis_dSnr )
        dphis_dSpr = gradstep[22] + np.dot( gradstep[4] , dphis_dSpr )
        dphis_deps = gradstep[5] + np.dot( gradstep[4] , dphis_deps )
        dphis_dChi = gradstep[6] + np.dot( gradstep[4] , dphis_dChi )
        dphis_dEg = gradstep[7] + np.dot( gradstep[4] , dphis_dEg )
        dphis_dNc = gradstep[8] + np.dot( gradstep[4] , dphis_dNc )
        dphis_dNv = gradstep[9] + np.dot( gradstep[4] , dphis_dNv )
        dphis_dNdop = gradstep[10] + np.dot( gradstep[4] , dphis_dNdop )
        dphis_dmn = gradstep[11] + np.dot( gradstep[4] , dphis_dmn )
        dphis_dmp = gradstep[12] + np.dot( gradstep[4] , dphis_dmp )
        dphis_dEt = gradstep[13] + np.dot( gradstep[4] , dphis_dEt )
        dphis_dtn = gradstep[14] + np.dot( gradstep[4] , dphis_dtn )
        dphis_dtp = gradstep[15] + np.dot( gradstep[4] , dphis_dtp )
        dphis_dBr = gradstep[16] + np.dot( gradstep[4] , dphis_dBr )
        dphis_dCn = gradstep[17] + np.dot( gradstep[4] , dphis_dCn )
        dphis_dCp = gradstep[18] + np.dot( gradstep[4] , dphis_dCp )
        dphis_dG = gradstep[23] + np.dot( gradstep[4] , dphis_dG )

        error = error_dx
        iter += 1
        print( '                {0:02d}              {1:.9f}           {2:.9f}'.format( iter , float( error_F ) , float( error_dx ) ) )


    grad_phis = {}
    grad_phis['neq0'] = dphis_dneq0
    grad_phis['neqL'] = dphis_dneqL
    grad_phis['peq0'] = dphis_dpeq0
    grad_phis['peqL'] = dphis_dpeqL
    grad_phis['phi_ini'] = dphis_dphiini
    grad_phis['eps'] = dphis_deps
    grad_phis['Chi'] = dphis_dChi
    grad_phis['Eg'] = dphis_dEg
    grad_phis['Nc'] = dphis_dNc
    grad_phis['Nv'] = dphis_dNv
    grad_phis['Ndop'] = dphis_dNdop
    grad_phis['mn'] = dphis_dmn
    grad_phis['mp'] = dphis_dmp
    grad_phis['Et'] = dphis_dEt
    grad_phis['tn'] = dphis_dtn
    grad_phis['tp'] = dphis_dtp
    grad_phis['Br'] = dphis_dBr
    grad_phis['Cn'] = dphis_dCn
    grad_phis['Cp'] = dphis_dCp
    grad_phis['G'] = dphis_dG

    return phis , grad_phis
