from .e_drift_diffusion import *
from .h_drift_diffusion import *
from .poisson import *
from .boundary_conditions import *
if USE_JAX:
    from jax.config import config
    config.update("jax_enable_x64", True)
    from jax import ops

def F( dgrid , neq_0 , neq_L , peq_0 , peq_L , phi_n , phi_p , phi , eps , Chi , Eg , Nc , Nv , Ndop , mn , mp , Et , tn , tp , Br , Cn , Cp , Snl , Spl , Snr , Spr , G ):
    """
    Computes the system of equations to solve for the out of equilibrium potentials.

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
        phi_n    : numpy array , shape = ( N )
            e- quasi-Fermi energy
        phi_p    : numpy array , shape = ( N )
            hole quasi-Fermi energy
        phi      : numpy array , shape = ( N )
            electrostatic potential
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
            out of equilibrium equation system at current value of potentials

    """
    _ddn = ddn( dgrid , phi_n , phi_p , phi , Chi , Eg , Nc , Nv , mn , Et , tn , tp , Br , Cn , Cp , G )
    _ddp = ddp( dgrid , phi_n , phi_p , phi , Chi , Eg , Nc , Nv , mp , Et , tn , tp , Br , Cn , Cp , G )
    _pois = pois( dgrid , phi_n , phi_p , phi , eps , Chi , Eg , Nc , Nv , Ndop )
    ctct_0_phin , ctct_L_phin = contact_phin( dgrid , neq_0 , neq_L , phi_n , phi , Chi , Nc , mn , Snl , Snr )
    ctct_0_phip , ctct_L_phip = contact_phip( dgrid , peq_0 , peq_L , phi_p , phi , Chi , Eg , Nv , mp , Spl , Spr )

    result = [ ctct_0_phin , ctct_0_phip , 0.0 ]
    for i in range( len(_pois) ):
        result.append( _ddn[i] )
        result.append( _ddp[i] )
        result.append( _pois[i] )
        print( _ddn[i] )
    quit()
    result = result + [ ctct_L_phin , ctct_L_phip , 0.0 ]
    return np.array( result )





def F_deriv( dgrid , neq_0 , neq_L , peq_0 , peq_L , phi_n , phi_p , phi , eps , Chi , Eg , Nc , Nv , Ndop , mn , mp , Et , tn , tp , Br , Cn , Cp , Snl , Spl , Snr , Spr , G ):
    """
    Computes the Jacobian of the system of equations to solve for the out of equilibrium potentials.

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
        phi_n    : numpy array , shape = ( N )
            e- quasi-Fermi energy
        phi_p    : numpy array , shape = ( N )
            hole quasi-Fermi energy
        phi      : numpy array , shape = ( N )
            electrostatic potential
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
        numpy array , shape = ( 3N x 3N )
            Jacobian matrix of the out of equilibrium equation system at current value of potentials

    """
    dde_phin_ , dde_phin__ , dde_phin___ , dde_phip__ , dde_phi_ , dde_phi__ , dde_phi___ = ddn_deriv( dgrid , phi_n , phi_p , phi , Chi , Eg , Nc , Nv , mn , Et , tn , tp , Br , Cn , Cp , G )
    ddp_phin__ , ddp_phip_ , ddp_phip__ , ddp_phip___ , ddp_phi_ , ddp_phi__ , ddp_phi___ = ddp_deriv( dgrid , phi_n , phi_p , phi , Chi , Eg , Nc , Nv , mp , Et , tn , tp , Br , Cn , Cp , G )
    dpois_phi_ , dpois_phi__ , dpois_phi___ , dpois_dphin__ , dpois_dphip__ = pois_deriv( dgrid , phi_n , phi_p , phi , eps , Chi , Eg , Nc , Nv )
    dctct_phin = contact_phin_deriv( dgrid , phi_n , phi , Chi , Nc , mn , Snl , Snr )
    dctct_phip = contact_phip_deriv( dgrid , phi_p , phi , Chi , Eg , Nv , mp , Spl , Spr )

    N = phi.size

    row = np.array( [ 0 , 0 , 0 , 0 ] )
    col = np.array( [ 0 , 2 , 3 , 5 ] )
    dF = np.array( [ dctct_phin[0] , dctct_phin[4] , dctct_phin[1] , dctct_phin[5] ] )

    row = np.concatenate( ( row , np.array( [ 1 , 1 , 1 , 1 ] ) ) )
    col = np.concatenate( ( col , np.array( [ 1 , 2 , 4 , 5 ] ) ) )
    dF =  np.concatenate( ( dF , np.array( [ dctct_phip[0] , dctct_phip[4] , dctct_phip[1] , dctct_phip[5] ] ) ) )

    row = np.concatenate( ( row , np.array( [ 2 ] ) ) )
    col = np.concatenate( ( col , np.array( [ 2 ] ) ) )
    dF =  np.concatenate( ( dF , np.array( [ 1.0 ] ) ) )

    row = np.concatenate( ( row , np.array( [ 3 * ( N - 1 ) , 3 * ( N - 1 ) , 3 * ( N - 1 ) , 3 * ( N - 1 ) ] ) ) )
    col = np.concatenate( ( col , np.array( [ 3 * ( N - 2 ) , 3 * ( N - 2 ) + 2 , 3 * ( N - 1 ) , 3 * ( N - 1 ) + 2 ] ) ) )
    dF = np.concatenate( ( dF , np.array( [ dctct_phin[2] , dctct_phin[6] , dctct_phin[3] , dctct_phin[7] ] ) ) )

    row = np.concatenate( ( row , np.array( [ 3 * ( N - 1 ) + 1 , 3 * ( N - 1 ) + 1 , 3 * ( N - 1 ) + 1 , 3 * ( N - 1 ) + 1 ] ) ) )
    col = np.concatenate( ( col , np.array( [ 3 * ( N - 2 ) + 1 , 3 * ( N - 2 ) + 2 , 3 * ( N - 1 ) + 1 , 3 * ( N - 1 ) + 2 ] ) ) )
    dF = np.concatenate( ( dF , np.array( [ dctct_phip[2] , dctct_phip[6] , dctct_phip[3] , dctct_phip[7] ] ) ) )

    row = np.concatenate( ( row , np.array( [ 3 * ( N - 1 ) + 2 ] ) ) )
    col = np.concatenate( ( col , np.array( [ 3 * ( N - 1 ) + 2 ] ) ) )
    dF =  np.concatenate( ( dF , np.array( [ 1.0 ] ) ) )


    row = np.concatenate( ( row , np.arange( 3 , 3 * ( N - 1 ) , 3 ) ) )
    col = np.concatenate( ( col , np.arange( 0 , 3 * ( N - 2 ) , 3 ) ) )
    dF = np.concatenate( ( dF , dde_phin_ ) )

    row = np.concatenate( ( row , np.arange( 3 , 3 * ( N - 1 ) , 3 ) ) )
    col = np.concatenate( ( col , np.arange( 3 , 3 * ( N - 1 ) , 3 ) ) )
    dF = np.concatenate( ( dF , dde_phin__ ) )

    row = np.concatenate( ( row , np.arange( 3 , 3 * ( N - 1 ) , 3 ) ) )
    col = np.concatenate( ( col , np.arange( 6 , 3 * N , 3 ) ) )
    dF = np.concatenate( ( dF , dde_phin___ ) )

    row = np.concatenate( ( row , np.arange( 3 , 3 * ( N - 1 ) , 3 ) ) )
    col = np.concatenate( ( col , np.arange( 4 , 3 * ( N - 1 ) + 1 , 3 ) ) )
    dF = np.concatenate( ( dF , dde_phip__ ) )

    row = np.concatenate( ( row , np.arange( 3 , 3 * ( N - 1 ) , 3 ) ) )
    col = np.concatenate( ( col , np.arange( 2 , 3 * ( N - 2 ) + 2 , 3 ) ) )
    dF = np.concatenate( ( dF , dde_phi_ ) )

    row = np.concatenate( ( row , np.arange( 3 , 3 * ( N - 1 ) , 3 ) ) )
    col = np.concatenate( ( col , np.arange( 5 , 3 * ( N - 1 ) + 2 , 3 ) ) )
    dF = np.concatenate( ( dF , dde_phi__ ) )

    row = np.concatenate( ( row , np.arange( 3 , 3 * ( N - 1 ) , 3 ) ) )
    col = np.concatenate( ( col , np.arange( 8 , 3 * N + 2 , 3 ) ) )
    dF = np.concatenate( ( dF , dde_phi___ ) )

    row = np.concatenate( ( row , np.arange( 4 , 3 * ( N - 1 ) + 1 , 3 ) ) )
    col = np.concatenate( ( col , np.arange( 3 , 3 * ( N - 1 ) , 3 ) ) )
    dF = np.concatenate( ( dF , ddp_phin__ ) )

    row = np.concatenate( ( row , np.arange( 4 , 3 * ( N - 1 ) + 1 , 3 ) ) )
    col = np.concatenate( ( col , np.arange( 1 , 3 * ( N - 2 ) + 1 , 3 ) ) )
    dF = np.concatenate( ( dF , ddp_phip_ ) )

    row = np.concatenate( ( row , np.arange( 4 , 3 * ( N - 1 ) + 1 , 3 ) ) )
    col = np.concatenate( ( col , np.arange( 4 , 3 * ( N - 1 ) + 1 , 3 ) ) )
    dF = np.concatenate( ( dF , ddp_phip__ ) )

    row = np.concatenate( ( row , np.arange( 4 , 3 * ( N - 1 ) + 1 , 3 ) ) )
    col = np.concatenate( ( col , np.arange( 7 , 3 * N + 1 , 3 ) ) )
    dF = np.concatenate( ( dF , ddp_phip___ ) )

    row = np.concatenate( ( row , np.arange( 4 , 3 * ( N - 1 ) + 1 , 3 ) ) )
    col = np.concatenate( ( col , np.arange( 2 , 3 * ( N - 2 ) + 2 , 3 ) ) )
    dF = np.concatenate( ( dF , ddp_phi_ ) )

    row = np.concatenate( ( row , np.arange( 4 , 3 * ( N - 1 ) + 1 , 3 ) ) )
    col = np.concatenate( ( col , np.arange( 5 , 3 * ( N - 1 ) + 2 , 3 ) ) )
    dF = np.concatenate( ( dF , ddp_phi__ ) )

    row = np.concatenate( ( row , np.arange( 4 , 3 * ( N - 1 ) + 1 , 3 ) ) )
    col = np.concatenate( ( col , np.arange( 8 , 3 * N + 2 , 3 ) ) )
    dF = np.concatenate( ( dF , ddp_phi___ ) )

    row = np.concatenate( ( row , np.arange( 5 , 3 * ( N - 1 ) + 2 , 3 ) ) )
    col = np.concatenate( ( col , np.arange( 2 , 3 * ( N - 2 ) + 2 , 3 ) ) )
    dF = np.concatenate( ( dF , dpois_phi_ ) )

    row = np.concatenate( ( row , np.arange( 5 , 3 * ( N - 1 ) + 2 , 3 ) ) )
    col = np.concatenate( ( col , np.arange( 5 , 3 * ( N - 1 ) + 2 , 3 ) ) )
    dF = np.concatenate( ( dF , dpois_phi__ ) )

    row = np.concatenate( ( row , np.arange( 5 , 3 * ( N - 1 ) + 2 , 3 ) ) )
    col = np.concatenate( ( col , np.arange( 8 , 3 * N + 2 , 3 ) ) )
    dF = np.concatenate( ( dF , dpois_phi___ ) )

    row = np.concatenate( ( row , np.arange( 5 , 3 * ( N - 1 ) + 2 , 3 ) ) )
    col = np.concatenate( ( col , np.arange( 3 , 3 * ( N - 1 ) , 3 ) ) )
    dF = np.concatenate( ( dF , dpois_dphin__ ) )

    row = np.concatenate( ( row , np.arange( 5 , 3 * ( N - 1 ) + 2 , 3 ) ) )
    col = np.concatenate( ( col , np.arange( 4 , 3 * ( N - 1 ) + 1 , 3 ) ) )
    dF = np.concatenate( ( dF , dpois_dphip__ ) )

    result = np.zeros( ( 3 * N , 3 * N ) )
    if USE_JAX:
        return ops.index_update( result , ( row , col ) , dF )
    else:
        for i in range( len( row ) ):
            result[ row[ i ] , col[ i ] ] = dF[ i ]
        return result
