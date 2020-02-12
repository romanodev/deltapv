from .e_drift_diffusion import *
from .h_drift_diffusion import *
from .poisson import *
from .boundary_conditions import *
if USE_JAX:
    from jax.config import config
    config.update("jax_enable_x64", True)
    import jax

### Compute the system of equations to solve for the out of equilibrium potentials
## Inputs :
#      phi_n (array:N) -> e- quasi-Fermi energy
#      phi_p (array:N) -> hole quasi-Fermi energy
#      phi (array:N) -> electrostatic potential
#      dgrid (array:N-1) -> array of distances between consecutive grid points
#      eps(array:N) -> relative dieclectric constant
#      Chi (array:N) -> electron affinity
#      Eg (array:N) -> band gap
#      Nc (array:N) -> e- density of states
#      Nv (array:N) -> hole density of states
#      Ndop (array:N) -> dopant density ( = donor density - acceptor density )
#      Et (array:N) -> trap state energy level (SHR)
#      tn (array:N) -> e- lifetime (SHR)
#      tp (array:N) -> hole lifetime (SHR)
#      mn (array:N) -> e- mobility
#      mp (array:N) -> hole mobility
#      G (array:N) -> electron-hole generation rate density
#      Snl (scalar) -> e- surface recombination velocity at left boundary
#      Spl (scalar) -> hole surface recombination velocity at left boundary
#      Snr (scalar) -> e- surface recombination velocity at right boundary
#      Spr (scalar) -> hole surface recombination velocity at right boundary
#      neq_0 (scalar) -> e- density at left boundary
#      neq_L (scalar) -> e- density at right boundary
#      peq_0 (scalar) -> hole density at left boundary
#      peq_L (scalar) -> hole density at right boundary
## Outputs :
#      1 (array:3N) -> out of equilibrium equation system at current value of potentials

def F( phi_n , phi_p , phi , dgrid , eps , Chi , Eg , Nc , Nv , Ndop , Et , tn , tp , mn , mp , G , Snl , Spl , Snr , Spr , neq_0 , neq_L , peq_0 , peq_L ):

    _ddn = ddn( phi_n , phi_p , phi , dgrid , Chi , Eg , Nc , Nv , Et , tn , tp , mn , G )
    _ddp = ddp( phi_n , phi_p , phi , dgrid , Chi , Eg , Nc , Nv , Et , tn , tp , mp , G )
    _pois = pois( phi_n , phi_p , phi , dgrid , eps , Chi , Eg , Nc , Nv , Ndop )
    ctct_0_phin , ctct_L_phin = contact_phin( phi_n , phi , dgrid , Chi , Nc , mn , Snl , Snr , neq_0 , neq_L )
    ctct_0_phip , ctct_L_phip = contact_phip( phi_p , phi , dgrid , Chi , Eg , Nv , mp , Spl , Spr , peq_0 , peq_L )

    result = [ ctct_0_phin , ctct_0_phip , _pois[0] ]
    for i in range( len(_pois) - 2 ):
        result.append( _ddn[i] )
        result.append( _ddp[i] )
        result.append( _pois[i+1] )
    result.append( ctct_L_phin )
    result.append( ctct_L_phip )
    result.append( _pois[-1] )
    return np.array( result )





### Compute the Jacobian of the system of equations to solve for the out of equilibrium potentials
## Inputs :
#      phi_n (array:N) -> e- quasi-Fermi energy
#      phi_p (array:N) -> hole quasi-Fermi energy
#      phi (array:N) -> electrostatic potential
#      dgrid (array:N-1) -> array of distances between consecutive grid points
#      eps(array:N) -> relative dieclectric constant
#      Chi (array:N) -> electron affinity
#      Eg (array:N) -> band gap
#      Nc (array:N) -> e- density of states
#      Nv (array:N) -> hole density of states
#      Ndop (array:N) -> dopant density ( = donor density - acceptor density )
#      Et (array:N) -> trap state energy level (SHR)
#      tn (array:N) -> e- lifetime (SHR)
#      tp (array:N) -> hole lifetime (SHR)
#      mn (array:N) -> e- mobility
#      mp (array:N) -> hole mobility
#      G (array:N) -> electron-hole generation rate density
#      Snl (scalar) -> e- surface recombination velocity at left boundary
#      Spl (scalar) -> hole surface recombination velocity at left boundary
#      Snr (scalar) -> e- surface recombination velocity at right boundary
#      Spr (scalar) -> hole surface recombination velocity at right boundary
#      neq_0 (scalar) -> e- density at left boundary
#      neq_L (scalar) -> e- density at right boundary
#      peq_0 (scalar) -> hole density at left boundary
#      peq_L (scalar) -> hole density at right boundary
## Outputs :
#      1 (matrix:3Nx3N) -> Jacobian of the out of equilibrium equation system at current value of potentials

def F_deriv( phi_n , phi_p , phi , dgrid , eps , Chi , Eg , Nc , Nv , Et , tn , tp , mn , mp , G , Snl , Spl , Snr , Spr , neq_0 , neq_L , peq_0 , peq_L ):

    dde_phin_ , dde_phin__ , dde_phin___ , dde_phip__ , dde_phi_ , dde_phi__ , dde_phi___ = ddn_deriv( phi_n , phi_p , phi , dgrid , Chi , Eg , Nc , Nv , Et , tn , tp , mn , G )
    ddp_phin__ , ddp_phip_ , ddp_phip__ , ddp_phip___ , ddp_phi_ , ddp_phi__ , ddp_phi___ = ddp_deriv( phi_n , phi_p , phi , dgrid , Chi , Eg , Nc , Nv , Et , tn , tp , mp , G )
    row_pois , col_pois , dpois = pois_deriv( phi_n , phi_p , phi , dgrid , eps , Chi , Eg , Nc , Nv , False )
    dctct_phin = contact_phin_deriv( phi_n , phi , dgrid , Chi , Nc , mn , Snl , Snr )
    dctct_phip = contact_phip_deriv( phi_p , phi , dgrid , Chi , Eg , Nv , mp , Spl , Spr )

    N = phi.size

    row = np.array( [ 0 , 0 , 0 , 0 ] )
    col = np.array( [ 0 , 2 , 3 , 5 ] )
    dF = np.array( [ dctct_phin[0] , dctct_phin[4] , dctct_phin[1] , dctct_phin[5] ] )

    row = np.concatenate( ( row , np.array( [ 1 , 1 , 1 , 1 ] ) ) )
    col = np.concatenate( ( col , np.array( [ 1 , 2 , 4 , 5 ] ) ) )
    dF =  np.concatenate( ( dF , np.array( [ dctct_phip[0] , dctct_phip[4] , dctct_phip[1] , dctct_phip[5] ] ) ) )

    row = np.concatenate( ( row , np.array( [ 3 * ( N - 1 ) , 3 * ( N - 1 ) , 3 * ( N - 1 ) , 3 * ( N - 1 ) ] ) ) )
    col = np.concatenate( ( col , np.array( [ 3 * ( N - 2 ) , 3 * ( N - 2 ) + 2 , 3 * ( N - 1 ) , 3 * ( N - 1 ) + 2 ] ) ) )
    dF = np.concatenate( ( dF , np.array( [ dctct_phin[2] , dctct_phin[6] , dctct_phin[3] , dctct_phin[7] ] ) ) )

    row = np.concatenate( ( row , np.array( [ 3 * ( N - 1 ) + 1 , 3 * ( N - 1 ) + 1 , 3 * ( N - 1 ) + 1 , 3 * ( N - 1 ) + 1 ] ) ) )
    col = np.concatenate( ( col , np.array( [ 3 * ( N - 2 ) + 1 , 3 * ( N - 2 ) + 2 , 3 * ( N - 1 ) + 1 , 3 * ( N - 1 ) + 2 ] ) ) )
    dF = np.concatenate( ( dF , np.array( [ dctct_phip[2] , dctct_phip[6] , dctct_phip[3] , dctct_phip[7] ] ) ) )


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

    row = np.concatenate( ( row , row_pois ) )
    col = np.concatenate( ( col , col_pois ) )
    dF = np.concatenate( ( dF , dpois ) )

    result = np.zeros( ( 3 * N , 3 * N ) )
    if USE_JAX:
        return jax.ops.index_update( result , ( row , col ) , dF )
    else:
        for i in range( len( row ) ):
            result[ row[ i ] , col[ i ] ] = dF[ i ]
        return result
