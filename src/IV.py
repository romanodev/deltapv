from .solve_eq import *
from .solve import *

### Compute I-V curve
## Inputs :
#      Vincrement (scalar) -> increment voltage for I-V curve
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
## Outputs :
#      1 (array:M<=max_iter) -> array of (total) currents, if length M < max_iter, current switched signs

def calc_IV( Vincrement , dgrid , eps , Chi , Eg , Nc , Nv , Ndop , Et , tn , tp , mn , mp , G , Snl , Spl , Snr , Spr ):

    phi_ini_left = - Chi[0] - Eg[0] - np.log( np.abs( Ndop[0] ) / Nv[0] )
    if ( Ndop[0] > 0 ):
        phi_ini_left = - Chi[0] + np.log( ( Ndop[0] ) / Nc[0] )
    phi_ini_right = - Chi[-1] - Eg[-1] - np.log( np.abs( Ndop[-1] ) / Nv[-1] )
    if ( Ndop[-1] > 0 ):
        phi_ini_right = - Chi[-1] + np.log( ( Ndop[-1] ) / Nc[-1] )
    phi_ini = np.linspace( phi_ini_left , phi_ini_right , dgrid.size + 1 )

    phi_eq = solve_eq( phi_ini , dgrid , eps , Chi , Eg , Nc , Nv , Ndop )

    phi_n = [ np.zeros( dgrid.size + 1 ) ]
    phi_p = [ np.zeros( dgrid.size + 1 ) ]
    phi = [ phi_eq ]
    neq = n( phi_n[-1] , phi[-1] , Chi , Nc )
    peq = p( phi_p[-1] , phi[-1] , Chi , Eg , Nv )

    max_iter = 100
    iter = 0
    current = []
    cond = True
    v = 0

    while cond and ( iter < max_iter ):
        new_phi_n , new_phi_p , new_phi = solve( phi_n[-1] , phi_p[-1] , phi[-1] , dgrid , eps , Chi , Eg , Nc , Nv , Ndop , Et , tn , tp , mn , mp , G , Snl , Spl , Snr , Spr , neq[0] , neq[-1] , peq[0] , peq[-1] )
        current.append( Jn( new_phi_n , new_phi , dgrid , Chi , Nc , mn )[0] + Jp( new_phi_p , new_phi , dgrid , Chi , Eg , Nv , mp )[0] )
        if ( len( current ) > 1 ):
            cond = ( current[-2] * current[-1] > 0 )

        iter += 1
        v = v + Vincrement
        phi_n.append(new_phi_n)
        phi_p.append(new_phi_p)
        if USE_JAX:
            phi.append( ops.index_update( new_phi , -1 , phi_eq[-1] + v ) )
        else:
            new_phi[-1] = phi_eq[-1] + v
            phi.append( new_phi )

    return np.array( current , dtype = np.float64 )
