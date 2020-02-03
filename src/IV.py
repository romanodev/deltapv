from solve_eq import *
from solve import *

def calc_IV( Vincrement , dgrid , eps , Chi , Eg , Nc , Nv , Ndop , Et , tn , tp , mn , mp , G , Snl , Spl , Snr , Spr ):

    phi_eq = solve_eq( dgrid , eps , Chi , Eg , Nc , Nv , Ndop )

    phi_n = [ np.zeros( dgrid.size + 1 ) ]
    phi_p = [ np.zeros( dgrid.size + 1 ) ]
    phi = [ phi_eq ]
    neq = n( phi_n[-1] , phi[-1] , Chi , Nc )
    peq = p( phi_p[-1] , phi[-1] , Chi , Eg , Nv )

    current = []
    cond = True
    v = 0

    while cond:
        new_phi_n , new_phi_p , new_phi = solve( phi_n[-1] , phi_p[-1] , phi[-1] , dgrid , eps , Chi , Eg , Nc , Nv , Ndop , Et , tn , tp , mn , mp , G , Snl , Spl , Snr , Spr , neq[0] , neq[-1] , peq[0] , peq[-1] )
        current.append( Jn( new_phi_n , new_phi , dgrid , Chi , Nc , mn )[0] + Jp( new_phi_p , new_phi , dgrid , Chi , Eg , Nv , mp )[0] )
        if ( len( current ) > 1 ):
            cond = ( current[-2] * current[-1] > 0 )

        v = v + Vincrement
        phi_n.append(new_phi_n)
        phi_p.append(new_phi_p)
        if USE_JAX:
            phi.append( jax.ops.index_update( new_phi , -1 , phi_eq[-1] + v ) )
        else:
            new_phi[-1] = phi_eq[-1] + v
            phi.append( new_phi )

    return np.array( current , dtype = np.float64 )
