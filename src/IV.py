from .solve_eq import *
from .solve import *
if USE_JAX:
    from jax import jacfwd

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

    N = dgrid.size + 1

    if ( Ndop[0] > 0 ):
        phi_ini_left = - Chi[0] + np.log( ( Ndop[0] ) / Nc[0] )
    else:
        phi_ini_left = - Chi[0] - Eg[0] - np.log( - Ndop[0] / Nv[0] )
    if ( Ndop[-1] > 0 ):
        phi_ini_right = - Chi[-1] + np.log( ( Ndop[-1] ) / Nc[-1] )
    else:
        phi_ini_right = - Chi[-1] - Eg[-1] - np.log( - Ndop[-1] / Nv[-1] )
    phi_ini = np.linspace( phi_ini_left , phi_ini_right , N )

    phi_eq = solve_eq( dgrid , phi_ini , eps , Chi , Eg , Nc , Nv , Ndop )
    neq_0 = Nc[0] * np.exp( Chi[0] + phi_eq[0] )
    neq_L = Nc[-1] * np.exp( Chi[-1] + phi_eq[-1] )
    peq_0 = Nv[0] * np.exp( - Chi[0] - Eg[0] - phi_eq[0] )
    peq_L = Nv[-1] * np.exp( - Chi[-1] - Eg[-1] - phi_eq[-1] )

    phis = np.concatenate( ( np.zeros( 2*N ) , phi_eq ) , axis = 0 )
    max_iter = 100
    iter = 0
    current = []
    cond = True
    v = 0

    while cond and ( iter < max_iter ):
        sol = solve( dgrid , phis , eps , Chi , Eg , Nc , Nv , Ndop , Et , tn , tp , mn , mp , G , Snl , Spl , Snr , Spr , neq_0 , neq_L , peq_0 , peq_L )
        current.append( Jn( sol[0:N] , sol[2*N:-1] , dgrid , Chi , Nc , mn )[0] + Jp( sol[N:2*N] , sol[2*N:-1] , dgrid , Chi , Eg , Nv , mp )[0] )
        if ( len( current ) > 1 ):
            cond = ( current[-2] * current[-1] > 0 )

        iter += 1
        v = v + Vincrement
        if USE_JAX:
            phis = ops.index_update( sol , -1 , phi_eq[-1] + v )
        else:
            sol[-1] = phi_eq[-1] + v
            phi = sol

    return np.array( current , dtype = np.float64 )


def grad_IV( Vincrement , dgrid , eps , Chi , Eg , Nc , Nv , Ndop , Et , tn , tp , mn , mp , G , Snl , Spl , Snr , Spr ):

    grad_phieq = jit( jacfwd( solve_eq , argnums = ( 1 , 2 , 3 , 4 , 5 , 6 , 7 ) ) )
    grad_solve = jit( jacfwd( solve , argnums = ( 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 19 , 20 , 21 ) ) )

    N = dgrid.size + 1

    if ( Ndop[0] > 0 ):
        phi_ini_left = - Chi[0] + np.log( ( Ndop[0] ) / Nc[0] )
        dphi_ini_left_dChi0 = - 1
        dphi_ini_left_dEg0 = 0
        dphi_ini_left_dNdop0 = 1 / Ndop[0]
        dphi_ini_left_dNc0 = - 1 / Nc[0]
        dphi_ini_left_dNv0 = 0
    else:
        phi_ini_left = - Chi[0] - Eg[0] - np.log( - Ndop[0] / Nv[0] )
        dphi_ini_left_dChi0 = - 1
        dphi_ini_left_dEg0 = - 1
        dphi_ini_left_dNdop0 = - 1 / Ndop[0]
        dphi_ini_left_dNc0 = 0
        dphi_ini_left_dNv0 = 1 / Nv[0]
    if ( Ndop[-1] > 0 ):
        phi_ini_right = - Chi[-1] + np.log( ( Ndop[-1] ) / Nc[-1] )
        dphi_ini_right_dChiL = - 1
        dphi_ini_right_dEgL = 0
        dphi_ini_right_dNdopL = 1 / Ndop[-1]
        dphi_ini_right_dNcL = - 1 / Nc[-1]
        dphi_ini_right_dNvL = 0
    else:
        phi_ini_right = - Chi[-1] - Eg[-1] - np.log( - Ndop[-1] / Nv[-1] )
        dphi_ini_right_dChiL = - 1
        dphi_ini_right_dEgL = - 1
        dphi_ini_right_dNdopL = - 1 / Ndop[-1]
        dphi_ini_right_dNcL = 0
        dphi_ini_right_dNvL = - 1 / Nv[-1]
    phi_ini = np.linspace( phi_ini_left , phi_ini_right , N )

    dphi_ini_dChi0 = np.linspace( dphi_ini_left_dChi0 , 0 , N )
    dphi_ini_dEg0 = np.linspace( dphi_ini_left_dEg0 , 0 , N )
    dphi_ini_dNdop0 = np.linspace( dphi_ini_left_dNdop0 , 0 , N )
    dphi_ini_dNc0 = np.linspace( dphi_ini_left_dNc0 , 0 , N )
    dphi_ini_dNv0 = np.linspace( dphi_ini_left_dNv0 , 0 , N )
    dphi_ini_dChiL = np.linspace( 0 , dphi_ini_right_dChiL , N )
    dphi_ini_dEgL = np.linspace( 0 , dphi_ini_right_dEgL , N )
    dphi_ini_dNdopL = np.linspace( 0 , dphi_ini_right_dNdopL , N )
    dphi_ini_dNcL = np.linspace( 0 , dphi_ini_right_dNcL , N )
    dphi_ini_dNvL = np.linspace( 0 , dphi_ini_right_dNvL , N )

    phi_eq = solve_eq( dgrid , phi_ini , eps , Chi , Eg , Nc , Nv , Ndop )
    neq_0 = Nc[0] * np.exp( Chi[0] + phi_eq[0] )
    neq_L = Nc[-1] * np.exp( Chi[-1] + phi_eq[-1] )
    peq_0 = Nv[0] * np.exp( - Chi[0] - Eg[0] - phi_eq[0] )
    peq_L = Nv[-1] * np.exp( - Chi[-1] - Eg[-1] - phi_eq[-1] )


    gradphieq = grad_phieq( dgrid , phi_ini , eps , Chi , Eg , Nc , Nv , Ndop )

    dphi_eq_deps = gradphieq[ 1 ]
    dphi_eq_dChi = gradphieq[ 2 ]
    dphi_eq_dChi[:,0] += np.dot( gradphieq[ 0 ] , dphi_ini_dChi0 )
    dphi_eq_dChi[:,-1] += np.dot( gradphieq[ 0 ] , dphi_ini_dChiL )
    dphi_eq_dEg = gradphieq[ 3 ]
    dphi_eq_dEg[:,0] += np.dot( gradphieq[ 0 ] , dphi_ini_dEg0 )
    dphi_eq_dEg[:,-1] += np.dot( gradphieq[ 0 ] , dphi_ini_dEgL )
    dphi_eq_dNc = gradphieq[ 4 ]
    dphi_eq_dNc[:,0] += np.dot( gradphieq[ 0 ] , dphi_ini_dNc0 )
    dphi_eq_dNc[:,-1] += np.dot( gradphieq[ 0 ] , dphi_ini_dNcL )
    dphi_eq_dNv = gradphieq[ 5 ]
    dphi_eq_dNv[:,0] += np.dot( gradphieq[ 0 ] , dphi_ini_dNv0 )
    dphi_eq_dNv[:,-1] += np.dot( gradphieq[ 0 ] , dphi_ini_dNvL )
    dphi_eq_dNdop = gradphieq[ 6 ]
    dphi_eq_dNdop[:,0] += np.dot( gradphieq[ 0 ] , dphi_ini_dNdop0 )
    dphi_eq_dNdop[:,-1] += np.dot( gradphieq[ 0 ] , dphi_ini_dNdopL )

    dneq0_dChi = np.concatenate( ( np.array( [ neq_0 ] ) , np.zeros( N - 1 ) ) , axis = 0 ) + neq_0 * dphi_eq_dChi[0,:]
    dneqL_dChi = np.concatenate( ( np.zeros( N - 1 ) , np.array( [ neq_L ] ) ) , axis = 0 ) + neq_L * dphi_eq_dChi[-1,:]
    dneq0_dEg = neq_0 * dphi_eq_dEg[0,:]
    dneqL_dEg = neq_L * dphi_eq_dEg[-1,:]
    dneq0_dNc = np.concatenate( ( np.array( [ np.exp( Chi[0] + phi_eq[0] ) ] ) , np.zeros( N - 1 ) ) , axis = 0 ) + neq_0 * dphi_eq_dNc[0,:]
    dneqL_dNc = np.concatenate( ( np.zeros( N - 1 ) , np.array( [ np.exp( Chi[-1] + phi_eq[-1] ) ] ) ) , axis = 0 ) + neq_L * dphi_eq_dNc[-1,:]
    dneq0_dNv = neq_0 * dphi_eq_dNv[0,:]
    dneqL_dNv = neq_L * dphi_eq_dNv[-1,:]
    dneq0_dNdop = neq_0 * dphi_eq_dNdop[0,:]
    dneqL_dNdop = neq_L * dphi_eq_dNdop[-1,:]

    dpeq0_dChi = np.concatenate( ( np.array( [ - peq_0 ] ) , np.zeros( N - 1 ) ) , axis = 0 ) - peq_0 * dphi_eq_dChi[0,:]
    dpeqL_dChi = np.concatenate( ( np.zeros( N - 1 ) , np.array( [ - peq_L ] ) ) , axis = 0 ) - peq_L * dphi_eq_dChi[-1,:]
    dpeq0_dEg = - peq_0 * dphi_eq_dEg[0,:]
    dpeqL_dEg = - peq_L * dphi_eq_dEg[-1,:]
    dpeq0_dNc = np.concatenate( ( np.array( [ np.exp( - Chi[0] - Eg[0] - phi_eq[0] ) ] ) , np.zeros( N - 1 ) ) , axis = 0 ) - peq_0 * dphi_eq_dNc[0,:]
    dpeqL_dNc = np.concatenate( ( np.zeros( N - 1 ) , np.array( [ np.exp( - Chi[-1] - Eg[-1] - phi_eq[-1] ) ] ) ) , axis = 0 ) - peq_L * dphi_eq_dNc[-1,:]
    dpeq0_dNv = - peq_0 * dphi_eq_dNv[0,:]
    dpeqL_dNv = - peq_L * dphi_eq_dNv[-1,:]
    dpeq0_dNdop = - peq_0 * dphi_eq_dNdop[0,:]
    dpeqL_dNdop = - peq_L * dphi_eq_dNdop[-1,:]

    phis = np.concatenate( ( np.zeros( 2*N ) , phi_eq ) , axis = 0 )
    jac_phis = {}
    jac_phis['eps'] = np.vstack( ( np.zeros( ( 2*N , N ) ) , dphi_eq_deps ) )
    jac_phis['Chi'] = np.vstack( ( np.zeros( ( 2*N , N ) ) , dphi_eq_dChi ) )
    jac_phis['Eg'] = np.vstack( ( np.zeros( ( 2*N , N ) ) , dphi_eq_dEg ) )
    jac_phis['Nc'] = np.vstack( ( np.zeros( ( 2*N , N ) ) , dphi_eq_dNc ) )
    jac_phis['Nv'] = np.vstack( ( np.zeros( ( 2*N , N ) ) , dphi_eq_dNv ) )
    jac_phis['Ndop'] = np.vstack( ( np.zeros( ( 2*N , N ) ) , dphi_eq_dNdop ) )
    jac_phis['Et'] = np.zeros( ( 3*N , N ) )
    jac_phis['tn'] = np.zeros( ( 3*N , N ) )
    jac_phis['tp'] = np.zeros( ( 3*N , N ) )
    jac_phis['mn'] = np.zeros( ( 3*N , N ) )
    jac_phis['mp'] = np.zeros( ( 3*N , N ) )
    jac_phis['G'] = np.zeros( ( 3*N , N ) )
    jac_phis['Snl'] = np.zeros( ( 3*N , N ) )
    jac_phis['Spl'] = np.zeros( ( 3*N , N ) )
    jac_phis['Snr'] = np.zeros( ( 3*N , N ) )
    jac_phis['Spr'] = np.zeros( ( 3*N , N ) )

    max_iter = 100
    iter = 0
    current = []
    current_jac = []
    cond = True
    v = 0

    while cond and ( iter < max_iter ):
        sol = solve( dgrid , phis , eps , Chi , Eg , Nc , Nv , Ndop , Et , tn , tp , mn , mp , G , Snl , Spl , Snr , Spr , neq_0 , neq_L , peq_0 , peq_L )
        gradsol = grad_solve( dgrid , phis , eps , Chi , Eg , Nc , Nv , Ndop , Et , tn , tp , mn , mp , G , Snl , Spl , Snr , Spr , neq_0 , neq_L , peq_0 , peq_L )

        current.append( Jn( sol[0:N] , sol[2*N:-1] , dgrid , Chi , Nc , mn )[0] + Jp( sol[N:2*N] , sol[2*N:-1] , dgrid , Chi , Eg , Nv , mp )[0] )

        jac_phis['eps'] = gradsol[1] + np.dot( gradsol[0] , jac_phis['eps'] )
        jac_phis['Chi'] = gradsol[2] + np.dot( gradsol[17] , dneq0_dChi ) + np.dot( gradsol[18] , dneqL_dChi ) + np.dot( gradsol[19] , dpeq0_dChi ) + np.dot( gradsol[20] , dpeqL_dChi ) + np.dot( gradsol[0] , jac_phis['Chi'] )
        jac_phis['Eg'] = gradsol[3] + np.dot( gradsol[17] , dneq0_dEg ) + np.dot( gradsol[18] , dneqL_dEg ) + np.dot( gradsol[19] , dpeq0_dEg ) + np.dot( gradsol[20] , dpeqL_dEg )+ np.dot( gradsol[0] , jac_phis['Eg'] )
        jac_phis['Nc'] = gradsol[4] + np.dot( gradsol[17] , dneq0_dNc ) + np.dot( gradsol[18] , dneqL_dNc ) + np.dot( gradsol[19] , dpeq0_dNc ) + np.dot( gradsol[20] , dpeqL_dNc )+ np.dot( gradsol[0] , jac_phis['Nc'] )
        jac_phis['Nv'] = gradsol[5] + np.dot( gradsol[17] , dneq0_dNv ) + np.dot( gradsol[18] , dneqL_dNv ) + np.dot( gradsol[19] , dpeq0_dNv ) + np.dot( gradsol[20] , dpeqL_dNv )+ np.dot( gradsol[0] , jac_phis['Nv'] )
        jac_phis['Ndop'] = gradsol[6] + np.dot( gradsol[17] , dneq0_dNdop ) + np.dot( gradsol[18] , dneqL_dNdop ) + np.dot( gradsol[19] , dpeq0_dNdop ) + np.dot( gradsol[20] , dpeqL_dNdop )+ np.dot( gradsol[0] , jac_phis['Ndop'] )
        jac_phis['Et'] = gradsol[7] + np.dot( gradsol[0] , jac_phis['Et'] )
        jac_phis['tn'] = gradsol[8] + np.dot( gradsol[0] , jac_phis['tn'] )
        jac_phis['tp'] = gradsol[9] + np.dot( gradsol[0] , jac_phis['tp'] )
        jac_phis['mn'] = gradsol[10] + np.dot( gradsol[0] , jac_phis['mn'] )
        jac_phis['mp'] = gradsol[11] + np.dot( gradsol[0] , jac_phis['mp'] )
        jac_phis['G'] = gradsol[12] + np.dot( gradsol[0] , jac_phis['G'] )
        jac_phis['Snl'] = gradsol[13] + np.dot( gradsol[0] , jac_phis['Snl'] )
        jac_phis['Spl'] = gradsol[14] + np.dot( gradsol[0] , jac_phis['Spl'] )
        jac_phis['Snr'] = gradsol[15] + np.dot( gradsol[0] , jac_phis['Snr'] )
        jac_phis['Spr'] = gradsol[16] + np.dot( gradsol[0] , jac_phis['Spr'] )

        new_current_jac = {}

        current_jac.append( new_current_jac )

        if ( len( current ) > 1 ):
            cond = ( current[-2] * current[-1] > 0 )

        iter += 1
        v = v + Vincrement
        phis = ops.index_update( sol , -1 , phi_eq[-1] + v )

    return current , current_jac
