from .physics import *

### Compute the left side term of the poisson equation
### (including electrostatic potential boundary condition ; which is constrained to be satisfied)
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
## Outputs :
#      1 (array:N) -> left side term of the poisson equation

def pois( phi_n , phi_p , phi , dgrid , eps , Chi , Eg , Nc , Nv , Ndop ):
    ave_dgrid = ( dgrid[:-1] + dgrid[1:] ) / 2.0
    ave_eps = 0.5 * ( eps[1:] + eps[:-1] )
    pois = ( ave_eps[:-1] * ( phi[1:-1] - phi[:-2] ) / dgrid[:-1] - ave_eps[1:] * ( phi[2:] - phi[1:-1] ) / dgrid[1:] ) / ave_dgrid - charge( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Ndop )[1:-1]
    return np.concatenate( ( np.array([0.0]) , pois , np.array([0.0]) ) , axis = 0 )





### Compute the derivatives of the left side term of the poisson equation w.r.t. e- and hole quasi-Fermi energy
### and electrostatic potential (only this one required for equilibrium calculation)
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
#      eq (boolean) -> True if computing derivative only w.r.t. phi (equilibrium) or False for all three potentials
## Outputs :
#      1 (array:N) -> left side term of the poisson equation

def pois_deriv( phi_n , phi_p , phi , dgrid , eps , Chi , Eg , Nc , Nv , eq ):
    ave_dgrid = ( dgrid[:-1] + dgrid[1:] ) / 2.0
    ave_eps = 0.5 * ( eps[1:] + eps[:-1] )
    _n = n( phi_n , phi , Chi , Nc )
    _p = p( phi_p , phi , Chi , Eg , Nv )

    dpois_phi_ =  - ave_eps[:-1] / dgrid[:-1] / ave_dgrid
    dpois_phi__ =  ( ave_eps[:-1] / dgrid[:-1] + ave_eps[1:] / dgrid[1:] ) / ave_dgrid
    dpois_phi___ =  - ave_eps[1:] / dgrid[1:] / ave_dgrid

    dchg_phi_n = - _n
    dchg_phi_p = - _p
    dchg_phi = - _n - _p

    if eq:
        row = np.array( [ 0 , phi.size - 1 ] )
        col = np.array( [ 0 , phi.size - 1 ] )
        dpois = np.array( [ 1 , 1 ] )

        row = np.concatenate( ( row , np.arange( 1 , phi.size - 1 , 1 ) ) )
        col = np.concatenate( ( col , np.arange( 1 , phi.size - 1 , 1 ) ) )
        dpois = np.concatenate( ( dpois , dpois_phi__ - dchg_phi[1:-1] ) )

        row = np.concatenate( ( row , np.arange( 1 , phi.size - 1 , 1 ) ) )
        col = np.concatenate( ( col , np.arange( 0 , phi.size - 2 , 1 ) ) )
        dpois = np.concatenate( ( dpois , dpois_phi_ ) )

        row = np.concatenate( ( row , np.arange( 1 , phi.size - 1 , 1 ) ) )
        col = np.concatenate( ( col , np.arange( 2 , phi.size , 1 ) ) )
        dpois = np.concatenate( ( dpois , dpois_phi___ ) )

    else:
        row = np.array( [ 2 , 3 * ( phi.size - 1 ) + 2 ] )
        col = np.array( [ 2 , 3 * ( phi.size - 1 ) + 2 ] )
        dpois = np.array( [ 1 , 1 ] )

        row = np.concatenate( ( row , np.arange( 5 , 3 * ( phi.size - 1 ) + 2 , 3 ) ) )
        col = np.concatenate( ( col , np.arange( 5 , 3 * ( phi.size - 1 ) + 2 , 3 ) ) )
        dpois = np.concatenate( ( dpois , dpois_phi__ - dchg_phi[1:-1] ) )

        row = np.concatenate( ( row , np.arange( 5 , 3 * ( phi.size - 1 ) + 2 , 3 ) ) )
        col = np.concatenate( ( col , np.arange( 2 , 3 * ( phi.size - 2 ) + 2 , 3 ) ) )
        dpois = np.concatenate( ( dpois , dpois_phi_ ) )

        row = np.concatenate( ( row , np.arange( 5 , 3 * ( phi.size - 1 ) + 2 , 3 ) ) )
        col = np.concatenate( ( col , np.arange( 8 , 3 * ( phi.size ) + 2 , 3 ) ) )
        dpois = np.concatenate( ( dpois , dpois_phi___ ) )

        row = np.concatenate( ( row , np.arange( 5 , 3 * ( phi.size - 1 ) + 2 , 3 ) ) )
        col = np.concatenate( ( col , np.arange( 3 , 3 * ( phi.size - 1 ) , 3 ) ) )
        dpois = np.concatenate( ( dpois , - dchg_phi_n[1:-1] ) )

        row = np.concatenate( ( row , np.arange( 5 , 3 * ( phi.size - 1 ) + 2 , 3 ) ) )
        col = np.concatenate( ( col , np.arange( 4 , 3 * ( phi.size - 1 ) + 1 , 3 ) ) )
        dpois = np.concatenate( ( dpois , - dchg_phi_p[1:-1] ) )

    return row , col , dpois
