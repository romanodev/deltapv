from .F_eq import *

### Compute the damped displacement of electrostatic potential based on the Newton step
## Inputs :
#      move (array:N) -> Newton-based change in electrostatic potential
## Outputs :
#      1 (array:N) -> damped change in electrostatic potential

def damp( move ):
    approx_sign = np.tanh( move )
    approx_abs = approx_sign * move
    approx_H = 1 - ( 1 + np.exp( - 500 * ( move**2 - 1 ) ) )**(-1)
    return np.log( 1 + approx_abs ) * approx_sign + approx_H * ( move - np.log( 1 + approx_abs ) * approx_sign )





### Compute the next electrostatic potential in the Newton method iterative scheme
## Inputs :
#      phi (array:N) -> current electrostatic potential
#      dgrid (array:N-1) -> array of distances between consecutive grid points
#      eps(array:N) -> relative dieclectric constant
#      Chi (array:N) -> electron affinity
#      Eg (array:N) -> band gap
#      Nc (array:N) -> e- density of states
#      Nv (array:N) -> hole density of states
#      Ndop (array:N) -> dopant density ( = donor density - acceptor density )
## Outputs :
#      1 (scalar) -> error (largest component of displacement)
#      2 (array:N) -> next electrostatic potential

def step_eq( phi , dgrid , eps , Chi , Eg , Nc , Nv , Ndop ):
    Feq = F_eq( np.zeros( phi.size ) , np.zeros( phi.size ) , phi , dgrid , eps , Chi , Eg , Nc , Nv , Ndop )
    gradFeq = F_eq_deriv( np.zeros( phi.size ) , np.zeros( phi.size ) , phi , dgrid , eps , Chi , Eg , Nc , Nv )
    move = np.linalg.solve( gradFeq , - Feq )
    error = max( np.abs( move ) )

    damp_move = damp(move)
    phi_new = phi + damp_move

    return error , phi_new





### Solve for the equilibrium electrostatic potential using the Newton method
## Inputs :
#      dgrid (array:N-1) -> array of distances between consecutive grid points
#      eps(array:N) -> relative dieclectric constant
#      Chi (array:N) -> electron affinity
#      Eg (array:N) -> band gap
#      Nc (array:N) -> e- density of states
#      Nv (array:N) -> hole density of states
#      Ndop (array:N) -> dopant density ( = donor density - acceptor density )
## Outputs :
#      1 (array:N) -> equilibrium electrostatic potential

def solve_eq( dgrid , eps , Chi , Eg , Nc , Nv , Ndop ):

    phi_ini_left = - Chi[0] - Eg[0] - np.log( np.abs( Ndop[0] ) / Nv[0] )
    if ( Ndop[0] > 0 ):
        phi_ini_left = - Chi[0] + np.log( ( Ndop[0] ) / Nc[0] )
    phi_ini_right = - Chi[-1] - Eg[-1] - np.log( np.abs( Ndop[-1] ) / Nv[-1] )
    if ( Ndop[-1] > 0 ):
        phi_ini_right = - Chi[-1] + np.log( ( Ndop[-1] ) / Nc[-1] )
    phi = np.linspace( phi_ini_left , phi_ini_right , dgrid.size + 1 )

    error = 1
    while (error > 1e-6):
        new_error , next_phi = step_eq( phi , dgrid , eps , Chi , Eg , Nc , Nv , Ndop )
        phi = next_phi
        error = new_error

    return phi
