from .poisson import *
if USE_JAX:
    from jax.config import config
    config.update("jax_enable_x64", True)
    from jax import ops

### Compute the system of equations to solve for the equilibrium electrostatic potential
### (i.e. the poisson equation)
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
#      1 (array:N) -> equilibrium equation system at current value of potentials

def F_eq( phi_n , phi_p , phi , dgrid , eps , Chi , Eg , Nc , Nv , Ndop ):
    return pois( phi_n , phi_p , phi , dgrid , eps , Chi , Eg , Nc , Nv , Ndop )





### Compute the Jacobian of the system of equations to solve for the equilibrium electrostatic potential
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
## Outputs :
#      1 (matrix:NxN) -> Jacobian of equilibrium equation system at current value of potentials

def F_eq_deriv( phi_n , phi_p , phi , dgrid , eps , Chi , Eg , Nc , Nv ):
    row , col , dpois = pois_deriv( phi_n , phi_p , phi , dgrid , eps , Chi , Eg , Nc , Nv , True )

    result = np.zeros( ( phi.size , phi.size ) )
    if USE_JAX:
        return ops.index_update( result , ( row , col ) , dpois )
    else:
        for i in range( len( row ) ):
            result[ row[ i ] , col[ i ] ] = dpois[ i ]
        return result
