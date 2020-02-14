from .scales import *
if USE_JAX:
    from jax.config import config
    config.update("jax_enable_x64", True)
    import jax.numpy as np
else:
    import numpy as np

### Compute the e- density
## Inputs :
#      phi_n (array:N) -> e- quasi-Fermi energy
#      phi (array:N) -> electrostatic potential
#      Chi (array:N) -> electron affinity
#      Nc (array:N) -> e- density of states
## Outputs :
#      1 (array:N) -> e- density

def n( phi_n , phi , Chi , Nc ):
    return Nc * np.exp( Chi + phi_n + phi )





### Compute the hole density
## Inputs :
#      phi_p (array:N) -> hole quasi-Fermi energy
#      phi (array:N) -> electrostatic potential
#      Chi (array:N) -> electron affinity
#      Eg (array:N) -> band gap
#      Nv (array:N) -> hole density of states
## Outputs :
#      1 (array:N) -> hole density

def p( phi_p , phi , Chi , Eg , Nv ):
    return Nv * np.exp( - Chi - Eg - phi_p - phi )





### Compute the charge density
## Inputs :
#      phi_n (array:N) -> e- quasi-Fermi energy
#      phi_p (array:N) -> hole quasi-Fermi energy
#      phi (array:N) -> electrostatic potential
#      Chi (array:N) -> electron affinity
#      Eg (array:N) -> band gap
#      Nc (array:N) -> e- density of states
#      Nv (array:N) -> hole density of states
#      Ndop (array:N) -> dopant density ( = donor density - acceptor density )
## Outputs :
#      1 (array:N) -> hole density

def charge( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Ndop ):
    _n = n( phi_n , phi , Chi , Nc )
    _p = p( phi_p , phi , Chi , Eg , Nv )
    return - _n + _p + Ndop





### Compute the intrinsic carrier density
## Inputs :
#      Eg (array:N) -> band gap
#      Nc (array:N) -> e- density of states
#      Nv (array:N) -> hole density of states
## Outputs :
#      1 (array:N) -> intrinsic carrier density

def ni( Eg , Nc , Nv ):
    return np.sqrt( Nc * Nv ) * np.exp( - Eg / 2 )
