from .scales import *
if USE_JAX:
    from jax.config import config
    config.update("jax_enable_x64", True)    
    import jax.numpy as np
else:
    import numpy as np

def n( phi_n , phi , Chi , Nc ):
    return Nc * np.exp( Chi + phi_n + phi )

def p( phi_p , phi , Chi , Eg , Nv ):
    return Nv * np.exp( - Chi - Eg - phi_p - phi )

def charge( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Ndop ):
    _n = n( phi_n , phi , Chi , Nc )
    _p = p( phi_p , phi , Chi , Eg , Nv )
    return - _n + _p + Ndop

def ni( Eg , Nc , Nv ):
    return np.sqrt( Nc * Nv ) * np.exp( - Eg / 2 )
