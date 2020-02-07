from .poisson import *
if USE_JAX:
    from jax.config import config
    config.update("jax_enable_x64", True)
    import jax
    
def F_eq( phi_n , phi_p , phi , dgrid , eps , Chi , Eg , Nc , Nv , Ndop ):
    return pois( phi_n , phi_p , phi , dgrid , eps , Chi , Eg , Nc , Nv , Ndop )

def F_eq_deriv( phi_n , phi_p , phi , dgrid , eps , Chi , Eg , Nc , Nv ):
    row , col , dpois = pois_deriv( phi_n , phi_p , phi , dgrid , eps , Chi , Eg , Nc , Nv , True )

    result = np.zeros( ( phi.size , phi.size ) )
    if USE_JAX:
        return jax.ops.index_update( result , ( row , col ) , dpois )
    else:
        for i in range( len( row ) ):
            result[ row[ i ] , col[ i ] ] = dpois[ i ]
        return result
