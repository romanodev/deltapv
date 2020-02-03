from physics import *

def SHR( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Et , tn , tp ):
    _ni = ni( Eg , Nc , Nv )
    _n = n( phi_n , phi , Chi , Nc )
    _p = p( phi_p , phi , Chi , Eg , Nv )
    nR = _ni * np.exp( Et ) + _n
    pR = _ni * np.exp( - Et ) + _p
    return ( _n * _p - _ni**2 ) / ( tp * nR + tn * pR )

def SHR_deriv( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Et , tn , tp ):
    _ni = ni( Eg , Nc , Nv )
    _n = n( phi_n , phi , Chi , Nc )
    _p = p( phi_p , phi , Chi , Eg , Nv )
    nR = _ni * np.exp( Et ) + _n
    pR = _ni * np.exp( - Et ) + _p
    num = _n * _p - _ni**2
    denom = ( tp * nR + tn * pR )

    DR_phin = ( ( _n * _p ) * denom - num * ( tp * _n ) ) * denom**-2
    DR_phip = ( ( - _n * _p ) * denom - num * ( - tn * _p ) ) * denom**-2
    DR_phi = ( - num * ( tp * _n - tn * _p ) ) * denom**-2 

    return DR_phin , DR_phip , DR_phi
