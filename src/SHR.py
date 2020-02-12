from .physics import *

### Compute the SHR electron-hole recombination rate density
## Inputs :
#      phi_n (array:N) -> e- quasi-Fermi energy
#      phi_p (array:N) -> hole quasi-Fermi energy
#      phi (array:N) -> electrostatic potential
#      eps(array:N) -> relative dieclectric constant
#      Chi (array:N) -> electron affinity
#      Eg (array:N) -> band gap
#      Nc (array:N) -> e- density of states
#      Nv (array:N) -> hole density of states
#      Et (array:N) -> trap state energy level (SHR)
#      tn (array:N) -> e- lifetime (SHR)
#      tp (array:N) -> hole lifetime (SHR)
## Outputs :
#      1 (array:N) -> SHR recombination rate density

def SHR( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Et , tn , tp ):
    _ni = ni( Eg , Nc , Nv )
    _n = n( phi_n , phi , Chi , Nc )
    _p = p( phi_p , phi , Chi , Eg , Nv )
    nR = _ni * np.exp( Et ) + _n
    pR = _ni * np.exp( - Et ) + _p
    return ( _n * _p - _ni**2 ) / ( tp * nR + tn * pR )





### Compute the derivatives of the SHR electron-hole recombination rate density w.r.t. e- and hole quasi-Fermi
### energy and electrostatic potential
## Inputs :
#      phi_n (array:N) -> e- quasi-Fermi energy
#      phi_p (array:N) -> hole quasi-Fermi energy
#      phi (array:N) -> electrostatic potential
#      eps(array:N) -> relative dieclectric constant
#      Chi (array:N) -> electron affinity
#      Eg (array:N) -> band gap
#      Nc (array:N) -> e- density of states
#      Nv (array:N) -> hole density of states
#      Et (array:N) -> trap state energy level (SHR)
#      tn (array:N) -> e- lifetime (SHR)
#      tp (array:N) -> hole lifetime (SHR)
## Outputs :
#      1 (array:N) -> derivative of SHR recombination at point i w.r.t phi_n[i]
#      2 (array:N) -> derivative of SHR recombination at point i w.r.t phi_p[i]
#      3 (array:N) -> derivative of SHR recombination at point i w.r.t phi[i]

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
