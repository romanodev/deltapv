from .physics import *
from .current import *

### Compute the left side term of the Dirichlet out of equilibrium boundary condition for e- quasi-Fermi energy
## Inputs :
#      phi_n (array:N) -> e- quasi-Fermi energy
#      phi (array:N) -> electrostatic potential
#      dgrid (array:N-1) -> array of distances between consecutive grid points
#      Chi (array:N) -> electron affinity
#      Nc (array:N) -> e- density of states
#      mn (array:N) -> e- mobility
#      Snl (scalar) -> e- surface recombination velocity at left boundary
#      Snr (scalar) -> e- surface recombination velocity at right boundary
#      neq_0 (scalar) -> e- density at left boundary
#      neq_L (scalar) -> e- density at right boundary
## Outputs :
#      1 (scalar) -> left side term of left boundary condition
#      2 (scalar) -> left side term of right boundary condition

def contact_phin( phi_n , phi , dgrid , Chi , Nc , mn , Snl , Snr , neq_0 , neq_L ):
    _n = n( phi_n , phi , Chi , Nc )
    _Jn = Jn( phi_n , phi , dgrid , Chi , Nc , mn )
    return _Jn[0] - Snl * ( _n[0] - neq_0 ) , _Jn[-1] + Snr * ( _n[-1] - neq_L )





### Compute the derivatives of the Dirichlet out of equilibrium boundary condition for e- quasi-Fermi energy
### w.r.t. the e- quasi-Fermi energy and the electrostatic potential
## Inputs :
#      phi_n (array:N) -> e- quasi-Fermi energy
#      phi (array:N) -> electrostatic potential
#      dgrid (array:N-1) -> array of distances between consecutive grid points
#      Chi (array:N) -> electron affinity
#      Nc (array:N) -> e- density of states
#      mn (array:N) -> e- mobility
#      Snl (scalar) -> e- surface recombination velocity at left boundary
#      Snr (scalar) -> e- surface recombination velocity at right boundary
## Outputs :
#      1 (scalar) -> derivative of left boundary condition w.r.t. phi_n[0]
#      2 (scalar) -> derivative of left boundary condition w.r.t. phi_n[1]
#      3 (scalar) -> derivative of right boundary condition w.r.t. phi_n[0]
#      4 (scalar) -> derivative of right boundary condition w.r.t. phi_n[1]
#      5 (scalar) -> derivative of left boundary condition w.r.t. phi[0]
#      6 (scalar) -> derivative of left boundary condition w.r.t. phi[1]
#      7 (scalar) -> derivative of right boundary condition w.r.t. phi[0]
#      8 (scalar) -> derivative of right boundary condition w.r.t. phi[1]

def contact_phin_deriv( phi_n , phi , dgrid , Chi , Nc , mn , Snl , Snr ):
    _n = n( phi_n , phi , Chi , Nc )
    dJn_phin_maindiag , dJn_phin_upperdiag , dJn_phi_maindiag , dJn_phi_upperdiag = Jn_deriv( phi_n , phi , dgrid , Chi , Nc , mn )
    return dJn_phin_maindiag[0] - Snl * _n[0] , dJn_phin_upperdiag[0] , \
    dJn_phin_maindiag[-1] + Snr * _n[-1] , dJn_phin_upperdiag[-1] , \
    dJn_phi_maindiag[0] - Snl * _n[0] , dJn_phi_upperdiag[0] , \
    dJn_phi_maindiag[-1] + Snr * _n[-1] , dJn_phi_upperdiag[-1]





### Computes left side term of the Dirichlet out of equilibrium boundary condition for hole quasi-Fermi energy
## Inputs :
#      phi_p (array:N) -> hole quasi-Fermi energy
#      phi (array:N) -> electrostatic potential
#      dgrid (array:N-1) -> array of distances between consecutive grid points
#      Chi (array:N) -> electron affinity
#      Eg (array:N) -> band gap
#      Nv (array:N) -> hole density of states
#      mp (array:N) -> hole mobility
#      Spl (scalar) -> hole surface recombination velocity at left boundary
#      Spr (scalar) -> hole surface recombination velocity at right boundary
#      peq_0 (scalar) -> hole density at left boundary
#      peq_L (scalar) -> hole density at right boundary
## Outputs :
#      1 (scalar) -> left side term of left boundary condition
#      2 (scalar) -> left side term of right boundary condition

def contact_phip( phi_p , phi , dgrid , Chi , Eg , Nv , mp , Spl , Spr , peq_0 , peq_L ):
    _p = p( phi_p , phi , Chi , Eg , Nv )
    _Jp = Jp( phi_p , phi , dgrid , Chi , Eg , Nv , mp )
    return _Jp[0] + Spl * ( _p[0] - peq_0 ) , _Jp[-1] - Spr * ( _p[-1] - peq_L )





### Computes the derivatives of the Dirichlet out of equilibrium boundary condition for hole quasi-Fermi energy
### w.r.t. the hole quasi-Fermi energy and the electrostatic potential
## Inputs :
#      phi_p (array:N) -> hole quasi-Fermi energy
#      phi (array:N) -> electrostatic potential
#      dgrid (array:N-1) -> array of distances between consecutive grid points
#      Chi (array:N) -> electron affinity
#      Eg (array:N) -> band gap
#      Nv (array:N) -> hole density of states
#      mp (array:N) -> hole mobility
#      Spl (scalar) -> hole surface recombination velocity at left boundary
#      Spr (scalar) -> hole surface recombination velocity at right boundary
## Outputs :
#      1 (scalar) -> derivative of left boundary condition w.r.t. phi_n[0]
#      2 (scalar) -> derivative of left boundary condition w.r.t. phi_n[1]
#      3 (scalar) -> derivative of right boundary condition w.r.t. phi_n[0]
#      4 (scalar) -> derivative of right boundary condition w.r.t. phi_n[1]
#      5 (scalar) -> derivative of left boundary condition w.r.t. phi[0]
#      6 (scalar) -> derivative of left boundary condition w.r.t. phi[1]
#      7 (scalar) -> derivative of right boundary condition w.r.t. phi[0]
#      8 (scalar) -> derivative of right boundary condition w.r.t. phi[1]

def contact_phip_deriv( phi_p , phi , dgrid , Chi , Eg , Nv , mp , Spl , Spr ):
    _p = p( phi_p , phi , Chi , Eg , Nv )
    dJp_phip_maindiag , dJp_phip_upperdiag , dJp_phi_maindiag , dJp_phi_upperdiag = Jp_deriv( phi_p , phi , dgrid , Chi , Eg , Nv , mp )

    return dJp_phip_maindiag[0] + Spl * ( - _p[0] ) , dJp_phip_upperdiag[0] , \
    dJp_phip_maindiag[-1] , dJp_phip_upperdiag[-1]  - Spr * ( - _p[-1] ) , \
    dJp_phi_maindiag[0] + Spl * ( - _p[0] ) , dJp_phi_upperdiag[0] , \
    dJp_phi_maindiag[-1] , dJp_phi_upperdiag[-1]  - Spr * ( - _p[-1] )
