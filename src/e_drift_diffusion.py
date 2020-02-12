from .SHR import *
from .current import *

### Compute the left side term of the drift-diffusion equation for e-
## Inputs :
#      phi_n (array:N) -> e- quasi-Fermi energy
#      phi_p (array:N) -> hole quasi-Fermi energy
#      phi (array:N) -> electrostatic potential
#      dgrid (array:N-1) -> array of distances between consecutive grid points
#      Chi (array:N) -> electron affinity
#      Eg (array:N) -> band gap
#      Nc (array:N) -> e- density of states
#      Nv (array:N) -> hole density of states
#      Et (array:N) -> trap state energy level (SHR)
#      tn (array:N) -> e- lifetime (SHR)
#      tp (array:N) -> hole lifetime (SHR)
#      mn (array:N) -> e- mobility
#      G (array:N) -> electron-hole generation rate density
## Outputs :
#      1 (array:N-2) -> left side term of the drift-diffusion equation for e-

def ddn( phi_n , phi_p , phi , dgrid , Chi , Eg , Nc , Nv , Et , tn , tp , mn , G ):
    R = SHR( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Et , tn , tp )
    _Jn = Jn( phi_n , phi , dgrid , Chi , Nc , mn )
    ave_dgrid = ( dgrid[:-1] + dgrid[1:] ) / 2.0
    return ( _Jn[1:] - _Jn[:-1] ) / ave_dgrid - R[1:-1] + G[1:-1]





### Compute the derivatives of the drift-diffusion equation for e- w.r.t. e- and hole quasi-Fermi energy
### and electrostatic potential
## Inputs :
#      phi_n (array:N) -> e- quasi-Fermi energy
#      phi_p (array:N) -> hole quasi-Fermi energy
#      phi (array:N) -> electrostatic potential
#      dgrid (array:N-1) -> array of distances between consecutive grid points
#      Chi (array:N) -> electron affinity
#      Eg (array:N) -> band gap
#      Nc (array:N) -> e- density of states
#      Nv (array:N) -> hole density of states
#      Et (array:N) -> trap state energy level (SHR)
#      tn (array:N) -> e- lifetime (SHR)
#      tp (array:N) -> hole lifetime (SHR)
#      mn (array:N) -> e- mobility
#      G (array:N) -> electron-hole generation rate density
## Outputs :
#      1 (array:N-2) -> derivative of drift-diffusion equation for e- at point i w.r.t. phi_n[i]
#      2 (array:N-2) -> derivative of drift-diffusion equation for e- at point i i w.r.t. phi_n[i+1]
#      3 (array:N-2) -> derivative of drift-diffusion equation for e- at point i i w.r.t. phi_p[i]
#      4 (array:N-2) -> derivative of drift-diffusion equation for e- at point i i w.r.t. phi[i-1]
#      5 (array:N-2) -> derivative of drift-diffusion equation for e- at point i i w.r.t. phi[i]
#      6 (array:N-2) -> derivative of drift-diffusion equation for e- at point i i w.r.t. phi[i+1]

def ddn_deriv( phi_n , phi_p , phi , dgrid , Chi , Eg , Nc , Nv , Et , tn , tp , mn , G ):
    DR_phin , DR_phip , DR_phi = SHR_deriv( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Et , tn , tp )
    dJn_phin_maindiag , dJn_phin_upperdiag , dJn_phi_maindiag , dJn_phi_upperdiag = Jn_deriv( phi_n , phi , dgrid , Chi , Nc , mn )

    ave_dgrid = ( dgrid[:-1] + dgrid[1:] ) / 2.0

    dde_phin_ = - dJn_phin_maindiag[:-1] / ave_dgrid
    dde_phin__ = ( - dJn_phin_upperdiag[:-1] + dJn_phin_maindiag[1:] ) / ave_dgrid - DR_phin[1:-1]
    dde_phin___ = dJn_phin_upperdiag[1:] / ave_dgrid

    dde_phi_ = - dJn_phi_maindiag[:-1] / ave_dgrid
    dde_phi__ = ( - dJn_phi_upperdiag[:-1] + dJn_phi_maindiag[1:] ) / ave_dgrid - DR_phi[1:-1]
    dde_phi___ = dJn_phi_upperdiag[1:] / ave_dgrid

    dde_phip__ = - DR_phip[1:-1]

    return dde_phin_ , dde_phin__ , dde_phin___ , dde_phip__ , dde_phi_ , dde_phi__ , dde_phi___
