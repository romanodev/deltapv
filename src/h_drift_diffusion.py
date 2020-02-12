from .SHR import *
from .current import *

### Compute the left side term of the drift-diffusion equation for holes
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
#      mp (array:N) -> hole mobility
#      G (array:N) -> electron-hole generation rate density
## Outputs :
#      1 (array:N-2) -> left side term of the drift-diffusion equation for holes

def ddp( phi_n , phi_p , phi , dgrid , Chi , Eg , Nc , Nv , Et , tn , tp , mp , G ):
    R = SHR( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Et , tn , tp )
    _Jp = Jp( phi_p , phi , dgrid , Chi , Eg , Nv , mp )
    ave_dgrid = ( dgrid[:-1] + dgrid[1:] ) / 2.0
    return ( _Jp[1:] - _Jp[:-1] ) / ave_dgrid + R[1:-1] - G[1:-1]





### Compute the derivatives of the drift-diffusion equation for holes w.r.t. e- and hole quasi-Fermi energy
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
#      mp (array:N) -> hole mobility
#      G (array:N) -> electron-hole generation rate density
## Outputs :
#      1 (array:N-2) -> derivative of drift-diffusion equation for holes at point i i w.r.t. phi_n[i]
#      2 (array:N-2) -> derivative of drift-diffusion equation for holes at point i i w.r.t. phi_p[i]
#      3 (array:N-2) -> derivative of drift-diffusion equation for holes at point i i w.r.t. phi_p[i+1]
#      4 (array:N-2) -> derivative of drift-diffusion equation for holes at point i i w.r.t. phi[i-1]
#      5 (array:N-2) -> derivative of drift-diffusion equation for holes at point i i w.r.t. phi[i]
#      6 (array:N-2) -> derivative of drift-diffusion equation for holes at point i i w.r.t. phi[i+1]

def ddp_deriv( phi_n , phi_p , phi , dgrid , Chi , Eg , Nc , Nv , Et , tn , tp , mp , G ):
    DR_phin , DR_phip , DR_phi = SHR_deriv( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Et , tn , tp )
    dJp_phip_maindiag , dJp_phip_upperdiag , dJp_phi_maindiag , dJp_phi_upperdiag = Jp_deriv( phi_p , phi , dgrid , Chi , Eg , Nv , mp )

    ave_dgrid = ( dgrid[:-1] + dgrid[1:] ) / 2.0

    ddp_phip_ = - dJp_phip_maindiag[:-1] / ave_dgrid
    ddp_phip__ = ( - dJp_phip_upperdiag[:-1] + dJp_phip_maindiag[1:] ) / ave_dgrid + DR_phip[1:-1]
    ddp_phip___ = dJp_phip_upperdiag[1:] / ave_dgrid

    ddp_phi_ = - dJp_phi_maindiag[:-1] / ave_dgrid
    ddp_phi__ = ( - dJp_phi_upperdiag[:-1] + dJp_phi_maindiag[1:] ) / ave_dgrid + DR_phi[1:-1]
    ddp_phi___ = dJp_phi_upperdiag[1:] / ave_dgrid

    ddp_phin__ = DR_phin[1:-1]

    return ddp_phin__ , ddp_phip_ , ddp_phip__ , ddp_phip___ , ddp_phi_ , ddp_phi__ , ddp_phi___
