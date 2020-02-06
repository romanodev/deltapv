from .SHR import *
from .current import *

def ddp( phi_n , phi_p , phi , dgrid , Chi , Eg , Nc , Nv , Et , tn , tp , mp , G ):
    R = SHR( phi_n , phi_p , phi , Chi , Eg , Nc , Nv , Et , tn , tp )
    _Jp = Jp( phi_p , phi , dgrid , Chi , Eg , Nv , mp )
    ave_dgrid = ( dgrid[:-1] + dgrid[1:] ) / 2.0
    return ( _Jp[1:] - _Jp[:-1] ) / ave_dgrid + R[1:-1] - G[1:-1]

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
