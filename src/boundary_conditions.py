from physics import *
from current import *

def contact_phin( phi_n , phi , dgrid , Chi , Nc , mn , Snl , Snr , neq_0 , neq_L ):
    _n = n( phi_n , phi , Chi , Nc )
    _Jn = Jn( phi_n , phi , dgrid , Chi , Nc , mn )
    return _Jn[0] - Snl * ( _n[0] - neq_0 ) , _Jn[-1] + Snr * ( _n[-1] - neq_L )

def contact_phin_deriv( phi_n , phi , dgrid , Chi , Nc , mn , Snl , Snr ):
    _n = n( phi_n , phi , Chi , Nc )
    dJn_phin_maindiag , dJn_phin_upperdiag , dJn_phi_maindiag , dJn_phi_upperdiag = Jn_deriv( phi_n , phi , dgrid , Chi , Nc , mn )
    return dJn_phin_maindiag[0] - Snl * _n[0] , dJn_phin_upperdiag[0] , \
    dJn_phin_maindiag[-1] + Snr * _n[-1] , dJn_phin_upperdiag[-1] , \
    dJn_phi_maindiag[0] - Snl * _n[0] , dJn_phi_upperdiag[0] , \
    dJn_phi_maindiag[-1] + Snr * _n[-1] , dJn_phi_upperdiag[-1]

def contact_phip( phi_p , phi , dgrid , Chi , Eg , Nv , mp , Spl , Spr , peq_0 , peq_L ):
    _p = p( phi_p , phi , Chi , Eg , Nv )
    _Jp = Jp( phi_p , phi , dgrid , Chi , Eg , Nv , mp )
    return _Jp[0] + Spl * ( _p[0] - peq_0 ) , _Jp[-1] - Spr * ( _p[-1] - peq_L )

def contact_phip_deriv( phi_p , phi , dgrid , Chi , Eg , Nv , mp , Spl , Spr ):
    _p = p( phi_p , phi , Chi , Eg , Nv )
    dJp_phip_maindiag , dJp_phip_upperdiag , dJp_phi_maindiag , dJp_phi_upperdiag = Jp_deriv( phi_p , phi , dgrid , Chi , Eg , Nv , mp )

    return dJp_phip_maindiag[0] + Spl * ( - _p[0] ) , dJp_phip_upperdiag[0] , \
    dJp_phip_maindiag[-1] , dJp_phip_upperdiag[-1]  - Spr * ( - _p[-1] ) , \
    dJp_phi_maindiag[0] + Spl * ( - _p[0] ) , dJp_phi_upperdiag[0] , \
    dJp_phi_maindiag[-1] , dJp_phi_upperdiag[-1]  - Spr * ( - _p[-1] )
