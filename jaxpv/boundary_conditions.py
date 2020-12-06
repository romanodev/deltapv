from . import physics
from . import current


def contact_phin(data, neq0, neqL, phi_n, phi):
    
    Snl = data["Snl"]
    Snr = data["Snr"]
    n = physics.n(data, phi_n, phi)
    Jn = current.Jn(data, phi_n, phi)
    return Jn[0] - Snl * (n[0] - neq0), Jn[-1] + Snr * (n[-1] - neqL)


def contact_phin_deriv(data, phi_n, phi):
    
    Snl = data["Snl"]
    Snr = data["Snr"]
    n = physics.n(data, phi_n, phi)
    dJn_phin_maindiag, dJn_phin_upperdiag, dJn_phi_maindiag, dJn_phi_upperdiag = current.Jn_deriv(data, phi_n, phi)
    
    return dJn_phin_maindiag[0] - Snl * n[0] , dJn_phin_upperdiag[0] , \
    dJn_phi_maindiag[0] - Snl * n[0] , dJn_phi_upperdiag[0] , \
    dJn_phin_maindiag[-1] , dJn_phin_upperdiag[-1] + Snr * n[-1] , \
    dJn_phi_maindiag[-1] , dJn_phi_upperdiag[-1] + Snr * n[-1]


def contact_phip(data, peq0, peqL, phi_p, phi):
    
    Spl = data["Spl"]
    Spr = data["Spr"]
    p = physics.p(data, phi_p, phi)
    Jp = current.Jp(data, phi_p, phi)
    return Jp[0] + Spl * (p[0] - peq0), Jp[-1] - Spr * (p[-1] - peqL)


def contact_phip_deriv(data, phi_p, phi):
    
    Spl = data["Spl"]
    Spr = data["Spr"]
    p = physics.p(data, phi_p, phi)
    dJp_phip_maindiag, dJp_phip_upperdiag, dJp_phi_maindiag, dJp_phi_upperdiag = current.Jp_deriv(data, phi_p, phi)
    
    return dJp_phip_maindiag[0] - Spl * p[0] , dJp_phip_upperdiag[0] , \
    dJp_phi_maindiag[0] - Spl * p[0] , dJp_phi_upperdiag[0] , \
    dJp_phip_maindiag[-1] , dJp_phip_upperdiag[-1] + Spr * p[-1] , \
    dJp_phi_maindiag[-1] , dJp_phi_upperdiag[-1] + Spr * p[-1]
