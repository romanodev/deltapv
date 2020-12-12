from . import physics


def comp_auger(data, phi_n, phi_p, phi):

    Cn = data["Cn"]
    Cp = data["Cp"]
    ni = physics.ni(data)
    n = physics.n(data, phi_n, phi)
    p = physics.p(data, phi_p, phi)
    return (Cn * n + Cp * p) * (n * p - ni**2)


def comp_auger_deriv(data, phi_n, phi_p, phi):

    Cn = data["Cn"]
    Cp = data["Cp"]
    ni = physics.ni(data)
    n = physics.n(data, phi_n, phi)
    p = physics.p(data, phi_p, phi)

    DR_phin = (Cn * n) * (n * p - ni**2) + (Cn * n + Cp * p) * (n * p)
    DR_phip = (-Cp * p) * (n * p - ni**2) + (Cn * n + Cp * p) * (-n * p)
    DR_phi = (Cn * n - Cp * p) * (n * p - ni**2)

    return DR_phin, DR_phip, DR_phi
