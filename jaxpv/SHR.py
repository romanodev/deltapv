from . import physics


def comp_SHR(data, phi_n, phi_p, phi):

    Et = data["Et"]
    tp = data["tp"]
    tn = data["tn"]
    ni = physics.ni(data)
    n = physics.n(data, phi_n, phi)
    p = physics.p(data, phi_p, phi)
    nR = ni * np.exp(Et) + n
    pR = ni * np.exp(-Et) + p
    return (n * p - ni**2) / (tp * nR + tn * pR)


def comp_SHR_deriv(data, phi_n, phi_p, phi):

    Et = data["Et"]
    tp = data["tp"]
    tn = data["tn"]
    ni = physics.ni(data)
    n = physics.n(data, phi_n, phi)
    p = physics.p(data, phi_p, phi)
    nR = ni * np.exp(Et) + n
    pR = ni * np.exp(-Et) + p
    num = n * p - ni**2
    denom = tp * nR + tn * pR

    DR_phin = ((n * p) * denom - num * (tp * n)) * denom**(-2)
    DR_phip = ((-n * p) * denom - num * (-tn * p)) * denom**(-2)
    DR_phi = (-num * (tp * n - tn * p)) * denom**(-2)

    return DR_phin, DR_phip, DR_phi
