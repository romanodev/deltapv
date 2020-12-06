import jax.numpy as np


def n(data, phi_n, phi):
    
    Chi = data["Chi"]
    Nc = data["Nc"]
    return Nc * np.exp(Chi + phi_n + phi)


def p(data, phi_p, phi):
    
    Chi = data["Chi"]
    Eg = data["Eg"]
    Nv = data["Nv"]
    return Nv * np.exp(-Chi - Eg - phi_p - phi)


def charge(data, phi_n, phi_p, phi):
    
    Ndop = data["Ndop"]
    _n = n(data, phi_n, phi)
    _p = p(data, phi_p, phi)
    return -_n + _p + Ndop


def ni(data):
    
    Nc = data["Nc"]
    Nv = data["Nv"]
    Eg = data["Eg"]
    return np.sqrt(Nc * Nv) * np.exp(-Eg / 2)
