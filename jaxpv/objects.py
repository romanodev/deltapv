from jaxpv import dataclasses, util
from jax import numpy as np

Array = util.Array
f64 = util.f64


@dataclasses.dataclass
class PVCell:

    grid: Array
    dgrid: Array
    eps: Array
    Chi: Array
    Eg: Array
    Nc: Array
    Nv: Array
    mn: Array
    mp: Array
    tn: Array
    tp: Array
    Et: Array
    Br: Array
    Cn: Array
    Cp: Array
    A: Array
    G: Array
    Ndop: Array
    Snl: f64
    Snr: f64
    Spl: f64
    Spr: f64


@dataclasses.dataclass
class LightSource:

    Lambda: Array = np.zeros(1)
    P_in: Array = np.zeros(1)


@dataclasses.dataclass
class Material:
    eps: f64 = f64(1)
    Chi: f64 = f64(1)
    Eg: f64 = f64(1)
    Nc: f64 = f64(1e17)
    Nv: f64 = f64(1e17)
    mn: f64 = f64(1e2)
    mp: f64 = f64(1e2)
    tn: f64 = f64(1e-8)
    tp: f64 = f64(1e-8)
    Et: f64 = f64(0)
    Br: f64 = f64(0)
    Cn: f64 = f64(0)
    Cp: f64 = f64(0)
    A: f64 = f64(0)

    def __iter__(self):
        return self.__dict__.items().__iter__()


@dataclasses.dataclass
class Potentials:
    phi: Array
    phi_n: Array
    phi_p: Array


@dataclasses.dataclass
class Boundary:
    neq0: f64
    neqL: f64
    peq0: f64
    peqL: f64
