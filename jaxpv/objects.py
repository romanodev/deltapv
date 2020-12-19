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
    G: Array = -np.ones(1)