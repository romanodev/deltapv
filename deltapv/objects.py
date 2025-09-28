from deltapv import util
from deltapv import dataclasses_dpv as dataclasses
from jax import numpy as jnp
from typing import Union
from dataclasses import field

Array = util.Array
f64 = float   # safer for dataclass defaults
i64 = int     # safer for dataclass defaults


# -----------------------------------------------------------------------------
# Core device design and cell structures
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class PVDesign:
    grid: Array
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
    alpha: Array
    Ndop: Array
    Snl: f64 = 0.0
    Snr: f64 = 0.0
    Spl: f64 = 0.0
    Spr: f64 = 0.0
    PhiM0: f64 = 0.0
    PhiML: f64 = 0.0


@dataclasses.dataclass
class PVCell:
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
    Ndop: Array
    G: Array
    Snl: f64 = 0.0
    Snr: f64 = 0.0
    Spl: f64 = 0.0
    Spr: f64 = 0.0
    PhiM0: f64 = 0.0
    PhiML: f64 = 0.0


def zero_cell(n: i64) -> PVCell:
    nz = jnp.zeros(n)
    mn1z = jnp.zeros(n - 1)
    return PVCell(
        dgrid=mn1z,
        eps=nz,
        Chi=nz,
        Eg=nz,
        Nc=nz,
        Nv=nz,
        mn=nz,
        mp=nz,
        tn=nz,
        tp=nz,
        Et=nz,
        Br=nz,
        Cn=nz,
        Cp=nz,
        Ndop=nz,
        G=nz,
    )


# -----------------------------------------------------------------------------
# Light, material, and potential definitions
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class LightSource:
    Lambda: Array = field(default_factory=lambda: jnp.ones(1))
    P_in: Array = field(default_factory=lambda: jnp.zeros(1))


@dataclasses.dataclass
class Material:
    eps: f64 = 1.0
    Chi: f64 = 1.0
    Eg: f64 = 1.0
    Nc: f64 = 1e17
    Nv: f64 = 1e17
    mn: f64 = 1e2
    mp: f64 = 1e2
    tn: f64 = 1e-8
    tp: f64 = 1e-8
    Et: f64 = 0.0
    Br: f64 = 0.0
    Cn: f64 = 0.0
    Cp: f64 = 0.0
    A: f64 = 0.0
    alpha: Array = field(default_factory=lambda: jnp.zeros(100))

    def __iter__(self):
        return iter(self.__dict__.items())


@dataclasses.dataclass
class Potentials:
    phi: Array
    phi_n: Array
    phi_p: Array


def zero_pot(n: i64) -> Potentials:
    nz = jnp.zeros(n)
    return Potentials(phi=nz, phi_n=nz, phi_p=nz)


@dataclasses.dataclass
class Boundary:
    phi0: f64 = 0.0
    phiL: f64 = 0.0
    neq0: f64 = 0.0
    neqL: f64 = 0.0
    peq0: f64 = 0.0
    peqL: f64 = 0.0


# -----------------------------------------------------------------------------
# Update helper
# -----------------------------------------------------------------------------

def update(
    obj: Union[PVDesign, PVCell, Material], **kwargs
) -> Union[PVDesign, PVCell, Material]:
    return obj.__class__(
        **{
            key: kwargs[key] if key in kwargs else value
            for key, value in obj.__dict__.items()
        }
    )