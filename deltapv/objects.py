from deltapv import equinox_objects, util
from jax import numpy as jnp
from typing import Union
import equinox as eqx

Array = util.Array
f64 = util.f64
i64 = util.i64


@equinox_objects.eqx_dataclass
class PVDesign(eqx.Module):

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
    Snl: f64
    Snr: f64
    Spl: f64
    Spr: f64
    PhiM0: f64
    PhiML: f64


@equinox_objects.eqx_dataclass
class PVCell(eqx.Module):

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
    Snl: f64
    Snr: f64
    Spl: f64
    Spr: f64
    PhiM0: f64
    PhiML: f64


def zero_cell(n: i64) -> PVCell:
    nz = jnp.zeros(n)
    mn1z = jnp.zeros(n - 1)
    zc = PVCell(mn1z, nz, nz, nz, nz, nz, nz, nz, nz, nz, nz, nz, nz, nz, nz,
                nz, 0., 0., 0., 0., 0., 0.)
    return zc


@equinox_objects.eqx_dataclass
class LightSource(eqx.Module):

    Lambda: Array = jnp.ones(1)
    P_in: Array = jnp.zeros(1)


@equinox_objects.eqx_dataclass
class Material(eqx.Module):
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
    alpha: Array = jnp.zeros(100)

    def __iter__(self):
        return self.__dict__.items().__iter__()


@equinox_objects.eqx_dataclass
class Potentials(eqx.Module):
    phi: Array
    phi_n: Array
    phi_p: Array


def zero_pot(n: i64) -> Potentials:
    nz = jnp.zeros(n)
    zp = Potentials(nz, nz, nz)
    return zp


@equinox_objects.eqx_dataclass
class Boundary(eqx.Module):
    phi0: f64
    phiL: f64
    neq0: f64
    neqL: f64
    peq0: f64
    peqL: f64


def update(obj: Union[PVDesign, PVCell, Material], **kwargs) -> Union[PVDesign, PVCell, Material]:

    return obj.__class__(
        **{
            key: kwargs[key] if key in kwargs else value
            for key, value in obj.__dict__.items()
        })
