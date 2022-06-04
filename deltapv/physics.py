from deltapv import objects, scales, util
from jax import numpy as jnp, custom_jvp

PVCell = objects.PVCell
LightSource = objects.LightSource
Potentials = objects.Potentials
Array = util.Array
f64 = util.f64


@custom_jvp
def n(cell: PVCell, pot: Potentials) -> Array:

    return cell.Nc * jnp.exp(cell.Chi + pot.phi_n + pot.phi)


@n.defjvp
def n_jvp(primals, tangents):
    cell, pot = primals
    dcell, dpot = tangents
    expterm = jnp.exp(cell.Chi + pot.phi_n + pot.phi)
    primal_out = cell.Nc * expterm
    tangent_out = expterm * (dcell.Nc + cell.Nc
                             * (dcell.Chi + dpot.phi_n + dpot.phi))
    return primal_out, tangent_out


@custom_jvp
def p(cell: PVCell, pot: Potentials) -> Array:

    return cell.Nv * jnp.exp(-cell.Chi - cell.Eg - pot.phi_p - pot.phi)


@p.defjvp
def p_jvp(primals, tangents):
    cell, pot = primals
    dcell, dpot = tangents
    expterm = jnp.exp(-cell.Chi - cell.Eg - pot.phi_p - pot.phi)
    primal_out = cell.Nv * expterm
    tangent_out = expterm * (dcell.Nv - cell.Nv
                             * (dcell.Chi + dcell.Eg + dpot.phi_p + dpot.phi))
    return primal_out, tangent_out


def charge(cell: PVCell, pot: Potentials) -> Array:

    _n = n(cell, pot)
    _p = p(cell, pot)
    return -_n + _p + cell.Ndop


def ni(cell: PVCell) -> Array:

    return jnp.sqrt(cell.Nc * cell.Nv) * jnp.exp(-cell.Eg / 2)


def Ec(cell: PVCell) -> Array:

    return -cell.Chi


def Ev(cell: PVCell) -> Array:

    return -cell.Chi - cell.Eg


def EFi(cell: PVCell) -> Array:

    return Ec(cell) - cell.Eg / 2 + scales.kB * scales.T / (
        2 * scales.q * scales.energy) * jnp.log(cell.Nc / cell.Nv)


def EF(cell: PVCell) -> Array:

    _ni = ni(cell)
    Ndop_nz = jnp.where(cell.Ndop != 0, cell.Ndop, _ni)

    dEF = scales.kB * scales.T / (scales.q * scales.energy) * jnp.where(
        Ndop_nz > 0, jnp.log(jnp.abs(Ndop_nz) / _ni),
        -jnp.log(jnp.abs(Ndop_nz) / _ni))

    return EFi(cell) + dEF


def flatband_wf(Nc, Nv, Eg, Chi, N):

    kBT = scales.kB * scales.T / scales.q
    ni = jnp.sqrt(Nc * Nv) * jnp.exp(-Eg / (2 * kBT))
    Ec = -Chi
    EFi = Ec - Eg / 2 + (kBT / 2) * jnp.log(Nc / Nv)
    dEF = kBT * jnp.where(N > 0, jnp.log(jnp.abs(N) / ni),
                          -jnp.log(jnp.abs(N) / ni))

    return -EFi - dEF
