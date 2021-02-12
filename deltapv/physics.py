from deltapv import objects, scales, util
from jax import numpy as np, custom_jvp

PVCell = objects.PVCell
LightSource = objects.LightSource
Potentials = objects.Potentials
Array = util.Array
f64 = util.f64


@custom_jvp
def n(cell: PVCell, pot: Potentials) -> Array:

    return cell.Nc * np.exp(cell.Chi + pot.phi_n + pot.phi)


@n.defjvp
def n_jvp(primals, tangents):
    cell, pot = primals
    dcell, dpot = tangents
    expterm = np.exp(cell.Chi + pot.phi_n + pot.phi)
    primal_out = cell.Nc * expterm
    tangent_out = expterm * (dcell.Nc + cell.Nc * (dcell.Chi + dpot.phi_n + dpot.phi))
    return primal_out, tangent_out


@custom_jvp
def p(cell: PVCell, pot: Potentials) -> Array:

    return cell.Nv * np.exp(-cell.Chi - cell.Eg - pot.phi_p - pot.phi)


@p.defjvp
def p_jvp(primals, tangents):
    cell, pot = primals
    dcell, dpot = tangents
    expterm = np.exp(-cell.Chi - cell.Eg - pot.phi_p - pot.phi)
    primal_out = cell.Nv * expterm
    tangent_out = expterm * (dcell.Nv - cell.Nv * (dcell.Chi + dcell.Eg + dpot.phi_p + dpot.phi))
    return primal_out, tangent_out


def charge(cell: PVCell, pot: Potentials) -> Array:

    _n = n(cell, pot)
    _p = p(cell, pot)
    return -_n + _p + cell.Ndop


def ni(cell: PVCell) -> Array:

    return np.sqrt(cell.Nc * cell.Nv) * np.exp(-cell.Eg / 2)


def Ec(cell: PVCell) -> Array:

    return -cell.Chi


def Ev(cell: PVCell) -> Array:

    return -cell.Chi - cell.Eg


def EFi(cell: PVCell) -> Array:

    return Ec(cell) - cell.Eg / 2 + scales.kB * scales.T / (
        2 * scales.q * scales.energy) * np.log(cell.Nc / cell.Nv)


def EF(cell: PVCell) -> Array:

    _ni = ni(cell)
    Ndop_nz = np.where(cell.Ndop != 0, cell.Ndop, _ni)

    dEF = scales.kB * scales.T / (scales.q * scales.energy) * np.where(
        Ndop_nz > 0, np.log(np.abs(Ndop_nz) / _ni),
        -np.log(np.abs(Ndop_nz) / _ni))

    return EFi(cell) + dEF
