from jaxpv import objects, IV, scales, util
from jax import numpy as np, lax

PVCell = objects.PVCell
LightSource = objects.LightSource
Array = util.Array
f64 = util.f64


def Vincrement(cell: PVCell, num_vals: int=50) -> f64:
    
    phi_ini_left = util.switch(cell.Ndop[0] > 0,
                               -cell.Chi[0] + np.log(cell.Ndop[0] / cell.Nc[0]),
                               -cell.Chi[0] - cell.Eg[0] - np.log(np.abs(cell.Ndop[0]) / cell.Nv[0]))
    
    phi_ini_right = util.switch(cell.Ndop[-1] > 0,
                                -cell.Chi[-1] + np.log(cell.Ndop[-1] / cell.Nc[-1]),
                                -cell.Chi[-1] - cell.Eg[-1] - np.log(np.abs(cell.Ndop[-1]) / cell.Nv[-1]))
    
    incr_step = np.abs(phi_ini_right - phi_ini_left) / num_vals
    incr_sign = (-1) ** (phi_ini_right > phi_ini_left)

    return incr_sign * incr_step


def comp_eff(cell: PVCell, Vincrement: f64) -> f64:

    current = IV.calc_IV(cell, Vincrement)
    voltages = np.linspace(start=0,
                           stop=(current.size - 1) * Vincrement,
                           num=current.size)
    Pmax = np.max(scales.E * voltages * scales.J * current) * 1e4  # W/m2
    eff = Pmax / 1e3

    return eff
