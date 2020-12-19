from jaxpv import objects, physics, util
from jax import numpy as np

PVCell = objects.PVCell
LightSource = objects.LightSource
Array = util.Array
f64 = util.f64


def eq_init_phi(cell: PVCell) -> Array:

    phi_ini_left = util.switch(cell.Ndop[0] > 0,
                               -cell.Chi[0] + np.log(cell.Ndop[0] / cell.Nc[0]),
                               -cell.Chi[0] - cell.Eg[0] - np.log(-cell.Ndop[0] / cell.Nv[0]))
    
    phi_ini_right = util.switch(cell.Ndop[-1] > 0,
                                -cell.Chi[-1] + np.log(cell.Ndop[-1] / cell.Nc[-1]),
                                -cell.Chi[-1] - cell.Eg[-1] - np.log(-cell.Ndop[-1] / cell.Nv[-1]))
    
    return np.linspace(phi_ini_left, phi_ini_right, cell.grid.size)
