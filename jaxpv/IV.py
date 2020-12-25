from jaxpv import objects, solver, scales, current, physics, bcond, util
from jax import numpy as np, ops

PVCell = objects.PVCell
LightSource = objects.LightSource
Potentials = objects.Potentials
Boundary = objects.Boundary
Array = util.Array
f64 = util.f64


def Vincrement(cell: PVCell, num_vals: int = 50) -> f64:

    phi_ini_left = util.switch(
        cell.Ndop[0] > 0, -cell.Chi[0] + np.log(np.abs(cell.Ndop[0] / cell.Nc[0])),
        -cell.Chi[0] - cell.Eg[0] - np.log(np.abs(cell.Ndop[0]) / cell.Nv[0]))

    phi_ini_right = util.switch(
        cell.Ndop[-1] > 0, -cell.Chi[-1] + np.log(np.abs(cell.Ndop[-1] / cell.Nc[-1])),
        -cell.Chi[-1] - cell.Eg[-1] -
        np.log(np.abs(cell.Ndop[-1]) / cell.Nv[-1]))

    incr_step = np.abs(phi_ini_right - phi_ini_left) / num_vals
    incr_sign = (-1)**(phi_ini_right > phi_ini_left)

    return incr_sign * incr_step


def calc_IV(cell: PVCell, Vincrement: f64) -> Array:

    N = cell.grid.size
    
    bound_eq = bcond.boundary_eq(cell)
    pot_ini = Potentials(np.linspace(bound_eq.phi0, bound_eq.phiL, cell.grid.size),
                         np.zeros(N),
                         np.zeros(N))
    print("Solving equilibrium...")
    pot = solver.solve_eq(cell, bound_eq, pot_ini)

    jcurve = np.array([], dtype=f64)
    voltages = np.array([], dtype=f64)
    vstep = 0

    while vstep < 100:

        V = Vincrement * vstep
        scaled_V = V * scales.E
        print(f"Solving for V = {scaled_V}...")
        
        bound = bcond.boundary(cell, bound_eq, V)

        sol = solver.solve(cell, bound, pot)
        total_j, _ = current.total_current(cell, sol)

        jcurve = np.append(jcurve, total_j)
        voltages = np.append(voltages, Vincrement * vstep)
        
        if jcurve.size > 2:
            if (jcurve[-2] * jcurve[-1]) <= 0:
                break
        
        vstep += 1
        V = Vincrement * vstep
        pot = Potentials(
            ops.index_update(sol.phi, ops.index[-1], bound_eq.phiL + V),
            sol.phi_n, sol.phi_p)

    return jcurve, voltages
