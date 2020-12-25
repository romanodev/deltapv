from jaxpv import objects, solver, scales, current, physics, bcond, util
from jax import numpy as np, ops

PVCell = objects.PVCell
LightSource = objects.LightSource
Potentials = objects.Potentials
Boundary = objects.Boundary
Array = util.Array
f64 = util.f64
i32 = util.i32


def vincr(cell: PVCell, num_vals: i32 = 50) -> f64:

    phi_ini_left, phi_ini_right = bcond.boundary_phi(cell)
    incr_step = np.abs(phi_ini_right - phi_ini_left) / num_vals
    incr_sign = (-1)**(phi_ini_right > phi_ini_left)

    return incr_sign * incr_step


def calc_iv(cell: PVCell, vincr: f64) -> Array:

    N = cell.grid.size

    print("Solving equilibrium...")
    bound_eq = bcond.boundary_eq(cell)
    pot_ini = Potentials(
        np.linspace(bound_eq.phi0, bound_eq.phiL, cell.grid.size), np.zeros(N),
        np.zeros(N))
    pot = solver.solve_eq(cell, bound_eq, pot_ini)

    jcurve = np.array([], dtype=f64)
    voltages = np.array([], dtype=f64)
    vstep = 0

    while vstep < 100:

        v = vincr * vstep
        scaled_v = v * scales.E
        print(f"Solving for v = {scaled_v}...")
        bound = bcond.boundary(cell, v)
        pot = solver.solve(cell, bound, pot)

        total_j, _ = current.total_current(cell, pot)
        jcurve = np.append(jcurve, total_j)
        voltages = np.append(voltages, vincr * vstep)
        vstep += 1

        if jcurve.size > 2:
            if (jcurve[-2] * jcurve[-1]) <= 0:
                break

    return jcurve, voltages
