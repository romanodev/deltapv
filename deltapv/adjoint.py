from deltapv import objects, solver, bcond, current, residual, linalg, util
from jax import numpy as jnp, custom_jvp, jvp, jacrev

import logging
logger = logging.getLogger("deltapv")

PVCell = objects.PVCell
Potentials = objects.Potentials
Array = util.Array
f64 = util.f64


def solve_pdd(cell: PVCell, v: f64, pot_ini: Potentials):
    """Solve PDD system at a specified voltage, with IFT for gradient

    Args:
        cell (PVCell): An initialized cell
        v (f64): Voltage to solve at, in dimensionless form
        pot_ini (Potentials): Initial guess of solution

    Returns:
        (f64, Potentials): Tuple of current found, in dimensionless form,
        and solution
    """
    # Determine boundary conditions
    bound = bcond.boundary(cell, v)

    # Solve system
    pot = solver.solve(cell, bound, pot_ini)

    # Compute total current
    flux = current.total_current(cell, pot)

    return flux, pot


@custom_jvp
def solve_pdd_adjoint(cell: PVCell, v: f64, pot_ini: Potentials):
    """Solve PDD system at a specified voltage, with adjoint method for gradient

    Args:
        cell (PVCell): An initialized cell
        v (f64): Voltage to solve at, in dimensionless form
        pot_ini (Potentials): Initial guess of solution

    Returns:
        (f64, Potentials): Tuple of current found, in dimensionless form,
        and solution
    """
    # Determine boundary conditions
    bound = bcond.boundary(cell, v)

    # Solve system
    pot = solver.solve(cell, bound, pot_ini)

    # Compute total current
    flux = current.total_current(cell, pot)

    return flux, pot


def F_wb(cell, v, pot):
    """Calculates residual of PDD system for a cell and solution guess at a
    specified voltage

    Args:
        cell (PVCell): An initialized cell
        v (f64): Voltage to solve at, in dimensionless form
        pot (Potentials): Guess of solution

    Returns:
        Array: Residual of PDD system
    """
    bound = bcond.boundary(cell, v)
    F = residual.comp_F(cell, bound, pot)
    return F


@solve_pdd_adjoint.defjvp
def solve_pdd_adjoint_jvp(primals, tangents):
    """Custom JVP implementing adjoint method

    Args:
        primals (Tuple): Ijnput arguments
        tangents (Tuple): Tangent of ijnput arguments

    Returns:
        Tuple: Value and tangent of PDD system solution
    """
    cell, v, pot_ini = primals
    dcell, _, _ = tangents

    # Solve forward problem
    bound = bcond.boundary(cell, v)
    pot = solver.solve(cell, bound, pot_ini)
    flux = current.total_current(cell, pot)

    # Compute gradients with adjoint method
    n = cell.Eg.size
    zp = objects.zero_pot(n)

    _, delF = jvp(F_wb, (cell, v, pot), (dcell, 0., zp))  # Fp @ dp
    _, delg = jvp(current.total_current, (cell, pot), (dcell, zp))  # gp @ dp

    gx_pot = jacrev(current.total_current, argnums=1)(cell, pot)
    gx = solver.pot2vec(gx_pot)  # vector form

    spFx = residual.comp_F_deriv(cell, bound, pot)
    FxT = linalg.sparse2dense(spFx).T

    lam = jnp.linalg.solve(FxT, gx)

    dg = delg - jnp.dot(lam, delF)  # total derivative

    primals_out = flux, pot
    tangents_out = dg, solver.vec2pot(jnp.zeros_like(gx))

    return primals_out, tangents_out
