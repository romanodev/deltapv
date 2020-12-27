from jaxpv import objects, bcond, current, residual, solver, linalg, util
from jax import numpy as np, ops, lax, vmap, grad, jacfwd, custom_jvp

PVCell = objects.PVCell
Potentials = objects.Potentials
Array = util.Array
f64 = util.f64


def solve_equilibrium(cell):
    
    # cell.G is assumed to be already calculated
    N = cell.grid.size
    bound_eq = bcond.boundary_eq(cell)
    pot_ini = Potentials(
        np.linspace(bound_eq.phi0, bound_eq.phiL, cell.grid.size), np.zeros(N),
        np.zeros(N))
    pot = solver.solve_eq(cell, bound_eq, pot_ini)

    return pot


def solve_bias(cell, v):
    
    pot_eq = solve_equilibrium(cell)
    bound_0 = bcond.boundary(cell, 0)
    pot_ini = solver.solve(cell, bound_0, pot_eq)
    bound = bcond.boundary(cell, v)
    pot = solver.solve(cell, bound, pot_ini)
    
    return pot, bound


jaccur = jit(jacfwd(current.total_current, argnums=(0, 1)))
jacF = jit(jacfwd(residual.comp_F, argnums=(0, 1)))
jacbound = jit(jacfwd(bcond.boundary, argnums=(0,)))


@custom_jvp
def current_bias(cell, v):
    pot, _ = solve_bias(cell, v)
    j = current.total_current(cell, pot)
    return j


@current_bias.defjvp
def current_bias_jvp(primals, tangents):
    cell, v = primals
    dcell, dv = tangents
    pot, bound = solve_bias(cell, v)
    primal_out = current.total_current(cell, pot)
    
    # adjoint method
    
    J_cell, _J_pot = jaccur(cell, pot)
    J_pot = solver.pot2vec(_J_pot)
    
    F_pot = residual.comp_F_deriv(cell, bound, pot)
    lam = linalg.transol(F_pot, J_pot)
    
    F_cell, F_bound = jacF(cell, bound, pot)
    bound_cell = jacbound(cell, v)
    
    dJdNc = _J_cell.Nc - lam @ F_cell.Nc \
        - np.dot(lam, F_bound.phi0) * bound_cell.phi0[0].Nc \
        - np.dot(lam, F_bound.phiL) * bound_cell.phiL[0].Nc \
        - np.dot(lam, F_bound.neq0) * bound_cell.neq0[0].Nc \
        - np.dot(lam, F_bound.neqL) * bound_cell.neqL[0].Nc \
        - np.dot(lam, F_bound.peq0) * bound_cell.peq0[0].Nc \
        - np.dot(lam, F_bound.peqL) * bound_cell.peqL[0].Nc \
    
    tangent_out = dJdNc @ dcell.Nc
    
    return primal_out, tangent_out