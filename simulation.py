import jax.numpy as np
from Newton_method import *

T = 300
grid = np.linspace(0.0, 1.0, num=10)
eps = np.ones(grid.size)
N_a = np.zeros(grid.size)
N_d = np.ones(grid.size)
G = np.zeros(grid.size)
E_g = np.ones(grid.size)
N_c = np.ones(grid.size)
N_v = np.ones(grid.size)
mu_e = np.ones(grid.size)
mu_h = np.ones(grid.size)
E_T = np.zeros(grid.size)
tau_n = np.ones(grid.size)
tau_p = np.ones(grid.size)
S_n = 0
S_p = 0
V = 0
phi_eq_bound_cond = "Neumann"

state = State( temperature = T , grid = grid , dielectric_cst = eps , acceptor_concentration = N_a , donor_concentration = N_d ,
generation = G , band_gap = E_g , conduction_DOS = N_c , valence_DOS = N_v , electron_mobility = mu_e , hole_mobility = mu_h,
SHR_trap_level = E_T , e_lifetime_level = tau_n , h_lifetime_level = tau_p , e_surface_recomb_velocity_at_contact = S_n ,
h_surface_recomb_velocity_at_contact = S_p , voltage = V , phi_eq_bound_cond = phi_eq_bound_cond )
state.simulation()
