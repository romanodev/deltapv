import jax.numpy as np
from numpy.linalg import lstsq
from physics_functions import *
from jax import grad , jit , jacfwd , jacrev

class State:

    ## initialize material properties
    ## TO DO : need to add default values

    def __init__( self , **argv ):
        if 'temperature' in argv.keys():
            self.T = argv['temperature']
        if 'grid' in argv.keys():
            self.grid = argv['grid']
        if 'dielectric_cst' in argv.keys():
            self.eps = argv['dielectric_cst']
        if 'acceptor_concentration' in argv.keys():
            self.N_a = argv['acceptor_concentration']
        if 'donor_concentration' in argv.keys():
            self.N_d = argv['donor_concentration']
        if 'generation' in argv.keys():
            self.G = argv['generation']
        if 'band_gap' in argv.keys():
            self.E_g = argv['band_gap']
        if 'conduction_DOS' in argv.keys():
            self.N_c = argv['conduction_DOS']
        if 'valence_DOS' in argv.keys():
            self.N_v = argv['valence_DOS']
        if 'electron_mobility' in argv.keys():
            self.mu_e = argv['electron_mobility']
        if 'hole_mobility' in argv.keys():
            self.mu_h = argv['hole_mobility']
        if 'SHR_trap_level' in argv.keys():
            self.E_T = argv['SHR_trap_level']
        if 'e_lifetime_level' in argv.keys():
            self.tau_n = argv['e_lifetime_level']
        if 'h_lifetime_level' in argv.keys():
            self.tau_p = argv['h_lifetime_level']
        if 'e_surface_recomb_velocity_at_contact' in argv.keys():
            self.S_n = argv['e_surface_recomb_velocity_at_contact']
        if 'h_surface_recomb_velocity_at_contact' in argv.keys():
            self.S_p = argv['h_surface_recomb_velocity_at_contact']
        if 'voltage' in argv.keys():
            self.V = argv['voltage']
        if 'phi_eq_bound_cond' in argv.keys():
            self.phi_eq_bound_cond = argv['phi_eq_bound_cond']



## in equilibrium, the only variable is the electrostatic potential phi, the rest is zero
## in this case we use a reduced matrix to solve

    ## update state following Newton-method in the equilibrium case

    def take_step_eq( self ):

        ## compute function F(x) for which we want the root, at current phase space position x
        if ( self.phi_eq_bound_cond == 'Neumann' ):
            F_root_eq_jit = jit( F_root_eq_Dirichlet )
            grad_F_eq_jit = jit( jacrev( F_root_eq_Dirichlet , argnums = 14 ) )

        elif ( self.phi_eq_bound_cond == 'Dirichlet' ):
            F_root_eq_jit = jit( F_root_eq_Neumann )
            grad_F_eq_jit = jit( jacrev( F_root_eq_Neumann , argnums = 14 ) )
        F_x_eq = F_root_eq_jit( self.T , self.grid , self.eps , self.N_a , self.N_d , self.G , self.E_g ,
        self.N_c , self.N_v , self.mu_e , self.mu_h , self.E_T , self.tau_n , self.tau_p , self.phi[1:-1] )

        ## compute the Jacobian matrix for the function Grad[f][x]
        grad_F_eq = grad_F_eq_jit( self.T , self.grid , self.eps , self.N_a , self.N_d , self.G , self.E_g ,
        self.N_c , self.N_v , self.mu_e , self.mu_h , self.E_T , self.tau_n , self.tau_p , self.phi[1:-1] )

        ## Newton method : x -= Grad[f][x]^[-1](F(x)) w.r.t. phi
        displ , c = lstsq( grad_F_eq , F_x_eq , rcond = None )
        self.phi[1:-1] = self.phi[1:-1] - displ

    ## solve for the equilibrium state of the solar cell

    def solve_eq( self , step_num ):
        self.e_fermi = np.zeros(self.grid.size)
        self.h_fermi = np.zeros(self.grid.size)
        self.phi = np.zeros(self.grid.size)
        for _ in range( step_num ):
            self.take_step_eq()
        if ( self.phi_eq_bound_cond == 'Neumann' ):
            self.phi[0] = self.phi[1]
            self.phi[-1] = self.phi[-2]
        elif ( self.phi_eq_bound_cond == 'Dirichlet' ):
            self.phi[0] = sc.k * T * log( N_d[0] / N_c[0])
            self.phi[-1] = sc.k * T * log( N_a[-1] / N_c[-1])

## out of equilibrium, we solve for the steady state and the three unknowns
## we start from the previous equilibrium solution ( phi_eq , e/h_fermi = 0 )
## from boundary conditions for phi, we restrict to changing only phi[1:-1] (phi[0] and phi[-1] are fixed)

    ## update state following Newton-method in steady state

    def take_step( self , n_eq_0 , n_eq_L , h_eq_0 , h_eq_L ):

        ## compute function F(x) for which we want the root, at current phase space position x
        F_root_jit = jit(F_root)
        F_x = F_root_jit( self.T , self.grid , self.eps , self.N_a , self.N_d , self.G , self.E_g , self.N_c ,
        self.N_v , self.mu_e , self.mu_h , self.E_T , self.tau_n , self.tau_p , self.S_n, self.S_p , n_eq_0 ,
        n_eq_L , h_eq_0 , h_eq_L , self.phi[0] , self.phi[-1] , self.e_fermi , self.h_fermi , self.phi[1:-1] )

        ## compute the Jacobian matrix for the function Grad[f][x] w.r.t e/h-fermi and phi[1:-1]
        grad_F_jit = jit( jacrev( F_root , argnums = ( 22 , 23 , 24 ) ) )
        grad_F = grad_F_jit( self.T , self.grid , self.eps , self.N_a , self.N_d , self.G , self.E_g , self.N_c ,
        self.N_v , self.mu_e , self.mu_h , self.E_T , self.tau_n , self.tau_p , self.S_n, self.S_p , n_eq_0 ,
        n_eq_L , h_eq_0 , h_eq_L , self.phi[0] , self.phi[-1] , self.e_fermi , self.h_fermi , self.phi[1:-1] )

        ## Newton method : x -= Grad[f][x]^[-1](F(x))
        displ , c = lstsq( grad_F , F_x , rcond = None )
        self.e_fermi = self.e_fermi - displ[0:(self.grid.size-1)]
        self.h_fermi = self.h_fermi - displ[self.grid.size:(2*self.grid.size-1)]
        self.phi[1:-1] = self.phi[1:-1] - displ[2*self.grid.size:-1]


    ## solve for the steady state of the solar cell

    def solve( self , step_num ):
        self.solve_eq( step_num )
        n_eq_0 = e_dens( self.T , self.N_c , self.e_fermi , self.phi )[0]
        n_eq_L = e_dens( self.T , self.N_c , self.e_fermi , self.phi )[-1]
        h_eq_0 = h_dens( T , self.E_g , self.N_v , self.h_fermi , self.phi )[0]
        h_eq_L = h_dens( T , self.E_g , self.N_v , self.h_fermi , self.phi )[-1]
        phi[-1] += self.V
        for _ in range( step_num ):
            self.take_step( n_eq_0 , n_eq_L , h_eq_0 , h_eq_L )

    ## Display results

    def printing( self ):
        np.savetxt( "electro_potential" , self.phi )
        np.savetxt( "e_fermi" , self.e_fermi )
        np.savetxt( "h_fermi" , self.h_fermi )
        np.savetxt( "e_density" , e_dens( self.T , self.N_c , self.e_fermi , self.phi ) )
        np.savetxt( "h_density" , h_dens( T , self.E_g , self.N_v , self.h_fermi , self.phi ) )
        np.savetxt( "e_current" , e_current( self.T , self.grid , self.N_c , self.mu_e , self.e_fermi , self.phi ) )
        np.savetxt( "h_current" , h_current( self.T , self.grid , self.E_g , self.N_v , self.mu_h , self.h_fermi , self.phi ) )

    def simulation( self ):
        step_num = 100
        self.solve( step_num )
        printing( self )
