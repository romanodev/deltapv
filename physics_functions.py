import jax.numpy as np
import scipy.constants as sc

# Define the energy level describing the electron population
# Args : 2 arrays (length : N) -> return : array (length : N)
def e_pot( e_fermi , phi ):
    return e_fermi + sc.e * phi

# Define the energy level describing the hole population
# Args : 3 arrays (length : N) -> return : array (length : N)
def h_pot( E_g , h_fermi , phi ):
    return - h_fermi - sc.e * phi - E_g

# Define the dimensionless energy level describing the electron population
# Args : 1 scalar, 2 arrays (length : N) -> return : array (length : N)
def e_pot_dimless( T , e_fermi , phi ):
    return e_pot( e_fermi , phi ) / ( sc.k * T )

# Define the dimensionless energy level describing the hole population
# Args : 1 scalar, 3 arrays (length : N) -> return : array (length : N)
def h_pot_dimless( T , E_g , h_fermi , phi ):
    return h_pot( E_g , h_fermi , phi ) / ( sc.k * T )

# Define the function giving the electron density
# Args : 1 scalar, 3 arrays (length : N) -> return : array (length : N)
def e_dens( T , N_c , e_fermi , phi ):
    return N_c * np.exp( e_pot_dimless( T , e_fermi , phi ) )

# Define the function giving the hole density
# Args : 1 scalar, 4 arrays (length : N) -> return : array (length : N)
def h_dens( T , E_g , N_v , h_fermi , phi ):
    return N_v * np.exp( h_pot_dimless( T , E_g , h_fermi , phi ) )

# Define the function giving the equilibrium density (n=p)
# Args : 1 scalar, 3 arrays (length : N) -> return : array (length : N)
def eq_dens( T , E_g , N_c , N_v ):
    return np.sqrt( N_c * N_v ) * np.exp( E_g / ( 2 * sc.k * T ) )

# Define auxiliary function to describe electron trapping for SHR recombination
# Args : 1 scalar, 2 arrays (length : N) -> return : array (length : N)
def n_R( T , E_T , eq_dens ):
    return eq_dens * np.exp( E_T / ( sc.k * T ) )

# Define auxiliary function to describe hole trapping for SHR recombination
# Args : 1 scalar, 2 arrays (length : N) -> return : array (length : N)
def p_R( T , E_T , eq_dens ):
    return eq_dens * np.exp( - E_T / ( sc.k * T ) )

# Define SHR recombination rate
# Args : 7 arrays (length : N) -> return : array (length : N)
def R_SHR( tau_p , tau_n , eq_dens , n_R , p_R , e_dens, h_dens ):
    return ( e_dens * h_dens + eq_dens ** 2 ) / ( tau_p * ( e_dens + n_R ) + tau_n * ( h_dens + p_R ) )

# Define -kT*log(electron density)
# Args : 1 scalar, 3 arrays (length : N) -> return : array (length : N)
def e_psi( T , N_c , e_fermi , phi ):
    return - e_fermi - sc.e * phi - sc.k * T * np.log( N_c )

# Define -kT*log(hole density)
# Args : 1 scalar, 4 arrays (length : N) -> return : array (length : N)
def h_psi( T , E_g , N_v , h_fermi , phi ):
    return E_g + h_fermi + sc.e * phi - sc.k * T * np.log( N_v )

# Define electron current
# Args : 1 scalar, 5 arrays (length : N) -> return : array (length : N-1)
def e_current( T , grid , N_c , mu_e , e_fermi , phi ):
    e_p = e_psi( T , N_c , e_fermi , phi )
    e_p_exp = np.exp(- e_p / ( sc.k * T ) )
    fermi = np.exp( e_fermi / ( sc.k * T ) )
    return sc.e / 2 * ( mu_e[:-1] + mu_e[1:] ) * ( e_p[1:] - e_p[:-1] ) * ( fermi[1:] - fermi[:-1] ) \
    / ( ( grid[1:] - grid[:-1] ) * ( e_p_exp[1:] - e_p_exp[:-1] ) )

# Define hole current
# Args : 1 scalar, 6 arrays (length : N) -> return : array (length : N-1)
def h_current( T , grid , E_g , N_v , mu_h , h_fermi , phi ):
    h_p = h_psi( T , E_g , N_v , h_fermi , phi )
    h_p_exp = h_p_exp = np.exp(- h_p / ( sc.k * T ) )
    fermi = np.exp( - h_fermi / ( sc.k * T ) )
    return - sc.e / 2 * ( mu_h[:-1] + mu_h[1:]) * ( h_p[1:] - h_p[:-1] ) * ( fermi[1:] - fermi[:-1] ) \
    / ( ( grid[1:] - grid[:-1] ) * ( h_p_exp[1:] - h_p_exp[:-1] ) )

# Define local charge
# Args : 4 arrays (length : N) -> return : array (length : N)
def loc_charge( N_a , N_d , e_dens , h_dens ):
    return e_dens - h_dens + N_a - N_d

# Define electron drift-diffusion equation (left-side term ; must equal zero)
# Args : 10 arrays (length : N) -> return : array (length : N-2)
def e_dd( grid , G , tau_n , tau_p , e_dens , h_dens , eq_dens , J_e , n_R , p_R ):
    return 2 * ( J_e[1:] - J_e[:-1] ) / ( grid[2:] - grid[:-2] ) - G[1:-1] \
    + R_SHR( tau_p , tau_n , eq_dens , n_R , p_R , e_dens , h_dens )[1:-1]

# Define hole drift-diffusion equation (left-side term ; must equal zero)
# Args : 10 arrays (length : N) -> return : array (length : N-2)
def h_dd( grid , G , tau_n , tau_p , e_dens , h_dens , eq_dens , J_h , n_R , p_R ):
    return 2 * ( J_h[1:] - J_h[:-1] ) / ( grid[2:] - grid[:-2] ) + G[1:-1] \
    - R_SHR( tau_p , tau_n , eq_dens , n_R , p_R , e_dens , h_dens )[1:-1]

# Define poisson equation (left-side term ; must equal zero)
# Args : 7 arrays (length : N) -> return : array (length : N-2)
def pois( grid , eps , N_a , N_d , phi , e_dens , h_dens ):
    first_deriv = ( eps[1:] + eps[:-1] ) * ( phi[1:] - phi[:-1] ) / ( grid[1:] - grid[:-1] )
    return ( first_deriv[1:] - first_deriv[:-1] ) / ( grid[2:] - grid[:-2] ) \
    - loc_charge( e_dens , h_dens , N_a , N_d )[1:-1]

# Define out of equilibrium boundary conditions (left-side term ; must equal zero)
# Args : 2 scalars , 2 arrays (length : N) -> return : scalar
def e_fermi_at_contact_0_condition( S_n , n_eq_0 , J_e , n ):
    return J_e[0] - sc.e * S_n * ( n[0] - n_eq_0 )
def e_fermi_at_contact_L_condition( S_n , n_eq_L , J_e , n ):
    return J_e[-1] + sc.e * S_n * ( n[-1] - n_eq_L )
def h_fermi_at_contact_0_condition( S_p , p_eq_0 , J_h , p ):
    return J_h[0] + sc.e * S_p * ( p[0] - p_eq_0 )
def h_fermi_at_contact_L_condition( S_p , p_eq_L , J_h , p ):
    return J_h[-1] - sc.e * S_p * ( p[-1] - p_eq_L )

# Define the function for which we must find the root to solve for the equilibrium state of the solar cell
# Args : all material parameters -> return : array
def F_root_eq_Dirichlet( T , grid , eps , N_a , N_d , G , E_g , N_c , N_v , mu_e , mu_h , E_T , tau_n , tau_p , phi_reduced ):

    phi = np.concatenate( (  np.array([phi_reduced[0]]) , phi_reduced , np.array([phi_reduced[-1]])  ) , axis = 0 )

    eq_density = eq_dens( T , E_g , N_c , N_v )
    n_SHR = n_R( T , E_T , eq_density )
    p_SHR = p_R( T , E_T , eq_density )
    n = e_dens( T , N_c , np.zeros(grid.size) , phi )
    p = h_dens( T , E_g , N_v , np.zeros(grid.size) , phi )
    J_e = e_current( T , grid , N_c , mu_e , np.zeros(grid.size) , phi )
    J_h = h_current( T , grid , E_g , N_v , mu_h , np.zeros(grid.size) , phi )

    e_driftdiff = e_dd( grid , G , tau_n , tau_p , n , p , eq_density , J_e , n_SHR , p_SHR )
    h_driftdiff = h_dd( grid , G , tau_n , tau_p , n , p , eq_density , J_h , n_SHR , p_SHR )
    poisson = pois( grid , eps , N_a , N_d , phi , n , p )

    return np.concatenate( ( e_driftdiff , h_driftdiff , poisson ) , axis = 0 )

# Define the function for which we must find the root to solve for the equilibrium state of the solar cell
# Args : all material parameters -> return : array
def F_root_eq_Neumann( T , grid , eps , N_a , N_d , G , E_g , N_c , N_v , mu_e , mu_h , E_T , tau_n , tau_p , phi_reduced ):

    phi = np.concatenate( (  np.array([ sc.k * T * log( N_d[0] / N_c[0] ) ]) , phi_reduced , \
    np.array([ sc.k * T * log( N_a[-1] / N_c[-1] ) ])  ) , axis = 0  )

    eq_density = eq_dens( T , E_g , N_c , N_v )
    n_SHR = n_R( T , E_T , eq_density )
    p_SHR = p_R( T , E_T , eq_density )
    n = e_dens( T , N_c , np.zeros(grid.size) , phi )
    p = h_dens( T , E_g , N_v , np.zeros(grid.size) , phi )
    J_e = e_current( T , grid , N_c , mu_e , np.zeros(grid.size) , phi )
    J_h = h_current( T , grid , E_g , N_v , mu_h , np.zeros(grid.size) , phi )

    e_driftdiff = e_dd( grid , G , tau_n , tau_p , n , p , eq_density , J_e , n_SHR , p_SHR )
    h_driftdiff = h_dd( grid , G , tau_n , tau_p , n , p , eq_density , J_h , n_SHR , p_SHR )
    poisson = pois( grid , eps , N_a , N_d , phi , n , p )

    return np.concatenate( ( e_driftdiff , h_driftdiff , poisson ) , axis = 0 )

# Define the function for which we must find the root to solve for the out of equilibrium state of the solar cell
# Args : all material parameters -> return : array
def F_root( T , grid , eps , N_a , N_d , G , E_g , N_c , N_v , mu_e , mu_h , E_T , tau_n , tau_p , S_n , S_p, \
    n_eq_0 , n_eq_L , p_eq_0 , p_eq_L , phi_0 , phi_L , e_fermi , h_fermi , phi_reduced ):

    phi = np.concatenate( [phi_0] , phi_reduced , [phi_L] , axis = 0  )

    eq_density = eq_dens( T , E_g , N_c , N_v )
    n_SHR = n_R( T , E_T , eq_density )
    p_SHR = p_R( T , E_T , eq_density )
    n = e_dens( T , N_c , e_fermi , phi )
    p = h_dens( T , E_g , N_v , h_fermi , phi )
    J_e = e_current( T , grid , N_c , mu_e , e_fermi , phi )
    J_h = h_current( T , grid , E_g , N_v , mu_h , h_fermi , phi )

    e_driftdiff = e_dd( grid , G , tau_n , tau_p , n , p , eq_density , J_e , n_SHR , p_SHR )
    h_driftdiff = h_dd( grid , G , tau_n , tau_p , n , p , eq_density , J_h , n_SHR , p_SHR )
    poisson = pois( grid , eps , N_a , N_d , phi , n , p )
    boundary_cond_e_0 = e_fermi_at_contact_0_condition( S_n , n_eq_0 , J_e , n )
    boundary_cond_e_L = e_fermi_at_contact_L_condition( S_n , n_eq_L , J_e , n )
    boundary_cond_h_0 = h_fermi_at_contact_0_condition( S_p , p_eq_0 , J_h , p )
    boundary_cond_h_L = h_fermi_at_contact_L_condition( S_p , p_eq_L , J_h , p )

    return np.concatenate( ( e_driftdiff , h_driftdiff , poisson , boundary_cond_e_0 , boundary_cond_e_L , \
    boundary_cond_h_0 , boundary_cond_h_L ) , axis = 0 )
