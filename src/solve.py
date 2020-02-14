from .F import *
if USE_JAX:
    from jax import jit

### Compute the damped displacement of potentials based on the Newton step
## Inputs :
#      move (array:3N) -> Newton-based change in e-/hole quais-Fermi energy and electrostatic potential
## Outputs :
#      1 (array:3N) -> damped change in potentials

def damp( move ):
    approx_sign = np.tanh( move )
    approx_abs = approx_sign * move
    approx_H = 1 - ( 1 + np.exp( - 500 * ( move**2 - 1 ) ) )**(-1)
    return np.log( 1 + approx_abs ) * approx_sign + approx_H * ( move - np.log( 1 + approx_abs ) * approx_sign )





### Compute the next potentials in the Newton method iterative scheme
## Inputs :
#      phi_n (array:N) -> current e- quasi-Fermi energy
#      phi_p (array:N) -> current hole quasi-Fermi energy
#      phi (array:N) -> current electrostatic potential
#      dgrid (array:N-1) -> array of distances between consecutive grid points
#      eps(array:N) -> relative dieclectric constant
#      Chi (array:N) -> electron affinity
#      Eg (array:N) -> band gap
#      Nc (array:N) -> e- density of states
#      Nv (array:N) -> hole density of states
#      Ndop (array:N) -> dopant density ( = donor density - acceptor density )
#      Et (array:N) -> trap state energy level (SHR)
#      tn (array:N) -> e- lifetime (SHR)
#      tp (array:N) -> hole lifetime (SHR)
#      mn (array:N) -> e- mobility
#      mp (array:N) -> hole mobility
#      G (array:N) -> electron-hole generation rate density
#      Snl (scalar) -> e- surface recombination velocity at left boundary
#      Spl (scalar) -> hole surface recombination velocity at left boundary
#      Snr (scalar) -> e- surface recombination velocity at right boundary
#      Spr (scalar) -> hole surface recombination velocity at right boundary
#      neq_0 (scalar) -> e- density at left boundary
#      neq_L (scalar) -> e- density at right boundary
#      peq_0 (scalar) -> hole density at left boundary
#      peq_L (scalar) -> hole density at right boundary
## Outputs :
#      1 (array:N) -> next e- quasi-Fermi energy
#      2 (array:N) -> next hole quasi-Fermi energy
#      3 (array:N) -> next electrostatic potential
#      4 (scalar) -> error

@jit
def step( phi_n , phi_p , phi , dgrid , eps , Chi , Eg , Nc , Nv , Ndop , Et , tn , tp , mn , mp , G , Snl , Spl , Snr , Spr , neq_0 , neq_L , peq_0 , peq_L ):

    _F = F( phi_n , phi_p , phi , dgrid , eps , Chi , Eg , Nc , Nv , Ndop , Et , tn , tp , mn , mp , G , Snl , Spl , Snr , Spr , neq_0 , neq_L , peq_0 , peq_L )
    gradF = F_deriv( phi_n , phi_p , phi , dgrid , eps , Chi , Eg , Nc , Nv , Et , tn , tp , mn , mp , G , Snl , Spl , Snr , Spr , neq_0 , neq_L , peq_0 , peq_L )

    move = np.linalg.solve( gradF , - _F )
    error = max( np.abs( move ) )

    damp_move = damp( move )

    N = phi_n.size
    phi_n_new = phi_n + damp_move[0:3*N:3]
    phi_p_new = phi_p + damp_move[1:3*N:3]
    phi_new = phi + damp_move[2:3*N:3]

    return phi_n_new , phi_p_new , phi_new , error





### Solve for the e-/hole quasi-Fermi energies and electrostatic potential using the Newton method
## Inputs :
#      phi_n_ini (array:N) -> initial guess for the e- quasi-Fermi energy
#      phi_p_ini (array:N) -> initial guess for the hole quasi-Fermi energy
#      phi_ini (array:N) -> initial guess for the electrostatic potential (with applied voltage boundary condition)
#      dgrid (array:N-1) -> array of distances between consecutive grid points
#      eps(array:N) -> relative dieclectric constant
#      Chi (array:N) -> electron affinity
#      Eg (array:N) -> band gap
#      Nc (array:N) -> e- density of states
#      Nv (array:N) -> hole density of states
#      Ndop (array:N) -> dopant density ( = donor density - acceptor density )
#      Et (array:N) -> trap state energy level (SHR)
#      tn (array:N) -> e- lifetime (SHR)
#      tp (array:N) -> hole lifetime (SHR)
#      mn (array:N) -> e- mobility
#      mp (array:N) -> hole mobility
#      G (array:N) -> electron-hole generation rate density
#      Snl (scalar) -> e- surface recombination velocity at left boundary
#      Spl (scalar) -> hole surface recombination velocity at left boundary
#      Snr (scalar) -> e- surface recombination velocity at right boundary
#      Spr (scalar) -> hole surface recombination velocity at right boundary
#      neq_0 (scalar) -> e- density at left boundary
#      neq_L (scalar) -> e- density at right boundary
#      peq_0 (scalar) -> hole density at left boundary
#      peq_L (scalar) -> hole density at right boundary
## Outputs :
#      1 (array:N) -> e- quasi-Fermi energy
#      2 (array:N) -> hole quasi-Fermi energy
#      3 (array:N) -> electrostatic potential

def solve( phi_n_ini , phi_p_ini , phi_ini , dgrid , eps , Chi , Eg , Nc , Nv , Ndop , Et , tn , tp , mn , mp , G , Snl , Spl , Snr , Spr , neq_0 , neq_L , peq_0 , peq_L ):
    phi_n = phi_n_ini
    phi_p = phi_p_ini
    phi = phi_ini
    error = 1

    while (error > 1e-6):
        next_phi_n , next_phi_p , next_phi , error_new = step( phi_n , phi_p , phi , dgrid , eps , Chi , Eg , Nc , Nv , Ndop , Et , tn , tp , mn , mp , G , Snl , Spl , Snr , Spr , neq_0 , neq_L , peq_0 , peq_L )
        phi_n = next_phi_n
        phi_p = next_phi_p
        phi = next_phi
        error = error_new

    return phi_n , phi_p , phi
