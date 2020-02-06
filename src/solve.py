from .F import *

def damp( move ):
    approx_sign = np.tanh( move )
    approx_abs = approx_sign * move
    approx_H = 1 - ( 1 + np.exp( - 500 * ( move**2 - 1 ) ) )**(-1)
    return np.log( 1 + approx_abs ) * approx_sign + approx_H * ( move - np.log( 1 + approx_abs ) * approx_sign )


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
