from .IV import *
from .scales import *


def Vincrement( Chi , Eg , Nc , Nv , Ndop , num_vals = 50 ):
    phi_ini_left = - Chi[0] - Eg[0] - np.log( np.abs( Ndop[0] ) / Nv[0] )
    if ( Ndop[0] > 0 ):
        phi_ini_left = - Chi[0] + np.log( ( Ndop[0] ) / Nc[0] )
    phi_ini_right = - Chi[-1] - Eg[-1] - np.log( np.abs( Ndop[-1] ) / Nv[-1] )
    if ( Ndop[-1] > 0 ):
        phi_ini_right = - Chi[-1] + np.log( ( Ndop[-1] ) / Nc[-1] )
    incr_sign = 1
    incr_step = np.abs( phi_ini_right - phi_ini_left ) / num_vals
    if ( phi_ini_right > phi_ini_left ):
        incr_sign = -1
    return incr_sign * incr_step

def efficiency( Vincrement , P_in , dgrid , eps , Chi , Eg , Nc , Nv , Ndop , Et , tn , tp , mn , mp , G , Snl , Spl , Snr , Spr ):
    scale = scales()
    current = calc_IV( Vincrement , dgrid , eps , Chi , Eg , Nc , Nv , Ndop , Et , tn , tp , mn , mp , G , Snl , Spl , Snr , Spr )
    voltages = np.linspace( start = 0 , stop = len(current) * Vincrement , num = len(current) )
    Pmax = np.max( scale['E'] * voltages * scale['J'] * current )
    area_cell = ( scale['d'] * np.sum( dgrid ) * 1e-2 ) * 1 # m2
    Pin = P_in * area_cell
    return Pmax / Pin
