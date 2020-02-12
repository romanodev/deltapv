from .IV import *
from .scales import *

### Compute the increment of voltages when compute I-V curve. It si defined as
### the equilibrium potential difference divided by a fixed number of point calculations
### (less calculations could be done if current changes sign)
## Inputs :
#      Chi (array:N) -> electron affinity
#      Eg (array:N) -> band gap
#      Nc (array:N) -> e- density of states
#      Nv (array:N) -> hole density of states
#      Ndop (array:N) -> dopant density ( = donor density - acceptor density )
#      num_vals (scalar ; optional) -> number of subivisions for the voltage
#                                      ( Vincr * num_vals = equilibrium potential difference )
## Outputs :
#      1 (scalar) -> voltage increment

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





### Compute the photovoltaic efficiency of the system (with input P_in)
## Inputs :
#      Vincrement (scalar) -> increment voltage for I-V curve
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
## Outputs :
#      1 (scalar) -> efficiency

def efficiency( Vincrement , P_in , dgrid , eps , Chi , Eg , Nc , Nv , Ndop , Et , tn , tp , mn , mp , G , Snl , Spl , Snr , Spr ):
    scale = scales()
    current = calc_IV( Vincrement , dgrid , eps , Chi , Eg , Nc , Nv , Ndop , Et , tn , tp , mn , mp , G , Snl , Spl , Snr , Spr )
    voltages = np.linspace( start = 0 , stop = len(current) * Vincrement , num = len(current) )
    Pmax = np.max( scale['E'] * voltages * scale['J'] * current )
    area_cell = ( scale['d'] * np.sum( dgrid ) * 1e-2 ) * 1 # m2
    Pin = P_in * area_cell
    return Pmax / Pin
