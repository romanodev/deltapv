from .IV import *
from .scales import *

def Vincrement( Chi , Eg , Nc , Nv , Ndop , num_vals = 50 ):
    """
    Compute the increment of voltages when compute I-V curve.

    It si defined as the equilibrium potential difference divided by a fixed number of point calculations.

    Parameters
    ----------
        Chi      : numpy array , shape = ( N )
            electron affinity
        Eg       : numpy array , shape = ( N )
            band gap
        Nc       : numpy array , shape = ( N )
            e- density of states
        Nv       : numpy array , shape = ( N )
            hole density of states
        Ndop     : numpy array , shape = ( N )
            dopant density ( positive for donors , negative for acceptors )
        num_vals : float ( default = 50 )
            number of subivisions for the voltage ( Vincr * num_vals = equilibrium potential difference )

    Returns
    -------
        float
            voltage increment

    """
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





def efficiency( dgrid , Vincrement , eps , Chi , Eg , Nc , Nv , Ndop , mn , mp , Et , tn , tp , Br , Cn , Cp , Snl , Spl , Snr , Spr , G_used ):
    """
    Compute the photovoltaic efficiency of the system.

    Parameters
    ----------
        dgrid      : numpy array , shape = ( N - 1 )
            array of distances between consecutive grid points
        Vincrement : float
            increment voltage for I-V curve
        eps        : numpy array , shape = ( N )
            relative dieclectric constant
        Chi        : numpy array , shape = ( N )
            electron affinity
        Eg         : numpy array , shape = ( N )
            band gap
        Nc         : numpy array , shape = ( N )
            e- density of states
        Nv         : numpy array , shape = ( N )
            hole density of states
        Ndop       : numpy array , shape = ( N )
            dopant density ( positive for donors , negative for acceptors )
        mn         : numpy array , shape = ( N )
            e- mobility
        mp         : numpy array , shape = ( N )
            hole mobility
        Et         : numpy array , shape = ( N )
            SHR trap state energy level
        tn         : numpy array , shape = ( N )
            SHR e- lifetime
        tp         : numpy array , shape = ( N )
            SHR hole lifetime
        Br         : numpy array , shape = ( N )
            radiative recombination coefficient
        Cn         : numpy array , shape = ( N )
            electron Auger coefficient
        Cp         : numpy array , shape = ( N )
            hole Auger coefficient
        Snl        : float
            e- surface recombination velocity at left boundary
        Spl        : float
            hole surface recombination velocity at left boundary
        Snr        : float
            e- surface recombination velocity at right boundary
        Spr        : float
            hole surface recombination velocity at right boundary
        G_used     : numpy array , shape = ( N )
            e-/hole pair generation rate density ( computed or user defined )

    Returns
    -------
        float
            efficiency

    """
    scale = scales()
    current = calc_IV( dgrid , Vincrement , eps , Chi , Eg , Nc , Nv , Ndop , mn , mp , Et , tn , tp , Br , Cn , Cp , Snl , Spl , Snr , Spr , G_used )
    voltages = np.linspace( start = 0 , stop = len(current) * Vincrement , num = len(current) )
    Pmax = np.max( scale['E'] * voltages * scale['J'] * current ) * 1e4 # W/m2
    return Pmax / np.sum( P_in )
