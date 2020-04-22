from .physics import *

def eq_init_phi( Chi , Eg , Nc , Nv , Ndop ):
    """
    Returns the initial guess for the electrostatic potential at equilibrium.

    Parameters
    ----------
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

    Returns
    -------
        numpy array , shape = ( N )
            initial guess for the electrostatic potential at equilibrium

    """
    if ( Ndop[0] > 0 ):
        phi_ini_left = - Chi[0] + np.log( Ndop[0] / Nc[0] )
    else:
        phi_ini_left = - Chi[0] - Eg[0] - np.log( - Ndop[0] / Nv[0] )
    if ( Ndop[-1] > 0 ):
        phi_ini_right = - Chi[-1] + np.log( Ndop[-1] / Nc[-1] )
    else:
        phi_ini_right = - Chi[-1] - Eg[-1] - np.log( - Ndop[-1] / Nv[-1] )
    return np.linspace( phi_ini_left , phi_ini_right , Chi.size )





def eq_init_phi_deriv( Chi , Eg , Nc , Nv , Ndop ):
    """
    Returns the derivatives of the initial guess for the electrostatic potential at equilibrium.

    Parameters
    ----------
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

    Returns
    -------
        dphi_ini_dChi0  : numpy array , shape = ( N )
            derivative of the initial guess for the electrostatic potential with respect to Chi[0]
        dphi_ini_dEg0   : numpy array , shape = ( N )
            derivative of the initial guess for the electrostatic potential with respect to Eg[0]
        dphi_ini_dNc0   : numpy array , shape = ( N )
            derivative of the initial guess for the electrostatic potential with respect to Nc[0]
        dphi_ini_dNv0   : numpy array , shape = ( N )
            derivative of the initial guess for the electrostatic potential with respect to Nv[0]
        dphi_ini_dNdop0 : numpy array , shape = ( N )
            derivative of the initial guess for the electrostatic potential with respect to Ndop[0]
        dphi_ini_dChiL  : numpy array , shape = ( N )
            derivative of the initial guess for the electrostatic potential with respect to Chi[N-1]
        dphi_ini_dEgL   : numpy array , shape = ( N )
            derivative of the initial guess for the electrostatic potential with respect to Eg[N-1]
        dphi_ini_dNcL   : numpy array , shape = ( N )
            derivative of the initial guess for the electrostatic potential with respect to Nc[N-1]
        dphi_ini_dNvL   : numpy array , shape = ( N )
            derivative of the initial guess for the electrostatic potential with respect to Nv[N-1]
        dphi_ini_dNdopL : numpy array , shape = ( N )
            derivative of the initial guess for the electrostatic potential with respect to Ndop[N-1]

    """
    if ( Ndop[0] > 0 ):
        dphi_ini_left_dChi0 = - 1
        dphi_ini_left_dEg0 = 0
        dphi_ini_left_dNdop0 = 1 / Ndop[0]
        dphi_ini_left_dNc0 = - 1 / Nc[0]
        dphi_ini_left_dNv0 = 0
    else:
        dphi_ini_left_dChi0 = - 1
        dphi_ini_left_dEg0 = - 1
        dphi_ini_left_dNdop0 = - 1 / Ndop[0]
        dphi_ini_left_dNc0 = 0
        dphi_ini_left_dNv0 = 1 / Nv[0]
    if ( Ndop[-1] > 0 ):
        dphi_ini_right_dChiL = - 1
        dphi_ini_right_dEgL = 0
        dphi_ini_right_dNdopL = 1 / Ndop[-1]
        dphi_ini_right_dNcL = - 1 / Nc[-1]
        dphi_ini_right_dNvL = 0
    else:
        dphi_ini_right_dChiL = - 1
        dphi_ini_right_dEgL = - 1
        dphi_ini_right_dNdopL = - 1 / Ndop[-1]
        dphi_ini_right_dNcL = 0
        dphi_ini_right_dNvL = - 1 / Nv[-1]

    N = Chi.size
    dphi_ini_dChi0 = np.linspace( dphi_ini_left_dChi0 , 0 , N )
    dphi_ini_dEg0 = np.linspace( dphi_ini_left_dEg0 , 0 , N )
    dphi_ini_dNc0 = np.linspace( dphi_ini_left_dNc0 , 0 , N )
    dphi_ini_dNv0 = np.linspace( dphi_ini_left_dNv0 , 0 , N )
    dphi_ini_dNdop0 = np.linspace( dphi_ini_left_dNdop0 , 0 , N )
    dphi_ini_dChiL = np.linspace( 0 , dphi_ini_right_dChiL , N )
    dphi_ini_dEgL = np.linspace( 0 , dphi_ini_right_dEgL , N )
    dphi_ini_dNcL = np.linspace( 0 , dphi_ini_right_dNcL , N )
    dphi_ini_dNvL = np.linspace( 0 , dphi_ini_right_dNvL , N )
    dphi_ini_dNdopL = np.linspace( 0 , dphi_ini_right_dNdopL , N )

    return dphi_ini_dChi0 , dphi_ini_dEg0 , dphi_ini_dNc0 , dphi_ini_dNv0 , dphi_ini_dNdop0 , dphi_ini_dChiL , dphi_ini_dEgL , dphi_ini_dNcL , dphi_ini_dNvL , dphi_ini_dNdopL
