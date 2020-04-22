import sys
from .scales import *
from .efficiency import *
from .optical import *
from .sun import *
from .initial_guess import *
import matplotlib.pyplot as plt
if USE_JAX:
    from jax.config import config
    config.update("jax_enable_x64", True)
    import jax.numpy as np
    from jax import grad , jit
else:
    import numpy as np


class JAXPV( object ):
    """
    Object associated with a jaxpv simulation.

    Attributes
    ----------
        grid   : numpy array , shape = ( N )
            array of positions of grid points
        eps    : numpy array , shape = ( N )
            relative dieclectric constant
        Chi    : numpy array , shape = ( N )
            electron affinity
        Eg     : numpy array , shape = ( N )
            band gap
        Nc     : numpy array , shape = ( N )
            e- density of states
        Nv     : numpy array , shape = ( N )
            hole density of states
        Ndop   : numpy array , shape = ( N )
            dopant density ( positive for donors , negative for acceptors )
        mn     : numpy array , shape = ( N )
            e- mobility
        mp     : numpy array , shape = ( N )
            hole mobility
        Et     : numpy array , shape = ( N )
            SHR trap state energy level
        tn     : numpy array , shape = ( N )
            SHR e- lifetime
        tp     : numpy array , shape = ( N )
            SHR hole lifetime
        Br     : numpy array , shape = ( N )
            radiative recombination coefficient
        Cn     : numpy array , shape = ( N )
            electron Auger coefficient
        Cp     : numpy array , shape = ( N )
            hole Auger coefficient
        Snl    : float
            e- surface recombination velocity at left boundary
        Spl    : float
            hole surface recombination velocity at left boundary
        Snr    : float
            e- surface recombination velocity at right boundary
        Spr    : float
            hole surface recombination velocity at right boundary
        Lambda : numpy array , shape = ( M )
            array of light wavelengths
        P_in   : numpy array , shape = ( M )
            array of incident power for every wavelength
        A      : numpy array , shape = ( N )
            array of coefficients for direct band gap absorption coefficient model
        G      : numpy array , shape = ( N )
            e-/hole pair generation rate density ( only used if user defined G )
        opt    : string
            describes which type of generation density should be used

    """





    def __init__( self , grid ):
        """
        Initialization method for the jaxpv class.

        Parameters
        ----------
            grid   : numpy array , shape = ( N )
                array of positions of grid points

        """
        scale = scales()

        self.grid = np.float64( 1 / scale['d'] * grid )
        N = grid.size

        self.eps = [ 0.0 for i in range( N ) ]
        self.Chi = [ 0.0 for i in range( N ) ]
        self.Eg = [ 0.0 for i in range( N ) ]
        self.Nc = [ 0.0 for i in range( N ) ]
        self.Nv = [ 0.0 for i in range( N ) ]
        self.mn = [ 0.0 for i in range( N ) ]
        self.mp = [ 0.0 for i in range( N ) ]
        self.Ndop = [ 0.0 for i in range( N ) ]

        self.Et = [ 0.0 for i in range( N ) ]
        self.tn = [ 0.0 for i in range( N ) ]
        self.tp = [ 0.0 for i in range( N ) ]
        self.Br = [ 0.0 for i in range( N ) ]
        self.Cn = [ 0.0 for i in range( N ) ]
        sefl.Cp = [ 0.0 for i in range( N ) ]

        self.Snl = 0.0
        self.Snr = 0.0
        self.Spl = 0.0
        self.Spr = 0.0

        self.Lambda = [ 0.0 ]
        self.P_in = [ 0.0 ]
        self.A = [ 0.0 for i in range( N ) ]
        self.G = [ 0.0 for i in range( N ) ]
        self.opt = 'user'





    def add_material( self , properties , subgrid ):
        """
        Define material properties for a subset of the grid.

        The properties dictionnary need not define all possible material properties.

        Parameters
        ----------
            properties : dictionnary of floats ( 15 keys )
                dictionnary of the properties of the material
                'eps'  -> relative dieclectric constant
                'Chi'  -> electron affinity
                'Eg'   -> band gap
                'Nc'   -> e- density of states
                'Nv'   -> hole density of states
                'Ndop' -> dopant density
                'mn'   -> e- mobility
                'mp'   -> hole mobility
                'Et'   -> trap state energy level (SHR)
                'tn'   -> e- lifetime (SHR)
                'tp'   -> hole lifetime (SHR)
                'Br'   -> radiative recombination coefficient
                'Cn'   -> electron Auger coefficient
                'Cp'   -> hole Auger coefficient
                'A'    -> array of coefficients for direct band gap absorption coefficient model
            subgrid    : numpy array , shape = ( L <= N )
                array of indices of the grid where we define the material

        """
        scale = scales()
        for i in range( len( subgrid ) ):
            if 'eps' in properties:
                self.eps[ subgrid[ i ] ] = np.float64( properties['eps'] )
            if 'Chi' in properties:
                self.Chi[ subgrid[ i ] ] = np.float64( 1 / scale['E'] * properties['Chi'] )
            if 'Eg' in properties:
                self.Eg[ subgrid[ i ] ] = np.float64( 1 / scale['E'] * properties['Eg'] )
            if 'Nc' in properties:
                self.Nc[ subgrid[ i ] ] = np.float64( 1 / scale['n'] * properties['Nc'] )
            if 'Nv' in properties:
                self.Nv[ subgrid[ i ] ] = np.float64( 1 / scale['n'] * properties['Nv'] )
            if 'mn' in properties:
                self.mn[ subgrid[ i ] ] = np.float64( 1 / scale['m'] * properties['mn'] )
            if 'mp' in properties:
                self.mp[ subgrid[ i ] ] = np.float64( 1 / scale['m'] * properties['mp'] )
            if 'Et' in properties:
                self.Et[ subgrid[ i ] ] = np.float64( 1 / scale['E'] * properties['Et'] )
            if 'tn' in properties:
                self.tn[ subgrid[ i ] ] = np.float64( 1 / scale['t'] * properties['tn'] )
            if 'tp' in properties:
                self.tp[ subgrid[ i ] ] = np.float64( 1 / scale['t'] * properties['tp'] )
            if 'Br' in properties:
                self.Br[ subgrid[ i ] ] = np.float64( scale['t'] * scale['n'] * properties['Br'] )
            if 'Cn' in properties:
                self.Cn[ subgrid[ i ] ] = np.float64( scale['t'] * scale['n'] * scale['n'] * properties['Cn'] )
            if 'Cp' in properties:
                self.Cp[ subgrid[ i ] ] = np.float64( scale['t'] * scale['n'] * scale['n'] * properties['Cp'] )
            if 'A' in properties:
                self.A[ subgrid[ i ] ] = np.float64( scale['d'] * properties['A'] )





    def contacts( self , Snl , Snr , Spl , Spr ):
        """
        Define recombination velocities for carriers at the contacts.

        Parameters
        ----------
        Snl    : float
            e- surface recombination velocity at left boundary
        Spl    : float
            hole surface recombination velocity at left boundary
        Snr    : float
            e- surface recombination velocity at right boundary
        Spr    : float
            hole surface recombination velocity at right boundary

        """
        scale = scales()
        self.Snl = np.float64( 1 / scale['v'] * Snl )
        self.Snr = np.float64( 1 / scale['v'] * Snr )
        self.Spl = np.float64( 1 / scale['v'] * Spl )
        self.Spr = np.float64( 1 / scale['v'] * Spr )





    def single_pn_junction( self , Nleft , Nright , junction_position ):
        """
        Define the doping profile as a single pn junction.

        Parameters
        ----------
            Nleft             : float
                dopant density at the left of the junction ( positive for donors , negative for acceptors )
            Nright            : float
                dopant density at the right of the junction ( positive for donors , negative for acceptors )
            junction_position : float
                position of the junction

        """
        scale = scales()
        index = 0
        while ( self.grid[ index ] < 1 / scale['d'] * junction_position ):
            self.Ndop[ index ] = np.float64( 1 / scale['n'] * Nleft )
            index += 1
        for i in range( index , self.grid.size ):
            self.Ndop[ i ] = np.float64( 1 / scale['n'] * Nright )





    def doping_profile( self , doping , subgrid ):
        """
        Define a general doping profile over a subset of the grid.

        Parameters
        ----------
            doping  : numpy array , shape = ( L <= N )
                array dopant density ( positive for donors , negative for acceptors )
            subgrid : numpy array , shape = ( L <= N )
                array of indices of the grid where we define the material

        """
        scale = scales()
        for i in range( len( subgrid ) ):
            self.Ndop[ subgrid[ i ] ] = np.float64( 1 / scale['n'] * doping[ i ] )





    def incident_light( self , type = 'sun' , Lambda = None , P_in = None ):
        """
        Define the incident light on the system.

        The light can be defined either as the sun light, white light, monochromatic or user-derfined.
        In all cases, the total power is normalized to 1 sun = 1000 W/m2.

        sun light     -> Lambda / P_in not required
        white         -> if Lambda not defined, defaults to visible range ; else Lambda is used
        monochromatic -> if Lambda not defined, defaults to blue ; else Lambda is used
        user          -> uses Lambda and P_in

        Parameters
        ----------
        type   : string
            type of incident light : sun , white , monochromatic , user ( default : sun )
        Lambda : numpy array , shape = ( M )
            array of light wavelengths
        P_in   : numpy array , shape = ( M )
            array of incident power for every wavelength

        """
        if ( type == 'sun' ):
            Lambda_sun , P_in_sun = sun()
            self.Lambda = Lambda_sun
            self.P_in = P_in_sun
        else if ( type == 'white' ):
            if ( Lambda is None ):
                self.Lambda = np.linspace( 400.0 , 800.0 , num = 5 )
                self.P_in = np.linspace( 200.0 , 200.0 , num = 5 )
            else:
                self.Lambda = Lambda
                power = 1000.0 / Lambda.size
                self.P_in = np.linspace( power , power , num = Lambda.size )
        else if ( type == 'monochromatic' ):
            if ( Lambda is None ):
                self.Lambda = np.array( [ 400.0 ] )
            else:
                self.Lambda = Lambda
            self.P_in = np.array( [ 1000.0 ] )
        else if ( type == 'user' ):
            if ( ( Lambda is None ) or ( P_in is None ) ):
                print( 'Lambda or Pin not defined' )
                sys.exit()
            else:
                self.Lambda = Lambda
                self.P_in = 1000.0 / np.sum( P_in ) * P_in





    def optical_G( self , type = 'direct' , G = None ):
        """
        Define how the generation rate density is computed.

        Currently, G can either be user defined ( type = user ) or can be computed
        in the case of a direct band gap semi-conductor model from material parameters
        ( type = direct ).

        Parameters
        ----------
        type    : string
            describes which type of generation density should be used
        G      : numpy array , shape = ( N )
            e-/hole pair generation rate density ( only used if user defined G )

        """
        self.opt = type
        if ( type = 'user' ):
            self.G = np.float64( 1 / scale['U'] * G )





    def efficiency( self ):
        """
        Computes the efficiency of the system.

        Returns
        -------
            float
                efficiency of the system

        """
        Vincr = Vincrement( np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) )

        if ( opt is 'user' ):
            G_used = np.array( self.G )
        else:
            G_used = compute_G( np.array( self.Lambda ) , np.array( self.P_in ) , np.array( self.grid[1:] - self.grid[:-1] ) , np.array( self.Eg ) , np.array( self.A ) )

        return efficiency( np.array( self.grid[1:] - self.grid[:-1] ) , Vincr , np.array( self.eps ) , np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) , np.array( self.mn ) , np.array( self.mp ) , np.array( self.Et ) , np.array( self.tn ) , np.array( self.tp ) , np.array( self.Br ) , np.array( self.Cn ) , np.array( self.Cp ) , np.array( self.Snl ) , np.array( self.Spl ) , np.array( self.Snr ) , np.array( self.Spr ) , G_used )





    def grad_efficiency( self , jit = 'true' ):
        """
        Computes the gradient of the efficiency of the system.

        Parameters
        ----------
            jit : boolean
                use jit for the calculation ( defaut : true )

        Returns
        -------
            dictionnary of numpy arrays , shape = ( N ) or ( 1 ) ( 19 keys )
                dictionnary of the derivatives of the efficiency
                'eps'  -> derivative with respect to the relative dieclectric constant
                'Chi'  -> derivative with respect to the electron affinity
                'Eg'   -> derivative with respect to the band gap
                'Nc'   -> derivative with respect to the e- density of states
                'Nv'   -> derivative with respect to the hole density of states
                'Ndop' -> derivative with respect to the dopant density
                'mn'   -> derivative with respect to the e- mobility
                'mp'   -> derivative with respect to the hole mobility
                'Et'   -> derivative with respect to the trap state energy level (SHR)
                'tn'   -> derivative with respect to the e- lifetime (SHR)
                'tp'   -> derivative with respect to the hole lifetime (SHR)
                'Br'   -> derivative with respect to the radiative recombination coefficient
                'Cn'   -> derivative with respect to the electron Auger coefficient
                'Cp'   -> derivative with respect to the hole Auger coefficient
                'Snl'  -> derivative with respect to left-side e- recombination velocity
                'Spl'  -> derivative with respect to left-side hole recombination velocity
                'Snr'  -> derivative with respect to right-side e- recombination velocity
                'Spr'  -> derivative with respect to right-side hole recombination velocity
                'G'    -> derivative with respect to the generation rate density

        """
        scale = scales()
        Vincr = Vincrement( np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) )
        if ( opt is 'user' ):
            G_used = np.array( self.G )
        else:
            G_used = compute_G( np.array( self.Lambda ) , np.array( self.P_in ) , np.array( self.grid[1:] - self.grid[:-1] ) , np.array( self.Eg ) , np.array( self.A ) )

        if jit:
            cur , cur_grad = grad_IV( np.array( self.grid[1:] - self.grid[:-1] ) , Vincr , np.array( self.eps ) , np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) , np.array( self.mn ) , np.array( self.mp ) , np.array( self.Et ) , np.array( self.tn ) , np.array( self.tp ) , np.array( self.Br ) , np.array( self.Cn ) , np.array( self.Cp ) , np.array( self.Snl ) , np.array( self.Spl ) , np.array( self.Snr ) , np.array( self.Spr ) , G_used )
            voltages = np.linspace( start = 0 , stop = len(current) * Vincrement , num = len(current) )
            coef = scale['E'] * scale['J'] * 1e4 / np.sum( P_in )
            P = coef * voltages * current
            Pmax = np.max( P )
            index = np.where( P == Pmax )
            efficiency = Pmax
            result = {}
            result['eps'] = cur_grad['eps'][ index ] * coef * voltages[ index ]
            result['Chi'] = cur_grad['Chi'][ index ] * coef * voltages[ index ]
            result['Eg'] = cur_grad['Eg'][ index ] * coef * voltages[ index ]
            result['Nc'] = cur_grad['Nc'][ index ] * coef * voltages[ index ]
            result['Nv'] = cur_grad['Nv'][ index ] * coef * voltages[ index ]
            result['Ndop'] = cur_grad['Ndop'][ index ] * coef * voltages[ index ]
            result['mn'] = cur_grad['mn'][ index ] * coef * voltages[ index ]
            result['mp'] = cur_grad['mp'][ index ] * coef * voltages[ index ]
            result['Et'] = cur_grad['Et'][ index ] * coef * voltages[ index ]
            result['tn'] = cur_grad['tn'][ index ] * coef * voltages[ index ]
            result['tp'] = cur_grad['tp'][ index ] * coef * voltages[ index ]
            result['Br'] = cur_grad['Br'][ index ] * coef * voltages[ index ]
            result['Cn'] = cur_grad['Cn'][ index ] * coef * voltages[ index ]
            result['Cp'] = cur_grad['Cp'][ index ] * coef * voltages[ index ]
            result['Snl'] = cur_grad['Snl'][ index ] * coef * voltages[ index ]
            result['Spl'] = cur_grad['Spl'][ index ] * coef * voltages[ index ]
            result['Snr'] = cur_grad['Snr'][ index ] * coef * voltages[ index ]
            result['Spr'] = cur_grad['Spr'][ index ] * coef * voltages[ index ]
            result['G'] = cur_grad['G'][ index ] * coef * voltages[ index ]

            print( efficiency )
            print( result )

        else:
            gradeff = grad( efficiency , argnums = ( 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 19 , 20 ) )
            deriv = gradeff( np.array( self.grid[1:] - self.grid[:-1] ) , Vincr , np.array( self.eps ) , np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) , np.array( self.mn ) , np.array( self.mp ) , np.array( self.Et ) , np.array( self.tn ) , np.array( self.tp ) , np.array( self.Br ) , np.array( self.Cn ) , np.array( self.Cp ) , np.array( self.Snl ) , np.array( self.Spl ) , np.array( self.Snr ) , np.array( self.Spr ) , G_used )
            result = {}
            result['eps'] = deriv[0]
            result['Chi'] = deriv[1]
            result['Eg'] = deriv[2]
            result['Nc'] = deriv[3]
            result['Nv'] = deriv[4]
            result['Ndop'] = deriv[5]
            result['mn'] = deriv[6]
            result['mp'] = deriv[7]
            result['Et'] = deriv[9]
            result['tn'] = deriv[10]
            result['tp'] = deriv[11]
            result['Br'] = deriv[12]
            result['Cn'] = deriv[13]
            result['Cp'] = deriv[14]
            result['Snl'] = deriv[15]
            result['Spl'] = deriv[16]
            result['Snr'] = deriv[17]
            result['Spr'] = deriv[18]
            result['G'] = deriv[19]

            print( result )



    def IV_curve( self , title = None ):
        """
        Computes the IV curve for the system.

        Parameters
        ----------
            title    : string
                if defined, the IV plot is saved to this filename

        Returns
        -------
            voltages : numpy array , shape = ( L )
                values of the voltage over which the current is computed
            current  : numpy array , shape = ( L )
                total current

        """
        scale = scales()
        Vincr = Vincrement( np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) )
        if ( opt is 'user' ):
            G_used = np.array( self.G )
        else:
            G_used = compute_G( np.array( self.Lambda ) , np.array( self.P_in ) , np.array( self.grid[1:] - self.grid[:-1] ) , np.array( self.Eg ) , np.array( self.A ) )

        current = scale['J'] * calc_IV( np.array( self.grid[1:] - self.grid[:-1] ) , Vincr , np.array( self.eps ) , np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) , np.array( self.mn ) , np.array( self.mp ) , np.array( self.Et ) , np.array( self.tn ) , np.array( self.tp ) , np.array( self.Br ) , np.array( self.Cn ) , np.array( self.Cp ) , np.array( self.Snl ) , np.array( self.Spl ) , np.array( self.Snr ) , np.array( self.Spr ) , G_used )
        voltages = scale['E'] * np.linspace( start = 0 , stop = len( current ) * Vincr , num = len( current ) )
        fig = plt.figure()
        plt.plot( voltages , current , color='blue' , marker='.' )
        plt.xlabel( 'Voltage (V)' )
        plt.ylabel( 'current (A.cm-2)' )
        plt.show()
        if title is not None:
            plt.savefig( title )
        return voltages , current





    def solve( self , V , equilibrium = False ):
        """
        Solves the solar cell equations and outputs common observables.

        Parameters
        ----------
            V   : float
                value of the voltage ( not used if computing for te equilibrium of the system )
            equilibrium      : boolean
                defines if solving the equilibrium problem ( default : False )

        Returns
        -------
            dictionnary of numpy arrays , shape = ( N ) ( 7 keys )
                dictionnary of the observables outputed
                'phi_n' -> e- quasi-Fermi energy
                'phi_p' -> e- quasi-Fermi energy
                'phi'   -> e- quasi-Fermi energy
                'n'     -> e- density
                'p'     -> hole density
                'Jn'    -> e- current
                'Jp'    -> hole current

        """
        scale = scales()
        Vincr = Vincrement( np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) )
        if ( opt is 'user' ):
            G_used = np.array( self.G )
        else:
            G_used = compute_G( np.array( self.Lambda ) , np.array( self.P_in ) , np.array( self.grid[1:] - self.grid[:-1] ) , np.array( self.Eg ) , np.array( self.A ) )

        N = self.grid.size

        phi_ini = phi_ini = eq_init_phi( np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) )

        phi_eq = solve_eq( np.array( self.grid[1:] - self.grid[:-1] ) , phi_ini , np.array( self.eps ) , np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) )

        result = {}

        if equilibrium:
            result['phi_n'] = np.zeros( N )
            result['phi_p'] = np.zeros( N )
            result['phi'] = scale['E'] * phi_eq
            result['n'] = scale['n'] * n( np.zeros( N ) , phi_eq , np.array( self.Chi ) , np.array( self.Nc ) )
            result['p'] = scale['n'] * p( np.zeros( N ) , phi_eq , np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nv ) )
            result['Jn'] = np.zeros( N - 1 )
            result['Jp'] = np.zeros( N - 1 )
            return result
        else:
            num_steps = math.floor( V / Vincr )

            phis = np.concatenate( ( np.zeros( 2*N ) , phi_eq ) , axis = 0 )
            neq_0 = self.Nc[0] * np.exp( self.Chi[0] + phi_eq[0] )
            neq_L = self.Nc[-1] * np.exp( self.Chi[-1] + phi_eq[-1] )
            peq_0 = self.Nv[0] * np.exp( - self.Chi[0] - self.Eg[0] - phi_eq[0] )
            peq_L = self.Nv[-1] * np.exp( - self.Chi[-1] - self.Eg[-1] - phi_eq[-1] )

            volt = [ i * Vincr for i in range( num_steps ) ]
            volt.append( V )

            for v in volt:
                print( 'V = {0:.7f}   Iteration       |F(x)|                Residual     '.format( scale['E'] * v ) )
                print( '-------------------------------------------------------------------' )
                sol = solve( np.array( self.grid[1:] - self.grid[:-1] ) , neq_0 , neq_L , peq_0 , peq_L , phis , np.array( self.eps ) , np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) , np.array( self.mn ) , np.array( self.mp ) , np.array( self.Et ) , np.array( self.tn ) , np.array( self.tp ) , np.array( self.Br ) , np.array( self.Cn ) , np.array( self.Cp ) , np.array( self.Snl ) , np.array( self.Spl ) , np.array( self.Snr ) , np.array( self.Spr ) , G_used )
                if USE_JAX:
                    phis = ops.index_update( sol , -1 , phi_eq[-1] + v )
                else:
                    sol[-1] = phi_eq[-1] + v
                    phis = sol

            result['phi_n'] = scale['E'] * phis[0:N]
            result['phi_p'] = scale['E'] * phis[N:2*N]
            result['phi'] = scale['E'] * phis[2*N:]
            result['n'] = scale['n'] * n( phis[0:N] , phis[2*N:] , np.array( self.Chi ) , np.array( self.Nc ) )
            result['p'] = scale['n'] * p( phis[N:2*N] , phis[2*N:] , np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nv ) )
            result['Jn'] = scale['J'] * Jn( np.array( self.grid[1:] - self.grid[:-1] ) , phis[0:N] , phis[2*N:] , np.array( self.Chi ) , np.array( self.Nc ) , np.array( self.mn ) )
            result['Jp'] = scale['J'] * Jp( np.array( self.grid[1:] - self.grid[:-1] ) , phis[N:2*N] , phis[2*N:] , np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nv ) , np.array( self.mp ) )
            return result





    def plot_band_diagram( self , result , title = None ):
        """
        Plots the band diagram of the system.

        This function requires as a parameter the output of the solve method.

        Parameters
        ----------
            result : dictionnary of numpy arrays , shape = ( N ) ( 7 keys )
                dictionnary of the observables outputed from the solve method
                'phi_n' -> e- quasi-Fermi energy
                'phi_p' -> e- quasi-Fermi energy
                'phi'   -> e- quasi-Fermi energy
                'n'     -> e- density
                'p'     -> hole density
                'Jn'    -> e- current
                'Jp'    -> hole current
            title  : string
                if defined, the band diagram plot is saved to this filename

        """
        scale = scales()
        Ec = - scale['E'] * np.array( self.Chi ) - result['phi']
        Ev = - scale['E'] * np.array( self.Chi ) - scale['E'] * np.array( self.Eg ) - result['phi']
        fig = plt.figure()
        plt.plot( scale['d'] * self.grid , Ec , color='red' , label = 'conduction band' , linestyle='dashed' )
        plt.plot( scale['d'] * self.grid , Ev , color = 'blue' , label = 'valence band' , linestyle='dashed' )
        plt.plot( scale['d'] * self.grid , result['phi_n'] , color='red' , label = 'e- quasiFermi energy')
        plt.plot( scale['d'] * self.grid , result['phi_p'] , color = 'blue' , label = 'hole quasiFermi energy' )
        plt.xlabel( 'thickness (cm)' )
        plt.ylabel( 'energy (eV)' )
        plt.legend()
        plt.show()
        if title is not None:
            plt.savefig( title )





    def plot_concentration_profile( self , result , title = None ):
        """
        Plots the concentration profile of carriers in the system.

        This function requires as a parameter the output of the solve method.

        Parameters
        ----------
            result : dictionnary of numpy arrays , shape = ( N ) ( 7 keys )
                dictionnary of the observables outputed from the solve method
                'phi_n' -> e- quasi-Fermi energy
                'phi_p' -> e- quasi-Fermi energy
                'phi'   -> e- quasi-Fermi energy
                'n'     -> e- density
                'p'     -> hole density
                'Jn'    -> e- current
                'Jp'    -> hole current
            title  : string
                if defined, the concentration profile plot is saved to this filename

        """
        scale = scales()
        fig = plt.figure()
        plt.yscale('log')
        plt.plot( scale['d'] * self.grid , result['n'] , color='red' , label = 'e-' )
        plt.plot( scale['d'] * self.grid , result['p'] , color='blue' , label = 'hole' )
        plt.xlabel( 'thickness (cm)' )
        plt.ylabel( 'concentration (cm-3)' )
        plt.legend()
        plt.show()
        if title is not None:
            plt.savefig( title )





    def plot_current_profile( self , result , title = None ):
        """
        Plots the current profile in the system.

        This function requires as a parameter the output of the solve method.

        Parameters
        ----------
            result : dictionnary of numpy arrays , shape = ( N ) ( 7 keys )
                dictionnary of the observables outputed from the solve method
                'phi_n' -> e- quasi-Fermi energy
                'phi_p' -> e- quasi-Fermi energy
                'phi'   -> e- quasi-Fermi energy
                'n'     -> e- density
                'p'     -> hole density
                'Jn'    -> e- current
                'Jp'    -> hole current
            title  : string
                if defined, the current profile plot is saved to this filename

        """
        scale = scales()
        fig = plt.figure()
        plt.plot( scale['d'] * 0.5 * ( self.grid[1:] + self.grid[:-1] ) , result['Jn'] , color='red' , label = 'e-' )
        plt.plot( scale['d'] * 0.5 * ( self.grid[1:] + self.grid[:-1] ) , result['Jp'] , color='blue' , label = 'hole' )
        plt.plot( scale['d'] * 0.5 * ( self.grid[1:] + self.grid[:-1] ) , result['Jn'] + result['Jp'] , color='green' , label = 'total' , linestyle='dashed' )
        plt.xlabel( 'thickness (cm)' )
        plt.ylabel( 'current (A.cm-2)' )
        plt.legend()
        plt.show()
        if title is not None:
            plt.savefig( title )
