from .scales import *
from .efficiency import *
import matplotlib.pyplot as plt
if USE_JAX:
    from jax.config import config
    config.update("jax_enable_x64", True)
    import jax.numpy as np
    from jax import grad , jit
else:
    import numpy as np

### Class for defining the system and computing photovoltaic properties

class JAXPV( object ):
    """docstring for JAXPV."""

    ### Initialization defines the grid
    ## Inputs :
    #      grid (array:N) -> grid points

    def __init__( self , grid , P_in = 1.0 ):
        scale = scales()
        self.grid = np.float64( 1 / scale['d'] * grid )
        self.P_in = np.float64( P_in )
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
        self.Snl = 0.0
        self.Snr = 0.0
        self.Spl = 0.0
        self.Spr = 0.0
        self.G = [ 0.0 for i in range( N ) ]

    ### Initialization defines the grid
    ## Inputs :
    #      properties (dictionnary:P<10keys) -> material properties (scalars)
    #            'eps' -> relative dieclectric constant
    #            'Chi' -> electron affinity
    #            'Eg' -> band gap
    #            'Nc'  -> e- density of states
    #            'Nv' -> hole density of states
    #            'Et' -> trap state energy level (SHR)
    #            'tn' -> e- lifetime (SHR)
    #            'tp' -> hole lifetime (SHR)
    #            'mn' -> e- mobility
    #            'mp' -> hole mobility
    #      subgrid (array:M<N) -> list of grid point indices for which the material is defined

    def add_material( self , properties , subgrid ):
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

    ### Initialization of recombination velocities at contact
    ## Inputs :
    #      Snl (scalar) -> e- surface recombination velocity at left boundary
    #      Spl (scalar) -> hole surface recombination velocity at left boundary
    #      Snr (scalar) -> e- surface recombination velocity at right boundary
    #      Spr (scalar) -> hole surface recombination velocity at right boundary

    def contacts( self , Snl , Snr , Spl , Spr ):
        scale = scales()
        self.Snl = np.float64( 1 / scale['v'] * Snl )
        self.Snr = np.float64( 1 / scale['v'] * Snr )
        self.Spl = np.float64( 1 / scale['v'] * Spl )
        self.Spr = np.float64( 1 / scale['v'] * Spr )

    ### Initialization of generation rate density
    ## Inputs :
    #      G (array:M<N) -> electron-hole generation rate density
    #      subgrid (array:M<N) -> list of grid point indices for which G is defined

    def generation_rate( self , G , subgrid ):
        scale = scales()
        for i in range( len( subgrid ) ):
            self.G[ subgrid[ i ] ] = np.float64( 1 / scale['U'] * G[ i ] )

    ### Initialization of doping profile in the case of a single p-n junction
    ## Inputs :
    #      Nleft (scalar) -> left side doping density
    #      Nright (scalar) -> right side doping density
    #      junction_position (scalar) -> junction boundary position (cm)

    def single_pn_junction( self , Nleft , Nright , junction_position ):
        scale = scales()
        index = 0
        while ( self.grid[ index ] < 1 / scale['d'] * junction_position ):
            self.Ndop[ index ] = np.float64( 1 / scale['n'] * Nleft )
            index += 1
        for i in range( index , self.grid.size ):
            self.Ndop[ i ] = np.float64( 1 / scale['n'] * Nright )

    ### Initialization of doping profile
    ## Inputs :
    #      doping (array:M<N) -> doping density = donor density - acceptor density
    #                            (i.e. Ndop > 0 for n-type doping and < 0 for p-type doping)
    #      subgrid (array:M<N) -> list of grid point indices for which Ndop is defined

    def doping_profile( self , doping , subgrid ):
        scale = scales()
        for i in range( len( subgrid ) ):
            self.Ndop[ subgrid[ i ] ] = np.float64( 1 / scale['n'] * doping[ i ] )

    ### Compute efficiency
    ## Outputs :
    #      1 (scalar) -> efficiency for the system

    def efficiency( self ):
        Vincr = Vincrement( np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) )
        return efficiency( Vincr , self.P_in , np.array( self.grid[1:] - self.grid[:-1] ) , np.array( self.eps ) , np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) , np.array( self.Et ) , np.array( self.tn ) , np.array( self.tp ) , np.array( self.mn ) , np.array( self.mp ) , np.array( self.G ) , np.array( self.Snl ) , np.array( self.Spl ) , np.array( self.Snr ) , np.array( self.Spr ) )

    ### Compute derivatives of efficiency w.r.t. material properties
    ## Outputs :
    #      1 (dictionnary:) -> efficiency for the system
    #            'eps' -> derivative w.r.t. relative dieclectric constant
    #            'Chi' -> derivative w.r.t. electron affinity
    #            'Eg' -> derivative w.r.t. band gap
    #            'Nc'  -> derivative w.r.t. e- density of states
    #            'Nv' -> derivative w.r.t. hole density of states
    #            'Ndop' -> derivative w.r.t. doping density
    #            'Et' -> derivative w.r.t. trap state energy level (SHR)
    #            'tn' -> derivative w.r.t. e- lifetime (SHR)
    #            'tp' -> derivative w.r.t. hole lifetime (SHR)
    #            'mn' -> derivative w.r.t. e- mobility
    #            'mp' -> derivative w.r.t. hole mobility
    #            'G' -> derivative w.r.t. generation rate density
    #            'Snl' -> derivative w.r.t. e- surface recombination velocity at left boundary
    #            'Spl' -> derivative w.r.t. hole surface recombination velocity at left boundary
    #            'Snr' -> derivative w.r.t. e- surface recombination velocity at right boundary
    #            'Spr' -> derivative w.r.t. hole surface recombination velocity at right boundary

    def grad_efficiency( self ):
        Vincr = Vincrement( np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) )
        if USE_JAX:
            gradeff = grad( efficiency , argnums = ( 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 ) )
            deriv = gradeff( Vincr , self.P_in , np.array( self.grid[1:] - self.grid[:-1] ) , np.array( self.eps ) , np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) , np.array( self.Et ) , np.array( self.tn ) , np.array( self.tp ) , np.array( self.mn ) , np.array( self.mp ) , np.array( self.G ) , np.array( self.Snl ) , np.array( self.Spl ) , np.array( self.Snr ) , np.array( self.Spr ) )
            result = {}
            result['eps'] = deriv[0]
            result['Chi'] = deriv[1]
            result['Eg'] = deriv[2]
            result['Nc'] = deriv[3]
            result['Nv'] = deriv[4]
            result['Ndop'] = deriv[5]
            result['Et'] = deriv[6]
            result['tn'] = deriv[7]
            result['tp'] = deriv[8]
            result['mn'] = deriv[9]
            result['mp'] = deriv[10]
            result['G'] = deriv[11]
            result['Snl'] = deriv[12]
            result['Spl'] = deriv[13]
            result['Snr'] = deriv[14]
            result['Spr'] = deriv[15]
            return result
        else:
            return "Error: JAX not loaded"

    ### Compute the I-V curve for the system and plots it (current are current densities)
    ## Inputs :
    #      title (string ; optional) -> if defined, saves the plot to the file named 'title'
    ## Outputs :
    #      1 (array:M) -> voltage array
    #      2 (array:M) -> (total) current array

    def IV_curve( self , title = None ):
        scale = scales()
        Vincr = Vincrement( np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) )
        current = scale['J'] * calc_IV( Vincr , np.array( self.grid[1:] - self.grid[:-1] ) , np.array( self.eps ) , np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) , np.array( self.Et ) , np.array( self.tn ) , np.array( self.tp ) , np.array( self.mn ) , np.array( self.mp ) , np.array( self.G ) , np.array( self.Snl ) , np.array( self.Spl ) , np.array( self.Snr ) , np.array( self.Spr ) )
        voltages = scale['E'] * np.linspace( start = 0 , stop = len( current ) * Vincr , num = len( current ) )
        fig = plt.figure()
        plt.plot( voltages , current , color='blue' , marker='.' )
        plt.xlabel( 'Voltage (V)' )
        plt.ylabel( 'current (A.cm-2)' )
        plt.show()
        if title is not None:
            plt.savefig( title )
        return voltages , current

    ### Solve for the potentials for the system at a given voltage or at equilibrium
    ## Inputs :
    #      V (scalar) -> Voltage ; not used if equilibrium is selected
    #      equilibrium (boolean) -> if defined as True, computes equilibrium
    ## Outputs :
    #      1 (dictionnary:7keys) -> outputs for system state
    #            'phi_n' -> e- quai-Fermi energy (array:N)
    #            'phi_p' -> ehole quai-Fermi energy (array:N)
    #            'phi' -> electrostatic potential (array:N)
    #            'n' -> e- density (array:N)
    #            'p' -> hole density (array:N)
    #            'Jn' -> e- current (array:N)
    #            'Jp' -> hole current (array:N)

    def solve( self , V , equilibrium = False ):
        scale = scales()

        N = self.grid.size

        if ( self.Ndop[0] > 0 ):
            phi_ini_left = - self.Chi[0] + np.log( ( self.Ndop[0] ) / self.Nc[0] )
        else:
            phi_ini_left = - self.Chi[0] - self.Eg[0] - np.log( - self.Ndop[0] / self.Nv[0] )
        if ( self.Ndop[-1] > 0 ):
            phi_ini_right = - self.Chi[-1] + np.log( ( self.Ndop[-1] ) / self.Nc[-1] )
        else:
            phi_ini_right = - self.Chi[-1] - self.Eg[-1] - np.log( - self.Ndop[-1] / self.Nv[-1] )
        phi_ini = np.linspace( phi_ini_left , phi_ini_right , N )

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
            Vincr = Vincrement( np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) )
            num_steps = math.floor( V / Vincr )

            phis = np.concatenate( ( np.zeros( 2*N ) , phi_eq ) , axis = 0 )
            neq_0 = self.Nc[0] * np.exp( self.Chi[0] + phi_eq[0] )
            neq_L = self.Nc[-1] * np.exp( self.Chi[-1] + phi_eq[-1] )
            peq_0 = self.Nv[0] * np.exp( - self.Chi[0] - self.Eg[0] - phi_eq[0] )
            peq_L = self.Nv[-1] * np.exp( - self.Chi[-1] - self.Eg[-1] - phi_eq[-1] )

            volt = [ i * Vincr for i in range( num_steps ) ]
            volt.append( V )

            for v in volt:
                sol = solve( np.array( self.grid[1:] - self.grid[:-1] ) , phis , np.array( self.eps ) , np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) , np.array( self.Et ) , np.array( self.tn ) , np.array( self.tp ) , np.array( self.mn ) , np.array( self.mp ) , np.array( self.G ) , np.array( self.Snl ) , np.array( self.Spl ) , np.array( self.Snr ) , np.array( self.Spr ) , neq_0 , neq_L , peq_0 , peq_L )
                if USE_JAX:
                    phis = ops.index_update( sol , -1 , phi_eq[-1] + v )
                else:
                    sol[-1] = phi_eq[-1] + v
                    phis = sol

            result['phi_n'] = scale['E'] * phis[0:N]
            result['phi_p'] = scale['E'] * phis[N:2*N]
            result['phi'] = scale['E'] * phis[2*N:]
            result['n'] = scale['n'] * n( phis[0:N] , phis[2*N:] , np.array( self.Chi ) , np.array( self.Nc ) )
            result['p'] = scale['n'] * p( phis[N:2*N] , phis[2*N:] , np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nv )  )
            result['Jn'] = scale['J'] * Jn( phis[0:N] , phis[2*N:] , self.grid[1:] - self.grid[:-1] , np.array( self.Chi ) , np.array( self.Nc ) , np.array( self.mn )  )
            result['Jp'] = scale['J'] * Jp( phis[N:2*N] , phis[2*N:] , self.grid[1:] - self.grid[:-1] , np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nv )  , np.array( self.mp )  )
            return result

    ### Plot band diagram from previous calculation of system state (equilibrium or at given voltage)
    ## Inputs :
    #      result (dictionnary:7keys) -> output dictionnary from member function solve
    #      title (string ; optional) -> if defined, saves the plot to the file named 'title'

    def plot_band_diagram( self , result , title = None ):
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

    ### Plot concentration profile from previous calculation of system state (equilibrium or at given voltage)
    ## Inputs :
    #      result (dictionnary:7keys) -> output dictionnary from member function solve
    #      title (string ; optional) -> if defined, saves the plot to the file named 'title'

    def plot_concentration_profile( self , result , title = None ):
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

    ### Plot current profile from previous calculation of system state (equilibrium or at given voltage)
    ## Inputs :
    #      result (dictionnary:7keys) -> output dictionnary from member function solve
    #      title (string ; optional) -> if defined, saves the plot to the file named 'title'

    def plot_current_profile( self , result , title = None ):
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
