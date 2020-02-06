from .scales import *
from .efficiency import *
import matplotlib.pyplot as plt
if USE_JAX:
    import jax.numpy as np
    from jax import grad
else:
    import numpy as np

class JAXPV( object ):
    """docstring for JAXPV."""

    def __init__( self , grid , P_in=1 ):
        scale = scales()
        self.grid = 1 / scale['d'] * grid
        self.P_in = P_in
        N = grid.size
        self.eps = [ 0 for i in range( N ) ]
        self.Chi = [ 0 for i in range( N ) ]
        self.Eg = [ 0 for i in range( N ) ]
        self.Nc = [ 0 for i in range( N ) ]
        self.Nv = [ 0 for i in range( N ) ]
        self.mn = [ 0 for i in range( N ) ]
        self.mp = [ 0 for i in range( N ) ]
        self.Ndop = [ 0 for i in range( N ) ]
        self.Et = [ 0 for i in range( N ) ]
        self.tn = [ 0 for i in range( N ) ]
        self.tp = [ 0 for i in range( N ) ]
        self.Snl = 0.0
        self.Snr = 0.0
        self.Spl = 0.0
        self.Spr = 0.0
        self.G = [ 0 for i in range( N ) ]

    def add_material( self , properties , subgrid ):
        scale = scales()
        for i in range( len( subgrid ) ):
            if 'eps' in properties:
                self.eps[ subgrid[ i ] ] = properties['eps']
            if 'Chi' in properties:
                self.Chi[ subgrid[ i ] ] = 1 / scale['E'] * properties['Chi']
            if 'Eg' in properties:
                self.Eg[ subgrid[ i ] ] = 1 / scale['E'] * properties['Eg']
            if 'Nc' in properties:
                self.Nc[ subgrid[ i ] ] = 1 / scale['n'] * properties['Nc']
            if 'Nv' in properties:
                self.Nv[ subgrid[ i ] ] = 1 / scale['n'] * properties['Nv']
            if 'mn' in properties:
                self.mn[ subgrid[ i ] ] = properties['mn']
            if 'mp' in properties:
                self.mp[ subgrid[ i ] ] = properties['mp']
            if 'Et' in properties:
                self.Et[ subgrid[ i ] ] = 1 / scale['E'] * properties['Et']
            if 'tn' in properties:
                self.tn[ subgrid[ i ] ] = 1 / scale['t'] * properties['tn']
            if 'tp' in properties:
                self.tp[ subgrid[ i ] ] = 1 / scale['t'] * properties['tp']

    def contacts( self , Snl , Snr , Spl , Spr ):
        scale = scales()
        self.Snl = 1 / scale['v'] * Snl
        self.Snr = 1 / scale['v'] * Snr
        self.Spl = 1 / scale['v'] * Spl
        self.Spr = 1 / scale['v'] * Spr

    def generation_rate( self , G , subgrid ):
        scale = scales()
        for i in range( len( subgrid ) ):
            self.G[ subgrid[ i ] ] = 1 / scale['U'] * G[ i ]

    def single_pn_junction( self , Nleft , Nright , junction_position ):
        scale = scales()
        index = 0
        while ( self.grid[ index ] < 1 / scale['d'] * junction_position ):
            self.Ndop[ index ] = 1 / scale['n'] * Nleft
            index += 1
        for i in range( index , self.grid.size ):
            self.Ndop[ i ] = 1 / scale['n'] * Nright

    def doping_profile( self , doping , subgrid ):
        scale = scales()
        for i in range( len( subgrid ) ):
            self.Ndop[ subgrid[ i ] ] = 1 / scale['n'] * doping[ i ]

    def efficiency( self ):
        Vincr = Vincrement( np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) )
        return efficiency( Vincr , self.P_in , self.grid[1:] - self.grid[:-1] , np.array( self.eps ) , np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) , np.array( self.Et ) , np.array( self.tn ) , np.array( self.tp ) , np.array( self.mn ) , np.array( self.mp ) , np.array( self.G ) , np.array( self.Snl ) , np.array( self.Spl ) , np.array( self.Snr ) , np.array( self.Spr ) )

    def grad_efficiency( self ):
        Vincr = Vincrement( np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) )
        if USE_JAX:
            gradeff = grad( efficiency , argnums = ( 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 ) )
            return gradeff( Vincr , self.P_in , self.grid[1:] - self.grid[:-1] , np.array( self.eps ) , np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) , np.array( self.Et ) , np.array( self.tn ) , np.array( self.tp ) , np.array( self.mn ) , np.array( self.mp ) , np.array( self.G ) , np.array( self.Snl ) , np.array( self.Spl ) , np.array( self.Snr ) , np.array( self.Spr ) )
        else:
            return "Error: JAX not loaded"

    def IV_curve( self , title = 'IV.pdf' ):
        scale = scales()
        Vincr = Vincrement( np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) )
        current = calc_IV( Vincr , self.grid[1:] - self.grid[:-1] , np.array( self.eps ) , np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) , np.array( self.Et ) , np.array( self.tn ) , np.array( self.tp ) , np.array( self.mn ) , np.array( self.mp ) , np.array( self.G ) , np.array( self.Snl ) , np.array( self.Spl ) , np.array( self.Snr ) , np.array( self.Spr ) )
        voltages = np.linspace( start = 0 , stop = len(current) * Vincr , num = len(current) )
        fig = plt.figure()
        plt.plot( scale['E'] * voltages , scale['J'] * current , color='blue' , marker='.' )
        plt.xlabel( 'Voltage (V)' )
        plt.ylabel( 'current (A.cm-2)' )
        plt.show()
        quit()
        plt.savefig(title)

    def solve( self , V , equilibrium = False ):
        scale = scales()
        phi_eq = solve_eq( self.grid[1:] - self.grid[:-1] , np.array( self.eps ) , np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) )

        result = {}

        if equilibrium:
            result['phi_n'] = np.zeros( self.grid.size )
            result['phi_p'] = np.zeros( self.grid.size )
            result['phi'] = scale['E'] * phi_eq
            result['n'] = scale['n'] * n( np.zeros( self.grid.size ) , phi_eq , np.array( self.Chi ) , np.array( self.Nc ) )
            result['p'] = scale['n'] * p( np.zeros( self.grid.size ) , phi_eq , np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nv ) )
            result['Jn'] = np.zeros( self.grid.size )
            result['Jp'] = np.zeros( self.grid.size )
            return result
        else:
            Vincr = Vincrement( np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) )
            num_steps = math.floor( V / Vincr )

            phi_n = [ np.zeros( self.grid.size ) ]
            phi_p = [ np.zeros( self.grid.size ) ]
            phi = [ phi_eq ]
            neq = n( phi_n[-1] , phi[-1] , np.array( self.Chi ) , np.array( self.Nc ) )
            peq = p( phi_p[-1] , phi[-1] , np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nv ) )

            volt = [ i * Vincr for i in range( num_steps ) ]
            volt.append( V )

            for v in volt:
                new_phi_n , new_phi_p , new_phi = solve( phi_n[-1] , phi_p[-1] , phi[-1] , self.grid[1:] - self.grid[:-1] , np.array( self.eps ) , np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nc ) , np.array( self.Nv ) , np.array( self.Ndop ) , np.array( self.Et ) , np.array( self.tn ) , np.array( self.tp ) , np.array( self.mn ) , np.array( self.mp ) , np.array( self.G ) , np.array( self.Snl ) , np.array( self.Spl ) , np.array( self.Snr ) , np.array( self.Spr ) , neq[0] , neq[-1] , peq[0] , peq[-1] )
                phi_n.append(new_phi_n)
                phi_p.append(new_phi_p)
                if USE_JAX:
                    phi.append( jax.ops.index_update( new_phi , -1 , phi_eq[-1] + v ) )
                else:
                    new_phi[-1] = phi_eq[-1] + v
                    phi.append( new_phi )

            result['phi_n'] = scale['E'] * phi_n[-1]
            result['phi_p'] = scale['E'] * phi_p[-1]
            result['phi'] = scale['E'] * phi[-1]
            result['n'] = scale['n'] * n( phi_n[-1] , phi[-1] , np.array( self.Chi ) , np.array( self.Nc ) )
            result['p'] = scale['n'] * p( phi_p[-1] , phi[-1] , np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nv )  )
            result['Jn'] = scale['J'] * Jn( phi_n[-1] , phi[-1] , self.grid[1:] - self.grid[:-1] , np.array( self.Chi ) , np.array( self.Nc ) , np.array( self.mn )  )
            result['Jp'] = scale['J'] * Jp( phi_p[-1] , phi[-1] , self.grid[1:] - self.grid[:-1] , np.array( self.Chi ) , np.array( self.Eg ) , np.array( self.Nv )  , np.array( self.mp )  )
            return result

    def plot_band_diagram( self , result , title='band_diagram.pdf' ):
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
        quit()
        plt.savefig(title)

    def plot_concentration_profile( self , result , title='concentration_profile.pdf' ):
        scale = scales()
        fig = plt.figure()
        plt.yscale('log')
        plt.plot( scale['d'] * self.grid , result['n'] , color='red' , label = 'e-' )
        plt.plot( scale['d'] * self.grid , result['p'] , color='blue' , label = 'hole' )
        plt.xlabel( 'thickness (cm)' )
        plt.ylabel( 'concentration (cm-3)' )
        plt.legend()
        plt.show()
        quit()
        plt.savefig(title)

    def plot_current_profile( self , result , title='current_profile.pdf' ):
        scale = scales()
        fig = plt.figure()
        plt.plot( scale['d'] * 0.5 * ( self.grid[1:] + self.grid[:-1] ) , result['Jn'] , color='red' , label = 'e-' )
        plt.plot( scale['d'] * 0.5 * ( self.grid[1:] + self.grid[:-1] ) , result['Jp'] , color='blue' , label = 'hole' )
        plt.plot( scale['d'] * 0.5 * ( self.grid[1:] + self.grid[:-1] ) , result['Jn'] + result['Jp'] , color='green' , label = 'total' , linestyle='dashed' )
        plt.xlabel( 'thickness (cm)' )
        plt.ylabel( 'current (A.cm-2)' )
        plt.legend()
        plt.show()
        quit()
        plt.savefig(title)
