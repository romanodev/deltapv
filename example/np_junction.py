from jaxpv.src.jaxpv import *
grid =  np.concatenate( ( np.linspace(0,1.2e-4, 100, endpoint=False, dtype=np.float64) , np.linspace(1.2e-4, 3e-4, 50, dtype=np.float64) ) )  #cm
simu = JAXPV( grid )
material = {
    'Chi' : 3.9 , #eV
    'Eg'  : 1.5 , #eV
    'eps' : 9.4 ,
    'Nc'  : 8e17 , #cm-3
    'Nv'  : 1.8e19 , #cm-3
    'mn'  : 100 , #cm2V-1s-1
    'mp'  : 100 , #cm2V-1s-1
    'Et'  : 0 , #eV
    'tn'  : 10e-9 , #s
    'tp'  : 10e-9 , #s
    'Br'  : 0 ,
    'Cn'  : 0 ,
    'Cp'  : 0 ,
    'A'   : 0
}

Snl = 1e7  #cm/s
Spl = 0  #cm/s
Snr = 0  #cm/s
Spr = 1e7 #cm/s

simu.add_material( material , range( len( grid ) ) )
simu.contacts( Snl , Snr , Spl , Spr )
simu.single_pn_junction( 1e17 , - 1e15 , 50e-7 )

phi = 1e17       # photon flux [cm-2 s-1)]
alpha = 2.3e4    # absorption coefficient [cm-1]
G = phi * alpha * np.exp( - alpha * grid ) # cm-3 s-1
simu.optical_G( 'user' , G )

#result = simu.solve( 0 , equilibrium=True )
#result = simu.solve( 0 )
#simu.plot_band_diagram( result )
#simu.plot_concentration_profile( result )
#simu.plot_current_profile( result )
IV = simu.IV_curve()
#efficiency = simu.efficiency()
#print(efficiency)
