import os
import numpy as np
os.environ['JAX'] = 'NO'
import jaxpv
import sesame
import argparse
    
    
def jaxpv_1v(G=None, V=1):
    L = 3e-4
    grid = np.linspace(0, L, 500)
    simu = jaxpv.JAXPV(grid)

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
    }

    Snl = 1e7  #cm/s
    Spl = 0  #cm/s
    Snr = 0  #cm/s
    Spr = 1e7 #cm/s

    simu.add_material( material , range( len( grid ) ) )
    simu.contacts( Snl , Snr , Spl , Spr )
    simu.single_pn_junction( 1e17 , - 1e15 , 50e-7 )
    
    if G is None:
        phi = 1e17       # photon flux [cm-2 s-1)]
        alpha = 2.3e4    # absorption coefficient [cm-1]
        G = phi * alpha * np.exp( - alpha * grid ) # cm-3 s-1
    else:
        G = G * np.ones(grid.shape)
    print(G)
    simu.optical_G( 'user' , G )

    result = simu.solve(V=V)
    
    return result


def sesame_1v(G=None, V=1):
    L = 3e-4
    # grid = np.concatenate((np.linspace(0, 1.2e-4, 100, endpoint=False, dtype=np.float64),
    #                        np.linspace(1.2e-4, L, 50, dtype=np.float64)))  # cm
    grid = np.linspace(0, L, 500)

    material = {
        'affinity': 3.9,  # eV
        'Eg': 1.5,  # eV
        'epsilon': 9.4,
        'Nc': 8e17,  # cm-3
        'Nv': 1.8e19,  # cm-3
        'mu_e': 100,  # cm2V-1s-1
        'mu_h': 100,  # cm2V-1s-1
        'Et': 0,  # eV
        'tau_e': 10e-9,  # s
        'tau_h': 10e-9,  # s
    }

    sys = sesame.Builder(grid)

    junction_position = 50e-7

    sys.add_material(material)

    n_region = lambda x: x < junction_position
    p_region = lambda x: x >= junction_position

    sys.add_donor(1e17, n_region)
    sys.add_acceptor(1e15, p_region)

    Snl = 1e7  # cm/s
    Spl = 0  # cm/s
    Snr = 0  # cm/s
    Spr = 1e7  # cm/s

    sys.contact_type('Ohmic', 'Ohmic')

    Sn_left, Sp_left, Sn_right, Sp_right = 1e7, 0, 0, 1e7
    sys.contact_S(Sn_left, Sp_left, Sn_right, Sp_right)

    if G is None:
        phi = 1e17  # photon flux [cm-2 s-1)]
        alpha = 2.3e4  # absorption coefficient [cm-1]
        f = lambda x: phi * alpha * np.exp(- alpha * x)  # cm-3 s-1
    else:
        f = lambda x: G
    print(f(0))
    sys.generation(f)
    
    voltages = np.array([V])

    os.makedirs('sesame_output', exist_ok=True)

    j = sesame.IVcurve(sys, voltages, 'sesame_output/1dhomo', verbose=True)
    j = j * sys.scaling.current

    return voltages, j


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save")
    parser.add_argument('--G')
    parser.add_argument('--V')
    args = parser.parse_args()
    
    G = float(args.G) if args.G else None
    V = float(args.V) if args.V else 1
    
    jaxpv_result = jaxpv_1v(G=G, V=V)
    sesame_result = sesame_1v(G=G, V=V)
    
    print(jaxpv_result)
    print(sesame_result)