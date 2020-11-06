import os
import numpy as np
os.environ['JAX'] = 'YES'
import jaxpv
import argparse
import matplotlib.pyplot as plt
import pickle
    
    
def jaxpv_pnj_gradeff(G=None):
    L = 3e-4
    grid = np.concatenate((np.linspace(0, 1.2e-4, 100, endpoint=False, dtype=np.float64),
                            np.linspace(1.2e-4, L, 50, dtype=np.float64)))  #cm
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

    simu.optical_G( 'user' , G )

    gradeff = simu.grad_efficiency(jit=False)

    return gradeff

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save")
    parser.add_argument('--G')
    args = parser.parse_args()
    
    G = float(args.G) if args.G else None
    
    gradeff = jaxpv_pnj_gradeff()

    print(gradeff)

    if args.save is not None:
        with open(args.save, 'wb') as f:
            pickle.dump(gradeff, f)
    
    print('saved to', args.save)