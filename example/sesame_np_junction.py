import sesame
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
    
    
def sesame_pnj(G=None, voltages=None):
    L = 3e-4
    grid = np.concatenate((np.linspace(0, 1.2e-4, 100, endpoint=False, dtype=np.float64),
                           np.linspace(1.2e-4, L, 50, dtype=np.float64)))  # cm

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
    
    if voltages is None:
        voltages = np.linspace(0, 0.95, 40)

    os.makedirs('sesame_output', exist_ok=True)

    j = sesame.IVcurve(sys, voltages, 'sesame_output/1dhomo')
    j = j * sys.scaling.current

    return voltages, j

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save')
    parser.add_argument('--G')
    args = parser.parse_args()
    
    G = float(args.G) if args.G else None
    
    voltages, j = sesame_pnj(G=G)
    
    plt.plot(voltages, j, '-o')
    plt.xlabel('Voltage [V]')
    plt.ylabel('Current [A/cm^2]')
    plt.title('SESAME pn-junction example')
    if args.save is not None:
        plt.savefig(args.save)
    plt.show()