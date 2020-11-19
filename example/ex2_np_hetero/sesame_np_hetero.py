import sesame
import numpy as np
import scipy.io
from scipy.io import savemat
import matplotlib.pyplot as plt
import os
import argparse
    
    
def sesame_pn_hetero(G=None, voltages=None):
    t1 = 25*1e-7    # thickness of CdS
    t2 = 4*1e-4     # thickness of CdTe

    # Heterojunctions require dense mesh near the interface
    dd = 1e-7
    
    x = np.concatenate((np.linspace(0, dd, 10, endpoint=False),                        # L contact interface
                        np.linspace(dd, t1-dd, 50, endpoint=False),                    # material 1
                        np.linspace(t1 - dd, t1 + dd, 10, endpoint=False),             # interface 1
                        np.linspace(t1 + dd, (t1+t2) - dd, 100, endpoint=False),       # material 2
                        np.linspace((t1+t2) - dd, (t1+t2), 10)))                       # R contact interface

    sys = sesame.Builder(x)

    CdS = {'Nc': 2.2e18, 'Nv':1.8e19, 'Eg':2.4, 'epsilon':10, 'Et': 0,
            'mu_e':100, 'mu_h':25, 'tau_e':1e-8, 'tau_h':1e-13,
            'affinity': 4.}
    
    CdTe = {'Nc': 8e17, 'Nv': 1.8e19, 'Eg':1.5, 'epsilon':9.4, 'Et': 0,
            'mu_e':320, 'mu_h':40, 'tau_e':5e-9, 'tau_h':5e-9,
            'affinity': 3.9}

    CdS_region = lambda x: x<=t1
    CdTe_region = lambda x: x>t1

    sys.add_material(CdS, CdS_region)
    sys.add_material(CdTe, CdTe_region)

    nD = 1e17  # donor density [cm^-3]
    sys.add_donor(nD, CdS_region)
    nA = 1e15  # acceptor density [cm^-3]
    sys.add_acceptor(nA, CdTe_region)

    sys.contact_type('Ohmic', 'Ohmic')

    Scontact = 1.16e7
    Sn_left, Sp_left, Sn_right, Sp_right = Scontact, Scontact, Scontact, Scontact
    sys.contact_S(Sn_left, Sp_left, Sn_right, Sp_right)

    if G is None:
        phi0 = 1e17
        alpha = 2.3e4
        f = lambda x: phi0*alpha*np.exp(-x*alpha)
    else:
        f = lambda x: G
    sys.generation(f)

    if voltages is None:
        voltages = np.linspace(0,1,21)
    
    os.makedirs('sesame_output', exist_ok=True)
    j = sesame.IVcurve(sys, voltages, 'sesame_output/1dhetero_V')
    j = j * sys.scaling.current
    
    return voltages, j


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save')
    parser.add_argument('--G')
    args = parser.parse_args()
    
    G = float(args.G) if args.G else None
    
    voltages, j = sesame_pn_hetero(G=G)
    
    plt.plot(voltages, j, '-o')
    plt.xlabel('Voltage [V]')
    plt.ylabel('Current [A/cm^2]')
    plt.title('SESAME pn-heterojunction example')
    if args.save is not None:
        plt.savefig(args.save)
    plt.show()