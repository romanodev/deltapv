import sesame
import numpy as np
import matplotlib.pyplot as plt

t1 = 25 * 1e-7  # thickness of CdS
t2 = 4 * 1e-4  # thickness of CdTe
dd = 1e-7  # 2*dd is the distance over which mesh is refined
# Define the mesh
x = np.concatenate((np.linspace(0, dd, 10, endpoint=False),  # L contact interface
                    np.linspace(dd, t1 - dd, 50, endpoint=False),  # material 1
                    np.linspace(t1 - dd, t1 + dd, 10, endpoint=False),  # interface 1
                    np.linspace(t1 + dd, (t1 + t2) - dd, 100, endpoint=False),  # material 2
                    np.linspace((t1 + t2) - dd, (t1 + t2), 10)))  # R contact interface

# CdS material dictionary
CdS = {'Nc': 2.2e18, 'Nv': 1.8e19, 'Eg': 2.4, 'epsilon': 10, 'Et': 0,
       'mu_e': 100, 'mu_h': 25, 'tau_e': 1e-8, 'tau_h': 1e-13,
       'affinity': 4.}
# CdTe material dictionary
CdTe = {'Nc': 8e17, 'Nv': 1.8e19, 'Eg': 1.5, 'epsilon': 9.4, 'Et': 0,
        'mu_e': 320, 'mu_h': 40, 'tau_e': 5e-9, 'tau_h': 5e-9,
        'affinity': 3.9}

nD = 1e17  # donor density [cm^-3]

nA = 1e15  # acceptor density [cm^-3]


class Solver(object):
    def __init__(self, t1, t2, dd, mesh):
        self.t1 = t1
        self.t2 = t2
        self.dd = dd
        self.mesh = mesh
        self.CdS_region = lambda x: x <= t1
        self.CdTe_region = lambda x: x > t1

    def solve(self, CdS, CdTe, nD, nA, voltages=np.linspace(0, 1, 21)):
        sys = sesame.Builder(self.mesh)

        sys.add_material(CdS, self.CdS_region)  # adding CdS
        sys.add_material(CdTe, self.CdTe_region)  # adding CdTe

        sys.add_donor(nD, self.CdS_region)
        sys.add_acceptor(nA, self.CdTe_region)

        Lcontact_type, Rcontact_type = 'Ohmic', 'Ohmic'
        Lcontact_workFcn, Rcontact_workFcn = 0, 0  # Work function irrelevant because L contact is Ohmic
        sys.contact_type(Lcontact_type, Rcontact_type, Lcontact_workFcn, Rcontact_workFcn)

        Scontact = 1.16e7  # [cm/s]
        Sn_left, Sp_left, Sn_right, Sp_right = Scontact, Scontact, Scontact, Scontact
        sys.contact_S(Sn_left, Sp_left, Sn_right, Sp_right)

        phi0 = 1e17  # incoming flux [1/(cm^2 sec)]
        alpha = 2.3e4  # absorbtion coefficient [1/cm]
        f = lambda x: phi0 * alpha * np.exp(-x * alpha)
        sys.generation(f)

        # Perform I-V calculation
        j = sesame.IVcurve(sys, voltages, '1dhetero_V')
        j = j * sys.scaling.current

        result = {'v': voltages, 'j': j}
        np.save('IV_values', result)

        # plot I-V curve
        plt.plot(voltages, j, '-o')
        plt.xlabel('Voltage [V]')
        plt.ylabel('Current [A/cm^2]')
        plt.grid()  # add grid
        plt.show()  # show the plot on the screen

        return j, voltages


if __name__ == '__main__':
    solver = Solver(t1, t2, dd, x)
    j, v = solver.solve(CdS, CdTe, nD, nA, voltages=np.linspace(0, 1, 51))
    plt.plot(v, j * v)
    plt.show()
