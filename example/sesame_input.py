import sesame
import numpy as np
import matplotlib.pyplot as plt


ETM_THICKNESS = 0.1701e-4
PEROV_THICKNESS = 0.7301e-4
HTM_THICKNESS = 0.4989e-4
CELL_THICKNESS = ETM_THICKNESS + PEROV_THICKNESS + HTM_THICKNESS
NUM_POINTS = 500
ALL_INDICES = np.arange(NUM_POINTS, dtype=int)
ETM_START_INDEX = int(0)
PEROV_START_INDEX = int(NUM_POINTS * ETM_THICKNESS // CELL_THICKNESS)
HTM_START_INDEX = int(NUM_POINTS * (ETM_THICKNESS + PEROV_THICKNESS) // CELL_THICKNESS)
ETM_RANGE = np.arange(0, PEROV_START_INDEX, dtype=int)
PEROV_RANGE = np.arange(PEROV_START_INDEX, HTM_START_INDEX, dtype=int)
HTM_RANGE = np.arange(HTM_START_INDEX, NUM_POINTS, dtype=int)

GRID = np.linspace(0, CELL_THICKNESS, num=NUM_POINTS)
sys = sesame.Builder(GRID)

PEROV_PROP = {'epsilon': 10,
              'affinity': 3.9,
              'Eg': 1.5,
              'Nc': 3.9e18,
              'Nv': 2.7e18,
              'mu_e': 2,
              'mu_h': 2,
              #'Br': 2.3e-9,
              'Et': 1,
              'tau_e': 1e-3,
              'tau_h': 1e-3}

ETM_PROP = {'epsilon': 7.3402,
            'affinity': 4.0303,
            'Eg': 3.9375,
            'Nc': 3.7034e18,
            'Nv': 3.3427e18,
            'mu_e': 175.4944,
            'mu_h': 5.4444,
            'Et': 1,
            'tau_e': 1e-3,
            'tau_h': 1e-3}

HTM_PROP = {'epsilon': 16.1612,
            'affinity': 2.0355,
            'Eg': 3.3645,
            'Nc': 9.097e19,
            'Nv': 1.6106e18,
            'mu_e': 4.9727,
            'mu_h': 436.1325,
            'Et': 1,
            'tau_e': 1e-3,
            'tau_h': 1e-3}

ETM_REGION = lambda x: x <= ETM_THICKNESS
PEROV_REGION = lambda x: np.logical_and(ETM_THICKNESS < x, x <= ETM_THICKNESS + PEROV_THICKNESS)
HTM_REGION = lambda x: np.logical_and(ETM_THICKNESS + PEROV_THICKNESS < x, x <= CELL_THICKNESS)

sys.add_material(ETM_PROP, ETM_REGION)
sys.add_material(PEROV_PROP, PEROV_REGION)
sys.add_material(HTM_PROP, HTM_REGION)

sys.add_donor(3.7025e18, ETM_REGION)
sys.add_acceptor(1.6092e18, HTM_REGION)

Lcontact_type, Rcontact_type = 'Ohmic', 'Ohmic'
Lcontact_workFcn, Rcontact_workFcn = 0, 0  # Work function irrelevant because L contact is Ohmic
sys.contact_type(Lcontact_type, Rcontact_type, Lcontact_workFcn, Rcontact_workFcn)

Scontact = 1.e8  # [cm/s]
Sn_left, Sp_left, Sn_right, Sp_right = Scontact, Scontact, Scontact, Scontact
sys.contact_S(Sn_left, Sp_left, Sn_right, Sp_right)

#f = lambda x: 0.
f = lambda x: 1e17
"""phi0 = 1e17  # incoming flux [1/(cm^2 sec)]
alpha = 2.3e4  # absorbtion coefficient [1/cm]
f = lambda x: phi0 * alpha * np.exp(-x * alpha)"""
sys.generation(f)

voltages = np.linspace(0, 1, 50)

j = sesame.IVcurve(sys, voltages, '1dhetero_V')
j = j * sys.scaling.current

plt.plot(voltages, j, '-o')
plt.xlabel('Voltage [V]')
plt.ylabel('Current [A/cm^2]')
plt.grid()
plt.show()