import os
os.environ['JAX'] = 'NO'
from jaxpv.utils import *
from jaxpv import JAXPV

ETM_THICKNESS = 0.1701e-4
PEROV_THICKNESS = 0.7301e-4
HTM_THICKNESS = 0.4989e-4
CELL_THICKNESS = ETM_THICKNESS + PEROV_THICKNESS + HTM_THICKNESS
NUM_POINTS = 500
ALL_INDICES = np.arange(NUM_POINTS, dtype=int)
ETM_START_INDEX = int(0)
PEROV_START_INDEX = int(NUM_POINTS * ETM_THICKNESS // CELL_THICKNESS)
HTM_START_INDEX = int(NUM_POINTS * (ETM_THICKNESS + PEROV_THICKNESS) //
                      CELL_THICKNESS)
ETM_RANGE = np.arange(0, PEROV_START_INDEX, dtype=int)
PEROV_RANGE = np.arange(PEROV_START_INDEX, HTM_START_INDEX, dtype=int)
HTM_RANGE = np.arange(HTM_START_INDEX, NUM_POINTS, dtype=int)

GRID = np.linspace(0, CELL_THICKNESS, num=NUM_POINTS)

pv_obj = JAXPV(GRID)

PEROV_PROP = {
    'eps': 10,
    'Chi': 3.9,
    'Eg': 1.5,
    'Nc': 3.9e18,
    'Nv': 2.7e18,
    'mn': 2,
    'mp': 2,
    'Br': 0,
    'Et': 1,
    'tn': 1e-3,
    'tp': 1e-3
}

ETM_PROP = {
    'eps': 7.3402,
    'Chi': 4.0303,
    'Eg': 3.9375,
    'Nc': 3.7034e18,
    'Nv': 3.3427e18,
    'mn': 175.4944,
    'mp': 5.4444,
    'Et': 1,
    'tn': 1e-3,
    'tp': 1e-3
}

HTM_PROP = {
    'eps': 16.1612,
    'Chi': 2.0355,
    'Eg': 3.3645,
    'Nc': 9.097e19,
    'Nv': 1.6106e18,
    'mn': 4.9727,
    'mp': 436.1325,
    'Et': 1,
    'tn': 1e-3,
    'tp': 1e-3
}

pv_obj.add_material(PEROV_PROP, PEROV_RANGE)
pv_obj.add_material(ETM_PROP, ETM_RANGE)
pv_obj.add_material(HTM_PROP, HTM_RANGE)

ETM_DOPING = 3.7025e18 * np.ones(ETM_RANGE.size)
HTM_DOPING = -1.6092e18 * np.ones(HTM_RANGE.size)
pv_obj.doping_profile(ETM_DOPING, ETM_RANGE)
pv_obj.doping_profile(HTM_DOPING, HTM_RANGE)

pv_obj.contacts(1.e8, 1.e8, 1.e8, 1.e8)

#pv_obj.optical_G(type='user', G=1e18 * np.ones(NUM_POINTS, dtype=np.float32))
pv_obj.optical_G(type='user', G=np.zeros(NUM_POINTS, dtype=np.float32))

#eq_result = pv_obj.solve(equilibrium=False,V=1)
#pv_obj.plot_concentration_profile(eq_result)
#pv_obj.plot_band_diagram(eq_result)
#pv_obj.plot_current_profile(eq_result)

pv_obj.IV_curve()

#print(pv_obj.efficiency())
#print(pv_obj.grad_efficiency(jit=False))
