import sesame
import numpy as np
import matplotlib.pyplot as plt

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
    # 'Br'  : 0 ,
    # 'Cn'  : 0 ,
    # 'Cp'  : 0 ,
    # 'A'   : 0
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

phi = 1e17  # photon flux [cm-2 s-1)]
alpha = 2.3e4  # absorption coefficient [cm-1]
f = lambda x: phi * alpha * np.exp(- alpha * x)  # cm-3 s-1
#f = lambda x: 0
sys.generation(f)

voltages = np.linspace(0, 1, 50)

j = sesame.IVcurve(sys, voltages, '1dhetero_V')
j = j * sys.scaling.current

plt.plot(voltages, j, '-o')
plt.xlabel('Voltage [V]')
plt.ylabel('Current [A/cm^2]')
plt.grid()
plt.show()

v_sesame = voltages
j_sesame = j