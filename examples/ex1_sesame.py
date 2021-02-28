import sesame
import numpy as np

L = 2e-4  # length of the system in the x-direction [cm]

# Mesh
grid = np.linspace(0, L, 500)

# Create a system
sys = sesame.Builder(grid)

# Dictionary with the material parameters
material = {
    "Nc": 8e17,
    "Nv": 1.8e19,
    "Eg": 1.5,
    "affinity": 3.9,
    "epsilon": 9.4,
    "mu_e": 100,
    "mu_h": 100,
    "tau_e": 1e-8,
    "tau_h": 1e-8,
    "Et": 0
}

# Add the material to the system
sys.add_material(material)

junction = 1e-4  # extent of the junction from the left contact [m]


def n_region(pos):
    x = pos
    return x < junction


def p_region(pos):
    x = pos
    return x >= junction


# Add the donors
nD = 1e17  # [cm^-3]
sys.add_donor(nD, n_region)

# Add the acceptors
nA = 1e17  # [cm^-3]
sys.add_acceptor(nA, p_region)

# Define Ohmic contacts
sys.contact_type("Ohmic", "Ohmic")

# Define the surface recombination velocities for electrons and holes [cm/s]
Sn_left, Sp_left, Sn_right, Sp_right = 1e7, 0, 0, 1e7
sys.contact_S(Sn_left, Sp_left, Sn_right, Sp_right)

# Define a function for the generation rate
phi = 1e17  # photon flux [1/(cm^2 s)]
alpha = 2.3e4  # absorption coefficient [1/cm]

G = np.load("outputs/G.npy")

# Define a function for the generation rate
def gfcn(x, y):
    return np.interp(x, grid, G)

# add generation to system
sys.generation(gfcn)

# IV curve
voltages = np.arange(23) * 0.05
j = sesame.IVcurve(sys, voltages, "outputs/example/iv")

# convert dimensionless current to dimension-ful current
j = j * sys.scaling.current
# save voltage and current values to dictionary
result = {"v": voltages, "j": j}

# save data to python data file
np.save("outputs/example/IV_values", result)

# save data to an ascii txt file
np.savetxt("outputs/example/IV_values.txt", (voltages, j))

# plot I-V curve
try:
    import matplotlib.pyplot as plt
    plt.plot(voltages, j, "-o")
    plt.xlabel("Voltage [V]")
    plt.ylabel("Current [A/cm^2]")
    plt.grid()  # add grid
    plt.show()  # show the plot on the screen
# no matplotlib installed
except ImportError:
    print("Matplotlib not installed, can't make plot")

sys, result = sesame.load_sim("outputs/example/iv_0.gzip")
az = sesame.Analyzer(sys,result)
p1 = (0, 0)
p2 = (L, 0)
az.band_diagram((p1,p2))
