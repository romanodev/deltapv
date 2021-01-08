import sesame
import numpy as np
import argparse
import matplotlib.pyplot as plt

GaP_EMIT = 1e-5  # cm
GaP_BASE = 6e-5
InP_EMIT = 2e-5
InP_BASE = 3e-4
Ge_EMIT = 4e-5
Ge_BASE = 1e-2

nodes = np.cumsum(
    np.array([0, GaP_EMIT, GaP_BASE, InP_EMIT, InP_BASE, Ge_EMIT, Ge_BASE]))

dd = 5e-6
grid = np.concatenate([
    np.linspace(0, nodes[1] - dd, 100, endpoint=False),
    np.linspace(nodes[1] - dd, nodes[1] + dd, 100, endpoint=False),
    np.linspace(nodes[1] + dd, nodes[2] - dd, 100, endpoint=False),
    np.linspace(nodes[2] - dd, nodes[2] + dd, 100, endpoint=False),
    np.linspace(nodes[2] + dd, nodes[3] - dd, 100, endpoint=False),
    np.linspace(nodes[3] - dd, nodes[3] + dd, 100, endpoint=False),
    np.linspace(nodes[3] + dd, nodes[4] - dd, 100, endpoint=False),
    np.linspace(nodes[4] - dd, nodes[4] + dd, 100, endpoint=False),
    np.linspace(nodes[4] + dd, nodes[5] - dd, 100, endpoint=False),
    np.linspace(nodes[5] - dd, nodes[5] + dd, 100, endpoint=False),
    np.linspace(nodes[5] + dd, nodes[6], 100, endpoint=True)
])


sys = sesame.Builder(grid)

GaP = {"Nc":1.8e19, "Nv":1.9e19, "Eg":2.26, "affinity":3.8, "epsilon":11.1,
        "mu_e":250., "mu_h":150., "tau_e":1e-7, "tau_h":1e-6, "Et":0}
InP = {"Nc":5.7e17, "Nv":1.1e19, "Eg":1.344, "affinity":4.38, "epsilon":12.5,
        "mu_e":5400., "mu_h":100, "tau_e":3e-6, "tau_h":2e-9, "Et":0}
Ge = {"Nc":1e19, "Nv":5e18, "Eg":0.661, "affinity":4., "epsilon":16.2,
        "mu_e":3900., "mu_h":1900., "tau_e":1e-3, "tau_h":1e-3, "Et":0}

sys.add_material(GaP, lambda x: x < nodes[2])
sys.add_material(InP, lambda x: np.logical_and(nodes[2] <= x, x < nodes[4]))
sys.add_material(Ge, lambda x: nodes[4] <= x)

sys.add_donor(2e18, lambda x: x < nodes[1])
sys.add_acceptor(1e17, lambda x: np.logical_and(nodes[1] <= x, x < nodes[2]))
sys.add_donor(3e17, lambda x: np.logical_and(nodes[2] <= x, x < nodes[3]))
sys.add_acceptor(1e17, lambda x: np.logical_and(nodes[3] <= x, x < nodes[4]))
sys.add_donor(2e18, lambda x: np.logical_and(nodes[4] <= x, x < nodes[5]))
sys.add_acceptor(1e17, lambda x: nodes[5] <= x)

sys.contact_type("Ohmic", "Ohmic")
sys.contact_S(1e7, 0, 0, 1e7)

phi = 1e17
alpha = 2.3e4

def gfcn(x):
    return phi * alpha * np.exp(-alpha * x)

sys.generation(gfcn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save")
    args = parser.parse_args()

    voltages = np.linspace(0, 0.5, 20)
    j = sesame.IVcurve(sys, voltages, "outputs/3J_V")
    j = j * sys.scaling.current

    plt.plot(voltages, j, "-o")
    plt.xlabel("Voltage [V]")
    plt.ylabel("Current [A/cm^2]")
    plt.title("3J")
    if args.save is not None:
        plt.savefig(args.save)
    plt.show()
