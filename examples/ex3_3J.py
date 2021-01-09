import jaxpv
import jax.numpy as np
import argparse
import matplotlib.pyplot as plt

GaP_EMIT = 1e-5  # cm
GaP_BASE = 6e-5
InP_EMIT = 2e-5
InP_BASE = 3e-4
Ge_EMIT = 4e-5
Ge_BASE = 1e-4  # was 1e-2

nodes = np.cumsum(
    np.array([0, GaP_EMIT, GaP_BASE, InP_EMIT, InP_BASE, Ge_EMIT, Ge_BASE]))

dd = 2e-6
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

des = jaxpv.simulator.create_design(grid)

GaP = jaxpv.materials.load_material("GaP")
InP = jaxpv.materials.load_material("InP")
Ge = jaxpv.materials.load_material("Ge")

des = jaxpv.simulator.add_material(des, GaP, lambda x: x < nodes[2])
des = jaxpv.simulator.add_material(
    des, InP, lambda x: np.logical_and(nodes[2] <= x, x < nodes[4]))
des = jaxpv.simulator.add_material(des, Ge, lambda x: nodes[4] <= x)

des = jaxpv.simulator.doping(des, 2e18, lambda x: x < nodes[1])
des = jaxpv.simulator.doping(
    des, -1e17, lambda x: np.logical_and(nodes[1] <= x, x < nodes[2]))
des = jaxpv.simulator.doping(
    des, 3e17, lambda x: np.logical_and(nodes[2] <= x, x < nodes[3]))
des = jaxpv.simulator.doping(
    des, -1e17, lambda x: np.logical_and(nodes[3] <= x, x < nodes[4]))
des = jaxpv.simulator.doping(
    des, 2e18, lambda x: np.logical_and(nodes[4] <= x, x < nodes[5]))
des = jaxpv.simulator.doping(des, -1e17, lambda x: nodes[5] <= x)

des = jaxpv.simulator.contacts(des, 1e7, 0, 0, 1e7)

ls = jaxpv.simulator.incident_light()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save")
    args = parser.parse_args()

    results = jaxpv.simulator.simulate(des, ls)
    v, j = results["iv"]

    jaxpv.plotting.plot_iv_curve(v, j)
    jaxpv.plotting.plot_bars(des)
    jaxpv.plotting.plot_band_diagram(des, results["eq"], eq=True)
    jaxpv.plotting.plot_band_diagram(des, results["Voc"])
