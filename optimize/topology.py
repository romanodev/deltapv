import os
os.environ["DEBUGNANS"] = "TRUE"
import deltapv
import jax
from jax import numpy as np, lax
from jax.experimental import optimizers
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import ast
logger = logging.getLogger("deltapv")
logger.setLevel("INFO")

L = 1e-4
JPOS = 5e-5
N = 500
S = 1e7
grid = np.linspace(0, L, N)
base = deltapv.simulator.create_design(grid)
Eg_0 = 1.12
Eg_min = 1
Eg_max = 5
MAT = {
    "eps": 11.7,
    "Chi": 4.05,
    "Eg": Eg_0,
    "Nc": 10**19.5051499783,
    "Nv": 10**19.2552725051,
    "mn": 10**3.1461280357,
    "mp": 10**2.6532125138,
    "tn": 10**(-8.0),
    "tp": 10**(-8.0),
    "A": 20000.0
}


def f(Eg):

    mat = deltapv.materials.create_material(**MAT)
    des = deltapv.simulator.add_material(base, mat, lambda _: True)
    des = deltapv.simulator.single_pn_junction(des, 1e18, -1e18, JPOS)
    des = deltapv.simulator.contacts(des, S, S, S, S)
    des = deltapv.objects.update(des, Eg=Eg / deltapv.scales.energy)
    ls = deltapv.simulator.incident_light()

    results = deltapv.simulator.simulate(des, ls)
    eta = 100 * results["eff"]

    """deltapv.plotting.plot_band_diagram(des, results["eq"], eq=True)
    deltapv.plotting.plot_band_diagram(des, results["Voc"])
    deltapv.plotting.plot_charge(des, results["eq"])
    deltapv.plotting.plot_charge(des, results["Voc"])
    deltapv.plotting.plot_iv_curve(*results["iv"])"""

    return -eta


df = jax.value_and_grad(f)


def barrier(x, l, u, sigma=1e-3):
    cost = -sigma * (np.log(u - x) + np.log(x - l))
    return cost


def penalty(x, l, u, sigma=1e4):
    cost = sigma * (np.maximum(0, x - u)**2 + np.maximum(0, l - x)**2)
    return cost


def adam(niters, lr=1e-3, filename=None):
    if filename is not None:
        logger.addHandler(logging.FileHandler(f"logs/{filename}"))
    opt_init, opt_update, get_params = optimizers.adam(lr,
                                                       b1=0.9,
                                                       b2=0.999,
                                                       eps=1e-8)
    opt_state = opt_init(np.linspace(Eg_0, Eg_0, N))
    trajs = []
    derivs = []
    growth = []

    def take_step(step, opt_state):
        param = get_params(opt_state)
        logger.info(f"param = {param}")
        value, grads = df(param)
        logger.info(f"value = {value}")
        logger.info(f"grads = {grads}")
        growth.append(value)
        trajs.append(param)
        derivs.append(grads)
        opt_state = opt_update(step, grads, opt_state)
        return value, opt_state

    for step in range(niters):
        try:
            value, opt_state = take_step(step, opt_state)
        except:
            logger.info("failed! early termination")
            break

    trajs = np.array(trajs)
    derivs = np.array(derivs)

    logger.info("done")
    logger.info("growth:")
    logger.info([float(i) for i in growth])
    logger.info("trajs:")
    logger.info(trajs)
    logger.info("derivs:")
    logger.info(derivs)

    return growth, trajs, derivs


def analyze_adam(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    growth = ast.literal_eval(lines[-5])
    trajs = ast.literal_eval(lines[-3])
    derivs = ast.literal_eval(lines[-1])
    return growth, trajs, derivs


if __name__ == "__main__":
    pass
