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
M0 = {
    "eps": 11.7,
    "Chi": 4.05,
    "Eg": 1.12,
    "logNc": 19.5051499783,
    "logNv": 19.2552725051,
    "logmn": 3.1461280357,
    "logmp": 2.6532125138,
    "logtn": -8.0,
    "logtp": -8.0,
}
LB = {
    "eps": 1.0,
    "Chi": 1.0,
    "Eg": 1.0,
    "logNc": 17.0,
    "logNv": 17.0,
    "logmn": 0.0,
    "logmp": 0.0,
    "logtn": -9.0,
    "logtp": -9.0,
}
UB = {
    "eps": 20.0,
    "Chi": 5.0,
    "Eg": 5.0,
    "logNc": 20.0,
    "logNv": 20.0,
    "logmn": 3.6989700043,
    "logmp": 3.6989700043,
    "logtn": -6.0,
    "logtp": -6.0,
}


def f(matdict):
    pen = 0
    for param in matdict:
        pen += penalty(matdict[param], LB[param], UB[param])

    matdict["Nc"] = 10**matdict["logNc"]
    matdict["Nv"] = 10**matdict["logNv"]
    matdict["tn"] = 10**matdict["logtn"]
    matdict["tp"] = 10**matdict["logtp"]
    matdict["mn"] = 10**matdict["logmn"]
    matdict["mp"] = 10**matdict["logmp"]
    matdict.pop("logNc")
    matdict.pop("logNv")
    matdict.pop("logtn")
    matdict.pop("logtp")
    matdict.pop("logmn")
    matdict.pop("logmp")
    matdict["A"] = 20000.0

    mat = deltapv.materials.create_material(**matdict)
    des = deltapv.simulator.add_material(base, mat, lambda _: True)
    des = deltapv.simulator.single_pn_junction(des, 1e18, -1e18, JPOS)
    des = deltapv.simulator.contacts(des, S, S, S, S)
    ls = deltapv.simulator.incident_light()

    results = deltapv.simulator.simulate(des, ls)
    eta = 100 * results["eff"]

    return pen - eta


df = jax.value_and_grad(f)


def barrier(x, l, u, sigma=1e-3):
    cost = -sigma * (np.log(u - x) + np.log(x - l))
    return cost


def penalty(x, l, u, sigma=1e4):
    cost = sigma * (np.maximum(0, x - u)**2 + np.maximum(0, l - x)**2)
    return cost


def graddesc(lr=1e-4):
    M = M0.copy()
    trajs = {param: [] for param in M}
    derivs = {param: [] for param in M}
    growth = []

    for _ in range(100):
        logger.info(M)
        eff, deff = df(M)
        growth.append(eff)
        logger.info(eff)
        logger.info(deff)
        for param in M:
            trajs[param].append(M[param])
            derivs[param].append(deff[param])
            new = M[param] - lr * deff[param]
            M[param] = np.clip(new, LB[param], UB[param])

    logger.info(trajs)
    logger.info(derivs)
    logger.info(growth)


def adam(niters, lr=1e-3, filename=None):
    if filename is not None:
        logger.addHandler(logging.FileHandler(f"logs/{filename}"))
    opt_init, opt_update, get_params = optimizers.adam(lr,
                                                       b1=0.9,
                                                       b2=0.999,
                                                       eps=1e-8)
    opt_state = opt_init(M0)
    trajs = {param: [] for param in M0}
    derivs = {param: [] for param in M0}
    growth = []

    def take_step(step, opt_state):
        param = get_params(opt_state)
        logger.info(f"param = {param}")
        value, grads = df(param)
        logger.info(f"value = {value}")
        logger.info(f"grads = {grads}")
        growth.append(np.float64(value))
        for key in M0:
            trajs[key].append(np.float64(param[key]))
            derivs[key].append(np.float64(grads[key]))
        opt_state = opt_update(step, grads, opt_state)
        return value, opt_state

    for step in range(niters):
        value, opt_state = take_step(step, opt_state)

    logger.info("done")
    logger.info("growth:")
    logger.info([float(i) for i in growth])
    logger.info("trajs:")
    logger.info(
        {key: [float(i) for i in value]
         for key, value in trajs.items()})
    logger.info("derivs:")
    logger.info(
        {key: [float(i) for i in value]
         for key, value in derivs.items()})


def random(niters, filename=None, seed=0):
    if filename is not None:
        logger.addHandler(logging.FileHandler(f"logs/{filename}"))
    key = jax.random.PRNGKey(seed)
    vals = []
    for _ in range(niters):
        key, subkey = jax.random.split(key)
        unif = jax.random.uniform(subkey, shape=(9, ))
        matnew = {}
        for param, d in zip(M0, unif):
            matnew[param] = LB[param] + d * (UB[param] - LB[param])
        try:
            value = f(matnew)
        except:
            logger.error(f"failed: {matnew}")
            continue
        vals.append(float(value))
        logger.info(f"param = {matnew}")
        logger.info(f"value = {value}")
    logger.info("done")
    logger.info("values:")
    logger.info(vals)


def analyze_adam(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    growth = ast.literal_eval(lines[-5])
    trajs = ast.literal_eval(lines[-3])
    derivs = ast.literal_eval(lines[-1])
    return growth, trajs, derivs


def analyze_random(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    growth = ast.literal_eval(lines[-1])
    return growth


if __name__ == "__main__":

    # adam(niters=100, lr=1e-1, filename="adam_1em1_100iter.log")
    # random(500, filename="random_500iter.log")
    growth, trajs, derivs = analyze_adam("logs/adam_1em1_100iter.log")
    values = analyze_random("logs/random_500iter.log")
    growth = np.array(growth)
    values = np.array(values)

    plt.plot(growth)
    plt.xlabel("iterations")
    plt.ylabel("objective")
    plt.ylim(top=0)
    plt.show()

    plt.plot(values)
    plt.xlabel("iterations")
    plt.ylabel("objective")
    plt.ylim(top=0)
    plt.show()

    plt.plot(lax.cummin(growth), label="adam")
    plt.plot(lax.cummin(values), label="random")
    plt.xlabel("iterations")
    plt.ylabel("objective")
    plt.legend()
    plt.show()
