import deltapv
from jax import numpy as np, random, value_and_grad, grad
from jax.experimental import optimizers
import numpy as onp
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
logger = logging.getLogger("deltapv")
logger.setLevel("INFO")

key = random.PRNGKey(0)

PARAMS = [
    "Eg_ETM", "Chi_ETM", "eps_ETM", "logNc_ETM", "logNv_ETM", "logmn_ETM",
    "logmp_ETM", "Eg_HTM", "Chi_HTM", "eps_HTM", "logNc_HTM", "logNv_HTM",
    "logmn_HTM", "logmp_HTM", "logNd_ETM", "logNa_HTM"
]
n_params = len(PARAMS)

L_ETM = 5e-5
L_Perov = 1.1e-4
L_HTM = 5e-5
N = 500
A = 2e4
tau = 1e-6
S = 1e7
Eg_P = 1.5
Chi_P = 3.9
eps_P = 10
Nc_P = 3.9e18
Nv_P = 2.7e18
mn_P = 2
mp_P = 2
Br_P = 2.3e-9

bounds = [(1., 5.), (1., 5.), (1., 20.), (17., 20.), (17., 20.), (0., 3.),
          (0., 3.), (1., 5.), (1., 5.), (1., 20.), (17., 20.), (17., 20.),
          (0., 3.), (0., 3.), (17., 20.), (17., 20.)]

vl = np.array([tup[0] for tup in bounds])
vu = np.array([tup[1] for tup in bounds])

Perov = deltapv.materials.create_material(Eg=Eg_P,
                                          Chi=Chi_P,
                                          eps=eps_P,
                                          Nc=Nc_P,
                                          Nv=Nv_P,
                                          mn=mn_P,
                                          mp=mp_P,
                                          tn=tau,
                                          tp=tau,
                                          Br=Br_P,
                                          A=A)

region_ETM = lambda x: x <= L_ETM
region_Perov = lambda x: np.logical_and(L_ETM < x, x <= L_ETM + L_Perov)
region_HTM = lambda x: L_ETM + L_Perov < x


def EF(Nc, Nv, Eg, Chi, N):

    kBT = deltapv.scales.kB * deltapv.scales.T / deltapv.scales.q
    ni = np.sqrt(Nc * Nv) * np.exp(-Eg / (2 * kBT))
    Ec = -Chi
    EFi = Ec - Eg / 2 + (kBT / 2) * np.log(Nc / Nv)
    dEF = kBT * np.where(N > 0, np.log(np.abs(N) / ni),
                         -np.log(np.abs(N) / ni))

    return EFi + dEF


def getPhis(params):

    params = np.array(params, dtype=np.float64)
    Eg_ETM = params[0]
    Chi_ETM = params[1]
    Nc_ETM = 10**params[3]
    Nv_ETM = 10**params[4]
    Eg_HTM = params[7]
    Chi_HTM = params[8]
    Nc_HTM = 10**params[10]
    Nv_HTM = 10**params[11]
    Nd_ETM = 10**params[14]
    Na_HTM = 10**params[15]

    PhiM0 = -EF(Nc_ETM, Nv_ETM, Eg_ETM, Chi_ETM, Nd_ETM)
    PhiML = -EF(Nc_HTM, Nv_HTM, Eg_HTM, Chi_HTM, -Na_HTM)

    return PhiM0, PhiML


def g1(x):

    Chi_ETM = x[1]
    PhiM0, _ = getPhis(x)

    return Chi_ETM - PhiM0


def g2(x):

    Chi_HTM = x[8]

    return Chi_HTM - Chi_P


def g3(x):

    Eg_HTM = x[7]
    Chi_HTM = x[8]
    _, PhiML = getPhis(x)

    return PhiML - Chi_HTM - Eg_HTM


def g4(x):

    Eg_HTM = x[7]
    Chi_HTM = x[8]

    return Chi_HTM + Eg_HTM - Chi_P - Eg_P


def g5(x):

    Chi_ETM = x[1]

    return Chi_P - Chi_ETM


def g(x):

    r = np.array([g1(x), g2(x), g3(x), g4(x), g5(x)])

    return r


def feasible(x):

    return np.alltrue(g(x) <= 0) and np.alltrue(vl <= x) and np.alltrue(
        x <= vu)


def penalty(x, sigma=1e4):

    upper = np.clip(x - vu, a_min=0)
    lower = np.clip(vl - x, a_min=0)
    cons = np.clip(g(x), a_min=0)
    pen = sigma * (np.sum(upper**2) + np.sum(lower**2) + np.sum(cons**2))

    return pen


def x2des(params, perov=Perov):

    Eg_ETM = params[0]
    Chi_ETM = params[1]
    eps_ETM = params[2]
    Nc_ETM = 10**params[3]
    Nv_ETM = 10**params[4]
    mn_ETM = 10**params[5]
    mp_ETM = 10**params[6]
    Eg_HTM = params[7]
    Chi_HTM = params[8]
    eps_HTM = params[9]
    Nc_HTM = 10**params[10]
    Nv_HTM = 10**params[11]
    mn_HTM = 10**params[12]
    mp_HTM = 10**params[13]
    Nd_ETM = 10**params[14]
    Na_HTM = 10**params[15]

    ETM = deltapv.materials.create_material(Eg=Eg_ETM,
                                            Chi=Chi_ETM,
                                            eps=eps_ETM,
                                            Nc=Nc_ETM,
                                            Nv=Nv_ETM,
                                            mn=mn_ETM,
                                            mp=mp_ETM,
                                            tn=tau,
                                            tp=tau,
                                            A=A)
    HTM = deltapv.materials.create_material(Eg=Eg_HTM,
                                            Chi=Chi_HTM,
                                            eps=eps_HTM,
                                            Nc=Nc_HTM,
                                            Nv=Nv_HTM,
                                            mn=mn_HTM,
                                            mp=mp_HTM,
                                            tn=tau,
                                            tp=tau,
                                            A=A)

    grid = np.linspace(0, L_ETM + L_Perov + L_HTM, N)
    des = deltapv.simulator.create_design(grid)

    des = deltapv.simulator.add_material(des, ETM, region_ETM)
    des = deltapv.simulator.add_material(des, perov, region_Perov)
    des = deltapv.simulator.add_material(des, HTM, region_HTM)
    des = deltapv.simulator.doping(des, Nd_ETM, region_ETM)
    des = deltapv.simulator.doping(des, -Na_HTM, region_HTM)
    des = deltapv.simulator.contacts(des, S, S, S, S)

    return des


def f(params):

    params = np.array(params)
    des = x2des(params)
    ls = deltapv.simulator.incident_light()
    results = deltapv.simulator.simulate(des, ls)
    eff = results["eff"] * 100
    pen = penalty(params)

    return -eff + pen


df = value_and_grad(f)


def sample(key):
    n_points = 0
    while True:
        key, subkey = random.split(key)
        u = random.uniform(subkey, (n_params, ))
        sample = vl + (vu - vl) * u
        n_points += 1
        if feasible(sample):
            return sample, key


def adam(x0, niters, lr=1e-1, b1=0.9, b2=0.999, filename=None):
    if filename is not None:
        h = logging.FileHandler(f"logs/{filename}")
        logger.addHandler(h)
    opt_init, opt_update, get_params = optimizers.adam(lr,
                                                       b1=b1,
                                                       b2=b2,
                                                       eps=1e-8)
    opt_state = opt_init(x0)
    growth = []

    def take_step(step, opt_state):
        param = get_params(opt_state)
        logger.info(f"param = {list(param)}")
        logger.info(f"feasible = {feasible(param)}")
        value, grads = df(param)
        logger.info(f"value = {value}")
        logger.info(f"grads = {list(grads)}")
        opt_state = opt_update(step, grads, opt_state)
        return value, opt_state

    for step in range(niters):
        value, opt_state = take_step(step, opt_state)
        growth.append(value)

    logger.info("done")
    logger.info("growth:")
    logger.info([float(i) for i in growth])

    if filename is not None:
        logger.removeHandler(h)

    return growth


def random_sampling(niters, key, filename=None):
    if filename is not None:
        h = logging.FileHandler(f"logs/{filename}")
        logger.addHandler(h)
    growth = []
    for _ in range(niters):
        param, key = sample(key)
        logger.info(f"param = {list(param)}")
        try:
            value = f(param)
        except:
            value = 0.
        growth.append(value)
        logger.info(f"value = {value}")
    growth = np.array(growth)
    logger.info("done")
    logger.info("growth:")
    logger.info([float(i) for i in growth])
    if filename is not None:
        logger.removeHandler(h)
    return growth, key


def get_j(params):
    params = np.array(params)
    des = x2des(params)
    ls = deltapv.simulator.incident_light()
    results = deltapv.simulator.simulate(des, ls)
    _, j = results["iv"]
    return j


def residual(Eg_P, params, target_j):
    perov = deltapv.materials.create_material(Eg=Eg_P,
                                              Chi=Chi_P,
                                              eps=eps_P,
                                              Nc=Nc_P,
                                              Nv=Nv_P,
                                              mn=mn_P,
                                              mp=mp_P,
                                              tn=tau,
                                              tp=tau,
                                              Br=Br_P,
                                              A=A)
    des = x2des(params, perov=perov)
    ls = deltapv.simulator.incident_light()
    results = deltapv.simulator.simulate(des, ls, n_steps=target_j.size)
    _, j = results["iv"]
    rss = np.sum((j - target_j)**2)
    return rss


dr = value_and_grad(residual, argnums=0)


def adam_rss(x0, params, target_j, tol=1e-4, lr=1., clip=0.1, filename=None):
    if filename is not None:
        h = logging.FileHandler(f"logs/{filename}")
        logger.addHandler(h)
    
    growth = []
    step = 0
    curr = x0
    while True:
        logger.info(f"param = {float(curr)}")
        value, grads = dr(curr, params, target_j)
        logger.info(f"value = {value}")
        logger.info(f"grads = {float(grads)}")
        growth.append(value)
        curr = curr - np.clip(lr * grads, -clip, clip)
        step += 1
        if growth[-1] < tol:
            break

    logger.info("done")
    logger.info("growth:")
    logger.info([float(i) for i in growth])

    if filename is not None:
        logger.removeHandler(h)

    return growth


if __name__ == "__main__":
    x0, key = sample(key)
    adam(x0, 200, lr=1e-2, b1=0.1, b2=0.1, filename="adam_psc_lr1em2_b11em1_b21em1_200iter.log")
