import jaxpv
from jax import numpy as np, value_and_grad, jacobian, grad
import numpy as onp
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger("jaxpv")

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

Perov = jaxpv.materials.create_material(Eg=Eg_P,
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

    kBT = jaxpv.scales.kB * jaxpv.scales.T / jaxpv.scales.q
    ni = np.sqrt(Nc * Nv) * np.exp(-Eg / (2 * kBT))
    Ec = -Chi
    EFi = Ec - Eg / 2 + (kBT / 2) * np.log(Nc / Nv)
    dEF = kBT * np.where(N > 0, np.log(np.abs(N) / ni),
                         -np.log(np.abs(N) / ni))

    return EFi + dEF


def getPhis(params):

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


def x2des(params):

    Eg_ETM = params[0]
    Chi_ETM = params[1]
    eps_ETM = params[2]
    Nc_ETM = 10**params[3]
    Nv_ETM = 10**params[4]
    mn_ETM = params[5]
    mp_ETM = params[6]
    Eg_HTM = params[7]
    Chi_HTM = params[8]
    eps_HTM = params[9]
    Nc_HTM = 10**params[10]
    Nv_HTM = 10**params[11]
    mn_HTM = params[12]
    mp_HTM = params[13]
    Nd_ETM = 10**params[14]
    Na_HTM = 10**params[15]

    PhiM0, PhiML = getPhis(params)

    ETM = jaxpv.materials.create_material(Eg=Eg_ETM,
                                          Chi=Chi_ETM,
                                          eps=eps_ETM,
                                          Nc=Nc_ETM,
                                          Nv=Nv_ETM,
                                          mn=mn_ETM,
                                          mp=mp_ETM,
                                          tn=tau,
                                          tp=tau,
                                          A=A)
    HTM = jaxpv.materials.create_material(Eg=Eg_HTM,
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
    des = jaxpv.simulator.create_design(grid)

    des = jaxpv.simulator.add_material(des, ETM, region_ETM)
    des = jaxpv.simulator.add_material(des, Perov, region_Perov)
    des = jaxpv.simulator.add_material(des, HTM, region_HTM)
    des = jaxpv.simulator.doping(des, Nd_ETM, region_ETM)
    des = jaxpv.simulator.doping(des, -Na_HTM, region_HTM)
    des = jaxpv.simulator.contacts(des, S, S, S, S, PhiM0=PhiM0, PhiML=PhiML)

    return des


def f(params):

    feasibility = feasible(params)
    logger.info(f"Feasible: {feasibility}")
    if not feasibility:
        logger.error("Unfeasible point, returning zero.")
        return 0.
    try:
        des = x2des(params)
        ls = jaxpv.simulator.incident_light()
        results = jaxpv.simulator.simulate(des, ls)
        eff = results["eff"] * 100
    except:
        logger.error("Solver failed, returning zero.")
        return 0.

    return -eff


df = grad(f)
vagf = value_and_grad(f)


def g1(x):

    Chi_ETM = x[1]
    PhiM0, _ = getPhis(x)

    return -(Chi_ETM - PhiM0)


def g2(x):

    Chi_HTM = x[8]

    return -(Chi_HTM - Chi_P)


def g3(x):

    Eg_HTM = x[7]
    Chi_HTM = x[8]
    _, PhiML = getPhis(x)

    return -(PhiML - Chi_HTM - Eg_HTM)


def g4(x):

    Eg_HTM = x[7]
    Chi_HTM = x[8]

    return -(Chi_HTM + Eg_HTM - Chi_P - Eg_P)


def g5(x):

    Chi_ETM = x[1]

    return -(Chi_P - Chi_ETM)


def g(x):

    r = np.array([g1(x), g2(x), g3(x), g4(x), g5(x)])

    return r


jac1 = jacobian(g1)
jac2 = jacobian(g2)
jac3 = jacobian(g3)
jac4 = jacobian(g4)
jac5 = jacobian(g5)
jac = jacobian(g)

bounds = [(1, 5), (1, 5), (1, 20), (17, 20), (17, 20), (1, 500), (1, 500),
          (1, 5), (1, 5), (1, 20), (17, 20), (17, 20), (1, 500), (1, 500),
          (17, 20), (17, 20)]

vl = np.array([tup[0] for tup in bounds])
vu = np.array([tup[1] for tup in bounds])


def feasible(x):

    return np.alltrue(g(x) >= 0) and np.alltrue(vl <= x) and np.alltrue(
        x <= vu)


def sample():
    while True:
        u = onp.random.rand(n_params)
        sample = vl + (vu - vl) * u
        if feasible(sample):
            return sample


x_ref = np.array([
    4, 4.0692, 8.4, 18.8, 18, 191.4, 5.4, 3.3336, 2.0663, 19.9, 19.3, 18, 4.5,
    361, 17.8, 18
])

x_init = np.array([
    1.85164371, 4.98293216, 8.29670517, 18.84580405, 19.91887512, 102.60205287,
    473.07855517, 3.74772183, 1.02394888, 2.3392392, 17.76149694, 19.60516244,
    448.65494921, 311.63744301, 17.19468214, 17.57586159
])

n_params = x_ref.size

if __name__ == "__main__":

    for alpha in np.linspace(0, 1, 100):
        jaxpv.plotting.plot_bars(x2des(x_init + alpha * (x_ref - x_init)))
