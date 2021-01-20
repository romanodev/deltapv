import jaxpv
import jax
from jax import numpy as np, value_and_grad, jacobian
import matplotlib.pyplot as plt

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


def f(params):

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
    PhiM0 = params[16]
    PhiML = params[17]

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

    ls = jaxpv.simulator.incident_light()

    results = jaxpv.simulator.simulate(des, ls)
    neff = -results["eff"] * 100

    return neff


def h(params):

    return np.sum(params)


def g(x):

    r = np.array([g1(x), g2(x), g3(x), g4(x)])

    return r


def g1(x):

    Chi_ETM = x[1]
    PhiM0 = x[16]

    return -Chi_ETM + PhiM0


def g2(x):

    Chi_HTM = x[8]

    return -Chi_HTM + Chi_P


def g3(x):

    Eg_HTM = x[7]
    Chi_HTM = x[8]
    PhiML = x[17]

    return -PhiML + Chi_HTM + Eg_HTM


def g4(x):

    Eg_HTM = x[7]
    Chi_HTM = x[8]

    return -Chi_HTM - Eg_HTM + Chi_P + Eg_P


def feasible(x):

    return np.alltrue(g(x) >= 0)


gradf = value_and_grad(f)
gradh = value_and_grad(h)

x0 = np.array([
    4,
    4.0692,
    8.4,
    18.8,
    18,
    191.4,
    5.4,  # ETM
    3.3336,
    2.0663,
    20,
    19.3,
    18,
    4.5,
    361,  # HTM
    17.8,
    18,  # doping
    4.0763,
    5.3759  # contacts
])

jac1 = jacobian(g1)(x0)
jac2 = jacobian(g2)(x0)
jac3 = jacobian(g3)(x0)
jac4 = jacobian(g4)(x0)

n_params = x0.size

if __name__ == "__main__":

    print(feasible(x0))

    neff, grad = gradf(x0)

    print(x0)
    print(neff)
    print(grad)
