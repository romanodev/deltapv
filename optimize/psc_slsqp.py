import deltapv as dpv
from jax import numpy as jnp, value_and_grad, jacobian
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

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

Perov = dpv.create_material(Eg=Eg_P,
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

vl = jnp.array([1, 1, 1, 17, 17, 0, 0, 1, 1, 1, 17, 17, 0, 0, 17, 17],
               dtype=jnp.float64)
vu = jnp.array([5, 5, 20, 20, 20, 3, 3, 5, 5, 20, 20, 20, 3, 3, 20, 20],
               dtype=jnp.float64)


def x2des(x, perov=Perov):
    Eg_ETM = x[0]
    Chi_ETM = x[1]
    eps_ETM = x[2]
    Nc_ETM = 10**x[3]
    Nv_ETM = 10**x[4]
    mn_ETM = 10**x[5]
    mp_ETM = 10**x[6]
    Eg_HTM = x[7]
    Chi_HTM = x[8]
    eps_HTM = x[9]
    Nc_HTM = 10**x[10]
    Nv_HTM = 10**x[11]
    mn_HTM = 10**x[12]
    mp_HTM = 10**x[13]
    Nd_ETM = 10**x[14]
    Na_HTM = 10**x[15]

    ETM = dpv.create_material(Eg=Eg_ETM,
                              Chi=Chi_ETM,
                              eps=eps_ETM,
                              Nc=Nc_ETM,
                              Nv=Nv_ETM,
                              mn=mn_ETM,
                              mp=mp_ETM,
                              tn=tau,
                              tp=tau,
                              A=A)
    HTM = dpv.create_material(Eg=Eg_HTM,
                              Chi=Chi_HTM,
                              eps=eps_HTM,
                              Nc=Nc_HTM,
                              Nv=Nv_HTM,
                              mn=mn_HTM,
                              mp=mp_HTM,
                              tn=tau,
                              tp=tau,
                              A=A)

    des = dpv.make_design(n_points=N,
                          Ls=[L_ETM, L_Perov, L_HTM],
                          mats=[ETM, perov, HTM],
                          Ns=[Nd_ETM, 0, -Na_HTM],
                          Snl=S,
                          Snr=S,
                          Spl=S,
                          Spr=S)

    return des


def f(x):
    des = x2des(x)
    results = dpv.simulate(des, verbose=False)
    eff = results["eff"] * 100
    return -eff


df = value_and_grad(f)

x0 = np.array([
    1.661788237392516, 4.698293002285373, 19.6342803183675, 18.83471869026531,
    19.54569869328745, 0.7252792557586427, 1.6231392299175988,
    2.5268524699070234, 2.51936429069554, 6.933634938056497, 19.41835918276137,
    18.271793488422656, 0.46319949214386513, 0.2058139980642224,
    18.63975340175838, 17.643726318153238
])

xs = []
ys = []


def f_np(x):
    y, dy = df(x)
    result = float(y), np.array(dy)
    xs.append(x)
    ys.append(float(y))
    print(-result[0])
    return result


def g(x):
    Eg_ETM = x[0]
    Chi_ETM = x[1]
    Nc_ETM = 10**x[3]
    Nv_ETM = 10**x[4]
    Eg_HTM = x[7]
    Chi_HTM = x[8]
    Nc_HTM = 10**x[10]
    Nv_HTM = 10**x[11]
    Nd_ETM = 10**x[14]
    Na_HTM = 10**x[15]

    PhiM0 = dpv.physics.flatband_wf(Nc_ETM, Nv_ETM, Eg_ETM, Chi_ETM, Nd_ETM)
    PhiML = dpv.physics.flatband_wf(Nc_HTM, Nv_HTM, Eg_HTM, Chi_HTM, -Na_HTM)

    g = -jnp.array([
        Chi_ETM - PhiM0, Chi_HTM - Chi_P, PhiML - Chi_HTM - Eg_HTM,
        Chi_HTM + Eg_HTM - Chi_P - Eg_P, Chi_P - Chi_ETM
    ])

    return g


dg = jacobian(g)
g_np = lambda x: np.array(g(jnp.array(x)))
dg_np = lambda x: np.array(dg(jnp.array(x)))

x0_np = np.array(x0)

slsqp_res = minimize(f_np,
                     x0=x0_np,
                     method="SLSQP",
                     jac=True,
                     bounds=list(zip(vl, vu)),
                     constraints=[{
                         "type": "ineq",
                         "fun": g_np,
                         "jac": dg_np
                     }],
                     options={
                         "maxiter": 50,
                         "disp": True
                     })

xs = np.array(xs)
ys = -np.array(ys)
print(slsqp_res)

from ast import literal_eval
import pickle

with open("logs/sample_psc_200iter.log", "r") as f:
    samples = -np.array(literal_eval(f.readlines()[-1]))

with open("optimize/results/psc_slsqp.pickle", "wb") as f:
    pickle.dump({"x": xs, "y": ys, "rs": samples}, f)

plt.plot(ys)
plt.show()
