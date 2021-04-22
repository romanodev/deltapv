import deltapv as dpv
import psc
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def f(params, pot_ini):
    x = params[:-1]
    v = params[-1]
    vprint = jnp.round(v, 2)
    print("evaluating for V = {:.2f}".format(vprint))
    des = psc.x2des(x)
    eff, pot = dpv.eff_at_bias(des, v, pot_ini, verbose=False)
    return -eff, pot


df = jax.value_and_grad(f, has_aux=True)  # returns (p, pot), dp


class StatefulSolver:
    def __init__(self, params_init):
        self.count = 0
        self.x = params_init[:-1]
        self.v = params_init[-1]
        _, self.guess = dpv.ramp_up(psc.x2des(jnp.array(self.x)), self.v)

    def eval(self, params):
        x = params[:-1]
        v = params[-1]
        print(f"called eval with x = {x} and v = {v}")
        nsteps = int(np.ceil(np.abs(v - self.v) / 0.01))
        print(f"    currently at {self.count} total pde solves")
        print(f"    splitting into {nsteps} steps")
        xs = np.linspace(self.x, x, nsteps + 1)
        vs = np.linspace(self.v, v, nsteps + 1)
        for xr, vr in zip(xs, vs):
            (eff, self.guess), deff = df(jnp.append(jnp.array(xr), vr), self.guess)
            self.x = xr
            self.v = vr
        self.count += nsteps
        print(f"    returning {float(eff)}")
        return float(eff), np.array(deff)


bounds = [(1, 5), (1, 5), (1, 20), (17, 20), (17, 20), (0, 3), (0, 3), (1, 5),
          (1, 5), (1, 20), (17, 20), (17, 20), (0, 3), (0, 3), (17, 20),
          (17, 20), (0, None)]

params_init = np.array([
    1.661788237392516, 4.698293002285373, 19.6342803183675, 18.83471869026531,
    19.54569869328745, 0.7252792557586427, 1.6231392299175988,
    2.5268524699070234, 2.51936429069554, 6.933634938056497, 19.41835918276137,
    18.271793488422656, 0.46319949214386513, 0.2058139980642224,
    18.63975340175838, 17.643726318153238, 0.5
])

solver = StatefulSolver(params_init)

slsqp_res = minimize(solver.eval,
                     x0=params_init,
                     method="SLSQP",
                     jac=True,
                     bounds=bounds,
                     constraints=[{
                         "type": "ineq",
                         "fun": psc.g_np,
                         "jac": psc.dg_np
                     }],
                     options={
                         "maxiter": 50,
                         "disp": True
                     })
