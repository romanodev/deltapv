import deltapv
import psc
from jax import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
logger = logging.getLogger("deltapv")
logger.setLevel("WARNING")
# logger.addHandler(logging.FileHandler("logs/____.log"))

x = np.array([
    2.350895882256301, 4.11097967520195, 18.903984677363855,
    18.482564864268923, 19.220892716843395, 91.07733066021372,
    34.3829527039204, 2.3965730062122157, 2.6511689197537693,
    13.846682523469422, 18.605637739262203, 18.457692800000242,
    184.64653199538557, 483.68218652565844, 18.880656556865407,
    18.440501506781022
])

n_dx = 30
dd = 0.1
objs = np.zeros((psc.n_params, n_dx))
derivs = np.zeros((psc.n_params, n_dx))

for j, dx in tqdm(enumerate(np.linspace(-dd, dd, n_dx))):
    for i in range(psc.n_params):
        xc = x.at[i].add(dx)
        y, dy = psc.vagf(xc)
        objs = objs.at[i, j].set(y)
        derivs = derivs.at[i, j].set(dy[i])

print(objs)
print(derivs)
np.save("debug/fdtest_objs_adjoint.npy", objs)
np.save("debug/fdtest_derivs_adjoint.npy", derivs)

"""
eff0 = -15.96076275050233
alpha = np.linspace(-dd, dd, n_dx)
objs = np.load("debug/fdtest_objs_hr.npy")
objs = objs - eff0
derivs = np.load("debug/fdtest_derivs_hr.npy")

fig, axs = plt.subplots(4, 4, sharex=True)

for i in range(psc.n_params):
    j, k = i // 4, i % 4
    axs[j, k].plot(alpha, objs[i], color="blue")
    twax = axs[j, k].twinx()
    twax.plot(alpha, derivs[i], color="red")
    twax.plot(alpha[:-1], np.diff(objs[i]) / (2 * dd / 29), linestyle="--", color="red")
    axs[j, k].ticklabel_format(useMathText=True, useOffset=True, scilimits=(-1, 1))
    twax.ticklabel_format(useMathText=True, useOffset=True, scilimits=(-1, 1))
    axs[j, k].set_title(psc.PARAMS[i], pad=20)
plt.subplots_adjust(left=0.05,
                    bottom=0.05, 
                    right=0.95, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
plt.show()
"""