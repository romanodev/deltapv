import deltapv as dpv
import psc
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

bounds = [(1, 5), (1, 5), (1, 20), (17, 20), (17, 20), (0, 3), (0, 3), (1, 5),
          (1, 5), (1, 20), (17, 20), (17, 20), (0, 3), (0, 3), (17, 20),
          (17, 20), (0, None)]

x_init = np.array([
    1.661788237392516, 4.698293002285373, 19.6342803183675, 18.83471869026531,
    19.54569869328745, 0.7252792557586427, 1.6231392299175988,
    2.5268524699070234, 2.51936429069554, 6.933634938056497, 19.41835918276137,
    18.271793488422656, 0.46319949214386513, 0.2058139980642224,
    18.63975340175838, 17.643726318153238
])

opt = dpv.util.StatefulOptimizer(x_init=x_init,
                                 convr=psc.x2des,
                                 constr=psc.g,
                                 bounds=bounds)

results = opt.optimize(niters=100)
y = -100 * opt.get_growth()
plt.plot(y, color="black", marker=".")
plt.axhline(y[-1], color="black", linestyle="--")
plt.xlabel("PDE solves")
plt.ylabel("PCE / %")
plt.xlim(0, len(y) - 1)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig("optimize/result.png", dpi=300)
plt.show()

dpv.plot_bars(psc.x2des(opt.x), filename="optimize/bars.png")
