import deltapv as dpv
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
import pickle

with open("outputs/discovery.pickle", "rb") as f:
    result = pickle.load(f)

x = result["x"]
f = result["f"]
dfdx = result["dfdx"]

plt.plot(f, color="black")
plt.xlabel("iterations")
plt.ylabel("$R(\hat{J}, J^*)$")
plt.tight_layout()
plt.savefig("optimize/results/multi_obj.png")
plt.show()

plt.plot(x[:, 0], color="cornflowerblue", label="$\log_{10} \mu_p$")
plt.axhline(jnp.log10(160), color="cornflowerblue", linestyle="--")
plt.plot(x[:, 1], color="lightcoral", label="$E_g$")
plt.axhline(1, color="lightcoral", linestyle="--")
plt.xlabel("iterations")
plt.ylabel("parameter")
plt.legend()
plt.tight_layout()
plt.savefig("optimize/results/multi_param.png")
plt.show()

with open("optimize/results/psc_slsqp.pickle", "rb") as f:
    result = pickle.load(f)

x = result["x"]
y = result["y"]
rs = result["rs"]

from scipy.stats import gaussian_kde

plt.plot(y, color="black", marker=".")
plt.axhline(y[-1], color="black", linestyle="--")
plt.ylabel("PCE / %")
plt.xlabel("function evaluations")

plt.xlim(left=0, right=len(y) - 1)
plt.ylim(bottom=0, top=25)
plt.tight_layout()
plt.show()

kde = gaussian_kde(rs)
pos = np.linspace(0, 25, 500)
prob = kde.evaluate(pos)
plt.fill_betweenx(pos, 500 * prob, color="lightgrey")

plt.scatter(np.arange(len(rs)), rs, color="black")
plt.axhline(max(rs), color="black", linestyle="--")
plt.axhline(y[-1], color="lightcoral", linestyle="--")
plt.xlim(0, len(rs) - 1)
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()

from psc import x2des

dpv.plot_bars(x2des(x[-1]))
