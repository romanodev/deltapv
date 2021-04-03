import deltapv as dpv
from jax import numpy as jnp, value_and_grad
from jax.experimental import optimizers
import matplotlib.pyplot as plt

unknown = dpv.create_material(Eg=1.0, Chi=3.0, eps=10.0, mn=200.0, mp=200.0, Nc=3e18, Nv=3e18, A=2e4)

def get_iv(mn, mp):
    mat = dpv.objects.update(unknown, mn=mn, mp=mp)
    des = dpv.make_design(n_points=500, Ls=[1e-4, 1e-4], mats=mat, Ns=[1e17, -1e17], Snl=1e7, Snr=0, Spl=0, Spr=1e7)
    results = dpv.simulate(des)
    return results["iv"][1]

Eg0 = unknown.Eg
mn0, mp0 = unknown.mn, unknown.mp
logNc0, logNv0 = jnp.log10(unknown.Nc), jnp.log10(unknown.Nv)
J0 = get_iv(mn0, mp0)

def f(x):
    J = get_iv(*x)
    return dpv.util.dhor(J, J0)

df = value_and_grad(f)

"""# xs, obj = dpv.util.adam(df, [2.0, 2.0], lr=1e-2, b1=0.5, b2=0.5, steps=100, tol=1e-3)
xs, obj = dpv.util.gd(df, [1.0, 1.0], lr=1e-1, steps=100, tol=1e-3)

plt.plot(obj, color="black")
plt.xlabel("iterations")
plt.ylabel("$L_A$")
plt.tight_layout()
plt.savefig("loss_mnmp_gd2_lr1em1.png", dpi=300)
plt.show()

plt.plot(xs[:, 0], color="lightcoral", label="$\mu_n$")
plt.plot(xs[:, 1], color="cornflowerblue", label="$\mu_p$")
plt.axhline(logmn0, color="lightcoral", linestyle="--")
plt.axhline(logmp0, color="cornflowerblue", linestyle="--")
plt.xlabel("iterations")
plt.ylabel("$\mu$/cm$^2$V$^{-1}$s$^{-1}$")
plt.legend()
plt.tight_layout()
plt.savefig("param_mnmp_gd2_lr1em1.png", dpi=300)
plt.show()"""

L = 11
mnvec = mn0 + jnp.linspace(-50, 50, L)
mpvec = mp0 + jnp.linspace(-50, 50, L)
Z = jnp.zeros((L, L))

for i, mn in enumerate(mnvec):
    for j, mp in enumerate(mpvec):
        print(mn, mp)
        Z = Z.at[j, i].set(f([mn, mp]))

plt.contour(mnvec, mpvec, Z, levels=20)
plt.xlabel("$\mu_n$")
plt.ylabel("$\mu_p$")
plt.colorbar()
plt.tight_layout()
plt.show()
