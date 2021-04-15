import deltapv as dpv
from jax import numpy as jnp, value_and_grad
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

material = dpv.create_material(Eg=1.0,
                               Chi=3.0,
                               eps=10.0,
                               Nc=1e18,
                               Nv=1e18,
                               mn=130.0,
                               mp=160.0,
                               A=2e4)


def get_iv(**kwargs):
    candidate = dpv.objects.update(material, **kwargs)
    des = dpv.make_design(n_points=500,
                          Ls=[1e-4, 1e-4],
                          mats=candidate,
                          Ns=[1e17, -1e17],
                          Snl=1e7,
                          Snr=0,
                          Spl=0,
                          Spr=1e7)
    results = dpv.simulate(des, verbose=False)
    return results["iv"][1]


J0 = get_iv()


def f(x):
    params = {}
    params["mp"] = 10**x[0]
    params["Eg"] = x[1]
    J = get_iv(**params)
    res = dpv.util.dpol(J, J0)
    return res


df = value_and_grad(f)


xs = []
ys = []
dys = []


def f_np(x):
    y, dy = df(x)
    result = float(y), np.array(dy)
    xs.append(x)
    ys.append(float(y))
    dys.append(np.array(dy))
    print(result[0])
    return result


if __name__ == "__main__":
    slsqp_res = minimize(f_np,
                         x0=np.array([2.0, 1.2]),
                         method="SLSQP",
                         jac=True,
                         bounds=[(1.0, 3.0), (0.5, 2.0)],
                         options={
                             "maxiter": 50,
                             "disp": True
                         })

    print(slsqp_res)

    xs = np.array(xs)
    ys = np.array(ys)

    plt.plot(ys, marker=".", color="black")
    plt.xlabel("iterations")
    plt.ylabel("$R(\hat{J}, J^*)$")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("optimize/results/multi_slsqp_obj.png")
    plt.show()

    plt.plot(xs[:, 0], color="cornflowerblue", marker=".", label="$\log_{10} \mu_p$")
    plt.axhline(np.log10(160), color="cornflowerblue", linestyle="--")
    plt.plot(xs[:, 1], color="lightcoral", marker=".", label="$E_g$")
    plt.axhline(1, color="lightcoral", linestyle="--")
    plt.xlabel("iterations")
    plt.ylabel("parameter")
    plt.legend()
    plt.tight_layout()
    plt.savefig("optimize/results/multi_slsqp_param.png")
    plt.show()
