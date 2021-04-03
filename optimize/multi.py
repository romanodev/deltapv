import deltapv as dpv
from jax import numpy as jnp, value_and_grad
from jax.experimental import optimizers
import matplotlib.pyplot as plt

material = dpv.create_material(Eg=1.0, Chi=3.0, eps=10.0, Nc=1e18, Nv=1e18, mn=130.0, mp=160.0, A=2e4)

def get_iv(**kwargs):
    """Gets the IV curve after changing the material parameters according to keyword arguments

    Returns:
        jnp.ndarray: Candidate IV curve
    """
    candidate = dpv.objects.update(material, **kwargs)
    des = dpv.make_design(n_points=500, Ls=[1e-4, 1e-4], mats=candidate, Ns=[1e17, -1e17], Snl=1e7, Snr=0, Spl=0, Spr=1e7)
    results = dpv.simulate(des)
    return results["iv"][1]

J0 = get_iv()

def f(x, measure="polar", norm=2.0):
    """Objective function.

    Args:
        x (list): Vector of parameters to pass in. It must be defined in this function how to convert the entries of x to a dictionary of parameters describing the candidate material.
        measure (str): Either "polar", "x", "y", or "xy". "polar" interpolates in polar coordinates, "x" takes the horizontal area, "y" takes the vertical area, and "xy" takes both.
        norm (float): The norm of the differences taken. Taking norm=2.0 corresponds to Euclidean distance.

    Returns:
        float: A measure of distance between the candidate and target IV curves.
    """
    params = {}
    params["mn"] = 10 ** x[0]
    params["mp"] = 10 ** x[1]
    J = get_iv(**params)
    if measure == "polar":
        res = dpv.util.dpol(J, J0, norm=norm)
    elif measure == "x":
        res = dpv.util.dhor(J, J0, norm=norm)
    elif measure == "y":
        res = dpv.util.dver(J, J0, norm=norm)
    else:
        res = dpv.util.dhor(J, J0, norm=norm) + dpv.util.dver(J, J0, norm=norm)
    return res

df = value_and_grad(f)

if __name__ == "__main__":
    # xs, obj = dpv.util.adam(df, [2.0, 2.0], lr=1, b1=0.9, b2=0.999, steps=50)
    xs, obj = dpv.util.gd(df, [2.0, 2.0], lr=1, steps=10)
    print(xs[-1])
    print(10**xs[-1, 0], 10**xs[-1, 1])
    print(obj)

    plt.plot(obj)
    plt.show()

    plt.plot(xs)
    plt.show()
