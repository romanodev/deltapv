import deltapv
import jax
from jax import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
logger = logging.getLogger("deltapv")
logger.setLevel("WARNING")

L = 1e-4
JPOS = 5e-5
N = 500
S = 1e7
grid = np.linspace(0, L, N)
base = deltapv.simulator.create_design(grid)
Sidict = {
    "eps": 11.7,
    "Chi": 4.05,
    "Eg": 1.12,
    "Nc": 3.2e19,
    "Nv": 1.8e19,
    "mn": 1400.0,
    "mp": 450.0,
    "tn": 1e-8,
    "tp": 1e-8,
    "A": 20000.0
}


def f(matdict):
    mat = deltapv.materials.create_material(**matdict)
    des = deltapv.simulator.add_material(base, mat, lambda _: True)
    des = deltapv.simulator.single_pn_junction(des, 1e18, -1e18, JPOS)
    des = deltapv.simulator.contacts(des, S, S, S, S)
    ls = deltapv.simulator.incident_light()

    results = deltapv.simulator.simulate(des, ls)
    eta = 100 * results["eff"]

    return eta


df = jax.value_and_grad(f)


def g(matdict):
    mat = deltapv.materials.create_material(**matdict)
    des = deltapv.simulator.add_material(base, mat, lambda _: True)
    des = deltapv.simulator.single_pn_junction(des, 1e18, -1e18, JPOS)
    des = deltapv.simulator.contacts(des, S, S, S, S)
    j = deltapv.simulator.solve_isc(des)
    return j


dg = jax.value_and_grad(g)


def test_all():
    eta, deta = df(Sidict)
    print(eta)
    print(deta)

    for key, _ in Sidict.items():
        print(key)
        matnew = Sidict.copy()
        dd = 1e-3 / np.abs(deta[key])
        print(f"dd = {dd}")
        matnew[key] += dd
        etanew = f(matnew)
        fdgrad = (etanew - eta) / dd
        print(f"etanew = {etanew}")
        print(f"empgrad = {fdgrad}")
        print(f"jaxgrad = {deta[key]}")


def test_all_current(targetdj=1e-4):
    j, dj = dg(Sidict)
    print(j)
    print(dj)

    for key, _ in Sidict.items():
        print(key)
        matnew = Sidict.copy()
        dd = targetdj / np.abs(dj[key])
        print(f"dd = {dd}")
        matnew[key] += dd
        jnew = g(matnew)
        fdgrad = (jnew - j) / dd
        print(f"jnew = {jnew}")
        print(f"empgrad = {fdgrad}")
        print(f"jaxgrad = {dj[key]}")


def test_param(key="Eg", targetdeff=1e-3, n_points=31, dxmax=None):
    eta, deta = df(Sidict)
    print(eta)
    print(deta)
    etas = []
    grads = []
    if dxmax is None:
        dxmax = targetdeff / np.abs(deta[key])
    dxs = np.linspace(-1, 1, n_points) * dxmax
    for dx in tqdm(dxs):
        matnew = Sidict.copy()
        matnew[key] += dx
        etanew, gradnew = df(matnew)
        print(etanew)
        print(gradnew)
        etas.append(etanew)
        grads.append(gradnew[key])

    etas = np.array(etas)
    grads = np.array(grads)

    plt.plot(dxs, etas)
    plt.show()

    plt.plot(dxs, grads)
    plt.show()

    plt.plot(dxs[:-1], np.diff(etas) / np.diff(dxs))
    plt.show()


def test_param_current(key="Eg", targetdj=1e-4, n_points=31):
    j, dj = dg(Sidict)
    print(j)
    print(dj)
    js = []
    grads = []
    dxmax = targetdj / np.abs(dj[key])
    dxs = np.linspace(-1, 1, n_points) * dxmax
    for dx in dxs:
        matnew = Sidict.copy()
        matnew[key] += dx
        jnew, gradnew = dg(matnew)
        print(jnew)
        print(gradnew)
        js.append(jnew)
        grads.append(gradnew[key])

    js = np.array(js)
    grads = np.array(grads)

    plt.plot(dxs, js)
    plt.show()

    plt.plot(dxs, grads)
    plt.show()

    plt.plot(dxs[:-1], np.diff(js) / np.diff(dxs))
    plt.show()


def test_lower(matdict):
    mat = deltapv.materials.create_material(**matdict)
    des = deltapv.simulator.add_material(base, mat, lambda _: True)
    des = deltapv.simulator.single_pn_junction(des, 1e18, -1e18, JPOS)
    des = deltapv.simulator.contacts(des, S, S, S, S)
    ls = deltapv.simulator.incident_light()
    cell = deltapv.simulator.init_cell(des, ls)

    bound_eq = deltapv.bcond.boundary_eq(cell)
    guess_eq = deltapv.solver.eq_guess(cell, bound_eq)
    pot_eq = deltapv.solver.solve_eq(cell, bound_eq, guess_eq)

    guess = deltapv.solver.ooe_guess(cell, pot_eq)
    bound = deltapv.bcond.boundary(cell, 0.)
    pot = deltapv.solver.solve(cell, bound, guess)

    return pot


def test_pot_Eg():

    dpotdeg = jax.jacfwd(test_lower)(Sidict)

    dd = 1e-4
    potold = test_lower(Sidict)
    newdict = Sidict.copy()
    newdict["Eg"] += dd

    potnew = test_lower(newdict)
    fdphin = (potnew.phi_n - potold.phi_n) / dd
    fdphip = (potnew.phi_p - potold.phi_p) / dd
    fdphi = (potnew.phi - potold.phi) / dd

    fig, axs = plt.subplots(3)

    axs[0].plot(fdphin, color="red", alpha=0.5, linewidth=3)
    axs[0].plot(dpotdeg.phi_n["Eg"], color="blue", alpha=0.5, linewidth=3)
    axs[0].set_title("phi_n")

    axs[1].plot(fdphip, color="red", alpha=0.5, linewidth=3)
    axs[1].plot(dpotdeg.phi_p["Eg"], color="blue", alpha=0.5, linewidth=3)
    axs[1].set_title("phi_p")

    axs[2].plot(fdphi, color="red", alpha=0.5, linewidth=3)
    axs[2].plot(dpotdeg.phi["Eg"], color="blue", alpha=0.5, linewidth=3)
    axs[2].set_title("phi")

    plt.show()


if __name__ == "__main__":
    test_all()
