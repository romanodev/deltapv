import deltapv as dpv
from jax import numpy as jnp, ops
import matplotlib.pyplot as plt
import argparse

t1 = 2.5e-6
t2 = 4e-4
dd = 1e-7

grid = jnp.concatenate(
    (jnp.linspace(0, dd, 10,
                 endpoint=False), jnp.linspace(dd, t1 - dd, 50, endpoint=False),
     jnp.linspace(t1 - dd, t1 + dd, 10, endpoint=False),
     jnp.linspace(t1 + dd, (t1 + t2) - dd, 100,
                 endpoint=False), jnp.linspace((t1 + t2) - dd, (t1 + t2), 10)))

CdS = dpv.create_material(Nc=2.2e18,
                          Nv=1.8e19,
                          Eg=2.4,
                          eps=10,
                          Et=0,
                          mn=100,
                          mp=25,
                          tn=1e-8,
                          tp=1e-13,
                          Chi=4.,
                          A=1e4)

CdTe = dpv.create_material(Nc=8e17,
                           Nv=1.8e19,
                           Eg=1.5,
                           eps=9.4,
                           Et=0,
                           mn=320,
                           mp=40,
                           tn=5e-9,
                           tp=5e-9,
                           Chi=3.9,
                           A=1e4)

des = dpv.make_design(n_points=500, Ls=[t1, t2], mats=[CdS, CdTe], Ns=[1e17, -1e15], Snl=1.16e7, Snr=1.16e7, Spl=1.16e7, Spr=1.16e7)
ls = dpv.incident_light()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save")
    args = parser.parse_args()

    results = dpv.simulate(des, ls)
    v, j = results["iv"]

    dpv.plot_iv_curve(v, j)
    dpv.plot_bars(des)
    dpv.plot_band_diagram(des, results["eq"], eq=True)
    dpv.plot_band_diagram(des, results["Voc"])
    dpv.plot_charge(des, results["eq"])
    eff = results["eff"] * 100
    print(f"efficiency: {eff}%")
