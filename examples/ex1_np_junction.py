import deltapv as dpv
import jax.numpy as jnp
import argparse
import matplotlib.pyplot as plt

material = dpv.create_material(Chi=3.9,
                               Eg=1.5,
                               eps=9.4,
                               Nc=8e17,
                               Nv=1.8e19,
                               mn=100,
                               mp=100,
                               Et=0,
                               tn=1e-8,
                               tp=1e-8,
                               A=2e4)
des = dpv.make_design(n_points=500,
                      Ls=[1e-4, 1e-4],
                      mats=material,
                      Ns=[1e17, -1e17],
                      Snl=1e7,
                      Snr=0,
                      Spl=0,
                      Spr=1e7)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save")
    args = parser.parse_args()

    results = dpv.simulate(des)

    dpv.plot_iv_curve(*results["iv"])
    dpv.plot_bars(des)
    dpv.plot_band_diagram(des, results["eq"], eq=True)
    dpv.plot_charge(des, results["eq"])
