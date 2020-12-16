import jaxpv
import jax.numpy as np
from jax import ops
import matplotlib.pyplot as plt
import argparse


def jaxpv_pn_hetero(G=None):
    t1 = 25 * 1e-7  # thickness of CdS
    t2 = 4 * 1e-4  # thickness of CdTe

    # Heterojunctions require dense mesh near the interface
    dd = 1e-7

    x = np.concatenate((
        np.linspace(0, dd, 10, endpoint=False),  # L contact interface
        np.linspace(dd, t1 - dd, 50, endpoint=False),  # material 1
        np.linspace(t1 - dd, t1 + dd, 10, endpoint=False),  # interface 1
        np.linspace(t1 + dd, (t1 + t2) - dd, 100,
                    endpoint=False),  # material 2
        np.linspace((t1 + t2) - dd, (t1 + t2), 10)))  # R contact interface

    simu = jaxpv.JAXPV(x)

    CdS = {
        "Nc": 2.2e18,
        "Nv": 1.8e19,
        "Eg": 2.4,
        "eps": 10,
        "Et": 0,
        "mn": 100,
        "mp": 25,
        "tn": 1e-8,
        "tp": 1e-13,
        "Chi": 4.
    }

    CdTe = {
        "Nc": 8e17,
        "Nv": 1.8e19,
        "Eg": 1.5,
        "eps": 9.4,
        "Et": 0,
        "mn": 320,
        "mp": 40,
        "tn": 5e-9,
        "tp": 5e-9,
        "Chi": 3.9
    }

    CdS_region = np.argwhere(x < t1).flatten()
    CdTe_region = np.argwhere(x >= t1).flatten()
    simu.add_material(CdS, CdS_region)
    simu.add_material(CdTe, CdTe_region)

    dope = np.where(x < t1, 1e17, -1e15)
    simu.doping_profile(dope, np.arange(len(x)))

    Scontact = 1.16e7
    simu.contacts(Scontact, Scontact, Scontact, Scontact)

    if G is None:
        phi0 = 1e17
        alpha = 2.3e4
        G = phi0 * alpha * np.exp(-alpha * x)
    else:
        G = G * np.ones(x.shape)

    simu.optical_G("user", G)

    voltages, j = jaxpv.IV_curve(simu.data, simu.opt)

    return voltages, j


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save")
    parser.add_argument("--G")
    args = parser.parse_args()

    G = float(args.G) if args.G else None

    voltages, j = jaxpv_pn_hetero(G=G)

    plt.plot(voltages, j, "-o")
    plt.xlabel("Voltage [V]")
    plt.ylabel("Current [A/cm^2]")
    plt.title("JAXPV pn-heterojunction example")
    if args.save is not None:
        plt.savefig(args.save)
    plt.show()
