from . import scales
from . import eta
from . import optical
from . import sun
from . import initial_guess
from . import IV

import matplotlib.pyplot as plt
import jax.numpy as np
from jax import ops, lax


class JAXPV(object):
    def __init__(self, grid):

        self.gparams = {"grid": scales.d, "dgrid": scales.d}
        self.vparams = {
            "eps": 1,
            "Chi": scales.E,
            "Eg": scales.E,
            "Nc": scales.n,
            "Nv": scales.n,
            "mn": scales.m,
            "mp": scales.m,
            "Ndop": scales.n,
            "Et": scales.E,
            "tn": scales.t,
            "tp": scales.t,
            "Br": 1 / (scales.t * scales.n),
            "Cn": 1 / (scales.t * scales.n**2),
            "Cp": 1 / (scales.t * scales.n**2),
            "A": 1 / scales.d,
            "G": scales.U
        }
        self.sparams = {
            "Snl": scales.v,
            "Snr": scales.v,
            "Spl": scales.v,
            "Spr": scales.v
        }
        self.oparams = {"Lambda": 1, "P_in": 1}
        self.opt = "user"

        N = grid.size
        self.data = {
            key: np.zeros(N, dtype=np.float64)
            for key in self.vparams
        }
        self.data.update({key: 0. for key in self.sparams})
        self.data.update(
            {key: np.zeros(1, dtype=np.float64)
             for key in self.oparams})
        self.data["grid"] = grid / self.gparams["grid"]
        self.data["dgrid"] = np.diff(self.data["grid"])

    def add_material(self, properties, subgrid):

        for prop in properties:
            if prop in self.vparams:
                self.data[prop] = ops.index_update(
                    self.data[prop], subgrid,
                    properties[prop] / self.vparams[prop])

    def contacts(self, Snl, Snr, Spl, Spr):

        self.data["Snl"] = np.float64(Snl / self.sparams["Snl"])
        self.data["Snr"] = np.float64(Snr / self.sparams["Snr"])
        self.data["Spl"] = np.float64(Spl / self.sparams["Spl"])
        self.data["Spr"] = np.float64(Spr / self.sparams["Spr"])

    def single_pn_junction(self, Nleft, Nright, junction_position):

        self.data["Ndop"] = np.where(
            self.data["grid"] < junction_position / self.gparams["grid"],
            Nleft / self.vparams["Ndop"], Nright / self.vparams["Ndop"])

    def doping_profile(self, doping, subgrid):

        self.data["Ndop"] = ops.index_update(self.data["Ndop"], subgrid,
                                             doping * self.vparams["Ndop"])

    def incident_light(self, kind="sun", Lambda=None, P_in=None):

        if kind == "sun":
            self.data["Lambda"] = sun.wavelength
            self.data["P_in"] = sun.power

        elif kind == "white":
            if Lambda is None:
                self.data["Lambda"] = np.linspace(400., 800., num=5)
                self.data["P_in"] = 200. * np.ones(5, dtype=np.float64)
            else:
                self.data["Lambda"] = Lambda
                power = 1000. / Lambda.size
                self.data["P_in"] = power * np.ones_like(Lambda,
                                                         dtype=np.float64)

        elif kind == "monochromatic":
            if Lambda is None:
                self.data["Lambda"] = np.array([400.])
            else:
                self.data["Lambda"] = Lambda
            self.data["P_in"] = np.array([1000.])

        elif kind == "user":
            if Lambda is None or P_in is None:
                raise Exception("Lambda or Pin not defined")
            else:
                self.data["Lambda"] = Lambda
                self.data["P_in"] = P_in * 1000. / np.sum(P_in)

    def optical_G(self, kind="direct", G=None):

        self.opt = kind
        if kind == "user":
            self.data["G"] = np.float64(G / self.vparams["G"])


def efficiency(data, opt):

    Vincr = eta.Vincrement(data)
        
    data["G"] = lax.cond(opt == "user",
                         lambda data: data["G"],
                         lambda data: optical.compute_G(data),
                         data)
    
    return eta.comp_eff(data, Vincr)


def IV_curve(data, opt):

    Vincr = eta.Vincrement(data)
        
    data["G"] = lax.cond(opt == "user",
                         lambda data: data["G"],
                         lambda data: optical.compute_G(data),
                         data)

    current = scales.J * IV.calc_IV(data, Vincr)

    voltages = scales.E * np.linspace(
        start=0, stop=(current.size - 1) * Vincr, num=current.size)

    return voltages, current


def solve_equilibrium(data):
    
    Vincr = eta.Vincrement(data)
        
    data["G"] = lax.cond(opt == "user",
                         lambda data: data["G"],
                         lambda data: optical.compute_G(data),
                         data)

    phi_ini = initial_guess.eq_init_phi(data)
    phi_eq = solver.solve_eq(data, phi_ini)

    N = data["grid"].size
    result = {}
    result["phi_n"] = np.zeros(N, dtype=np.float64)
    result["phi_p"] = np.zeros(N, dtype=np.float64)
    result["phi"] = scales.E * phi_eq
    result["n"] = scales.n * physics.n(self.data, np.zeros(N),
                                         phi_eq)
    result["p"] = scales.n * physics.p(self.data, np.zeros(N),
                                         phi_eq)
    result["Jn"] = np.zeros(N - 1, dtype=np.float64)
    result["Jp"] = np.zeros(N - 1, dtype=np.float64)

    return result


def solve_bias(data, V):
    
    Vincr = eta.Vincrement(data)
        
    data["G"] = lax.cond(opt == "user",
                         lambda data: data["G"],
                         lambda data: optical.compute_G(data),
                         data)

    phi_ini = initial_guess.eq_init_phi(data)
    phi_eq = solver.solve_eq(data, phi_ini)

    N = data["grid"].size
    result = {}
    V_dim = V / scales.E
    num_steps = V_dim // Vincr

    phis = np.concatenate((np.zeros(2 * N), phi_eq), axis=0)
    neq_0 = self.data["Nc"][0] * np.exp(self.data["Chi"][0] +
                                        phi_eq[0])
    neq_L = self.data["Nc"][-1] * np.exp(self.data["Chi"][-1] +
                                         phi_eq[-1])
    peq_0 = self.data["Nv"][0] * np.exp(-self.data["Chi"][0] -
                                        self.data["Eg"][0] - phi_eq[0])
    peq_L = self.data["Nv"][-1] * np.exp(-self.data["Chi"][-1] -
                                         self.data["Eg"][-1] -
                                         phi_eq[-1])

    volt = [i * Vincr for i in range(num_steps)]
    
    volt.append(V_dim)

    for v in volt:
        sol = solver.solve(self.data, neq_0, neq_L, peq_0, peq_L, phis)
        phis = ops.index_update(sol, ops.index[-1], phi_eq[-1] + v)

    result["phi_n"] = scales.E * phis[0:N]
    result["phi_p"] = scales.E * phis[N:2 * N]
    result["phi"] = scales.E * phis[2 * N:]
    result["n"] = scales.n * physics.n(self.data, phis[0:N],
                                         phis[2 * N:])
    result["p"] = scales.n * physics.p(self.data, phis[N:2 * N],
                                         phis[2 * N:])
    result["Jn"] = scales.J * current.Jn(self.data, phis[0:N],
                                           phis[2 * N:])
    result["Jp"] = scales.J * current.Jp(self.data, phis[N:2 * N],
                                           phis[2 * N:])

    return result


def plot_IV(voltages, current):
    
    fig = plt.figure()
    plt.plot(voltages, current, color="blue", marker=".")
    plt.xlabel("Voltage (V)")
    plt.ylabel("current (A.cm-2)")
    if title is not None:
        plt.savefig(title)
    plt.show()


def plot_band_diagram(self, result, title=None):

    Ec = -scales.E * self.data["Chi"] - result["phi"]
    Ev = -scales.E * self.data["Chi"] - scales.E * self.data[
        "Eg"] - result["phi"]
    fig = plt.figure()
    plt.plot(scales.d * self.data["grid"],
             Ec,
             color="red",
             label="conduction band",
             linestyle="dashed")
    plt.plot(scales.d * self.data["grid"],
             Ev,
             color="blue",
             label="valence band",
             linestyle="dashed")
    plt.plot(scales.d * self.data["grid"],
             result["phi_n"],
             color="red",
             label="e- quasiFermi energy")
    plt.plot(scales.d * self.data["grid"],
             result["phi_p"],
             color="blue",
             label="hole quasiFermi energy")
    plt.xlabel("thickness (cm)")
    plt.ylabel("energy (eV)")
    plt.legend()
    plt.show()
    if title is not None:
        plt.savefig(title)


def plot_concentration_profile(self, result, title=None):

    fig = plt.figure()
    plt.yscale("log")
    plt.plot(scales.d * self.data["grid"],
             result["n"],
             color="red",
             label="e-")
    plt.plot(scales.d * self.data["grid"],
             result["p"],
             color="blue",
             label="hole")
    plt.xlabel("thickness (cm)")
    plt.ylabel("concentration (cm-3)")
    plt.legend()
    plt.show()
    if title is not None:
        plt.savefig(title)


def plot_current_profile(self, result, title=None):

    fig = plt.figure()
    plt.plot(scales.d * 0.5 *
             (self.data["grid"][1:] + self.data["grid"][:-1]),
             result["Jn"],
             color="red",
             label="e-")
    plt.plot(scales.d * 0.5 *
             (self.data["grid"][1:] + self.data["grid"][:-1]),
             result["Jp"],
             color="blue",
             label="hole")
    plt.plot(scales.d * 0.5 *
             (self.data["grid"][1:] + self.data["grid"][:-1]),
             result["Jn"] + result["Jp"],
             color="green",
             label="total",
             linestyle="dashed")
    plt.xlabel("thickness (cm)")
    plt.ylabel("current (A.cm-2)")
    plt.legend()
    plt.show()
    if title is not None:
        plt.savefig(title)
