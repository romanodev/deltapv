from . import scaling
from . import efficiency
from . import optical
from . import sun
from . import initial_guess

import jax.numpy as np
from jax import ops

scale = scaling.scales()


class JAXPV(object):

    def __init__(self, grid):
        
        self.gparams = {"grid": scale["d"],
                        "dgrid": scale["d"]}
        self.vparams = {"eps": 1,
                        "Chi": scale["E"],
                        "Eg": scale["E"],
                        "Nc": scale["n"],
                        "Nv": scale["n"],
                        "mn": scale["m"],
                        "mp": scale["m"],
                        "Ndop": scale["n"],
                        "Et": scale["E"],
                        "tn": scale["t"],
                        "tp": scale["t"],
                        "Br": 1 / (scale["t"] * scale["n"]),
                        "Cn": 1 / (scale["t"] * scale["n"] ** 2),
                        "Cp": 1 / (scale["t"] * scale["n"] ** 2),
                        "A": scale["d"],
                        "G": scale["U"]}
        self.sparams = {"Snl": scale["v"],
                        "Snr": scale["v"],
                        "Spl": scale["v"],
                        "Spr": scale["v"]}
        self.oparams = {"Lambda": 1,
                        "P_in": 1}
        self.opt = "user"
        
        N = grid.size
        self.data = {key: np.zeros(N, dtype=np.float64) for key in self.vparams}
        self.data.update({key: 0. for key in self.sparams})
        self.data.update({key: np.zeros(1, dtype=np.float64) for key in self.oparams})
        self.data["grid"] = grid / self.gparams["grid"]
        self.data["dgrid"] = np.diff(self.data["grid"])

    def add_material(self, properties, subgrid):
        
        for prop in properties:
            if prop in self.vparams:
                self.data[prop] = ops.index_update(self.data[prop],
                                                   subgrid,
                                                   properties[prop] / self.vparams[prop])

    def contacts(self, Snl, Snr, Spl, Spr):
        
        self.data["Snl"] = np.float64(Snl / self.sparams["Snl"])
        self.data["Snr"] = np.float64(Snr / self.sparams["Snr"])
        self.data["Spl"] = np.float64(Spl / self.sparams["Spl"])
        self.data["Spr"] = np.float64(Spr / self.sparams["Spr"])

    def single_pn_junction(self, Nleft, Nright, junction_position):

        self.data["Ndop"] = np.where(self.grid < junction_position,
                                    Nleft * self.vparams["Ndop"],
                                    Nright * self.vparams["Ndop"])

    def doping_profile(self, doping, subgrid):
        
        self.data["Ndop"] = ops.index_update(self.data["Ndop"],
                                            subgrid,
                                            doping * self.vparams["Ndop"])

    def incident_light(self, kind="sun", Lambda=None, P_in=None):
        
        if kind == "sun":
            Lambda_sun, P_in_sun = sun.solar()
            self.data["Lambda"] = Lambda_sun
            self.data["P_in"] = P_in_sun
            
        elif kind == "white":
            if Lambda is None:
                self.data["Lambda"] = np.linspace(400., 800., num=5)
                self.data["P_in"] = 200. * np.ones(5, dtype=np.float64)
            else:
                self.data["Lambda"] = Lambda
                power = 1000. / Lambda.size
                self.data["P_in"] = power * np.ones_like(Lambda, dtype=np.float64)
                
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
            self.data["G"] = np.float64(G * self.vparams["G"])

    def efficiency(self):
        
        Vincr = efficiency.Vincrement(self.data)

        if self.opt != "user":
            self.data["G"] = optical.compute_G(data)

        return efficiency.comp_eff(data, Vincr)
    
    # TODO: below
    
    def grad_efficiency(self, jit=True):
        
        Vincr = efficiency.Vincrement(np.array(self.Chi), np.array(self.Eg),
                           np.array(self.Nc), np.array(self.Nv),
                           np.array(self.Ndop))
        if self.opt == "user":
            G_used = np.array(self.G)
        else:
            G_used = compute_G(np.array(self.grid[1:] - self.grid[:-1]),
                               np.array(self.Eg), np.array(self.Lambda),
                               np.array(self.P_in), np.array(self.A))

        if jit:
            current, cur_grad = grad_IV(
                np.array(self.grid[1:] - self.grid[:-1]), Vincr,
                np.array(self.eps), np.array(self.Chi), np.array(self.Eg),
                np.array(self.Nc), np.array(self.Nv), np.array(self.Ndop),
                np.array(self.mn), np.array(self.mp), np.array(self.Et),
                np.array(self.tn), np.array(self.tp), np.array(self.Br),
                np.array(self.Cn), np.array(self.Cp), np.array(self.Snl),
                np.array(self.Spl), np.array(self.Snr), np.array(self.Spr),
                G_used)
            voltages = np.linspace(start=0,
                                   stop=len(current) * Vincr,
                                   num=len(current))
            coef = scale["E"] * scale["J"] * 10.
            P = coef * voltages * current
            Pmax = np.max(P)
            index = np.where(P == Pmax)[0][0]
            index = 0
            eff = Pmax
            result = {}
            result["eps"] = cur_grad[index]["eps"] * coef * voltages[index]
            result["Chi"] = cur_grad[index]["Chi"] * coef * voltages[index]
            result["Eg"] = cur_grad[index]["Eg"] * coef * voltages[index]
            result["Nc"] = cur_grad[index]["Nc"] * coef * voltages[index]
            result["Nv"] = cur_grad[index]["Nv"] * coef * voltages[index]
            result["Ndop"] = cur_grad[index]["Ndop"] * coef * voltages[index]
            result["mn"] = cur_grad[index]["mn"] * coef * voltages[index]
            result["mp"] = cur_grad[index]["mp"] * coef * voltages[index]
            result["Et"] = cur_grad[index]["Et"] * coef * voltages[index]
            result["tn"] = cur_grad[index]["tn"] * coef * voltages[index]
            result["tp"] = cur_grad[index]["tp"] * coef * voltages[index]
            result["Br"] = cur_grad[index]["Br"] * coef * voltages[index]
            result["Cn"] = cur_grad[index]["Cn"] * coef * voltages[index]
            result["Cp"] = cur_grad[index]["Cp"] * coef * voltages[index]
            result["Snl"] = cur_grad[index]["Snl"] * coef * voltages[index]
            result["Spl"] = cur_grad[index]["Spl"] * coef * voltages[index]
            result["Snr"] = cur_grad[index]["Snr"] * coef * voltages[index]
            result["Spr"] = cur_grad[index]["Spr"] * coef * voltages[index]
            result["G"] = cur_grad[index]["G"] * coef * voltages[index]

            return eff, result

        else:
            gradeff = grad(efficiency,
                           argnums=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                    15, 16, 17, 18, 19, 20))
            deriv = gradeff(np.array(self.grid[1:] - self.grid[:-1]), Vincr,
                            np.array(self.eps), np.array(self.Chi),
                            np.array(self.Eg), np.array(self.Nc),
                            np.array(self.Nv), np.array(self.Ndop),
                            np.array(self.mn), np.array(self.mp),
                            np.array(self.Et), np.array(self.tn),
                            np.array(self.tp), np.array(self.Br),
                            np.array(self.Cn), np.array(self.Cp),
                            np.array(self.Snl), np.array(self.Spl),
                            np.array(self.Snr), np.array(self.Spr), G_used)
            result = {}
            result["eps"] = deriv[0]
            result["Chi"] = deriv[1]
            result["Eg"] = deriv[2]
            result["Nc"] = deriv[3]
            result["Nv"] = deriv[4]
            result["Ndop"] = deriv[5]
            result["mn"] = deriv[6]
            result["mp"] = deriv[7]
            result["Et"] = deriv[8]
            result["tn"] = deriv[9]
            result["tp"] = deriv[10]
            result["Br"] = deriv[11]
            result["Cn"] = deriv[12]
            result["Cp"] = deriv[13]
            result["Snl"] = deriv[14]
            result["Spl"] = deriv[15]
            result["Snr"] = deriv[16]
            result["Spr"] = deriv[17]
            result["G"] = deriv[18]

            return result

    def IV_curve(self, title=None):
        
        Vincr = efficiency.Vincrement(np.array(self.Chi), np.array(self.Eg),
                           np.array(self.Nc), np.array(self.Nv),
                           np.array(self.Ndop))
        if (self.opt is "user"):
            G_used = np.array(self.G)
        else:
            G_used = compute_G(np.array(self.grid[1:] - self.grid[:-1]),
                               np.array(self.Eg), np.array(self.Lambda),
                               np.array(self.P_in), np.array(self.A))

        current = scale["J"] * calc_IV(
            np.array(self.grid[1:] - self.grid[:-1]), Vincr, np.array(
                self.eps), np.array(self.Chi), np.array(self.Eg),
            np.array(self.Nc), np.array(self.Nv), np.array(self.Ndop),
            np.array(self.mn), np.array(self.mp), np.array(self.Et),
            np.array(self.tn), np.array(self.tp), np.array(self.Br),
            np.array(self.Cn), np.array(self.Cp), np.array(self.Snl),
            np.array(self.Spl), np.array(self.Snr), np.array(self.Spr), G_used)
        voltages = scale["E"] * np.linspace(
            start=0, stop=(len(current) - 1) * Vincr, num=len(current))
        fig = plt.figure()
        plt.plot(voltages, current, color="blue", marker=".")
        plt.xlabel("Voltage (V)")
        plt.ylabel("current (A.cm-2)")
        if title is not None:
            plt.savefig(title)
        plt.show()
        return voltages, current

    def solve(self, equilibrium=False, V=0):
        
        Vincr = efficiency.Vincrement(np.array(self.Chi), np.array(self.Eg),
                           np.array(self.Nc), np.array(self.Nv),
                           np.array(self.Ndop))
        if (self.opt is "user"):
            G_used = np.array(self.G)
        else:
            G_used = compute_G(np.array(self.grid[1:] - self.grid[:-1]),
                               np.array(self.Eg), np.array(self.Lambda),
                               np.array(self.P_in), np.array(self.A))

        N = self.grid.size

        phi_ini = eq_init_phi(np.array(self.Chi), np.array(self.Eg),
                              np.array(self.Nc), np.array(self.Nv),
                              np.array(self.Ndop))

        #Solve Equilibrium--
        phi_eq = solve_eq(np.array(self.grid[1:] - self.grid[:-1]), phi_ini,
                          np.array(self.eps), np.array(self.Chi),
                          np.array(self.Eg), np.array(self.Nc),
                          np.array(self.Nv), np.array(self.Ndop))

        result = {}

        V_dim = V / scale["E"]

        if equilibrium:
            result["phi_n"] = np.zeros(N)
            result["phi_p"] = np.zeros(N)
            result["phi"] = scale["E"] * phi_eq
            result["n"] = scale["n"] * n(np.zeros(N), phi_eq, np.array(
                self.Chi), np.array(self.Nc))
            result["p"] = scale["n"] * p(np.zeros(N), phi_eq, np.array(
                self.Chi), np.array(self.Eg), np.array(self.Nv))
            result["Jn"] = np.zeros(N - 1)
            result["Jp"] = np.zeros(N - 1)
            return result
        else:
            num_steps = math.floor(V_dim / Vincr)

            phis = np.concatenate((np.zeros(2 * N), phi_eq), axis=0)
            neq_0 = self.Nc[0] * np.exp(self.Chi[0] + phi_eq[0])
            neq_L = self.Nc[-1] * np.exp(self.Chi[-1] + phi_eq[-1])
            peq_0 = self.Nv[0] * np.exp(-self.Chi[0] - self.Eg[0] - phi_eq[0])
            peq_L = self.Nv[-1] * np.exp(-self.Chi[-1] - self.Eg[-1] -
                                         phi_eq[-1])

            volt = [i * Vincr for i in range(num_steps)]
            volt.append(V_dim)

            for v in volt:
                print(" ")
                print("V = {0:.3E}".format(scale["E"] * v) + " V")
                print(" ")
                print(" Iteration       |F(x)|                Residual     ")
                print(
                    " -------------------------------------------------------------------"
                )
                sol = solve(np.array(self.grid[1:] - self.grid[:-1]), neq_0,
                            neq_L, peq_0, peq_L, phis, np.array(self.eps),
                            np.array(self.Chi), np.array(self.Eg),
                            np.array(self.Nc), np.array(self.Nv),
                            np.array(self.Ndop), np.array(self.mn),
                            np.array(self.mp), np.array(self.Et),
                            np.array(self.tn), np.array(self.tp),
                            np.array(self.Br), np.array(self.Cn),
                            np.array(self.Cp), np.array(self.Snl),
                            np.array(self.Spl), np.array(self.Snr),
                            np.array(self.Spr), G_used)
                if os.environ["JAX"] == "YES":
                    phis = ops.index_update(sol, -1, phi_eq[-1] + v)
                else:
                    sol[-1] = phi_eq[-1] + v
                    phis = sol
            result["phi_n"] = scale["E"] * phis[0:N]
            result["phi_p"] = scale["E"] * phis[N:2 * N]
            result["phi"] = scale["E"] * phis[2 * N:]
            result["n"] = scale["n"] * n(phis[0:N], phis[2 * N:],
                                         np.array(self.Chi), np.array(self.Nc))
            result["p"] = scale["n"] * p(phis[N:2 * N], phis[2 * N:],
                                         np.array(self.Chi), np.array(self.Eg),
                                         np.array(self.Nv))
            result["Jn"] = scale["J"] * Jn(
                np.array(self.grid[1:] - self.grid[:-1]), phis[0:N],
                phis[2 * N:], np.array(self.Chi), np.array(self.Nc),
                np.array(self.mn))
            result["Jp"] = scale["J"] * Jp(
                np.array(self.grid[1:] - self.grid[:-1]), phis[N:2 * N],
                phis[2 * N:], np.array(self.Chi), np.array(self.Eg),
                np.array(self.Nv), np.array(self.mp))
            return result

    def plot_band_diagram(self, result, title=None):
        
        Ec = -scale["E"] * np.array(self.Chi) - result["phi"]
        Ev = -scale["E"] * np.array(self.Chi) - scale["E"] * np.array(
            self.Eg) - result["phi"]
        fig = plt.figure()
        plt.plot(scale["d"] * self.grid,
                 Ec,
                 color="red",
                 label="conduction band",
                 linestyle="dashed")
        plt.plot(scale["d"] * self.grid,
                 Ev,
                 color="blue",
                 label="valence band",
                 linestyle="dashed")
        plt.plot(scale["d"] * self.grid,
                 result["phi_n"],
                 color="red",
                 label="e- quasiFermi energy")
        plt.plot(scale["d"] * self.grid,
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
        plt.plot(scale["d"] * self.grid, result["n"], color="red", label="e-")
        plt.plot(scale["d"] * self.grid,
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
        plt.plot(scale["d"] * 0.5 * (self.grid[1:] + self.grid[:-1]),
                 result["Jn"],
                 color="red",
                 label="e-")
        plt.plot(scale["d"] * 0.5 * (self.grid[1:] + self.grid[:-1]),
                 result["Jp"],
                 color="blue",
                 label="hole")
        plt.plot(scale["d"] * 0.5 * (self.grid[1:] + self.grid[:-1]),
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
