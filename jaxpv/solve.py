from . import residual
from . import spilu

from scipy.sparse.linalg import gmres, LinearOperator
from scipy.sparse import csc_matrix

import matplotlib.pyplot as plt


def damp(move):
   
    tmp = 1e10
    approx_sign = np.tanh(tmp * move)
    approx_abs = approx_sign * move
    approx_H = 1 - (1 + tmp * np.exp(-(move**2 - 1)))**(-1)
    return np.log(1 + approx_abs) * approx_sign + approx_H * (
        move - np.log(1 + approx_abs) * approx_sign)


def step(data, neq0, neqL, peq0, peqL, phis):
    
    dgrid = data["dgrid"]
    N = dgrid.size + 1

    F = residual.F(data, neq0, neqL, peq0, peqL, phis[0:N], phis[N:2 * N], phis[2 * N:])
    
    values, indices, indptr = residual.F_deriv(data, neq0, neqL, peq0, peqL, phis[0:N],
                                    phis[N:2 * N], phis[2 * N:])

    gradF_jvp = lambda x: spdot(values, indices, indptr, x)
    invgradF_jvp = spilu0(values, indices, indptr)

    operator = LinearOperator((3 * N, 3 * N), gradF_jvp)
    precond = LinearOperator((3 * N, 3 * N), invgradF_jvp)

    move, conv_info = gmres(operator, -_F, tol=1e-10, maxiter=5, M=precond)

    error = np.linalg.norm(move)
    
    damp_move = damp(move)
    
    return np.concatenate(
        (phis[0:N] + damp_move[0:3 * N:3], phis[N:2 * N] +
         damp_move[1:3 * N:3], phis[2 * N:] + damp_move[2:3 * N:3]),
        axis=0), error


def solve(data, neq0, neqL, peq0, peqL, phis_ini):
    
    dgrid = data["dgrid"]
    N = dgrid.size + 1
    
    grad_step = jit(jacfwd(step, argnums=[0, 1, 2, 3, 4, 5], has_aux=True)

    dphis_dphiini = np.eye(3 * N)
    dphis_dneq0 = np.zeros((3 * N, 1))
    dphis_dneqL = np.zeros((3 * N, 1))
    dphis_dpeq0 = np.zeros((3 * N, 1))
    dphis_dpeqL = np.zeros((3 * N, 1))
    dphis_dSnl = np.zeros((3 * N, 1))
    dphis_dSpl = np.zeros((3 * N, 1))
    dphis_dSnr = np.zeros((3 * N, 1))
    dphis_dSpr = np.zeros((3 * N, 1))
    dphis_deps = np.zeros((3 * N, N))
    dphis_dChi = np.zeros((3 * N, N))
    dphis_dEg = np.zeros((3 * N, N))
    dphis_dNc = np.zeros((3 * N, N))
    dphis_dNv = np.zeros((3 * N, N))
    dphis_dNdop = np.zeros((3 * N, N))
    dphis_dmn = np.zeros((3 * N, N))
    dphis_dmp = np.zeros((3 * N, N))
    dphis_dEt = np.zeros((3 * N, N))
    dphis_dtn = np.zeros((3 * N, N))
    dphis_dtp = np.zeros((3 * N, N))
    dphis_dBr = np.zeros((3 * N, N))
    dphis_dCn = np.zeros((3 * N, N))
    dphis_dCp = np.zeros((3 * N, N))
    dphis_dG = np.zeros((3 * N, N))
    
    phis = phis_ini
    error = 1
    niter = 0

    while error > 1e-6 and ninter < 100:
                    
        gradstep = grad_step(data, neq0, neqL, peq0, peqL, phis)
        phis, error = step(data, neq0, neqL, peq0, peqL, phis)

        dphis_dphiini = np.dot(gradstep[5], dphis_dphiini)
        dphis_dneq0 = np.reshape(gradstep[1],
                                 (3 * N, 1)) + np.dot(gradstep[5], dphis_dneq0)
        dphis_dneqL = np.reshape(gradstep[2],
                                 (3 * N, 1)) + np.dot(gradstep[5], dphis_dneqL)
        dphis_dpeq0 = np.reshape(gradstep[3],
                                 (3 * N, 1)) + np.dot(gradstep[5], dphis_dpeq0)
        dphis_dpeqL = np.reshape(gradstep[4],
                                 (3 * N, 1)) + np.dot(gradstep[5], dphis_dpeqL)
        dphis_dSnl = np.reshape(gradstep[0]["Snl"],
                                (3 * N, 1)) + np.dot(gradstep[5], dphis_dSnl)
        dphis_dSpl = np.reshape(gradstep[0]["Spl"],
                                (3 * N, 1)) + np.dot(gradstep[5], dphis_dSpl)
        dphis_dSnr = np.reshape(gradstep[0]["Snr"],
                                (3 * N, 1)) + np.dot(gradstep[5], dphis_dSnr)
        dphis_dSpr = np.reshape(gradstep[0]["Spr"],
                                (3 * N, 1)) + np.dot(gradstep[5], dphis_dSpr)
        dphis_deps = gradstep[0]["eps"] + np.dot(gradstep[5], dphis_deps)
        dphis_dChi = gradstep[0]["Chi"] + np.dot(gradstep[5], dphis_dChi)
        dphis_dEg = gradstep[0]["Eg"] + np.dot(gradstep[5], dphis_dEg)
        dphis_dNc = gradstep[0]["Nc"] + np.dot(gradstep[5], dphis_dNc)
        dphis_dNv = gradstep[0]["Nv"] + np.dot(gradstep[5], dphis_dNv)
        dphis_dNdop = gradstep[0]["Ndop"] + np.dot(gradstep[5], dphis_dNdop)
        dphis_dmn = gradstep[0]["mn"] + np.dot(gradstep[5], dphis_dmn)
        dphis_dmp = gradstep[0]["mp"] + np.dot(gradstep[5], dphis_dmp)
        dphis_dEt = gradstep[0]["Et"] + np.dot(gradstep[5], dphis_dEt)
        dphis_dtn = gradstep[0]["tn"] + np.dot(gradstep[5], dphis_dtn)
        dphis_dtp = gradstep[0]["tp"] + np.dot(gradstep[5], dphis_dtp)
        dphis_dBr = gradstep[0]["Br"] + np.dot(gradstep[5], dphis_dBr)
        dphis_dCn = gradstep[0]["Cn"] + np.dot(gradstep[5], dphis_dCn)
        dphis_dCp = gradstep[0]["Cp"] + np.dot(gradstep[5], dphis_dCp)
        dphis_dG = gradstep[0]["G"] + np.dot(gradstep[5], dphis_dG)

        niter += 1
        print(f"    {niter}                         {float(error)}")

    grad_phis = {}
    grad_phis["neq0"] = dphis_dneq0
    grad_phis["neqL"] = dphis_dneqL
    grad_phis["peq0"] = dphis_dpeq0
    grad_phis["peqL"] = dphis_dpeqL
    grad_phis["phi_ini"] = dphis_dphiini
    grad_phis["eps"] = dphis_deps
    grad_phis["Chi"] = dphis_dChi
    grad_phis["Eg"] = dphis_dEg
    grad_phis["Nc"] = dphis_dNc
    grad_phis["Nv"] = dphis_dNv
    grad_phis["Ndop"] = dphis_dNdop
    grad_phis["mn"] = dphis_dmn
    grad_phis["mp"] = dphis_dmp
    grad_phis["Et"] = dphis_dEt
    grad_phis["tn"] = dphis_dtn
    grad_phis["tp"] = dphis_dtp
    grad_phis["Br"] = dphis_dBr
    grad_phis["Cn"] = dphis_dCn
    grad_phis["Cp"] = dphis_dCp
    grad_phis["Snl"] = dphis_dSnl
    grad_phis["Spl"] = dphis_dSpl
    grad_phis["Snr"] = dphis_dSnr
    grad_phis["Spr"] = dphis_dSpr
    grad_phis["G"] = dphis_dG

    return phis, grad_phis

                    
def step_eq(data, phi):
    
    dgrid = data["dgrid"]
    N = dgrid.size + 1

    Feq = residual.F_eq(data, np.zeros(phi.size), np.zeros(phi.size), phi)
                    
    values, indices, indptr = residual.F_eq_deriv(data, np.zeros(phi.size),
                                       np.zeros(phi.size))

    gradFeq_jvp = lambda x: spdot(values, indices, indptr, x)
    invgradFeq_jvp = spilu0(values, indices, indptr)

    operator = LinearOperator((N, N), gradFeq_jvp)
    precond = LinearOperator((N, N), invgradFeq_jvp)

    move, conv_info = gmres(operator, -Feq, tol=1e-10, maxiter=5, M=precond)

    if conv_info > 0:
        print(f"Early termination of GMRES at {conv_info} iterations")

    error = np.linalg.norm(move)

    damp_move = damp(move)

    return phi + damp_move, error


def solve_eq_forgrad(data, phi_ini):
                    
    dgrid = data["dgrid"]
    N = dgrid.size + 1
    
    print("Equilibrium     Iteration                             Residual   ")
    print("-----------------------------------------------------------------")
                    
    grad_step = jit(jacfwd(step_eq, argnums=[0, 1])

    dphi_dphiini = np.eye(N)
    dphi_deps = np.zeros((N, N))
    dphi_dChi = np.zeros((N, N))
    dphi_dEg = np.zeros((N, N))
    dphi_dNc = np.zeros((N, N))
    dphi_dNv = np.zeros((N, N))
    dphi_dNdop = np.zeros((N, N))
                    
    error = 1
    niter = 0
    phi = phi_ini

    while error > 1e-6 and ninter < 100:
                    
        gradstep = grad_step(data, phi)
        phi, error = step_eq(data, phi)
        
        dphi_dphiini = np.dot(gradstep[1], dphi_dphiini)
        dphi_deps = gradstep[0]["eps"] + np.dot(gradstep[1], dphi_deps)
        dphi_dChi = gradstep[0]["Chi"] + np.dot(gradstep[1], dphi_dChi)
        dphi_dEg = gradstep[0]["Eg"] + np.dot(gradstep[1], dphi_dEg)
        dphi_dNc = gradstep[0]["Nc"] + np.dot(gradstep[1], dphi_dNc)
        dphi_dNv = gradstep[0]["Nv"] + np.dot(gradstep[1], dphi_dNv)
        dphi_dNdop = gradstep[0]["Ndop"] + np.dot(gradstep[1], dphi_dNdop)

        niter += 1
        print(f"    {niter}                          {error}")

    grad_phi = {}
    grad_phi["phi_ini"] = dphi_dphiini
    grad_phi["eps"] = dphi_deps
    grad_phi["Chi"] = dphi_dChi
    grad_phi["Eg"] = dphi_dEg
    grad_phi["Nc"] = dphi_dNc
    grad_phi["Nv"] = dphi_dNv
    grad_phi["Ndop"] = dphi_dNdop

    return phi, grad_phi
