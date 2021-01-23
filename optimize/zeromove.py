from jaxpv import linalg
from jax import numpy as np
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt


def plotpot(vec):
    plt.plot(vec[::3], label="phi_n")
    plt.plot(vec[1::3], label="phi_p")
    plt.plot(vec[2::3], label="phi")
    plt.legend()
    plt.show()


def plotJ(mat):
    plt.matshow(mat[:100, :100])
    plt.colorbar()
    plt.show()


def rowscale(mat, vec, ord=2):

    D = 1 / np.linalg.norm(mat, ord=ord, axis=1, keepdims=True)
    mat_scaled = mat * D
    vec_scaled = vec * D.flatten()

    return mat_scaled, vec_scaled


spJ = np.load("spJ.npy")
J = np.load("J.npy")
F = np.load("F.npy")

spJp, Fp = rowscale(spJ, F, ord=2)
Jp = linalg.sparse2dense(spJp)

p, _ = gmres(J, -F)
plotpot(p)