from jaxpv import linalg
from jax import numpy as np
from scipy.sparse.linalg import gmres, spilu, LinearOperator
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


spJ = np.load("debug/spJ.npy")
J = np.load("debug/J.npy")
F = np.load("debug/F.npy")


def splinsol(A, b):
    precon = lambda x: spilu(A).solve(x)
    lo = LinearOperator(A.shape, precon)
    x, _ = gmres(A, b, M=lo)
    return x


fact = linalg.spilu(spJ)
ILU = linalg.sparse2dense(fact)

L = np.tril(ILU, k=-1) + np.eye(ILU.shape[0])
U = np.triu(ILU)

p = linalg.linsol(spJ, -F)

plt.plot(J @ p)
plt.plot(-F)
plt.show()
