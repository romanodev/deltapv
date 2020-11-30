import numpy as np


def ilu0(M):
    A = np.copy(M)
    n = M.shape[0]

    for i in range(1, n):
        for k in range(max(0, i - 3), i):
            if A[i, k] != 0 and A[k, k] != 0:
                A[i, k] = A[i, k] / A[k, k]
                for j in range(k + 1, min(n, i + 3)):
                    if A[i, j] != 0:
                        A[i, j] = A[i, j] - A[i, k] * A[k, j]

    U = np.triu(A)
    L = np.eye(n) + np.tril(A, k=-1)

    f = lambda x: bsub(U, fsub(L, x))

    return f


def fsub(L, b):
    n = L.shape[0]
    x = np.zeros_like(b)

    x[0] = b[0] / L[0, 0]

    for i in range(1, n):
        x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]

    return x


def bsub(U, b):
    n = U.shape[0]
    x = np.zeros_like(b)

    x[-1] = b[-1] / U[-1, -1]

    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x
