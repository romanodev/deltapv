from jax import numpy as np, grad, vmap, jit
from functools import partial
import matplotlib.pyplot as plt


def quad(x):

    return np.array([x**2, x, 1])


def qspline(x, y):

    n = x.size
    M = np.zeros((3 * (n - 1), 3 * (n - 1)))
    z = np.zeros(3 * (n - 1))

    M = M.at[0, 0].set(1)
    z = z.at[1].set(y[0])

    for i in range(n - 1):
        M = M.at[3 * i + 1, 3 * i:3 * i + 3].set(quad(x[i]))
        z = z.at[3 * i + 1].set(y[i])

        M = M.at[3 * i + 2, 3 * i:3 * i + 3].set(quad(x[i + 1]))
        z = z.at[3 * i + 2].set(y[i + 1])

    for i in range(n - 2):
        M = M.at[3 * i + 3, 3 * i:3 * i + 6].set(
            np.array([2 * x[i + 1], 1, 0, -2 * x[i + 1], -1, 0]))

    coef = np.linalg.solve(M, z)
    a = coef[::3]
    b = coef[1::3]
    c = coef[2::3]

    return a, b, c


@jit
def predict(x, xp, coef):

    a, b, c = coef
    idx = np.clip(np.searchsorted(xp, x) - 1, 0)
    y = a[idx] * x**2 + b[idx] * x + c[idx]

    return y


def ascent(df, x0=0., lr=1., niter=100):

    x = x0
    for _ in range(niter):
        x = x + lr * df(x)

    return x


def calcPmax(v, j):
    p = v * j
    coef = qspline(v, p)
    fun = partial(predict, xp=v, coef=coef)
    dfun = grad(fun)

    vbest = ascent(dfun, x0=v[np.argmax(p)])
    pmax = fun(vbest)
    return pmax
