from jax import numpy as jnp, grad, vmap, jit
from functools import partial
import matplotlib.pyplot as plt


def quadratic(x, coef):

    a, b, c = coef
    y = a * x**2 + b * x + c

    return y


def qspline(x, y):

    n = x.size
    M = jnp.zeros((3 * (n - 1), 3 * (n - 1)))
    z = jnp.zeros(3 * (n - 1))

    M = M.at[0, 0].set(1)
    z = z.at[1].set(y[0])

    for i in range(n - 1):
        M = M.at[3 * i + 1, 3 * i:3 * i + 3].set(jnp.array([x[i]**2, x[i], 1]))
        z = z.at[3 * i + 1].set(y[i])

        M = M.at[3 * i + 2, 3 * i:3 * i + 3].set([x[i + 1]**2, x[i + 1], 1])
        z = z.at[3 * i + 2].set(y[i + 1])

    for i in range(n - 2):
        M = M.at[3 * i + 3, 3 * i:3 * i + 6].set(
            jnp.array([2 * x[i + 1], 1, 0, -2 * x[i + 1], -1, 0]))

    coef = jnp.linalg.solve(M, z)
    a = coef[::3]
    b = coef[1::3]
    c = coef[2::3]

    return a, b, c


@jit
def predict(x, xp, coef):

    a, b, c = coef
    idx = jnp.clip(jnp.searchsorted(xp, x) - 1, 0)
    y = a[idx] * x**2 + b[idx] * x + c[idx]

    return y


def ascent(df, x0=0., lr=1., tol=1e-6, niter=100):

    x = x0
    for _ in range(niter):
        deriv = df(x)
        x = x + lr * deriv
        if jnp.abs(deriv) < 1e-6:
            break

    return x


def findmax(x, coef):

    a, b, _ = coef
    xl = x[:-1]
    xu = x[1:]
    filla = jnp.where(a != 0, a, 1)  # avoid divide-by-zero
    xm = jnp.clip(-b / (2 * filla), xl, xu)

    yl = quadratic(xl, coef)
    yu = quadratic(xu, coef)
    ym = quadratic(xm, coef)

    xall = jnp.concatenate([xl, xu, xm])
    yall = jnp.concatenate([yl, yu, ym])

    idxmax = jnp.argmax(yall)
    ymax = yall[idxmax]
    xmax = xall[idxmax]

    return ymax, xmax


def calcPmax_gd(v, j):
    p = v * j
    coef = qspline(v, p)
    fun = partial(predict, xp=v, coef=coef)
    dfun = grad(fun)

    vbest = ascent(dfun, x0=v[jnp.argmax(p)])
    pmax = fun(vbest)
    return pmax


def calcPmax(v, j):
    p = v * j
    coef = qspline(v, p)
    pmax, vmax = findmax(v, coef)
    return pmax, vmax


def qinterp(x, xp, yp):
    coef = qspline(xp, yp)
    y = predict(x, xp, coef)
    return y
