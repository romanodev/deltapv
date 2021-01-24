from jaxpv import util
from jax import numpy as np, ops, vmap, lax, jit
from jax.scipy.sparse.linalg import gmres
from functools import partial
from typing import Callable

Array = util.Array
f64 = util.f64
i64 = util.i64
_W = 13


@partial(jit, static_argnums=(3, ))
def coo2sparse(row: Array, col: Array, data: Array, n: i64) -> Array:

    disp = np.clip(col - row + _W // 2, 0, _W - 1)
    sparse = np.zeros((n, _W)).at[row, disp].set(data)
    return sparse


@jit
def sparse2dense(m: Array) -> Array:

    n = m.shape[0]

    def onerow(i):
        n = m.shape[0]
        row = np.zeros(n + _W - 1)
        row = lax.dynamic_update_slice(row, m[i], (i, ))[_W // 2:-(_W // 2)]
        return row

    return vmap(onerow)(np.arange(n))


@jit
def spmatvec(m: Array, x: Array) -> Array:
    def _onerow(m, x, i):
        return np.dot(m[i], lax.dynamic_slice(x, [i], [_W]))

    return vmap(_onerow, (None, None, 0))(m, np.pad(x, pad_width=_W // 2),
                                          np.arange(x.size))


@jit
def spget(m: Array, i: i64, j: i64) -> f64:

    disp = np.clip(j - i + _W // 2, 0, _W - 1)
    return m[i, disp]


@jit
def spwrite(m: Array, i: i64, j: i64, value: f64) -> Array:

    disp = np.clip(j - i + _W // 2, 0, _W - 1)
    mnew = m.at[i, disp].set(value)
    return mnew


@jit
def spilu(m: Array) -> Array:

    n = m.shape[0]

    def iloop(cmat, i):
        def kloop(crow, dispk):
            k = i + dispk - _W // 2

            def kli(k):
                mik, mkk = crow[dispk], spget(cmat, k, k)

                def processrow(row):
                    row = row.at[dispk].set(mik / mkk)

                    def jone(dispj):
                        j = i + dispj - _W // 2
                        mij = row[dispj]
                        return mij - (j > k) * (mij != 0) * row[dispk] * spget(
                            cmat, k, j)

                    return vmap(jone)(np.arange(_W))

                return lax.cond(mik * mkk != 0, processrow, lambda r: r, crow)

            return lax.cond(k < i, kli, lambda k: crow, k), None

        rowi, _ = lax.scan(kloop, cmat[i], np.arange(_W))
        return cmat.at[i].set(rowi), None

    result, _ = lax.scan(iloop, m, np.arange(n))

    return result


@jit
def fsub(m: Array, b: Array) -> Array:
    # Lower triangular, unit diagonal
    n = m.shape[0]

    def entry(xc, i):
        xcpad = np.pad(xc, pad_width=_W // 2)
        res = np.dot(m[i, :_W // 2], lax.dynamic_slice(xcpad, [i], [_W // 2]))
        entryi = b[i] - res
        xc = xc.at[i].set(entryi)
        return xc, None

    x, _ = lax.scan(entry, np.zeros(n), np.arange(n))
    return x


@jit
def bsub(m: Array, b: Array) -> Array:
    # Upper triangular
    n = m.shape[0]

    def entry(xc, i):
        xcpad = np.pad(xc, pad_width=_W // 2)
        res = np.dot(m[i, _W // 2 + 1:],
                     lax.dynamic_slice(xcpad, [i + _W // 2 + 1], [_W // 2]))
        entryi = (b[i] - res) / m[i, _W // 2]
        xc = xc.at[i].set(entryi)
        return xc, None

    x, _ = lax.scan(entry, np.zeros(n), np.flip(np.arange(n)))
    return x


@jit
def linsol(spmat: Array,
           vec: Array,
           tol=1e-12) -> Array:

    mvp = partial(spmatvec, spmat)
    fact = spilu(spmat)
    precond = lambda b: bsub(fact, fsub(fact, b))

    sol, _ = gmres(mvp,
                   vec,
                   M=precond,
                   tol=tol,
                   atol=0.,
                   maxiter=10,
                   solve_method="batched")

    return sol


@jit
def transpose(m: Array) -> Array:

    n = m.shape[0]
    fmp = np.fliplr(np.pad(m, ((_W // 2, _W // 2), (0, 0))))

    def onerow(i):
        return np.diag(lax.dynamic_slice_in_dim(fmp, i, _W, axis=0))

    return vmap(onerow)(np.arange(n))


@jit
def transol(spmat: Array, vec: Array) -> Array:

    tspmat = transpose(spmat)
    return linsol(tspmat, vec)
