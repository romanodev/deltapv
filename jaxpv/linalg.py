from jaxpv import util
from jax import numpy as np, ops, vmap, lax, jit
from jax.scipy.sparse.linalg import gmres
from functools import partial
from typing import Callable

Array = util.Array
f64 = util.f64
i32 = util.i32
_W = 13


@partial(jit, static_argnums=(3,))
def coo2sparse(row: Array, col: Array, data: Array, n: i32) -> Array:
    
    disp = np.clip(col - row + _W // 2, 0, _W - 1)
    sparse = ops.index_update(np.zeros((n, _W)), ops.index[row, disp], data)
    return sparse


@jit
def spmatvec(m: Array, x: Array) -> Array:
    
    def _onerow(m, x, i):
        return np.dot(m[i], lax.dynamic_slice(x, [i], [_W]))

    return vmap(_onerow, (None, None, 0))(m, np.pad(x, pad_width=_W // 2),
                                          np.arange(x.size))


@jit
def spget(m: Array, i: i32, j: i32) -> f64:
    
    disp = np.clip(j - i + _W // 2, 0, _W - 1)
    return m[i, disp]


@jit
def spwrite(m: Array, i: i32, j: i32, value: f64) -> Array:
    
    disp = np.clip(j - i + _W // 2, 0, _W - 1)
    mnew = ops.index_update(m, ops.index[i, disp], value)
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
                    row = ops.index_update(row, ops.index[dispk], mik / mkk)

                    def jone(dispj):
                        j = i + dispj - _W // 2
                        mij = row[dispj]
                        return mij - (j > k) * (mij != 0) * row[dispk] * spget(
                            cmat, k, j)

                    return vmap(jone)(np.arange(_W))

                return lax.cond(mik * mkk != 0, processrow, lambda r: r, crow)

            return lax.cond(k < i, kli, lambda k: crow, k), None

        rowi, _ = lax.scan(kloop, cmat[i], np.arange(_W))
        return ops.index_update(cmat, ops.index[i], rowi), None

    result, _ = lax.scan(iloop, m, np.arange(n))
    return result


@jit
def fsub(m: Array, b: Array) -> Array:

    n = m.shape[0]

    def entry(xc, i):
        xcpad = np.pad(xc, pad_width=_W // 2)
        res = np.dot(m[i, :_W // 2], lax.dynamic_slice(xcpad, [i], [_W // 2]))
        entryi = b[i] - res
        xc = ops.index_update(xc, ops.index[i], entryi)
        return xc, None

    x, _ = lax.scan(entry, np.zeros(n), np.arange(n))
    return x


@jit
def bsub(m: Array, b: Array) -> Array:

    n = m.shape[0]

    def entry(xc, i):
        xcpad = np.pad(xc, pad_width=_W // 2)
        res = np.dot(m[i, _W // 2 + 1:],
                     lax.dynamic_slice(xcpad, [i + _W // 2 + 1], [_W // 2]))
        entryi = (b[i] - res) / m[i, _W // 2]
        xc = ops.index_update(xc, ops.index[i], entryi)
        return xc, None

    x, _ = lax.scan(entry, np.zeros(n), np.flip(np.arange(n)))
    return x


def invmvp(sparse: Array) -> Callable[[Array], Array]:
    
    fact = spilu(sparse)
    jvp = lambda b: bsub(fact, fsub(fact, b))
    return jvp


@jit
def linsol(spmat: Array, vec: Array) -> Array:

    mvp = partial(spmatvec, spmat)
    precond = invmvp(spmat)

    sol, _ = gmres(mvp,
                   vec,
                   M=precond,
                   tol=1e-10,
                   atol=0.,
                   maxiter=5)

    return sol