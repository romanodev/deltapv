from deltapv import util
from jax import numpy as jnp, vmap, lax, jit
from jax.scipy.sparse.linalg import gmres
from functools import partial

Array = util.Array
f64 = util.f64
i64 = util.i64
_W = 13


@partial(jit, static_argnums=(3, ))
def coo2sparse(row: Array, col: Array, data: Array, n: i64) -> Array:

    disp = jnp.clip(col - row + _W // 2, 0, _W - 1)
    sparse = jnp.zeros((n, _W)).at[row, disp].set(data)
    return sparse


@jit
def sparse2dense(m: Array) -> Array:

    n = m.shape[0]

    def onerow(i):
        n = m.shape[0]
        row = jnp.zeros(n + _W - 1)
        row = lax.dynamic_update_slice(row, m[i], (i, ))[_W // 2:-(_W // 2)]
        return row

    return vmap(onerow)(jnp.arange(n))


@jit
def spmatvec(m: Array, x: Array) -> Array:
    def _onerow(m, x, i):
        return jnp.dot(m[i], lax.dynamic_slice(x, [i], [_W]))

    return vmap(_onerow, (None, None, 0))(m, jnp.pad(x, pad_width=_W // 2),
                                          jnp.arange(x.size))


@jit
def spget(m: Array, i: i64, j: i64) -> f64:

    disp = jnp.clip(j - i + _W // 2, 0, _W - 1)
    return m[i, disp]


@jit
def spwrite(m: Array, i: i64, j: i64, value: f64) -> Array:

    disp = jnp.clip(j - i + _W // 2, 0, _W - 1)
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
                        return mij - (j > k) * row[dispk] * spget(cmat, k, j)

                    return vmap(jone)(jnp.arange(_W))

                return lax.cond(jnp.logical_and(mik != 0, mkk != 0),
                                processrow,
                                lambda r: r, crow)

            return lax.cond(k < i, kli, lambda _: crow, k), None

        rowi, _ = lax.scan(kloop, cmat[i], jnp.arange(_W))
        return cmat.at[i].set(rowi), None

    result, _ = lax.scan(iloop, m, jnp.arange(n))

    return result


@jit
def fsub(m: Array, b: Array) -> Array:
    # Lower triangular, unit diagonal
    n = m.shape[0]

    def entry(xc, i):
        xcpad = jnp.pad(xc, pad_width=_W // 2)
        res = jnp.dot(m[i, :_W // 2], lax.dynamic_slice(xcpad, [i], [_W // 2]))
        entryi = b[i] - res
        xc = xc.at[i].set(entryi)
        return xc, None

    x, _ = lax.scan(entry, jnp.zeros(n), jnp.arange(n))
    return x


@jit
def bsub(m: Array, b: Array) -> Array:
    # Upper triangular
    n = m.shape[0]

    def entry(xc, i):
        xcpad = jnp.pad(xc, pad_width=_W // 2)
        res = jnp.dot(m[i, _W // 2 + 1:],
                      lax.dynamic_slice(xcpad, [i + _W // 2 + 1], [_W // 2]))
        entryi = (b[i] - res) / m[i, _W // 2]
        xc = xc.at[i].set(entryi)
        return xc, None

    x, _ = lax.scan(entry, jnp.zeros(n), jnp.flip(jnp.arange(n)))
    return x


@jit
def linsol(spmat: Array, vec: Array, tol=1e-12) -> Array:

    mvp = partial(spmatvec, spmat)
    fact = spilu(spmat)

    def precond(b):
        return bsub(fact, fsub(fact, b))

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
    fmp = jnp.fliplr(jnp.pad(m, ((_W // 2, _W // 2), (0, 0))))

    def onerow(i):
        return jnp.diag(lax.dynamic_slice_in_dim(fmp, i, _W, axis=0))

    return vmap(onerow)(jnp.arange(n))


@jit
def transol(spmat: Array, vec: Array) -> Array:

    tspmat = transpose(spmat)
    return linsol(tspmat, vec)
