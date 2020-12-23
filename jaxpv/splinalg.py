from jax import numpy as np, ops, vmap, lax, jit

_W = 13


def _coo2sparse(row, col, data, n):
    disp = np.clip(col - row + _W // 2, 0, _W - 1)
    sparse = ops.index_update(np.zeros((n, _W)), ops.index[row, disp], data)
    return sparse


coo2sparse = jit(_coo2sparse, static_argnums=[3])


@jit
def spmatvec(m, x):
    def _onerow(m, x, i):
        return np.dot(m[i], lax.dynamic_slice(x, [i], [_W]))

    return vmap(_onerow, (None, None, 0))(m, np.pad(x, pad_width=_W // 2),
                                          np.arange(x.size))


@jit
def spget(m, i, j):
    disp = np.clip(j - i + _W // 2, 0, _W - 1)
    return m[i, disp]


@jit
def spwrite(m, i, j, value):
    disp = np.clip(j - i + _W // 2, 0, _W - 1)
    mnew = ops.index_update(m, ops.index[i, disp], value)
    return mnew


@jit
def spilu(m):
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
def fsub(m, b):

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
def bsub(m, b):

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


def invjvp(sparse):
    fact = spilu(sparse)
    jvp = lambda b: bsub(fact, fsub(fact, b))
    return jvp
