import jax.numpy as np
from jax import ops


def spget(i, j, values, indices, indptr):

    try:
        loc = indptr[i] + np.argwhere(indices[indptr[i]:indptr[i +
                                                               1]] == j)[0][0]
        return loc, values[loc]
    except:
        return None, 0


def spdot(values, indices, indptr, x):

    n = indptr.size - 1

    def onerow(i):
        return np.dot(values[indptr[i]:indptr[i + 1]],
                      x[(indices[indptr[i]:indptr[i + 1]], )])

    vfunc = np.vectorize(onerow)

    return vfunc(np.arange(n))


def spilu(values, indices, indptr):

    n = indptr.size - 1
    fac = values

    for i in range(1, n):
        loc_start, loc_end = indptr[i], indptr[i + 1]
        for idx_in_row, k in enumerate(indices[loc_start:loc_end]):
            loc_ik = loc_start + idx_in_row
            if k >= i:
                break
            loc_kk, val_kk = spget(k, k, fac, indices, indptr)
            if loc_kk is not None:
                fac = ops.index_update(fac, ops.index[loc_ik],
                                       fac[loc_ik] / val_kk)
                for idx_in_row, j in enumerate(indices[loc_start:loc_end]):
                    loc_ij = indptr[i] + idx_in_row
                    if j <= k:
                        continue
                    _, val_kj = spget(k, j, fac, indices, indptr)
                    fac = ops.index_update(fac, ops.index[loc_ij],
                                           fac[loc_ij] - fac[loc_ik] * val_kj)

    sub = lambda b: spbsub(fac, indices, indptr, spfsub(
        fac, indices, indptr, b))

    return sub


def spfsub(values, indices, indptr, b):

    n = indptr.size - 1
    x = np.zeros_like(b)
    x = ops.index_update(x, ops.index[0], b[0] / values[0])

    for i in range(1, n):
        end = indptr[i] + np.where(indices[indptr[i]:indptr[i + 1]] == i)[0][0]
        x = ops.index_update(
            x, ops.index[i], b[i] -
            np.dot(values[indptr[i]:end], x[(indices[indptr[i]:end], )]))

    return x


def spbsub(values, indices, indptr, b):

    n = indptr.size - 1
    x = np.zeros_like(b)
    x = ops.index_update(x, ops.index[-1], b[-1] / values[-1])

    for i in range(n - 2, -1, -1):
        start = indptr[i] + np.where(
            indices[indptr[i]:indptr[i + 1]] == i)[0][0] + 1
        x = ops.index_update(
            x, ops.index[i],
            (b[i] - np.dot(values[start:indptr[i + 1]], x[
                (indices[start:indptr[i + 1]], )])) /
            spget(i, i, values, indices, indptr)[1])

    return x


def dense(values, indices, indptr):

    n = indptr.size - 1
    matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        matrix = ops.index_update(
            matrix, ops.index[i, indices[indptr[i]:indptr[i + 1]]],
            values[indptr[i]:indptr[i + 1]])

    return matrix
