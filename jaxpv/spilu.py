import numpy as np


def spget(i, j, data, indices, indptr):
    try:
        loc = indptr[i] + np.argwhere(indices[indptr[i]:indptr[i +
                                                               1]] == j)[0][0]
        return loc, data[loc]
    except:
        return None, 0


def spdot(data, indices, indptr, x):
    n = len(indptr) - 1
    onerow = lambda i: np.dot(data[indptr[i]:indptr[i + 1]], x[indices[indptr[
        i]:indptr[i + 1]]])
    vfunc = np.vectorize(onerow)

    return vfunc(np.arange(n))


def spilu0(data, indices, indptr):
    n = len(indptr) - 1
    fac = np.copy(data)

    for i in range(1, n):
        loc_start, loc_end = indptr[i], indptr[i + 1]

        for idx_in_row, k in enumerate(indices[loc_start:loc_end]):
            loc_ik = loc_start + idx_in_row
            if k >= i:
                break
            loc_kk, val_kk = spget(k, k, fac, indices, indptr)
            if loc_kk is not None:
                fac[loc_ik] /= val_kk

                for idx_in_row, j in enumerate(indices[loc_start:loc_end]):
                    loc_ij = indptr[i] + idx_in_row
                    if j <= k:
                        continue
                    _, val_kj = spget(k, j, fac, indices, indptr)
                    fac[loc_ij] -= fac[loc_ik] * val_kj

    sub = lambda b: spbsub(fac, indices, indptr, spfsub(
        fac, indices, indptr, b))

    return sub


def spfsub(data, indices, indptr, b):
    # for L component, which is unit diagonal
    n = len(indptr) - 1
    x = np.zeros_like(b)

    x[0] = b[0] / data[
        0]  # data[0] is always the left-top element as it is nonzero

    for i in range(1, n):
        end = indptr[i] + np.where(indices[indptr[i]:indptr[i + 1]] == i)[0][0]
        x[i] = (b[i] - np.dot(data[indptr[i]:end], x[indices[indptr[i]:end]])
                )  # L[i, i] = 1

    return x


def spbsub(data, indices, indptr, b):
    # for U component
    n = len(indptr) - 1
    x = np.zeros(n)

    x[-1] = b[-1] / data[
        -1]  # data[-1] is always the right-bottom element as it is nonzero

    for i in range(n - 2, -1, -1):
        start = indptr[i] + np.where(
            indices[indptr[i]:indptr[i + 1]] == i)[0][0] + 1
        x[i] = (b[i] - np.dot(data[start:indptr[i + 1]],
                              x[indices[start:indptr[i + 1]]])) / spget(
                                  i, i, data, indices, indptr)[1]

    return x
