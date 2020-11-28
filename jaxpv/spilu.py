import numpy as np


def spget(i, j, data, indices, indptr):
    try:
        loc = indptr[i] + np.argwhere(indices[indptr[i]:indptr[i +
                                                               1]] == j)[0][0]
        return loc, data[loc]
    except:
        return None, 0


def spilu0(n, data, indices, indptr):
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

    return fac, indices, indptr


def spfsub(n, data, indices, indptr, b):
    x = np.zeros_like(b)

    x[0] = b[0] / data[0]

    for i in range(1, n):
        x[i] = (b[i] -
                np.dot(data[indptr[i]:indptr[i + 1] - 1],
                       x[indices[indptr[i]:indptr[i + 1] - 1]])) / spget(
                           i, i, data, indices, indptr)[1]

    return x


def spbsub(n, data, indices, indptr, b):
    x = np.zeros_like(b)

    x[-1] = b[-1] / data[-1]

    for i in range(n - 2, -1, -1):
        x[i] = (b[i] -
                np.dot(data[indptr[i] + 1:indptr[i + 1]],
                       x[indices[indptr[i] + 1:indptr[i + 1]]])) / spget(
                           i, i, data, indices, indptr)[1]

    return x
