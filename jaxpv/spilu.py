import numpy as np
from scipy.linalg import lu_solve, inv

def ilu0(M):
    A = np.copy(M)
    n = M.shape[0]
    
    for i in range(1, n):
        for k in range(max(0, i - 3), i): # range(i)
            if A[i, k] != 0 and A[k, k] != 0:
                A[i, k] = A[i, k] / A[k, k]
                for j in range(k + 1, min(n, i + 3)): # range(k + 1, n)
                    if A[i, j] != 0:
                        A[i, j] = A[i, j] - A[i, k] * A[k, j]
    
    Ainv = lu_solve((A, np.arange(n)), np.eye(n))
    
    return Ainv