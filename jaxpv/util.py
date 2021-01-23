from jax import numpy as np, jit, custom_jvp, grad

Array = np.ndarray
f64 = np.float64
i64 = np.int64


@custom_jvp
@jit
def softmax(x: Array, alpha: f64 = 1) -> f64:

    C = np.max(alpha * x)
    expax = np.exp(alpha * x - C)
    num = np.sum(x * expax)
    denom = np.sum(expax)
    sm = num / denom

    return sm


@softmax.defjvp
def softmax_jvp(primals, tangents):

    x, alpha = primals
    dx, _ = tangents
    smx = softmax(x)

    C = np.max(alpha * x)
    expax = np.exp(alpha * x - C)
    pre = expax / np.sum(expax)
    term = 1 + alpha * (x - smx)
    dsmx = pre * term @ dx

    return smx, dsmx


@jit
def softabs(x: f64, alpha: f64 = 1) -> f64:

    pos = np.exp(alpha * x)
    neg = np.exp(-alpha * x)
    num = x * (pos - neg)
    denom = 2 + pos + neg
    sa = num / denom

    return sa


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    v = np.linspace(0, 1, 100)
    j = 10 * (2 - np.exp(20 * (v - 0.9653426)))
    p = v * j
    pmax = np.max(p)
    psmax = softmax(p, alpha=1)

    plt.plot(v, j)
    plt.plot(v, p)
    plt.axhline(psmax)
    plt.show()

    deriv = grad(softmax)(p, alpha=1)
    plt.plot(deriv)
    deriv2 = grad(np.max)(p)
    plt.plot(deriv2)
    plt.show()
