from jax import numpy as jnp, jit, custom_jvp, grad
from jax.experimental import optimizers

Array = jnp.ndarray
f64 = jnp.float64
i64 = jnp.int64


@custom_jvp
@jit
def softmax(x: Array, alpha: f64 = 1) -> f64:
    C = jnp.max(alpha * x)
    expax = jnp.exp(alpha * x - C)
    num = jnp.sum(x * expax)
    denom = jnp.sum(expax)
    sm = num / denom
    return sm


@softmax.defjvp
def softmax_jvp(primals, tangents):
    x, alpha = primals
    dx, _ = tangents
    smx = softmax(x)
    C = jnp.max(alpha * x)
    expax = jnp.exp(alpha * x - C)
    pre = expax / jnp.sum(expax)
    term = 1 + alpha * (x - smx)
    dsmx = pre * term @ dx
    return smx, dsmx


@jit
def softabs(x: f64, alpha: f64 = 1) -> f64:
    pos = jnp.exp(alpha * x)
    neg = jnp.exp(-alpha * x)
    num = x * (pos - neg)
    denom = 2 + pos + neg
    sa = num / denom
    return sa


def dhor(y1, y2):
    x1 = jnp.arange(y1.size) * 0.05
    x2 = jnp.arange(y2.size) * 0.05
    ymin = 0
    ymax = min(jnp.max(y1), jnp.max(y2))
    yint = jnp.linspace(ymin, ymax, 100)
    idx1 = jnp.argsort(y1)
    idx2 = jnp.argsort(y2)
    xint1 = jnp.interp(yint, y1[idx1], x1[idx1])
    xint2 = jnp.interp(yint, y2[idx2], x2[idx2])
    res = jnp.sum((xint1 - xint2)**2)
    return res


def gd(df, x0, lr=1, steps=jnp.inf, tol=0, gtol=0):
    i = 0
    y = dy = jnp.inf
    x = jnp.array(x0)
    obj = []
    xs = []
    while (i < steps) and (jnp.abs(y) > tol) and jnp.any(jnp.abs(dy) > gtol):
        print(x, y, dy)
        y, dy = df(x)
        obj.append(y)
        xs.append(x)
        x = x - lr * dy
        i = i + 1
    xs.append(x)
    xs = jnp.array(xs)
    obj = jnp.array(obj)
    return xs, obj


def adam(df, x0, lr=1, b1=0.9, b2=0.999, steps=jnp.inf, tol=0, gtol=0):
    opt_init, opt_update, get_params = optimizers.adam(step_size=lr, b1=b1, b2=b2)
    opt_state = opt_init(jnp.array(x0))
    i = 0
    y = dy = jnp.inf
    obj = []
    xs = []
    while (i < steps) and (jnp.abs(y) > tol) and jnp.any(jnp.abs(dy) > gtol):
        x = get_params(opt_state)
        y, dy = df(x)
        print(x, y, dy)
        opt_state = opt_update(i, dy, opt_state)
        xs.append(x)
        obj.append(y)
        i = i + 1
    xs.append(get_params(opt_state))
    xs = jnp.array(xs)
    obj = jnp.array(obj)
    return xs, obj
