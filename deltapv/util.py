from jax import numpy as jnp, jit, custom_jvp, grad
from jax.experimental import optimizers
from deltapv import spline
import matplotlib.pyplot as plt

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


def dhor(y1, y2, norm=2):
    x1 = jnp.arange(y1.size) * 0.05
    x2 = jnp.arange(y2.size) * 0.05
    ymin = 0
    ymax = min(jnp.max(y1), jnp.max(y2))
    yint = jnp.linspace(ymin, ymax, 100, endpoint=False)
    idx1 = jnp.argsort(y1)
    idx2 = jnp.argsort(y2)
    xint1 = jnp.interp(yint, y1[idx1], x1[idx1])
    xint2 = jnp.interp(yint, y2[idx2], x2[idx2])
    xint1 = spline.qinterp(yint, y1[idx1], x1[idx1])
    xint2 = spline.qinterp(yint, y2[idx2], x2[idx2])
    res = jnp.sum(jnp.power(jnp.abs(xint1 - xint2), norm))
    return res


def dver(y1, y2, norm=2):
    n = min(y1.size, y2.size)
    res = jnp.sum(jnp.power(jnp.abs(y1[:n] - y2[:n]), norm))
    return res


def polar(x, y):
    theta = jnp.arctan2(x, y)
    r = jnp.sqrt(x**2 + y**2)
    return theta, r


def cartesian(theta, r):
    x = r * jnp.sin(theta)
    y = r * jnp.cos(theta)
    return x, y


def dpol(y1, y2, norm=2):
    y1 = 10 * y1
    y2 = 10 * y2
    x1 = jnp.arange(y1.size) * 0.05
    x2 = jnp.arange(y2.size) * 0.05
    theta1, r1 = polar(x1, y1)
    theta2, r2 = polar(x2, y2)
    thetaint = jnp.linspace(0, jnp.pi / 2, 100)
    rint1 = spline.qinterp(thetaint, theta1, r1)
    rint2 = spline.qinterp(thetaint, theta2, r2)
    res = jnp.sum(jnp.power(jnp.abs(rint1 - rint2), norm))
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

def adagrad(df, x0, lr=1,steps=jnp.inf, tol=0, gtol=0):
    opt_init, opt_update, get_params = optimizers.adagrad(step_size=lr)
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
