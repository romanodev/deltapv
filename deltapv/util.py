from jax import numpy as jnp, jit, custom_jvp, grad
from jax.experimental import optimizers
import jax
import numpy as np
from deltapv import spline, simulator
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger("deltapv")

Array = jnp.ndarray
f64 = jnp.float64
i64 = jnp.int64


def print_ascii():
    print(
        "___________________    __\n______ \ __  __ \_ |  / /\n_  __  /__  /_/ /_ | / / \n/ /_/ / _  ____/__ |/ /  \n\__,_/  /_/     _____/   \n                         "
    )


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


def gd(df, x0, lr=1, steps=jnp.inf, tol=0, gtol=0, verbose=True):
    i = 0
    y = dy = jnp.inf
    x = jnp.array(x0)
    obj = []
    xs = []
    dys = []
    if verbose:
        logger.info("Starting gradient descent with:")
        logger.info(("    x0    =" + len(x) * " {:+.2e},").format(*x)[:-1])
        logger.info("    lr    = {:.2e}".format(lr))
        logger.info("    steps = {:3d}".format(steps))
        logger.info("    tol   = {:.2e}".format(tol))
        logger.info("    gtol  = {:.2e}".format(gtol))
    while (i < steps) and (jnp.abs(y) > tol) and jnp.any(jnp.abs(dy) > gtol):
        y, dy = df(x)
        obj.append(y)
        dys.append(dy)
        xs.append(x)
        x = x - lr * dy
        i = i + 1
        if verbose:
            logger.info("iteration {:3d}".format(i))
            logger.info("    f(x)  = {:.2e}".format(y))
            logger.info(("    x     =" + len(x) * " {:+.2e},").format(*x)[:-1])
            logger.info(("    df/dx =" + len(dy) * " {:+.2e},").format(*dy)[:-1])
    obj = jnp.array(obj)
    xs = jnp.array(xs)
    dys = jnp.array(dys)
    result = {"f": obj, "dfdx": dys, "x": xs}
    return result


def adagrad(df,
            x0,
            lr=1,
            steps=jnp.inf,
            tol=0,
            gtol=0,
            momentum=0.9,
            verbose=True):
    opt_init, opt_update, get_params = optimizers.adagrad(step_size=lr,
                                                          momentum=momentum)
    x = jnp.array(x0)
    opt_state = opt_init(x)
    i = 0
    y = dy = jnp.inf
    obj = []
    xs = []
    dys = []
    if verbose:
        logger.info("Starting Adagrad with:")
        logger.info(("    x0       =" + len(x) * " {:+.2e},").format(*x)[:-1])
        logger.info("    lr       = {:.2e}".format(lr))
        logger.info("    momentum = {:.2e}".format(momentum))
        logger.info("    steps    = {:3d}".format(steps))
        logger.info("    tol      = {:.2e}".format(tol))
        logger.info("    gtol     = {:.2e}".format(gtol))
    while (i < steps) and (jnp.abs(y) > tol) and jnp.any(jnp.abs(dy) > gtol):
        x = get_params(opt_state)
        y, dy = df(x)
        opt_state = opt_update(i, dy, opt_state)
        xs.append(x)
        obj.append(y)
        dys.append(dy)
        i = i + 1
        if verbose:
            logger.info("iteration {:3d}".format(i))
            logger.info("    f(x)  = {:.2e}".format(y))
            logger.info(("    x     =" + len(x) * " {:+.2e},").format(*x)[:-1])
            logger.info(("    df/dx =" + len(dy) * " {:+.2e},").format(*dy)[:-1])
    obj = jnp.array(obj)
    xs = jnp.array(xs)
    dys = jnp.array(dys)
    result = {"f": obj, "dfdx": dys, "x": xs}
    return result


def adam(df,
         x0,
         lr=1,
         b1=0.9,
         b2=0.999,
         steps=jnp.inf,
         tol=0,
         gtol=0,
         verbose=True):
    opt_init, opt_update, get_params = optimizers.adam(step_size=lr,
                                                       b1=b1,
                                                       b2=b2)
    x = jnp.array(x0)
    opt_state = opt_init(x)
    i = 0
    y = dy = jnp.inf
    obj = []
    xs = []
    dys = []
    if verbose:
        logger.info("Starting Adam with:")
        logger.info(("    x0    =" + len(x) * " {:+.2e},").format(*x)[:-1])
        logger.info("    lr    = {:.2e}".format(lr))
        logger.info("    b1    = {:.2e}".format(b1))
        logger.info("    b2    = {:.2e}".format(b2))
        logger.info("    steps = {:3d}".format(steps))
        logger.info("    tol   = {:.2e}".format(tol))
        logger.info("    gtol  = {:.2e}".format(gtol))
    while (i < steps) and (jnp.abs(y) > tol) and jnp.any(jnp.abs(dy) > gtol):
        x = get_params(opt_state)
        y, dy = df(x)
        opt_state = opt_update(i, dy, opt_state)
        xs.append(x)
        obj.append(y)
        dys.append(dy)
        i = i + 1
        if verbose:
            logger.info("iteration {:3d}".format(i))
            logger.info("    f(x)  = {:.2e}".format(y))
            logger.info(("    x     =" + len(x) * " {:+.2e},").format(*x)[:-1])
            logger.info(("    df/dx =" + len(dy) * " {:+.2e},").format(*dy)[:-1])
    obj = jnp.array(obj)
    xs = jnp.array(xs)
    dys = jnp.array(dys)
    result = {"f": obj, "dfdx": dys, "x": xs}
    return result


class StatefulOptimizer:
    def __init__(self, x_init, convr, constr, bounds, dv=0.01):
        self.count = 0
        self.growth = []
        self.dv = dv
        self.x = x_init
        self.bounds = bounds

        dg_jnp = jax.jacobian(constr)
        self.g = lambda x: np.array(constr(jnp.array(x)))
        self.dg = lambda x: np.array(dg_jnp(jnp.array(x)))

        def f(params, pot_ini):
            x = params[:-1]
            v = params[-1]
            vprint = jnp.round(v, 2)
            print("evaluating for V = {:.2f}".format(vprint))
            eff, pot = simulator.eff_at_bias(convr(x), v, pot_ini, verbose=False)
            return -eff, pot

        df_jnp = jax.value_and_grad(f, has_aux=True)  # returns (p, pot), dp

        def df_wrapper(params, guess):
            (eff, sol), deff = df_jnp(jnp.array(params), guess)
            return (float(eff), sol), np.array(deff)

        self.df = df_wrapper

        results = simulator.simulate(convr(jnp.array(self.x)))
        v, i = results["iv"]
        maxidx = jnp.argmax(v * i)
        self.guess = results["pots"][maxidx]
        self.v = v[maxidx]

    def eval(self, params):
        x = params[:-1]
        v = params[-1]
        print(f"called eval with x = {x} and v = {v}")
        nsteps = int(np.ceil(np.abs(v - self.v) / self.dv))
        print(f"    currently at {self.count} total pde solves")
        print(f"    splitting into {nsteps} steps")
        xs = np.linspace(self.x, x, nsteps + 1)
        vs = np.linspace(self.v, v, nsteps + 1)
        for xr, vr in zip(xs, vs):
            (eff, self.guess), deff = self.df(np.append(xr, vr), self.guess)
            self.x = xr
            self.v = vr
            self.growth.append(eff)
        self.count += nsteps + 1
        print("    efficiency = {:.2f}%\n".format(-100 * eff))
        return eff, deff

    def optimize(self, niters=100):
        slsqp_res = minimize(self.eval,
                             x0=self.get_params(),
                             method="SLSQP",
                             jac=True,
                             bounds=self.bounds,
                             constraints=[{
                                 "type": "ineq",
                                 "fun": self.g,
                                 "jac": self.dg
                             }],
                             options={
                                 "maxiter": niters,
                                 "disp": True
                             })
        return slsqp_res

    def get_params(self):
        return np.append(self.x, self.v)

    def get_growth(self):
        return np.array(self.growth)
