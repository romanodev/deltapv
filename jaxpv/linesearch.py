from jax import numpy as np, jacobian, vmap
import matplotlib.pyplot as plt


def accept(f, df, xold, xnew, alpha=1e-4):

    act = f(xnew) - f(xold)
    pred = alpha * np.dot(df, xnew - xold)

    return act <= pred or act == 0


def lnsrch(f, df, xold, p):

    # f: function
    # df: vector
    # xold: vector
    # p: vector

    if accept(f, df, xold, xold + p):
        print("accept original move")
        return p

    g0 = f(xold)
    gp0 = -2 * g0
    g1 = f(xold + p)

    l1 = np.maximum(-gp0 / (2 * (g1 - g0 - gp0)), 0.1)

    # lam_try = np.linspace(0, .1, 100)
    # plt.plot(lam_try, [f(xold + lam * p) for lam in lam_try])
    # plt.yscale("log")
    # plt.show()

    if accept(f, df, xold, xold + l1 * p):
        print(f"accept first step with lambda {l1}")
        return l1 * p

    # Use two most recent values to estimate cubic

    gl1 = f(xold + l1 * p)
    l2 = 1
    gl2 = g1

    while True:

        M = np.array([[1 / l1**2, -1 / l2**2], [-l2 / l1**2, l1 / l2**2]])
        coef = M @ np.array([gl1 - gp0 * l1 - g0, gl2 - gp0 * l2 - g0
                             ]) / (l1 - l2)
        a, b = coef.flatten()
        lnew = np.clip((-b + np.sqrt(b**2 - 3 * a * gp0)) / (3 * a), 0.1 * l1,
                       0.5 * l1)

        if accept(f, df, xold, xold + lnew * p):
            print(f"accept subsequent step with lambda {lnew}")
            return lnew * p

        l2 = l1
        gl2 = gl1
        l1 = lnew
        gl1 = f(xold + l1 * p)


def F(x):

    return np.sign(x) * np.log(np.abs(x) + 1)


def newton(F, x0):

    x = x0
    points = [x0]
    step = 1
    niter = 0

    while step > 1e-6:

        print(x)
        currF = F(x)
        f = lambda y: np.dot(F(y), F(y)) / 2
        df = jacobian(f)(x)
        J = jacobian(F)(x)
        p = np.linalg.solve(J, -currF)
        step = np.linalg.norm(p)

        # damping
        # dx = p
        # dx = logdamp(p)
        dx = lnsrch(f, df, x, p)

        x = x + dx
        points.append(x)
        niter += 1
    
    print(f"finished with {niter} iterations")

    return x, np.stack(points)


if __name__ == "__main__":

    sol, traj = newton(F, np.array([100., 88.]))

    plt.plot(traj[:, 0], traj[:, 1], marker="o")
    plt.scatter(traj[0, 0], traj[0, 1], color="red", zorder=10)
    plt.scatter(traj[-1, 0], traj[-1, 1], color="green", zorder=10)
    plt.show()