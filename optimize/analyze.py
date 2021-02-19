import psc
import numpy as np
import matplotlib.pyplot as plt
import ast

import logging
logger = logging.getLogger("deltapv")
logger.addHandler(logging.FileHandler("logs/tsuyoi.log"))


def analyzeRS(filename):

    effs = []
    fails = []

    with open(filename, "r") as f:
        for line in f.readlines():
            if line.startswith("Sample: "):
                res = line.split()[-1]
                if res != "None":
                    val = -float(res)
                    effs.append(val)
                else:
                    strlist = line.split("[")[1].split("]")[0].split(", ")
                    x = [float(y) for y in strlist]
                    fails.append(x)

    print(f"N = {len(effs)}")
    print(f"{len([x for x in effs if x == 0])} degenerate zero entries")

    return effs, fails


def analyzeOptim(filename):

    des = []
    effs = []

    with open(filename, "r") as f:
        for line in f.readlines():
            if line.startswith("["):
                strlist = line.strip("][\n").split(", ")
                x = [float(y) for y in strlist]
                des.append(x)
            elif line.startswith("Finished"):
                streff = line.split()[-1][:-1]
                effs.append(float(streff))
            elif "returning zero" in line:
                effs.append(0.)

    print(f"N = {len(des)}")

    return des, effs


def analyzeAdam(filename):

    effs = []
    value = []
    param = []
    grads = []

    with open(filename, "r") as f:
        for line in f.readlines():
            if line.startswith("value = "):
                y = ast.literal_eval(line.strip("value = "))
                value.append(y)
            elif line.startswith("param = "):
                x = ast.literal_eval(line.strip("param = "))
                param.append(x)
            elif line.startswith("grads = "):
                dydx = ast.literal_eval(line.strip("grads = "))
                grads.append(dydx)
            elif line.startswith("Finished simulation"):
                eff = float(line.split()[-1][:-1])
                effs.append(eff)
    
    effs = np.array(effs)
    value = np.array(value)
    param = np.array(param)
    grads = np.array(grads)

    return effs, value, param, grads


def analyzeRandom(filename):

    value = []
    param = []

    with open(filename, "r") as f:
        for line in f.readlines():
            if line.startswith("value = "):
                y = ast.literal_eval(line.strip("value = "))
                value.append(y)
            elif line.startswith("param = "):
                x = ast.literal_eval(line.strip("param = "))
                param.append(x)
    
    value = np.array(value)
    param = np.array(param)

    return value, param


def analyzeDiscovery(filename):

    value = []
    param = []
    grads = []

    with open(filename, "r") as f:
        for line in f.readlines():
            if line.startswith("value = "):
                y = ast.literal_eval(line.strip("value = "))
                value.append(y)
            elif line.startswith("param = "):
                x = ast.literal_eval(line.strip("param = "))
                param.append(x)
            elif line.startswith("grads = "):
                dydx = ast.literal_eval(line.strip("grads = "))
                grads.append(dydx)
    
    value = np.array(value)
    param = np.array(param)
    grads = np.array(grads)

    return value, param, grads
    

if __name__ == "__main__":

    """vrand, prand = analyzeRandom("logs/sample_psc_100iter.log")
    brand = np.minimum.accumulate(vrand)
    e, v, p, g = analyzeAdam("logs/adam_psc_sched_200iter.log")
    e = e[:100]
    v = v[:100]
    p = p[:100]
    g = g[:100]

    plt.plot(v, color="black", label="adam")
    plt.plot(-e, linestyle="--", color="black")
    plt.scatter(np.arange(100), vrand, color="black", marker=".")
    plt.plot(brand, linestyle="--", color="black", label="random")
    plt.ylim(top=0)
    plt.xlabel("iterations")
    plt.ylabel("objective / %")
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(4, 4)
    for i, traj in enumerate(p.T):
        j, k = i // 4, i % 4
        axs[j, k].plot(traj)
        axs[j, k].set_title(psc.PARAMS[i])
    fig.tight_layout()
    plt.show()

    fig, axs = plt.subplots(4, 4)
    for i, traj in enumerate(g.T):
        j, k = i // 4, i % 4
        axs[j, k].plot(traj)
        axs[j, k].set_title(psc.PARAMS[i])
    fig.tight_layout()
    plt.show()"""

    v, p, g = analyzeDiscovery("logs/discoverybay_1p03.log")

    plt.plot(p, color="black")
    plt.xlabel("iterations")
    plt.ylabel("$E_{g, P}$ / eV")
    plt.tight_layout()
    plt.show()

    ax1 = plt.gca()
    ax1.plot(v, color="black")
    ax2 = plt.gca().twinx()
    ax2.plot(g, color="black", linestyle="--")
    ax1.set_xlabel("iterations")
    ax1.set_ylabel("rss")
    ax2.set_ylabel("rss derivative")
    plt.tight_layout()
    plt.show()
