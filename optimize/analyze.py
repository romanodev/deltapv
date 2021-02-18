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
    

if __name__ == "__main__":

    e, v, p, g = analyzeAdam("logs/adam_psc_sched_200iter.log")

    plt.plot(v)
    plt.plot(-e, linestyle="--")
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
    plt.show()
