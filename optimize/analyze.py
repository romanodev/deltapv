import psc
import numpy as np
import matplotlib.pyplot as plt
import ast
plt.rcParams["lines.linewidth"] = 2
import logging
logger = logging.getLogger("deltapv")


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

    """vrand, prand = analyzeRandom("logs/sample_psc_200iter.log")
    vrand = -vrand
    brand = np.maximum.accumulate(vrand)

    efd, vfd, pfd, gfd = analyzeAdam("logs/adam_psc_fd_lr1em2_b11em1_b21em1_200iter.log")
    vfd = -vfd
    gfd = -gfd

    e, v, p, g = analyzeAdam("logs/adam_psc_lr1em2_b11em1_b21em1_200iter.log")
    v = -v
    g = -g

    plt.plot(v, color="red", alpha=0.5, label="adam")
    plt.plot(vfd, color="blue", alpha=0.5, label="adam, fd")
    plt.scatter(np.arange(200), vrand, color="black", marker=".")
    plt.plot(brand, linestyle="--", color="black", label="random")
    plt.ylim(bottom=0)
    plt.xlabel("iterations")
    plt.ylabel("objective / %")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("obj_adam.png")
    plt.show()

    fig, axs = plt.subplots(4, 4, figsize=(16, 10))
    for i, traj in enumerate(p.T):
        j, k = i // 4, i % 4
        axs[j, k].plot(traj, color="black")
        axs[j, k].set_title(psc.PARAMS_TEX[i])
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.savefig("params.png")
    plt.show()"""

    v, p, g = analyzeDiscovery("logs/discoverybay_1p03.log")

    plt.plot(p, color="black")
    plt.xlabel("iterations")
    plt.ylabel("$E_{g, P}$ / eV")
    plt.tight_layout()
    plt.savefig("gap.png")
    plt.show()

    ax1 = plt.gca()
    ax1.plot(v, color="black")
    ax2 = plt.gca().twinx()
    ax2.plot(g, color="black", linestyle="--")
    ax1.set_xlabel("iterations")
    ax1.set_ylabel("rss")
    ax2.set_ylabel("rss derivative")
    plt.tight_layout()
    plt.savefig("rss.png")
    plt.show()
