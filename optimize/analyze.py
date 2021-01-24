import os
os.environ["ALLOWNANS"] = "TRUE"
import psc
import numpy as np
import matplotlib.pyplot as plt


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


def analyzeSLSQP(filename):

    des = []

    with open(filename, "r") as f:
        for line in f.readlines():
            if line.startswith("["):
                strlist = line.strip("][\n").split(", ")
                x = [float(y) for y in strlist]
                des.append(x)

    print(f"N = {len(des)}")

    return des


def plot_stats(effs):

    plt.hist(effs, bins=20, histtype="step")
    plt.show()

    plt.boxplot(effs)
    plt.show()


if __name__ == "__main__":

    _, xs = analyzeRS("logs/randomsearch.log")
    whack = xs[1]
    psc.f(whack)
