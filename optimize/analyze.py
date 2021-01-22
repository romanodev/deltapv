import psc
import numpy as np
import matplotlib.pyplot as plt


def find_success(filename):

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


def plot_stats(effs):

    plt.hist(effs, bins=20, histtype="step")
    plt.show()

    plt.boxplot(effs)
    plt.show()



if __name__ == "__main__":

    effs, fails = find_success("randomsearch.log")

    for fail in fails:
        psc.f(np.array(fail))
