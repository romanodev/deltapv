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

    x = np.linspace(-1e-6, 1e-6, 100)
    val = np.array([
        -18.02634814, -18.02634797, -18.02634801, -18.02634825, -18.02634825,
        -18.0263482, -18.0263479, -18.02634834, -18.02634819, -18.02634821,
        -18.02634829, -18.0263482, -18.02634837, -18.02634846, -18.02634819,
        -18.02634822, -18.02634819, -18.02634846, -18.02634813, -18.02634835,
        -18.02634818, -18.02634825, -18.02634832, -18.02634811, -18.02634812,
        -18.02634805, -18.02634832, -18.02634805, -18.02634814, -18.02634789,
        -18.02634787, -18.02634833, -18.0263481, -18.02634815, -18.02634811,
        -18.02634814, -18.0263482, -18.02634787, -18.02634788, -18.02634809,
        -18.02634841, -18.02634807, -18.02634817, -18.02634822, -18.02634825,
        -18.02634805, -18.02634801, -18.02634807, -18.02634826, -18.02634842,
        -18.02634821, -18.02634818, -18.0263484, -18.02634821, -18.02634786,
        -18.02634813, -18.02634823, -18.02634824, -18.02634826, -18.02634839,
        -18.02634813, -18.02634809, -18.02634815, -18.02634796, -18.02634832,
        -18.02634823, -18.02634812, -18.02634805, -18.02634826, -18.02634805,
        -18.02634812, -18.02634821, -18.02634805, -18.02634822, -18.02634843,
        -18.02634825, -18.02634818, -18.02634817, -18.02634807, -18.02634799,
        -18.02634818, -18.02634823, -18.02634827, -18.02634847, -18.0263484,
        -18.02634783, -18.02634825, -18.02634808, -18.02634802, -18.02634807,
        -18.02634812, -18.02634818, -18.0263482, -18.02634813, -18.02634769,
        -18.02634821, -18.02634825, -18.0263481, -18.02634815, -18.02634836
    ])
    mean = np.mean(val)

    plt.plot(x, val)
    plt.show()

    plt.plot(x, (val - mean) / x)
    plt.show()
