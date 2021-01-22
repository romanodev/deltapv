import matplotlib.pyplot as plt

if __name__ == "__main__":

    effs = []

    with open("randomsearch.log", "r") as f:
        for line in f.readlines():
            if line.startswith("Sample: "):
                res = line.split()[-1]
                if res != "None":
                    val = -float(res)
                    effs.append(val)
    
    print(f"N = {len(effs)}")

    print(f"{len([x for x in effs if x == 0])} degenerate zero entries")

    plt.hist(effs, bins=20, histtype="step")
    plt.show()

    plt.boxplot(effs)
    plt.show()
