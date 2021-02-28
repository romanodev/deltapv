import deltapv
import sesame
import numpy as np
import matplotlib.pyplot as plt

iv = np.load("outputs/iv.npy")
ivbm = np.loadtxt("outputs/example/IV_values.txt")

plt.plot(iv[0], iv[1], color="red", label="deltapv", alpha=0.5)
plt.plot(ivbm[0], ivbm[1], color="blue", label="sesame", alpha=0.5)
plt.legend()
plt.show()
