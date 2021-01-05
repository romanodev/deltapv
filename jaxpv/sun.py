import jax.numpy as np
import pickle
import os

with open(os.path.join(os.path.dirname(__file__), "resources/solar.pickle"), "rb") as f:
    solar = pickle.load(f)

wavelength = np.array(solar["wavelength"])
raw_power = np.array(solar["irradiance"])
power = 1e3 * raw_power / np.sum(raw_power)
