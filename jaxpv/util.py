from jax import numpy as np, lax
from typing import Union

Array = np.ndarray
f64 = np.float64
i64 = np.int64


def log(a: Array) -> Array:
    return np.log(np.abs(a))
