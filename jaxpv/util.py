from jax import numpy as np, lax
from typing import Union

Array = np.ndarray
f64 = np.float64


def switch(condition: bool, val_true: Union[f64, Array], val_false: Union[f64, Array]) -> Union[f64, Array]:
    return lax.cond(condition, lambda _: val_true, lambda _: val_false, None)
