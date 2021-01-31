from deltapv import dataclasses, objects
from jax import numpy as np
import pandas as pd, yaml, glob, os

Material = objects.Material
Array = np.ndarray
f64 = np.float64
lam_interp = np.linspace(200, 1200, 100, endpoint=False)


def create_material(**kwargs) -> Material:
    return Material(**{
        key: f64(value)
        for key, value in kwargs.items() if value is not None
    })


def get_alpha(name: str) -> Array:

    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), f"resources/{name}.csv"))
    _lam = np.array(df[df.columns[0]])
    _alpha = np.array(df[df.columns[4]])
    alpha = np.interp(lam_interp, _lam, _alpha)

    return alpha


def load_material(name: str) -> Material:
    try:
        with open(
                os.path.join(os.path.dirname(__file__),
                             f"resources/{name}.yaml"), "r") as f:
            matdict = yaml.full_load(f)
            matdict["properties"]["alpha"] = get_alpha(name)
        return create_material(**matdict["properties"])
    except:
        raise FileNotFoundError(f"{name} is not an available material!")


def update(mat: Material, **kwargs) -> Material:
    return Material(
        **{
            key: f64(kwargs[key]) if key in kwargs else value
            for key, value in mat.__dict__.items()
        })
