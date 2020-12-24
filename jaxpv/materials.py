from jaxpv import dataclasses, objects
from jax import numpy as np
import yaml, glob, os

Array = np.ndarray
f64 = np.float64
Material = objects.Material


def create_material(**kwargs) -> Material:
    return Material(**{
        key: f64(value)
        for key, value in kwargs.items() if value is not None
    })


def load_material(name: str) -> Material:
    try:
        with open(
                os.path.join(os.path.dirname(__file__),
                             f"resources/{name}.yaml"), "r") as f:
            matdict = yaml.full_load(f)
        return create_material(**matdict["properties"])
    except:
        raise FileNotFoundError(f"{name} is not an available material!")


def update(mat: Material, **kwargs) -> Material:
    return Material(
        **{
            key: f64(kwargs[key]) if key in kwargs else value
            for key, value in mat.__dict__.items()
        })
