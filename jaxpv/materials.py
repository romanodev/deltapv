from . import dataclasses
from jax import numpy as np
import yaml
import glob
import os

Array = np.ndarray
f64 = np.float64

MATERIAL_FILES = glob.glob(
    os.path.join(os.path.dirname(__file__), "resources/*.yaml"))


@dataclasses.dataclass
class Material:
    eps: f64 = f64(1)
    Chi: f64 = f64(1)
    Eg: f64 = f64(1)
    Nc: f64 = f64(1e17)
    Nv: f64 = f64(1e17)
    mn: f64 = f64(1e2)
    mp: f64 = f64(1e2)
    tn: f64 = f64(1e-8)
    tp: f64 = f64(1e-8)
    Et: f64 = f64(0)
    Br: f64 = f64(0)
    Cn: f64 = f64(0)
    Cp: f64 = f64(0)
    A: f64 = f64(0)

    def __iter__(self):
        return self.__dict__.items().__iter__()


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
