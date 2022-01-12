from pathlib import Path

from deltapv import objects
from jax import numpy as jnp
import yaml

import pandas as pd


Material = objects.Material
Array = jnp.ndarray
f64 = jnp.float64
lam_interp = jnp.linspace(200, 1200, 100, endpoint=False)


def list_materials():
    """List materials that are saved in:
    Path(__file__).parent.joinpath("resources")
    """
    pth = Path(__file__).parent.joinpath("resources")

    try:
        assert pth.exists()
        for mat in pth.glob("*.yaml"):
            print(mat.stem)

    except AssertionError:
        raise FileNotFoundError("""
Path %s does not exist. Ensure you have downloaded the resources folder from
https://github.com/romanodev/deltapv/tree/master/deltapv/resources
            """ % pth)


def display_material(name: str) -> str:
    """Neatly print material properties"""
    try:
        with open(
            Path(__file__).parent.joinpath(f"resources/{name}.yaml"),
                "r") as f:
            matdict = yaml.full_load(f)
            return print(yaml.dump(matdict))
    except FileNotFoundError:
        print (f"{name} is not an available material!")


def create_material(**kwargs) -> Material:
    """Create a custom material. Possible arguments include the following:

        eps: unitless
        Chi , Eg , Et: eV
        Nc , Nv , Ndop: cm^(-3)
        mn , mp: cm^2 / (V s)
        tn , tp: s
        Br: cm^3 / s
        Cn, Cp: cm^6 / s
        alpha: 1 / cm. Should be given as an array of length 100, with the
        alphas for evenly spaced wavelengths from 200 to 1000nm

    Returns:
        Material: A material object
    """
    return Material(**{
        key: f64(value)
        for key, value in kwargs.items() if value is not None
    })


def get_alpha(name: str) -> Array:
    """Load alpha of a material from the materials library

    Args:
        name (str): Name of material

    Returns:
        Array: Array of absorption coefficients for material
    """
    df = pd.read_csv(
        Path(__file__).parent.joinpath(f"resources/{name}.csv"))
    _lam = jnp.array(df[df.columns[0]])
    _alpha = jnp.array(df[df.columns[4]])
    alpha = jnp.interp(lam_interp, _lam, _alpha)

    return alpha


def load_material(name: str) -> Material:
    """Load a material from the materials library

    Args:
        name (str): Name of material

    Raises:
        FileNotFoundError: If material is not found, raises an error

    Returns:
        Material: Loaded material
    """
    try:
        with open(
                Path(__file__).parent.joinpath(f"resources/{name}.yaml"),
                "r") as f:
            matdict = yaml.full_load(f)
            try:
                matdict["properties"]["alpha"] = get_alpha(name)
            except FileNotFoundError:
                print(f"{name}.csv does not exist!")
        return create_material(**matdict["properties"])
    except FileNotFoundError:
        print (f"{name} is not an available material!")


def update(mat: Material, **kwargs) -> Material:
    """Helper function for modifying a material.
    Keyword arguments are the same as "create_material"

    Args:
        mat (Material): Material to modify

    Returns:
        Material: Modified copy of the material
    """
    return Material(
        **{
            key: f64(kwargs[key]) if key in kwargs else value
            for key, value in mat.__dict__.items()
        })
