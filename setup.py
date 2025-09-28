from setuptools import setup, find_packages

setup(
    name="deltapv",
    version="0.0.5",
    description="Solar cell simulator with automatic differentiation",
    author=
    "Sean Mann, Eric Richard Fadel, Ekin Dogus Cubuk, Sam Schoenholz, and Giuseppe Romano",
    classifiers=["Programming Language :: Python :: 3.7"],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy", "jax", "jaxlib","matplotlib", "pandas", "equinox","pyyaml", "scipy"
    ],
    license="MIT",
    packages=["deltapv"],
    package_data={"deltapv": ["resources/*", "fonts/*"]}
    )
