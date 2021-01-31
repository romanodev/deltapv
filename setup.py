from setuptools import setup, find_packages

setup(
    name='deltapv',
    version='0.0.1',
    description='Solar cell simulator with automatic differentiation',
    author=
    'Ekin Dogus Cubuk, Sam Schoenholz, Eric Richard Fadel and Giuseppe Romano',
    classifiers=['Programming Language :: Python :: 3.6'],
    long_description=open('README.rst').read(),
    install_requires=['numpy', 'jax'],
    license='MIT',
    packages=['deltapv'])
