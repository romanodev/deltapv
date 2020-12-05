from jax.config import config
config.update("jax_enable_x64", True)

from .simulator import JAXPV
from .materials import Material