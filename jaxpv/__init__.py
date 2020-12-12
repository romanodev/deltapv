from jax.config import config
config.update("jax_enable_x64", True)

from .simulator import JAXPV
from .simulator import IV_curve, efficiency
from .materials import Material