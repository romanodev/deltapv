from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

from jaxpv import simulator
from jaxpv import materials