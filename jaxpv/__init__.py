from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

import logging
logger = logging.getLogger("jaxpv")
logger.setLevel("INFO")

from jaxpv import simulator, materials, plotting
