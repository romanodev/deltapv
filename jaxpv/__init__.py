import os
import logging
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

from jaxpv import simulator, materials, plotting