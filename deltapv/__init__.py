import logging
import os

from jax.config import config

config.update("jax_enable_x64", True)
if os.environ.get("DEBUGNANS") == "TRUE":
    config.update("jax_debug_nans", True)
if os.environ.get("NOJIT") == "TRUE":
    config.update('jax_disable_jit', True)

logging.basicConfig(format="")
logger = logging.getLogger("deltapv")
logger.setLevel("INFO")

from deltapv import (simulator, materials, plotting,
                     objects, spline, physics, util)
from deltapv.materials import create_material, load_material
from deltapv.plotting import (plot_band_diagram, plot_bars,
                              plot_charge, plot_iv_curve)
from deltapv.simulator import (make_design, incident_light, equilibrium,
                               simulate, eff_at_bias, empty_design,
                               add_material, doping, contacts)

util.print_ascii()
