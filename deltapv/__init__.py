import logging
import os
import jax
logging.basicConfig(format="")
logger = logging.getLogger("deltapv")
logger.setLevel("INFO")

jax.config.update("jax_enable_x64", True)


from deltapv import (simulator, materials, plotting,
                     objects, spline, physics, util)
from deltapv.materials import create_material, load_material
from deltapv.plotting import (plot_band_diagram, plot_bars,
                              plot_charge, plot_iv_curve)
from deltapv.simulator import (make_design, incident_light, equilibrium,
                               simulate, eff_at_bias, empty_design,
                               add_material, doping, contacts)

util.print_ascii()
