import os

if os.environ['JAX'] == 'YES':
    print("JAX")
    from jax.config import config
    config.update("jax_enable_x64", True)
    import jax.numpy as np
    from jax import grad , jit, jacfwd, ops
    ops
else:
    import numpy as np



