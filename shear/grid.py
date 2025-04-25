import jax.numpy as jnp

def get_X_harm(resxy, resz): 
  # Get grid from [-1, 1]

  linspxy = jnp.linspace(-1 + (1 / resxy), 1 - (1 / resxy), resxy)
  linspz = jnp.linspace(-1 + (1 / resz), 1 - (1 / resz), resz) * (resz / resxy)
  x, y, z = jnp.meshgrid(linspxy, linspxy, linspz, indexing='xy')
  pts = jnp.stack([x, y, z], axis=-1)
  X = pts.reshape((-1, 3))
  return X