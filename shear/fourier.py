import jax
from jax_lensing.inversion import ks93inv

ks93inv_batch = jax.vmap(ks93inv, in_axes=(2, None), out_axes=(2, 2))