from flax.traverse_util import flatten_dict
from jax import numpy as jnp

def summarize_params(params, label=""):
    flat = flatten_dict(params)
    print(f"\n===== PARAMETER STATS {label} =====")
    for k, v in flat.items():
        name = '/'.join(k)
        print(f"{name:50s} | mean: {jnp.mean(v): .6f}, std: {jnp.std(v): .6f}")
    print("="*70 + "\n")