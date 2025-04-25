from flax import linen as nn
import jax
from jax import numpy as jnp

from typing import Any, Callable
import functools

safe_sin = lambda x: jnp.sin(x % (100 * jnp.pi))

def posenc_z(x, degs): 
    out = [x]
    for j in range(3):
        if degs[j] == 0: 
            continue
        scales = jnp.array([2**i for i in range(degs[j])])
        xb = jnp.reshape((x[..., None, j] * scales), list(x.shape[:-1]) + [-1])
        sins = safe_sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
        out.append(sins)
    return jnp.concatenate(out, axis=-1)

class MLP(nn.Module):
    net_depth: int = 4
    net_width: int = 128
    activation: Callable[..., Any] = nn.relu
    out_channel: int = 1
    do_skip: bool = True
    deg_point: int = (4, 4, 4)

    sig_scale: int=3
    sig_shift: int=-1
  
    @nn.compact
    def __call__(self, x):
        #MLP with ior activation function builtin
        #Also with posenc builtin
        """A simple Multi-Layer Preceptron (MLP) network

        Parameters
        ----------
        x: jnp.ndarray(float32), 
            [batch_size * n_samples, feature], points.
        net_depth: int, 
            the depth of the first part of MLP.
        net_width: int, 
            the width of the first part of MLP.
        activation: function, 
            the activation function used in the MLP.
        out_channel: 
            int, the number of alpha_channels.
        do_skip: boolean, 
            whether or not to use a skip connection

        Returns
        -------
        out: jnp.ndarray(float32), 
            [batch_size * n_samples, out_channel].
        """
        dense_layer = functools.partial(
            nn.Dense, kernel_init=jax.nn.initializers.he_uniform())

        if self.do_skip:
            skip_layer = self.net_depth // 2

        x = posenc_z(x, self.deg_point)
        inputs = x
        for i in range(self.net_depth):
            x = dense_layer(self.net_width)(x)
            x = self.activation(x)
            if self.do_skip:
                if i % skip_layer == 0 and i > 0:
                    x = jnp.concatenate([x, inputs], axis=-1)
        out = dense_layer(self.out_channel)(x)
        out = jax.nn.sigmoid(out) * self.sig_scale + self.sig_shift

        return out           