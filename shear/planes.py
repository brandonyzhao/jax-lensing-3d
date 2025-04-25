from jax import numpy as jnp
import jax_cosmo as jc
import jax_cosmo.constants as constants

cosmo = jc.Planck15(Omega_c=0.2589, sigma8=0.8159)

get_coords = lambda lensplanes: jnp.stack([plane['coords'] for plane in lensplanes], axis=0)
get_planes = lambda lensplanes: jnp.stack([plane['plane'] for plane in lensplanes], axis=-1)

def unpack_lensplanes(lensplanes): 
    dz = jnp.array([plane['dz'] for plane in lensplanes])
    r = jnp.array([plane['r'] for plane in lensplanes])
    a = jnp.array([plane['a'] for plane in lensplanes])
    return dz, r, a

def get_source_scales(cosmo, lensplanes, z_source, photo_err=False): 
    dz, r, a = unpack_lensplanes(lensplanes)
    const_factor = 3 / 2 * cosmo.Omega_m * (constants.H0 / constants.c)**2
    norm_factor = dz * r / a
    if photo_err:
        # haven't yet implemented photometric redshift errors
        raise NotImplemented
    else:
        r_s = jc.background.radial_comoving_distance(cosmo, 1 / (1 + z_source.flatten())).reshape(z_source.shape)
        source_scales = jnp.clip(1. - (r.reshape(1, -1) / r_s.reshape(-1, 1)), 0, 1000).reshape([-1, 1, 1, len(r)])
    return const_factor * norm_factor.reshape((1, 1, 1, -1)) * source_scales