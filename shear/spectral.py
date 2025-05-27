from jax import numpy as jnp
import jax

# The following code is adapted from jax_lensing package: 
# https://github.com/CosmoStat/jax-lensing

def radial_profile(data):
  """
  Compute the radial profile of 2d image
  :param data: 2d image
  :return: radial profile
  """
  center = data.shape[0]/2
  y, x = jnp.indices((data.shape))
  r = jnp.sqrt((x - center)**2 + (y - center)**2)
  r = r.astype('int32')

  tbin = jnp.bincount(r.ravel(), data.ravel())
  nr = jnp.bincount(r.ravel())
  radialprofile = tbin / nr
  return radialprofile

def measure_power_spectrum(map_data):
  """
  measures power 2d data
  :param power: map
  :return: power spectrum
  """
  field_npix = map_data.shape[0]
  data_ft = jnp.fft.fft2(map_data) / field_npix
  data = jnp.real(data_ft * jnp.conj(data_ft))
  freqs = jnp.fft.fftfreq(field_npix, 1/field_npix)
  x, y = jnp.meshgrid(freqs, freqs)
  r = jnp.sqrt((x)**2 + (y)**2)
  r = r.astype('int32')

  tbin = jnp.bincount(r.ravel(), data.ravel())
  nr = jnp.bincount(r.ravel())
  radialprofile = tbin / nr

  return radialprofile / (2 * jnp.pi)**2

measure_spectra = jax.vmap(measure_power_spectrum, in_axes=(2,))