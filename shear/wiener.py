# Helper functions for wiener filtering
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import jax_cosmo.constants as constants

from shear.spectral import radial_profile

# These are hard-coded for now - be careful if using this code for other functions
plane_res = 64
field_npix = 75   

def get_Q(z_source_bin_id, pws, g1, g2):
    # Gets the lensing efficiency matrix from binned sources

    num_bins = z_source_bin_id.max() + 1

    g1hat = []
    g2hat = []
    Q_0 = []

    num_gals_list = []

    for i in range(num_bins):
        # id is the source bin index
        gal_id = jnp.where(z_source_bin_id==i, jnp.ones_like(g1), jnp.zeros_like(g1))
        num_gals = jnp.sum(gal_id, axis=0)
        num_gals_list.append(num_gals.sum())
        num_gals = num_gals.at[num_gals == 0].set(1) # dont divide by zero

        g1_bin = jnp.where(z_source_bin_id==i, g1, jnp.zeros_like(g1))
        g1_bin_reduce = (jnp.sum(g1_bin, axis=0) / num_gals).reshape(75, 75) # average the ellipticities
        g1hat_bin = jnp.fft.fft2(g1_bin_reduce)
        g1hat.append(g1hat_bin)

        g2_bin = jnp.where(z_source_bin_id==i, g2, jnp.zeros_like(g2))
        g2_bin_reduce = (jnp.sum(g2_bin, axis=0) / num_gals).reshape(75, 75) # average the ellipticities
        g2hat_bin = jnp.fft.fft2(g2_bin_reduce)
        g2hat.append(g2hat_bin)

        q_bin_large = jnp.where(z_source_bin_id[...,None,None] == i, pws, jnp.zeros_like(pws))
        q_bin = jnp.sum(q_bin_large, axis=(0,1)) / jnp.sum(num_gals)
        Q_0.append(q_bin.flatten())

    g1hat = jnp.array(g1hat)
    g2hat = jnp.array(g2hat)

    Q = jnp.array(Q_0)[:, :18]
    return Q, g1hat, g2hat

def measure_power_spectrum(map_data):
  """
  measures power 2d data
  :param power: map
  :return: power spectrum
  """
  data_ft = jnp.fft.fft2(map_data) / map_data.shape[0]
  data = jnp.real(data_ft * jnp.conj(data_ft))
  freqs = jnp.fft.fftfreq(field_npix, 1/field_npix)
  x, y = jnp.meshgrid(freqs, freqs)
  r = jnp.sqrt((x)**2 + (y)**2)
  r = r.astype('int32')

  tbin = jnp.bincount(r.ravel(), data.ravel())
  nr = jnp.bincount(r.ravel())
  radialprofile = tbin / nr

  return radialprofile

def matrix_stuff(gamma, alpha, S, Q, N):
    foo = Q.T @ jnp.linalg.solve(N, gamma)
    foo2 = Q.T @ jnp.linalg.solve(N, Q) + jnp.nan_to_num(alpha * jnp.linalg.inv(S), nan=0, posinf=0, neginf=0)
    foo3 = jnp.linalg.solve(foo2, foo)
    return foo3

def invert_single(g1hat, g2hat, S, p1, p2, k2, alpha, Q, N):
    g1hat_inv = matrix_stuff(g1hat, alpha, S, Q, N)
    g2hat_inv = matrix_stuff(g2hat, alpha, S, Q, N)
    od_out_E = (p1 * g1hat_inv + p2 * g2hat_inv) / k2
    od_out_B = -(p2 * g1hat_inv - p1 * g2hat_inv) / k2

    return od_out_E, od_out_B

invert_v = jax.vmap(invert_single, in_axes=(1, 1, 0, 0, 0, 0, None, None, None))