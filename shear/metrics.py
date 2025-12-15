from jax import numpy as jnp
from scipy.ndimage import gaussian_filter

def norm_fro(arr): 
    return jnp.sqrt(jnp.sum(jnp.square(arr)))

def pearson_cc(vol1, vol2): 
    vol2 = vol2.reshape(vol1.shape)
    return jnp.sum(vol1 * vol2) / (norm_fro(vol1) * norm_fro(vol2))

def optimal_blur(density_gt, density_recon, sigma_xys, sigma_zs):
    optimal_cc = 0.
    optimal_density_gt = None 
    optimal_sig_xy = None 
    optimal_sig_z = None
    for sig_xy in sigma_xys: 
        for sig_z in sigma_zs: 
            density_gt_blurred = gaussian_filter(density_gt, (sig_xy, sig_xy, sig_z))
            cc = pearson_cc(density_gt_blurred, density_recon)
            if cc > optimal_cc: 
                optimal_cc = cc
                optimal_density_gt = density_gt_blurred 
                optimal_sig_xy = sig_xy 
                optimal_sig_z = sig_z 
    return optimal_cc, optimal_density_gt, optimal_sig_xy, optimal_sig_z


def mean_subtract(arr, axis=1): 
    return arr - jnp.mean(arr, axis=axis, keepdims=True)

def shear_mse(arr1, arr2): 
    arr1 = mean_subtract(arr1)
    arr2 = mean_subtract(arr2)

    return jnp.mean(jnp.square(arr1 - arr2))

def shear_psnr(gt, recon): 
    maxval = jnp.abs(mean_subtract(gt)).max()
    mse = shear_mse(gt, recon)
    return 10 * jnp.log10(maxval**2 / (mse + 1e-8))