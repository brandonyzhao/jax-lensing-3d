import jax
from scipy.stats import gengamma 

def sample_photoz(num_samples, lb, ub, random_seed=0): 
    # Sample galaxy locations from generalized gamma distribution. 

    a = 2
    c = 1.5
    scale = (1/1.4)

    rv = gengamma(a, c, scale=scale)

    # Depending on the bounds, may have to change size to be larger
    samples = rv.rvs(size=2*num_samples, random_state=random_seed)
    samples = samples[(samples>lb) & (samples<ub)]

    return samples[:num_samples]

def weighted_proj(vol3d, pws): 
    ''' Weighted projection. 
    vol3d: shape (x*y*z, 2): flattened shear volume
    pws: shape (x*y, z, 1): one projection weight vector for every pixel location
    '''
    foo = vol3d.reshape(pws.shape[0], pws.shape[1], vol3d.shape[-1])
    vol_weighted  = foo * pws 
    return vol_weighted.sum(1) # Should be a sum!

map_coords_v = jax.vmap(jax.scipy.ndimage.map_coordinates, in_axes=(2, None, None), out_axes=1)

def coords_transform(coords, res): 
    ''' Map coordinates from [0, 1] to array indices [-0.5, res-0.5]'''
    return (coords * res) - 0.5

def interp_vol(vol3d, coords): 
    ''' Interpolate a 3D volume according to xy coordinates
    vol3D: shape (x, y, z, N)
    coords: shape (M, 2), ranging from 0 to 1
    '''
    vol3d_shape = vol3d.shape
    vol3d = vol3d.reshape(vol3d.shape[0], vol3d.shape[1], -1)
    coords = coords_transform(coords, vol3d.shape[0])
    out = map_coords_v(vol3d, coords.T, 1)
    return out.reshape(coords.shape[0], vol3d_shape[2], vol3d_shape[3])