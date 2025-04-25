from tqdm import tqdm

import os
from functools import partial
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, default='0', help='Visible GPU IDs')
parser.add_argument('-vol', type=str, default='pm0', help='Volume to recover')
parser.add_argument('-L', type=int, default=2, help='Positional degree (xy)')
parser.add_argument('-Lz', type=int, default=5, help='Positional encoding degree (z)')
parser.add_argument('-proj', type=str, default='gg', help='Projection Type')
parser.add_argument('-gpp', type=int, default=1, help='Galaxies per Shear Pixel')
parser.add_argument('-shape_sigma', type=float, default=0., help='Standard deviation of shape noise')
parser.add_argument('-lam', type=float, default=5., help='Power Spectrum Regularization Weight')
parser.add_argument('-init_seed', type=int, default=0, help='Initialization seed for MLP weights')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.d
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
from jax.tree_util import Partial
from jax import numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from optax._src import linear_algebra

from shear.grid import get_X_harm
from shear.network import MLP
from shear.helpers import makedir
from shear.photoz import sample_photoz, weighted_proj, interp_vol
from shear.fourier import ks93inv_batch
from shear.planes import cosmo, get_planes, get_source_scales
from shear.metrics import pearson_cc
from shear.spectral import measure_spectra

exp = ''
for key in vars(args).keys(): 
    if key != 'd':
        exp += key
        exp += '_'
        exp += str(vars(args)[key])
        exp += '_'
exp = exp[:-1]
print('Experiment Run: ', exp)

# Set up exp dir
exp_str = './exp/'
makedir(exp_str + exp)
makedir(exp_str + exp + '/checkpoints')
makedir(exp_str + exp + '/eta_out')
makedir(exp_str + exp + '/density')

rand_key = jax.random.PRNGKey(1)

# ========================================================
# Load Dark Matter Field
# ========================================================

# Particle mesh simulation output
if args.vol == 'pm0': 
    lensplanes = jnp.load('./pm_fields/pm0.npy', allow_pickle=True)
else: 
    raise ValueError

od_gt = get_planes(lensplanes)
eta_shape = od_gt.shape 
resxy = od_gt.shape[0]
res_z = od_gt.shape[2]

mean_od = jnp.mean(od_gt, axis=(0,1), keepdims=True)
gt_ps = measure_spectra(od_gt, None)
ps_len = gt_ps.shape[1]
deg_point = (args.L, args.L, args.Lz)

# Instantiate MLP
sig_scale = od_gt.max() - od_gt.min() + 0.4
sig_shift = -0.2

eta_MLP = MLP(net_depth=4,
                net_width=256,
                activation=nn.relu,
                out_channel=1,
                do_skip=False,
                deg_point=deg_point,
                sig_scale=sig_scale,
                sig_shift=sig_shift)

# Initialize parameters
init_key = jax.random.PRNGKey(args.init_seed)
mlp_params_eta = eta_MLP.init(init_key, jnp.ones([3]))['params']

num_iters = 10000
lr_init = 1e-4
lr_final = 5e-6
tx = optax.adam(lambda x : jnp.exp(jnp.log(lr_init) * (1 - (x/num_iters)) + jnp.log(lr_final) * (x/num_iters)), nesterov=True)

opt_state = train_state.TrainState.create(apply_fn=eta_MLP.apply, params=mlp_params_eta, tx=tx)

# ========================================================
# Functions for training
# ========================================================

X_nn = get_X_harm(resxy, res_z)

def get_pred_nn(apply_fn, params): 
    def predict_nn(x): 
        return apply_fn({'params': params}, x)
    return predict_nn    

shape_sigma = args.shape_sigma

def lossfn_chi2_psreg(params, apply_fn, fwd_model, pws, target): 
    predict_nn = get_pred_nn(apply_fn, params)
    model_out = fwd_model(predict_nn, pws)

    sq_err = jnp.square(jnp.abs(model_out - target) / shape_sigma)
    loss_meas = jnp.mean(sq_err)
    d_nn = predict_nn(X_nn).reshape(eta_shape)
    d_ps = measure_spectra(d_nn, ps_len)
    loss_reg = jnp.mean(jnp.square(d_ps - gt_ps))

    loss = loss_meas + args.lam * loss_reg
    
    return loss, [model_out, loss_meas, loss_reg]

def lossfn_meanreg(params, apply_fn, fwd_model, pws, target): 
    predict_nn = get_pred_nn(apply_fn, params)
    model_out = fwd_model(predict_nn, pws)

    loss_meas = jnp.mean(jnp.square(jnp.abs(model_out - target)) / 0.035)
    d_nn = predict_nn(X_nn).reshape(eta_shape)
    d_mean = jnp.mean(d_nn, axis=(0,1), keepdims=True)
    loss_reg = jnp.mean(jnp.square(d_mean - mean_od))

    loss = loss_meas + args.lam * loss_reg
    
    return loss, [model_out, loss_meas, loss_reg]

def lossfn_noreg(params, apply_fn, fwd_model, pws, target): 
    predict_nn = get_pred_nn(apply_fn, params)
    model_out = fwd_model(predict_nn, pws)

    loss = jnp.mean(jnp.square(jnp.abs(model_out - target)))
    
    return loss, [model_out, loss, 1e-10]

@partial(jax.jit, static_argnums=(0,4))
def train_step(fwd_model, pws, target, opt_state, lossfn, key): 
    key, new_key = jax.random.split(key)
    vals, grad = jax.value_and_grad(lossfn, argnums=(0), has_aux=True)(opt_state.params, opt_state.apply_fn, fwd_model, pws, target)
    grad_norm = linear_algebra.global_norm(grad)
    opt_state = opt_state.apply_gradients(grads=grad)
    loss, [model_out, loss_meas, loss_reg] = vals

    return loss, loss_meas, loss_reg, model_out, opt_state, new_key, grad_norm

@partial(jax.jit, static_argnums=(0,1,2))
def test_step(fwd_model_od, fwd_model_shear, fwd_model_train, pws, opt_state): 
    predict_nn = get_pred_nn(opt_state.apply_fn, opt_state.params)
    od_nn = fwd_model_od(predict_nn=predict_nn)
    shear_nn = fwd_model_shear(predict_nn=predict_nn)
    shear_map = fwd_model_train(predict_nn, pws)
    return od_nn, shear_nn, shear_map

# ========================================================
# Photometric Measurements 
# ========================================================

if args.proj == 'gg' or args.proj == 'gg_uniform': 
    pws = []
    zs = []
    for i in range(args.gpp):
        z_source = sample_photoz(resxy*resxy, 0., 2., random_seed = i).reshape((resxy*resxy, 1))
        source_scales = get_source_scales(cosmo, lensplanes, z_source)
        pw = source_scales.reshape((resxy*resxy, res_z, 1))
        pws.append(pw)
        zs.append(z_source)
    pws = jnp.array(pws)
    pws_vis = pws.sum(0)

    zs = jnp.array(zs)
    jnp.save(exp_str + exp + '/z_source.npy', zs)
    jnp.save(exp_str + exp + '/pws.npy', pws)
else:
    raise ValueError

# ========================================================
# Measurement model for NN output
# ========================================================

od_bm = jnp.zeros(od_gt.shape[:-1]) # Zeros for overdensity b-mode

def fwd_model_od(predict_nn): 
    return predict_nn(X_nn).reshape(-1, 1)
def fwd_model_shear_bm(predict_nn, od_bm):
    od_nn = predict_nn(X_nn).reshape(eta_shape)
    e1, e2 = ks93inv_batch(od_nn, od_bm)
    return jnp.stack([e1, e2], axis=-1).reshape(-1, 2)

fwd_model_shear = Partial(fwd_model_shear_bm, od_bm=od_bm)
e1, e2 = ks93inv_batch(od_gt, od_bm)

fwd_model_up = fwd_model_shear 
target = jnp.stack([e1, e2], axis=-1).reshape(-1, 2)

if args.proj == 'None': 
    fwd_model_train = lambda predict_nn, pws: fwd_model_up(predict_nn) 
elif args.proj == 'gg_uniform':
    # put galaxies at off-pixel locations and interpolate
    pws = pws.reshape(pws.shape[0]*pws.shape[1], pws.shape[2], pws.shape[3])
    gal_coords = jax.random.uniform(rand_key, shape=(args.gpp*resxy*resxy, 2))
    jnp.save(exp_str + exp + '/gal_coords.npy', gal_coords)
    def fwd_model_train(predict_nn, pws): 
        vol3d = fwd_model_up(predict_nn).reshape(resxy, resxy, res_z, -1)
        vol_interp = interp_vol(vol3d, gal_coords)
        return weighted_proj(vol_interp, pws)
    target = interp_vol(target.reshape(resxy, resxy, res_z, -1), gal_coords)
    target = weighted_proj(target, pws)
    fwd_model_train_batch = fwd_model_train
else:  
    def fwd_model_train(predict_nn, pws): 
        vol3d = fwd_model_up(predict_nn)
        return weighted_proj(vol3d, pws)
    target = jnp.array([weighted_proj(target, pw) for pw in pws])

    fwd_model_train_batch = jax.vmap(fwd_model_train, in_axes=(None, 0))

# ========================================================
# Shape Noise
# ========================================================
if args.shape_sigma > 0: 
    noise_key = jax.random.PRNGKey(0) 
    noise = jax.random.normal(noise_key, shape=target.shape) * args.shape_sigma 
    print('========================================')
    print('Mean Signal Amplitude: ', jnp.mean(jnp.abs(target)))
    print('Mean Noise Amplitude: ', jnp.mean(jnp.abs(noise)))
    print('SNR Level Per Measurement: ', jnp.mean(jnp.abs(target)) / jnp.mean(jnp.abs(noise)))
    print('========================================')
    target = target + noise

# ========================================================
# Save a copy of ground truth density
# ========================================================

jnp.save(open(exp_str + exp + '/density_gt.p', 'wb'), od_gt)

# Training Loop

test_iter = 200

if args.shape_sigma > 0: 
    lossfn = lossfn_chi2_psreg
elif args.lam == 0: 
    lossfn = lossfn_noreg 
else: 
    lossfn = lossfn_meanreg

early_stop = args.shape_sigma > 0.

for i in (pbar := tqdm(range(1, num_iters+1))):
    loss, loss_meas, loss_reg, _, opt_state, rand_key, grad_norm = train_step(fwd_model_train_batch, pws, target, opt_state, lossfn, rand_key)
    _, rand_key = jax.random.split(rand_key)

    if (i==1 or i % test_iter == 0):
        od_nn, shear_nn, shear_map = test_step(fwd_model_od, fwd_model_shear, fwd_model_train_batch, pws, opt_state) 
        od_nn = od_nn.reshape(od_gt.shape)
        shear_nn = shear_nn.reshape(od_gt.shape + (2,))

        pbar.set_description('Loss Meas: %f Loss Reg: %f'% (loss_meas, loss_reg))
        jnp.save(open(exp_str + exp + '/density/density_%d.p'%(i//test_iter), 'wb'), od_nn)