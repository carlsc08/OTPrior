import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from core import sampling, plotters
import matplotlib.pyplot as plt
from models import prior
import jax.random as jr
from ott.geometry import pointcloud
from ott.tools import sinkhorn_divergence
import functools as ft
from models import conv_vae, linear_vae
import yaml

config = sys.argv[1]
with open(config, 'r') as file:
    config = yaml.safe_load(file)
    
input_dim, hidden_dim, latent_dim, input_shape, lr, epochs, batch_size, batches, sae_prob_toggle, vae_prob_toggle, conv_toggle = config['autoencoders'].values()
scale_rn, scale_sd, scale_kl = config['scaling'].values()
num_plot_samples = config['plotting']['num_plot_samples']
params_path, config_name, metric = config['settings'].values()
data_type = config['data']['source']
data = {
    "mnist": sampling.get_mnist(labels=True)
}
train_data, train_labels, test_data, test_labels = data[data_type]
use_sinkhorn, max_iterations = config['sinkhorn'].values()
if use_sinkhorn:
    prob_toggle = sae_prob_toggle
else:
    prob_toggle = vae_prob_toggle

if conv_toggle:
    encoder = conv_vae.Encoder(input_dim, hidden_dim, latent_dim)
    decoder = conv_vae.Decoder(latent_dim, hidden_dim, input_dim)
    vae_model = conv_vae.VAEModel(encoder, decoder)
else:
    encoder = linear_vae.Encoder(input_dim, hidden_dim, latent_dim)
    decoder = linear_vae.Decoder(latent_dim, hidden_dim, input_dim)
    vae_model = linear_vae.VAEModel(encoder, decoder)

def sink_div(source, target, epsilon=None, max_iterations=max_iterations):
    """Return the Sinkhorn divergence cost and OT output given point clouds.
    Since ``y`` is fixed, we can use the flag ``static_b=True`` to avoid
    computing the ``reg_ot_cost(y, y)`` term.
    """
    geom = pointcloud.PointCloud(x=source, y=target, scale_cost='mean')
    ot = sinkhorn_divergence.sinkhorn_divergence(
        geom=geom,
        x=source,
        y=target,
        epsilon=epsilon,
        static_b=True,
        sinkhorn_kwargs={'max_iterations': max_iterations},
    )
    return ot.divergence

def loss_function(params, xs, p_key, rp_keys, prob_toggle=prob_toggle, use_sinkhorn=use_sinkhorn):
    if prob_toggle:
        x_hats, mus, log_vars = jax.vmap(lambda x, rp_key: vae_model.apply({'params': params['params']}, x, rp_key, prob_toggle=True))(xs, rp_keys)
        kld = jnp.mean(0.5 * jnp.sum(-1 - log_vars + mus**2 + jnp.exp(log_vars)))
        latent_vecs = mus
    else:
        x_hats, zs = jax.vmap(lambda x, rp_key: vae_model.apply({'params': params['params']}, x, rp_key, prob_toggle=False))(xs, rp_keys)
        kld = 0
        latent_vecs = zs
    reconstruction = jnp.mean(jnp.square(xs - x_hats))
    if use_sinkhorn:
        prior_samples = sampling.sample_learned_prior(batch_size, p_key, f'saved_params/prior_{config_name}_{data_type}_{metric}.npy', prior_model, latent_dim)
        sinkd = sink_div(latent_vecs, prior_samples) / batch_size
        return scale_rn*reconstruction + scale_sd*sinkd + scale_kl*kld
    else:
        return scale_rn*reconstruction + scale_kl*kld

@jax.jit
def train_step(state, xs, p_key, rp_keys):
    def batch_loss_fn(params):
        partial_loss_fn = ft.partial(loss_function, params)
        loss = partial_loss_fn(xs, p_key, rp_keys)
        return loss
    loss, grads = jax.value_and_grad(batch_loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state

def create_train_state(model, key, learning_rate=lr, prob_toggle=prob_toggle):
    rng, rp_rng = jr.split(key, 2)
    params = model.init(rng, jnp.ones(shape=input_shape), rp_rng, prob_toggle, test=False)
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params={'params': params['params']}, tx=tx)

def batch_generator(images):
    dataset_size = len(images)
    indices = jnp.arange(dataset_size)    
    for i in range(batches):
        batch_indices = indices[i * batch_size:(i + 1) * batch_size]
        yield images[batch_indices]

if use_sinkhorn:
    prior_in, prior_hid, prior_out = config['prior']['input_dim'], config['prior']['hidden_dim'], config['prior']['output_dim'] 
    prior_model = prior.PriorNet(prior_in, prior_hid, prior_out)

state = create_train_state(vae_model, key=jr.PRNGKey(0))
losses = []
p_key = jr.PRNGKey(0)
rp_key = jr.PRNGKey(1)

for epoch in range(epochs):
    train_gen = batch_generator(train_data)
    epoch_loss = 0
    for x_batch in train_gen:
        rp_key, rp_subkey = jr.split(rp_key)
        rp_subkeys = jr.split(rp_subkey, batch_size)
        p_key, p_subkey = jr.split(p_key)
        batch_loss, state = train_step(state, x_batch, p_subkey, rp_subkeys)
        epoch_loss += batch_loss
    epoch_loss /= batches
    print(f"Epoch {epoch + 1}\tAverage Loss: {epoch_loss}")
    losses.append(epoch_loss)

if use_sinkhorn:
    model_type = 'sae'
else:
    model_type = 'vae'
path = os.path.join("saved_params/", f'{model_type}_{config_name}_{data_type}_{metric}.npy')
jnp.save(path, state.params)
print("Model trained and weights saved.")
plt.figure()
plt.plot(losses)

reconstructed = []
latent = []
test_recon_loss = 0
test_gen = batch_generator(test_data)

if prob_toggle:
    for xs in test_gen:
        x_hats, mus, _ = jax.vmap(lambda x: vae_model.apply({'params': state.params['params']}, x, jr.PRNGKey(0), prob_toggle=True, test=True))(xs)
        test_recon_loss += jnp.mean(jnp.square(xs - x_hats))
        reconstructed.append(x_hats)
        latent.append(mus)
else:
    for xs in test_gen:
        x_hats, zs = jax.vmap(lambda x: vae_model.apply({'params': state.params['params']}, x, jr.PRNGKey(0), prob_toggle=False))(xs)
        test_recon_loss += jnp.mean(jnp.square(xs - x_hats))
        reconstructed.append(x_hats)
        latent.append(zs)

print("Average test reconstruction loss:", test_recon_loss / batches)
reconstructed = jnp.concatenate(reconstructed, axis=0)
latent = jnp.concatenate(latent, axis=0)

plotters.mnist_img_plot(test_data, reconstructed, 10)
plotters.latent_plot(train_data, train_labels, vae_model, state.params['params'], num_plot_samples, prob_toggle, use_tsne=True)
if latent_dim == 2:
    plotters.mnist_interp_euclidean(100, vae_model.decoder, state.params['params']['decoder'])
plt.show()