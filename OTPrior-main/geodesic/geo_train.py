"""
train perturbation term nn for geodesic dist
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
from jax import jit, vmap, random
import optax
import geo_net
import geo_length
from core import sampling
from flax.training import train_state
import matplotlib.pyplot as plt

vectorized_geodesic_objective = vmap(geo_length.geodesic_objective, in_axes=(0, 0, None))

def create_train_state(rng, learning_rate, correction_net):
    params = correction_net.init(rng, jnp.ones((1, 3)), jnp.ones((1, 3)), jnp.ones((1, 1)))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=correction_net.apply, params=params, tx=tx)

def geodesic_train(learning_rate=1e-2, num_epochs=200, batch_size=200, num_batches=2):
    correction_net = geo_net.CorrectionNet()
    state = create_train_state(random.PRNGKey(10), learning_rate, correction_net)

    def loss_fn(params, x0_batch, x1_batch):
        loss = jnp.mean(vectorized_geodesic_objective(x0_batch, x1_batch, lambda x0, x1, t: correction_net.apply({'params': params}, x0, x1, t)))
        return loss

    @jit
    def update(state, x0_batch, x1_batch):
        loss, grads = jax.value_and_grad(loss_fn)(state.params, x0_batch, x1_batch)
        state = state.apply_gradients(grads=grads)
        return state, loss

    losses = []
    for epoch in range(num_epochs):
        data_samples = sampling.hypersphere(num_samples=batch_size*num_batches)
        for batch_start in range(0, len(data_samples) - batch_size, batch_size):
            batch_end = batch_start + batch_size
            x0_batch = data_samples[batch_start:batch_end]
            x1_batch = data_samples[batch_end:batch_end + batch_size]
            state, loss = update(state, x0_batch, x1_batch)
            losses.append(loss)
        print(f"Epoch {epoch}, Loss: {jnp.mean(loss)}")
    
    jnp.save('geo_params_reg.npy', state.params)
    print("Model trained and weights saved to 'geo_params.npy'.")
    plt.plot(losses)
    plt.show()

geodesic_train()
