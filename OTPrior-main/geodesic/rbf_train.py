import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import jax
from jax import random, jit, vmap
import jax.numpy as jnp
from core import sampling
from geodesic import mtensor, rbf_net
from flax.training import train_state
import optax

num_samples = 1500
data_samples = sampling.moon_upper(num_samples=num_samples)

def create_train_state(rng, learning_rate, model):
    params = model.init(rng, jnp.ones((1, model.input_dim)))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

weight_net = rbf_net.WeightNet()
state = create_train_state(random.PRNGKey(8), learning_rate=1e-2, model=weight_net)

@jit
def weights_loss(weights, data_samples=data_samples):
    def single_loss(x):
        return (1 - mtensor.h_alpha_rbf(x, weights)) ** 2

    mean_loss = jnp.mean(vmap(single_loss)(data_samples))
    return mean_loss

@jit
def train_step(state, x):
    def loss_fn(params):
        weights = state.apply_fn({'params': params}, x)
        loss = jnp.mean(weights_loss(weights))
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def rbf_train(data=data_samples, num_epochs=100):
    "batch-wise updating???"
    global state
    for epoch in range(num_epochs):
        loss_vals = []
        for x in data:
            state, loss = train_step(state, x)
            loss_vals.append(loss)
        avg_loss = jnp.mean(jnp.array(loss_vals))
        print(f'Epoch {epoch}, Loss: {avg_loss}')

    jnp.save('rbf_params.npy', state.params)
    print("Model trained and weights saved to 'rbf_params.npy'.")

rbf_train()
