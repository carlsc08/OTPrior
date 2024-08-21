"""
Linear autoencoder with variational option
"""

import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn

class Encoder(nn.Module):
    input_dim: int
    hidden_dim: int
    latent_dim: int

    def setup(self):
        self.dense1 = nn.Dense(features=self.hidden_dim)
        self.dense2 = nn.Dense(features=self.hidden_dim)
        self.dense3 = nn.Dense(features=self.latent_dim)
        self.dense_log_var = nn.Dense(self.latent_dim)
        self.dense_mu = nn.Dense(features=self.latent_dim)

    def __call__(self, x, prob_toggle: bool = False):
        x = nn.relu(self.dense1(x))
        x = nn.relu(self.dense2(x))
        if prob_toggle:
            mu = self.dense_mu(x)
            log_var = self.dense_log_var(x)
            return mu, log_var
        else:
            return self.dense3(x)

class Decoder(nn.Module):
    latent_dim: int
    hidden_dim: int
    output_dim: int

    def setup(self):
        self.dense1 = nn.Dense(features=self.hidden_dim)
        self.dense2 = nn.Dense(features=self.hidden_dim)
        self.dense3 = nn.Dense(features=self.output_dim)

    def __call__(self, x):
        x = nn.relu(self.dense1(x))
        x = nn.relu(self.dense2(x))
        return self.dense3(x)

class VAEModel(nn.Module):
    encoder: Encoder
    decoder: Decoder

    def reparam(self, mu, std, key):
            eps = jr.normal(key)
            z = mu + std*eps
            return z
    
    def GaussianPosterior(self, x, key, test: bool = False):
        mu, log_var = self.encoder(x, prob_toggle=True)
        std = jnp.exp(0.5 * log_var)
        if test:
            z = mu
        else:
            z = self.reparam(mu, std, key)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var
    
    def __call__(self, x, key, prob_toggle: bool = False, test: bool = False):
        if prob_toggle:
            return self.GaussianPosterior(x, key, test)
        else:
            z = self.encoder(x)
            x_hat = self.decoder(z)
            return x_hat, z