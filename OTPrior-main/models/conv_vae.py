"""
Convolutional autoencoder with variational option
"""

import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn

class Encoder(nn.Module):
    input_dim: int
    hidden_dim: int
    latent_dim: int

    def setup(self):
        self.conv1 = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2))
        self.conv2 = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))
        self.dense3 = nn.Dense(features=128)
        self.dense4 = nn.Dense(features=self.latent_dim)
        self.dense_mu = nn.Dense(features=self.latent_dim)
        self.dense_log_var = nn.Dense(features=self.latent_dim)

    def __call__(self, x, prob_toggle: bool):
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = x.reshape((-1))
        x = nn.relu(self.dense3(x))
        if prob_toggle:
            mu = self.dense_mu(x)
            log_var = self.dense_log_var(x)
            return mu, log_var
        else:
            return self.dense4(x)

class Decoder(nn.Module):
    latent_dim: int
    hidden_dim: int
    output_dim: int

    def setup(self):
        self.dense1 = nn.Dense(features=7*7*64)
        self.convt1 = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME')
        self.convt2 = nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME')
        self.convt3 = nn.ConvTranspose(features=1, kernel_size=(3, 3), strides=(1, 1), padding='SAME')

    def __call__(self, x):
        x = nn.relu(self.dense1(x))
        x = x.reshape((7, 7, 64))
        x = nn.relu(self.convt1(x))
        x = nn.relu(self.convt2(x))
        return nn.sigmoid(self.convt3(x))

class VAEModel(nn.Module):
    encoder: Encoder
    decoder: Decoder

    def reparam(self, mu, std, key):
            eps = jr.normal(key)
            z = mu + std*eps
            return z
    
    def GaussianPosterior(self, x, key, test: bool):
        mu, log_var = self.encoder(x, prob_toggle=True)
        std = jnp.exp(0.5 * log_var)
        if test:
            z = mu
        else:
            z = self.reparam(mu, std, key)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var

    def __call__(self, x, key, prob_toggle: bool, test: bool = False):
        if prob_toggle:
            x_hat, mu, log_var = self.GaussianPosterior(x, key, test)
            return x_hat, mu, log_var
        else:
            z = self.encoder(x, prob_toggle=False)
            x_hat = self.decoder(z)
            return x_hat, z