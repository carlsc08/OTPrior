"""
Define data distribution for nn_learning_prior.py
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.random as jr
import jax.numpy as jnp
from core import distributions
import tensorflow_datasets as tfds

def moon_upper(num_samples):
    data = distributions.SklearnDistribution(
        name='moon_upper'
    )
    data_samples = data.generate_samples(
        jr.PRNGKey(1),
        num_samples=num_samples
    )
    rows = data_samples.shape[0]
    zeros_column = jnp.zeros((rows, 1))
    data_samples_matrix = jnp.concatenate((data_samples, zeros_column), axis=1)
    return data_samples_matrix

def s_curve(num_samples):
    data = distributions.SklearnDistribution(
        name='s_curve'
    )
    data_samples = data.generate_samples(
        jr.PRNGKey(1),
        num_samples=num_samples
    )
    rows = data_samples.shape[0]
    zeros_column = jnp.zeros((rows, 1))
    data_samples_matrix = jnp.concatenate((data_samples, zeros_column), axis=1)
    return data_samples_matrix

def hypersphere(num_samples, dim=3, key=jr.PRNGKey(42)):
    samples = jr.normal(key, (num_samples, dim))
    samples = samples / jnp.linalg.norm(samples, axis=1, keepdims=True)
    return samples

def multi_normal(num_samples):
    data_means = jnp.array(
        [
            (0, -1, -1),
            (0, -1, 1),
            (0, 1, -1),
            (0, 1, 1),
            (0, 0, 0)
        ]
    )
    data_covs = jnp.tile(
        .05 * jnp.eye(data_means.shape[1]),
        (len(data_means), 1, 1)
    )
    data = distributions.GaussianMixture(
        means=data_means,
        covs=data_covs,
        proportions=jnp.ones(len(data_means)) / len(data_means),
    )
    data_samples_matrix = data.generate_samples(
        jr.key(1), 
        num_samples=num_samples,
    )
    return data_samples_matrix

def noise_matrix(num_samples, key, input_dim):
    Gaussian = distributions.Gaussian(
        mean=jnp.zeros(shape=(input_dim,)),
        cov=jnp.eye(input_dim),
    )
    input_matrix = Gaussian.generate_samples(
        key, 
        num_samples=num_samples,
    )
    return input_matrix

def sample_learned_prior(num_samples, key, path, neural_prior, input_dim):
    loaded_params = jnp.load(path, allow_pickle=True).item()
    noise_vecs = noise_matrix(num_samples, key, input_dim)
    prior_samples_matrix = jax.vmap(lambda x: neural_prior.apply({'params': loaded_params}, x))(noise_vecs)
    return prior_samples_matrix

def get_mnist(labels=True):
    """
    Returns tuple: train_images[, train_labels], test_images[, test_labels]
    """
    data_dir = '/tmp/tfds'
    mnist_data, _ = tfds.load(name="mnist", batch_size=-1, data_dir=data_dir, with_info=True)
    mnist_data = tfds.as_numpy(mnist_data)
    train_data, test_data = mnist_data['train'], mnist_data['test']
    train_images = train_data['image'] / 255.0  # normalise to [0,1]
    train_images = jnp.reshape(train_images, (len(train_images), 28, 28, 1))
    train_labels = train_data['label']
    test_images = test_data['image'] / 255.0
    test_images = jnp.reshape(test_images, (len(test_images), 28, 28, 1))
    test_labels = test_data['label']
    if labels:
        return train_images, train_labels, test_images, test_labels
    return train_images, test_images

        



