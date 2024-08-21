import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from jax import jit, vmap
import jax.numpy as jnp
from core import sampling
from geodesic import rbf_net
from sklearn.cluster import KMeans

num_samples = 2000
data_samples_matrix = sampling.hypersphere(num_samples=num_samples)

# LAND
@jit    
def h_alpha(x, scale, alpha, sigma=0.075):
    diff = (data_samples_matrix[:, alpha] - x[alpha]) ** 2
    weights = jnp.exp(-jnp.sum((data_samples_matrix - x) ** 2, axis=1) / ( 2 * (scale * sigma) ** 2))
    h = jnp.sum(diff * weights[:, None], axis=0)
    return h

def G_LAND(x, scale, eps=1e-1):
    h = jnp.array([h_alpha(x, scale, alpha=a) for a in range(1, x.shape[0] + 1)])  # pg20 of metric FM paper: can take to arbitrary power
    diag_h = jnp.diag(h)
    G = jnp.linalg.inv(diag_h + eps*jnp.eye(len(h)))
    return G

# RBF
input_dim = 3

@jit
def h_alpha_rbf(x, weights, num_clusters=100):
    def centroids_card():
        "k-means clustering"
        k_means = KMeans(n_clusters=num_clusters, n_init='auto', random_state=1).fit(data_samples_matrix)
        centroids = jnp.array(k_means.cluster_centers_)
        cluster_labels = k_means.labels_
        cluster_cardinalities = jnp.array([jnp.sum(cluster_labels == k) for k in range(num_clusters)])
        return centroids, cluster_cardinalities

    def bandwidths(centroids, cluster_cardinalities, kappa=0.15):
        def bandwidth_k(centroid, cardinality):
            distances = jnp.sqrt(jnp.sum((x - centroid) ** 2, axis=0))
            bandwidth = (0.5 * kappa * distances / cardinality) ** -2
            return bandwidth
        bandwidths = vmap(bandwidth_k)(centroids, cluster_cardinalities)
        return bandwidths

    centroids, cardinalities = centroids_card()
    bandwidths = bandwidths(centroids, cardinalities)

    sq_dist = jnp.sum((centroids - x) ** 2, axis=1)
    exp_term = jnp.exp(-bandwidths * (sq_dist ** 2) / 2)
    h_alpha_rbf = weights.reshape(num_clusters, input_dim) * jnp.tile(exp_term[:, None], (1, 3))
    h_alpha_rbf = jnp.sum(h_alpha_rbf, axis=0)  # sum over clusters
    return h_alpha_rbf

def rbf_load_eval(x):
    loaded_params = jnp.load('rbf_params.npy', allow_pickle=True).item()
    weight_net = rbf_net.WeightNet()
    weights = weight_net.apply({'params': loaded_params}, x)
    return weights

def G_RBF(x, eps=1e-1):
    "post-training"
    weights = rbf_load_eval(x)
    h = h_alpha_rbf(x, weights)
    diag_h = jnp.diag(h)
    G = jnp.linalg.inv(diag_h + eps * jnp.eye(len(h)))
    return G
