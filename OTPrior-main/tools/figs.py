import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import jax.random as jr
from core import sampling, plotters
import matplotlib.pyplot as plt
import models.prior as prior

# prior
input_dim=3
num_samples=10000
path = '/Users/carlscandelius/Documents/HarvardUndergrad/OTPrior-main/saved_params/prior_base_cosine.npy'
prior_model = prior.PriorNet(3, 150, 3)
noise_batch = sampling.noise_matrix(num_samples, jr.PRNGKey(0), input_dim)
prior_samples = sampling.sample_learned_prior(num_samples, jr.PRNGKey(0), path, prior_model, input_dim)

plotters.kmeans_plot(prior_samples)
plt.show()

# latent and reconstruction
# plotters.mnist_img_plot(xs, reconstructed, 10)
# plotters.latent_plot(train_data, train_labels, vae_model, state.params['params'], num_plot_samples, prob_toggle, use_tsne=True)