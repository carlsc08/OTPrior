import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from jax import random
import jax.numpy as jnp
from geodesic import geo_net, geo_length
from core import sampling
import matplotlib.pyplot as plt

def geodesic_load_eval(x0, x1):
    loaded_params = jnp.load('geo_params_reg.npy', allow_pickle=True).item()
    correction_net = geo_net.CorrectionNet()

    def phi(x0, x1, t):
        return correction_net.apply({'params': loaded_params}, x0, x1, t)
    geodesic_len, x_h = geo_length.geodesic_length(x0, x1, phi)
    print(geodesic_len)
    return geodesic_len, x_h

data_samples_matrix = sampling.hypersphere(num_samples=200)
x0 = data_samples_matrix[0]
x1 = data_samples_matrix[150]

geodesic_len, x_h = geodesic_load_eval(x0, x1)
print(x_h)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_samples_matrix[:, 0], data_samples_matrix[:, 1], data_samples_matrix[:, 2], color='blue', label='Data Samples')
ax.scatter(x_h[:, 0], x_h[:, 1], x_h[:, 2], color='red', label='Geodesic Points')

# axes same scale
max_range = jnp.array([data_samples_matrix[:, 0].max() - data_samples_matrix[:, 0].min(),
                       data_samples_matrix[:, 1].max() - data_samples_matrix[:, 1].min(),
                       data_samples_matrix[:, 2].max() - data_samples_matrix[:, 2].min()]).max() / 2.0

mid_x = (data_samples_matrix[:, 0].max() + data_samples_matrix[:, 0].min()) * 0.5
mid_y = (data_samples_matrix[:, 1].max() + data_samples_matrix[:, 1].min()) * 0.5
mid_z = (data_samples_matrix[:, 2].max() + data_samples_matrix[:, 2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()
