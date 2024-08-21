"""
def neural net class for perturbation term \phi in geodesic metric func
"""
import jax.numpy as jnp
from flax import linen as nn

def zero_init(rng, shape, dtype=jnp.float32):
    return jnp.zeros(shape, dtype)

class CorrectionNet(nn.Module):
    input_dim: int = 3
    hidden_dim: int = 300

    def setup(self):
        self.dense1 = nn.Dense(features=self.hidden_dim, kernel_init=zero_init, bias_init=zero_init)
        self.dense2 = nn.Dense(features=self.hidden_dim, kernel_init=zero_init, bias_init=zero_init)
        self.dense3 = nn.Dense(features=self.hidden_dim, kernel_init=zero_init, bias_init=zero_init)
        self.dense4 = nn.Dense(features=self.hidden_dim, kernel_init=zero_init, bias_init=zero_init)
        self.dense5 = nn.Dense(features=self.input_dim, kernel_init=zero_init, bias_init=zero_init)

    def __call__(self, x0, x1, t):
        array_t = jnp.tile(t, self.input_dim)
        x = jnp.concatenate([x0, x1, array_t], axis=-1)
        h = nn.relu(self.dense1(x))
        h = nn.relu(self.dense2(h))
        h = nn.relu(self.dense3(h))
        h = nn.relu(self.dense4(h))
        return self.dense5(h)