"""
5-layer MLP for learning prior
"""

from flax import linen as nn

class PriorNet(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int

    def setup(self):
        self.dense1 = nn.Dense(features=self.hidden_dim)
        self.dense2 = nn.Dense(features=self.hidden_dim)
        self.dense3 = nn.Dense(features=self.hidden_dim)
        self.dense4 = nn.Dense(features=self.hidden_dim)
        self.dense5 = nn.Dense(features=self.output_dim)

    def __call__(self, x):
        x = nn.relu(self.dense1(x))
        x = nn.relu(self.dense2(x))
        x = nn.relu(self.dense3(x))
        x = nn.relu(self.dense4(x))
        return self.dense5(x)