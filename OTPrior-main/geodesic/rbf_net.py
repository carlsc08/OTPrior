from flax import linen as nn

class WeightNet(nn.Module):
    input_dim: int = 3
    num_clusters: int = 100
    hidden_dim: int = 100
    output_dim: int = input_dim*num_clusters

    def setup(self):
        self.dense1 = nn.Dense(features=self.hidden_dim)
        self.dense2 = nn.Dense(features=self.hidden_dim)
        self.dense3 = nn.Dense(features=self.hidden_dim)
        self.dense4 = nn.Dense(features=self.output_dim)

    def __call__(self, x):
        h = nn.relu(self.dense1(x))
        h = nn.relu(self.dense2(h))
        h = nn.relu(self.dense3(h))
        return self.dense4(h)