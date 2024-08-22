import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import jax
import jax.numpy as jnp
from ott import geometry
from ott.geometry import costs
from ott.problems.quadratic import quadratic_problem
from ott.solvers.quadratic import gromov_wasserstein
from typing import Optional, Literal, Mapping, Union
import optax
import matplotlib.pyplot as plt
from core import plotters, sampling
from geodesic import geo_class
import models.prior as prior
import jax.random as jr
import functools as ft
import yaml

config = sys.argv[1]
with open(config, 'r') as file:
    config = yaml.safe_load(file)

input_dim, hidden_dim, output_dim, lr, epochs, batch_size, batches = config['prior'].values()
num_plot_samples = config['plotting']['num_plot_samples']
params_path, config_name, metric = config['settings'].values()
data_type = config['data']['source']
data = {
    "mnist": sampling.get_mnist(labels=False)
}
train_data, test_data = data[data_type]
data = train_data
metrics = {
    "cosine": costs.Cosine(),
    "sqeuclidean": costs.SqEuclidean(),
    #"geodesic": geo_class.Geodesic()
}
CostScaling = Union[
    bool,
    int,
    float,
    Literal[
        "mean",
        "max_cost",
        "median",
        "mean_absolute",
        "max_cost_absolute",
        "median_absolute",
    ],
]

def get_data_batch(data, key, num_samples):
    num_data_samples = data.shape[0]
    random_indices = jr.choice(key, num_data_samples, shape=(num_samples,), replace=False)
    data = jnp.asarray(data)
    batch = data[random_indices]
    batch = batch.reshape(num_samples, -1)  # flatten for e.g. mnist
    return batch

def get_scale_cost(
    cost_matrix: jnp.ndarray, scale_cost: CostScaling
) -> CostScaling:
    r"""Get cost a float scaling to scale the ``cost_matrix``.

    Currently, OTT-JAX ``scale_cost`` ["mean", "median", "max_cost]
    cannot be computed on absolute value of cost matrix, which can lead to
    undesirable behavior when the cost may be negative.
    This function provides the possibility to manually computes one of these
    three ``scale_cost`` but based on the absolute value of the cost matrix.
    In that case, the returned float can be set as the ``scale_cost``
    attribute of ``geom``.
    If ``scale_cost`` is already in ["mean", "median", "max_cost], it computes
    the scaling as done in OTT-JAX.
    """
    if isinstance(scale_cost, float):
        return scale_cost
    if not scale_cost:
        return 1.0

    # prepare scaling computation depending on whether scaling is computed on
    # the absolute value of the cost matrix or not
    suffix = "_absolute"
    is_abs = scale_cost.endswith(suffix)
    pre_scale_cost = jnp.abs if is_abs else lambda arr: arr
    base_scale_cost = scale_cost[: -len(suffix)] if is_abs else scale_cost

    if base_scale_cost == "mean":
        scale_cost = jnp.nanmean(pre_scale_cost(cost_matrix))
    elif base_scale_cost == "median":
        scale_cost = jnp.nanmedian(pre_scale_cost(cost_matrix))
    elif base_scale_cost == "max_cost":
        scale_cost = jnp.nanmax(pre_scale_cost(cost_matrix))
    else:
        raise NotImplementedError(
            "Absolute cost scaling based on base cost scaling "
            f"{base_scale_cost} is not implemented."
        )
    return scale_cost

def create_geom(
    x: jnp.ndarray, y: jnp.ndarray, cost_fn: costs.CostFn, scale_cost: CostScaling
):
    r"""Define ``Geometry`` by manually setting ``scale_cost``.

    ``scale_cost`` is set posteriorly to be able to define custom scalings
    from the cost matrices, like ``scale_cost`` computed on the absolute
    value of the cost matrix using the ``get_scale_cost``function.
    """
    cost_matrix = cost_fn.all_pairs(x=x, y=y)
    if type(cost_fn) is costs.Cosine:
        cost_matrix = jnp.clip(cost_matrix, a_min=1e-6, a_max=None)
    _scale_cost = get_scale_cost(cost_matrix, scale_cost)
    _cost_matrix = cost_matrix / jax.lax.stop_gradient(_scale_cost)
    geom = geometry.geometry.Geometry(cost_matrix=_cost_matrix)
    return geom

def GW_loss(
        params, x: jnp.ndarray, key, scale_cost: Optional[Literal['mean', 'median', 'max_cost']] = "mean"
):
    output = jax.vmap(lambda x: prior_net.apply({'params': params}, x))(x)
    data_samples = get_data_batch(data, key, batch_size)

    if isinstance(scale_cost, Mapping):
        assert "scale_cost_xx" in scale_cost
        assert "scale_cost_yy" in scale_cost
        scale_cost_xx = scale_cost["cost_fn_xx"]
        scale_cost_yy = scale_cost["cost_fn_yy"]
    else:
        scale_cost_xx = scale_cost_yy = scale_cost

    geom_xx = create_geom(
        x=data_samples, y=data_samples, cost_fn=metrics[metric], scale_cost=scale_cost_xx
    )
    geom_yy = create_geom(
        x=output, y=output, cost_fn=costs.SqEuclidean(), scale_cost=scale_cost_yy
    )
    prob = quadratic_problem.QuadraticProblem(
        geom_xx, geom_yy, scale_cost=scale_cost
    )
    out = gromov_wasserstein.GromovWasserstein(epsilon=1e-2)(prob)
    return out.reg_gw_cost

prior_net = prior.PriorNet(input_dim, hidden_dim, output_dim)
params = prior_net.init(jr.PRNGKey(0), jnp.ones((1, input_dim)))['params']
adam = optax.adam(lr)
opt_state = adam.init(params)
losses = []

@jax.jit
def update(params, opt_states, xs, key):
    def batch_loss_fn(params):
        partial_loss_fn = ft.partial(GW_loss, params)
        loss = partial_loss_fn(xs, key)
        return loss
    loss, grads = jax.value_and_grad(batch_loss_fn)(params)
    updates, opt_states = adam.update(grads, opt_states, params)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_states

for epoch in range(epochs):
    epoch_loss = 0.0
    keys = jr.split(jr.PRNGKey(0), batches)
    for i in range(batches):
        batch_x = sampling.noise_matrix(batch_size, keys[i], input_dim)
        loss, params, opt_state = update(params, opt_state, batch_x, keys[i])
        epoch_loss += loss
    epoch_loss /= batches
    losses.append(float(epoch_loss))
    print("Epoch", epoch, "GW loss:", epoch_loss)

path = os.path.join("saved_params/", f'prior_{config_name}_{data_type}_{metric}.npy')
jnp.save(path, params)
print("Model trained and weights saved.")
plt.figure()
plt.plot(losses)

noise_vecs = sampling.noise_matrix(num_plot_samples, jr.PRNGKey(0), input_dim)
prior_samples = jax.vmap(lambda x: prior_net.apply({'params': params}, x))(noise_vecs)
if input_dim < 4:
    plotters.kmeans_plot(prior_samples)
plotters.tsne_kmeans_plot(prior_samples)
plt.show()