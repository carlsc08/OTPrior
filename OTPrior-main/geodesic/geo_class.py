"""
New class for geodesic metric for PointCloud object in GW solver
"""

import jax
import jax.numpy as jnp
from typing import Any, Tuple
from ott.geometry.costs import TICost
from geodesic import geo_net, geo_length

@jax.tree_util.register_pytree_node_class
class Geodesic(TICost):
  r"""Euclidean-based geodesic distance.
  Implemented as a translation invariant cost, :math:`h(z) = \|z\|^2`."""
  
  def __init__(self) -> None:
    super().__init__()
    self.correction_net = geo_net.CorrectionNet()
    self.params = jnp.load('geo_params_reg.npy', allow_pickle=True).item()

  # def norm(self, x: jnp.ndarray) -> Union[float, jnp.ndarray]:
  #   return jnp.sum(x ** 2, axis=-1)

  def pairwise(self, x, y: jnp.ndarray) -> float:
    """DIFFERS FROM SqEucl: THIS IS ENTIRE DISTANCE FUNCTION, NOT JUST PAIRWISE COMPONENT"""
    geodesic_len, _ = geo_length.geodesic_length(x, y, lambda x0, x1, t: self.correction_net.apply({'params': self.params}, x0, x1, t))
    # geodesic_len = geo_hop.geodesic_length(x, y)
    return geodesic_len
  
  # h for translation invariant (TI) maps
  def h(self, z: jnp.ndarray) -> float:  # noqa: D102
    """h := x - y"""
    return None#jnp.sum(z ** 2)

  def h_legendre(self, z: jnp.ndarray) -> float:  # noqa: D102
    return None#0.25 * jnp.sum(z ** 2)

  def barycenter(self, weights: jnp.ndarray,
                 xs: jnp.ndarray) -> Tuple[jnp.ndarray, Any]:
    """barycenter of vectors when using squared-Euclidean distance."""
    return jnp.average(xs, weights=weights, axis=0), None
  
