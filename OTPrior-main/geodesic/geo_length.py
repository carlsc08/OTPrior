"""
Main geodesic length solver, given a neural correction term
"""

import jax
from jax import vjp, vmap, jit
import jax.numpy as jnp
from geodesic import mtensor

# for training geo_net
def geodesic_objective(x0, x1, correction_net):
    t = jax.random.uniform(jax.random.key(5))   # t~[0,1]
    
    def phi(t):
      phi = correction_net(x0,x1,t)
      return phi
    
    #scale = jnp.tanh(norm)**0.25
    #scale = 0.5*(jnp.tanh(2*norm-2)+1)
    #scale = 0.5*(jnp.tanh(10*norm-7.5)+1)
    #scale = 0.5*(jnp.tanh(5*norm-5)+1.1)

    x_h = t*x1 + (1-t)*x0 + t*(1-t)*phi(t)
    G = mtensor.G_LAND(x_h, scale=1)
    ones = jnp.ones_like(phi(t))
    _, phi_vjp = vjp(phi, t)
    phi_vjp = phi_vjp(ones)[0]
    
    dx_dt = x1 - x0 + t*(1-t)*phi_vjp + (1-2*t)*phi(t)
    geo_loss = dx_dt.T @ G @ dx_dt

    # norm = jnp.linalg.norm(x1-x0)
    # ratio = (jnp.linalg.norm(x_h-x0) + jnp.linalg.norm(x_h-x1)) / norm
    #regulariser = ratio * ((1/jnp.tanh(12*norm))-1)
    #regulariser = 1 / (1 - jnp.tanh(jnp.abs(jnp.abs(geo_loss) - norm)))
    #regulariser = 0.1*jnp.exp(5*(ratio-15))
    #regulariser = jnp.where(regulariser > 1e-1, regulariser, 0)
    # exp = jnp.exp(0.5*(ratio-4))-1
    # regulariser = jnp.maximum(exp, 0)

    return geo_loss

# # approx geo length using learned correction term \phi_{h, t}
# def geodesic_length(x0, x1, correction_net, num_steps=20):
#     t = jnp.linspace(0, 1, num_steps).reshape(num_steps, 1)
#     x0_tiled = jnp.tile(x0, (num_steps, 1))
#     x1_tiled = jnp.tile(x1, (num_steps, 1))
    
#     # x_{t, h}
#     phi_output = correction_net(x0_tiled, x1_tiled, t)
#     x_h = t * x1_tiled + (1 - t) * x0_tiled + t * (1 - t) * phi_output  # Interpolated geodesic sampled coordinates
#     dx_dt = x_h[1:] - x_h[:-1]
#     distances = jnp.linalg.norm(dx_dt, axis=1)
#     geodesic_length = jnp.sum(distances)

#     return geodesic_length, x_h

def geodesic_length(x0, x1, correction_net, num_steps=20):
    t = jnp.linspace(0, 1, num_steps).reshape(num_steps, 1)
    x0_tiled = jnp.tile(x0, (num_steps, 1))
    x1_tiled = jnp.tile(x1, (num_steps, 1))
    SqEuclid = jnp.sqrt(jnp.sum((x0 - x1)**2, axis=0))
    
    def true_fn(SqEuclid):
        x_h = t * x1_tiled + (1 - t) * x0_tiled
        return SqEuclid, x_h

    def false_fn(SqEuclid):
        phi_output = correction_net(x0_tiled, x1_tiled, t)
        x_h = t * x1_tiled + (1 - t) * x0_tiled + t * (1 - t) * phi_output
        dx_dt = x_h[1:] - x_h[:-1]
        distances = jnp.linalg.norm(dx_dt, axis=1)
        geodesic_length = jnp.sum(distances)
        return geodesic_length, x_h
    
    return jax.lax.cond(SqEuclid < 0.8, true_fn, false_fn, SqEuclid)
