import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import jax
from jax import jit, random
import jax.numpy as jnp
import matplotlib.pyplot as plt
from collections import deque
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from core import sampling
from jax import lax

"""
kind of peristent homology? radius d around each sample
connected neigbourhoods continuous from x0 -> x1 form geodesic space
for 0 /le d /le a where d=a is the smallest d for non-null geodesic space
geodesic length is Euclidean between points whose neighbourhoods form geodesic space
equivalent to constructing nbhds, we construct pairwise distance between each point
and construct a path by pairwise distances where distances < d

better: persistence for different sample sizes: i.e. geodesic length for 2/le n /le N
tie-break for given n is shortest geodesic length at that d

increase num_samples, until geodesic_length stops improving by some margin
train a neural net on geo_len as loss function (i.e. learns optimal num_samples approx?)

Does regression defeat the point? Unless some way of non-hard-coding polynomial degree
i.e. better to use neural net for geo_length minimisation? or use some other smoothing function?
Or perhaps minimise
or increase num_samples until geo_length stops increasing proportionally much

Perhaps: once found minimum d, increase d by up to 5%, find all possible paths with d/le 1.05min_d, select shortest(by Euclidean distance)
    but then: searches paths of all possible cardinality--hugely expensive
"""

# @jit
# def pairwise_dist(data_samples):
#     squared_sum = jnp.sum(data_samples ** 2, axis=1)
#     squared_sum_expanded = jnp.expand_dims(squared_sum, axis=0) + jnp.expand_dims(squared_sum, axis=1)
#     dot_product = jnp.dot(data_samples, data_samples.T)
#     distances = jnp.sqrt(jnp.clip(squared_sum_expanded - 2 * dot_product, a_min=0.0))
#     return distances

# def find_chain(distances, d, start_idx, end_idx):
#     num_samples = distances.shape[0]
#     visited = set()
#     queue = deque([(start_idx, [start_idx])])

#     while queue:
#         current, path = queue.popleft()
#         if current == end_idx:
#             return path
#         for next_node in range(num_samples):
#             if next_node not in visited and distances[current, next_node] < d:
#                 visited.add(next_node)
#                 queue.append((next_node, path + [next_node]))
#     return None

# def min_d_search(samples, start_idx, end_idx, tolerance=1e-5):
#     distances = pairwise_dist(samples)
#     min_d, max_d = 0.0, jnp.max(distances)
    
#     # binary search
#     while max_d - min_d > tolerance:
#         mid_d = (min_d + max_d) / 2
#         if find_chain(distances, mid_d, start_idx, end_idx) is not None:
#             max_d = mid_d
#         else:
#             min_d = mid_d
#     return max_d, find_chain(distances, max_d, start_idx, end_idx), distances

# def path_len(path, distances):
#     length = jnp.sum(distances[jnp.array(path[:-1]), jnp.array(path[1:])])
#     return length

# def regress(chain_points, max_degree=2):
#     x = jnp.linspace(0, 1, len(chain_points))[:, None]
#     f = make_pipeline(PolynomialFeatures(max_degree), LinearRegression())
#     f.fit(x, chain_points)
#     curve = f.predict(jnp.linspace(0, 1, 100)[:, None])
#     return curve

# def curve_len(points):
#     differences = jnp.diff(points, axis=0)
#     distances = jnp.linalg.norm(differences, axis=1)
#     return jnp.sum(distances)

# num_samples = 50
# samples = sampling.moon_upper(num_samples=num_samples, key=random.PRNGKey(1))
# start_idx = 5
# end_idx = 10
# x_start = samples[start_idx]
# x_end = samples[end_idx]

# min_d, path, distances = min_d_search(samples, start_idx, end_idx)
# path_length = path_len(path, distances)
# path_points = samples[jnp.array(path)]

# print(f"Minimum d: {min_d:.5f}")
# print(f"Euclidean path length: {path_length:.5f}")

# chain_points_fit = regress(path_points)
# curve_length = curve_len(chain_points_fit)
# print(f"Total Euclidean distance of the regression curve: {curve_length:.5f}")


# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], color='blue', alpha=0.3, label='Data Samples')
# ax.scatter(samples[start_idx, 0], samples[start_idx, 1], samples[start_idx, 2], color='red', label='Start Point', s=100, edgecolors='k')
# ax.scatter(samples[end_idx, 0], samples[end_idx, 1], samples[end_idx, 2], color='red', label='End Point', s=100, edgecolors='k')
# ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], 'red', label='Path Points', markersize=0, linewidth=4)
# ax.plot(chain_points_fit[:, 0], chain_points_fit[:, 1], chain_points_fit[:, 2], 'green', label='Curve of Best Fit', linewidth=2)
# ax.set_xlabel('Dimension 1')
# ax.set_ylabel('Dimension 2')
# ax.set_zlabel('Dimension 3')
# ax.set_title(f'Data Samples and Constructed Path (d={min_d:.5f})')
# plt.legend()
# plt.show()


num_samples=25
samples = sampling.moon_upper(num_samples)

def geodesic_length(x0, x1):
   
    data_samples = jnp.vstack([samples, x0, x1])
    start_idx = num_samples
    end_idx = num_samples + 1

    @jit
    def pairwise_dist(data_samples):
        squared_sum = jnp.sum(data_samples ** 2, axis=1)
        squared_sum_expanded = jnp.expand_dims(squared_sum, axis=0) + jnp.expand_dims(squared_sum, axis=1)
        dot_product = jnp.dot(data_samples, data_samples.T)
        distances = jnp.sqrt(jnp.clip(squared_sum_expanded - 2 * dot_product, a_min=0.0))
        return distances

    def find_chain(distances, d, start_idx, end_idx):
        num_samples = distances.shape[0]
        visited = set()
        queue = deque([(start_idx, [start_idx])])

        while queue:
            current, path = queue.popleft()
            if current == end_idx:
                return path
            for next_node in range(num_samples):
                if next_node not in visited and distances[current, next_node] < d:
                    visited.add(next_node)
                    queue.append((next_node, path + [next_node]))
        return None

    def min_d_search(data_samples, start_idx, end_idx, tolerance=1e-5):
        distances = pairwise_dist(data_samples)
        min_d, max_d = 0.0, jnp.max(distances)
        
        # binary search
        while max_d - min_d > tolerance:
            mid_d = (min_d + max_d) / 2
            if find_chain(distances, mid_d, start_idx, end_idx) is not None:
                max_d = mid_d
            else:
                min_d = mid_d
        return max_d, find_chain(distances, max_d, start_idx, end_idx), distances

    def regress(chain_points, max_degree=3):
        x = jnp.linspace(0, 1, len(chain_points))[:, None]
        f = make_pipeline(PolynomialFeatures(max_degree), LinearRegression())
        f.fit(x, chain_points)
        curve = f.predict(jnp.linspace(0, 1, 100)[:, None])
        return curve

    def curve_len(points):
        differences = jnp.diff(points, axis=0)
        distances = jnp.linalg.norm(differences, axis=1)
        return jnp.sum(distances)

    _, path, _ = min_d_search(data_samples, start_idx, end_idx)
    path_points = data_samples[jnp.array(path)]
    chain_points_fit = regress(path_points)
    curve_length = curve_len(chain_points_fit)
    return curve_length#, path_points

x0 = jnp.array([1.0, 0., 0.])
x1 = jnp.array([-1.0, 0., 0.])
curve_length, path_points = geodesic_length(x0, x1)
print(curve_length)
print(path_points)

data_samples = jnp.vstack([samples, x0, x1])
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], color='blue', alpha=0.3, label='Data Samples')
ax.scatter(data_samples[:, 0], data_samples[:, 1], data_samples[:, 2], color='black', alpha=0.3, label='Data')
ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], 'red', label='Path Points', markersize=0, linewidth=4)
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
plt.legend()
plt.show()