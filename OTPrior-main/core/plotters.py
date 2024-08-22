import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import spherical_stats
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from scipy.stats import norm

def plot_3d(
    fig, points, num_subplots, 
    color=None, 
    label=None,
    title=None,
    view_init=None,
    title_size=20,
    legend_size=25,
    tick_size=20,
    lims=None,
    size_points=150,
    alpha=.8,
    set_equal_aspect=False,
    plot_sphere=False,
    plot_hypercube=False,
    n_grid=100,
    no_ticks=False,
):
    x, y, z = num_subplots
    ax = fig.add_subplot(x, y, z, projection="3d")
    if view_init is not None:
        ax.view_init(view_init[0], view_init[1])
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=color,
        edgecolors="k", 
        s=size_points, 
        alpha=alpha,
        label=label
    )
    if lims is not None:
        ax.axes.set_xlim3d(lims[0]) 
        ax.axes.set_ylim3d(lims[1]) 
        ax.axes.set_zlim3d(lims[2]) 
    ax.set_title(title, fontsize=title_size)
    ax.tick_params(axis="both", which="major", labelsize=tick_size)
    ax.grid(alpha=.4)
    if label is not None:
        ax.legend(fontsize=legend_size)
    if plot_sphere:
        x, y, z = spherical_stats.sphere(n_grid)
        ax.plot_wireframe(
            x, y, z,  
            # rstride=1, cstride=1, 
            color='grey', 
            alpha=0.2, 
            linewidth=.6
        )
    if plot_hypercube:
        axes = [1, 1, 1]
        data = np.ones(axes, dtype=np.bool)
        ax.voxels(data, facecolors="grey", alpha=.15)
    if set_equal_aspect:
        set_axes_equal(ax)
    if no_ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    
def plot_2d(
    fig, points, num_subplots, 
    color=None,
    label=None, 
    title=None,
    title_size=20,
    legend_size=25,
    tick_size=20,
    lims=None,
    size_points=150,
    alpha=.8,
    set_equal_aspect=False,
    no_ticks=False,
):
    x, y, z = num_subplots
    ax = fig.add_subplot(x, y, z)
    y_coordinates = (
        points[:, 1] if points.shape[1] == 2
        else jnp.zeros(points.shape[0])
    )
    ax.scatter(
        points[:, 0],
        y_coordinates,
        c=color,
        edgecolors="k", 
        s=size_points, 
        alpha=alpha,
        label=label
    )
    if lims is not None:
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
    ax.set_title(title, fontsize=title_size)
    ax.tick_params(axis="both", which="major", labelsize=tick_size)
    ax.grid(alpha=.4)
    if label is not None:
        ax.legend(fontsize=legend_size),
    if set_equal_aspect:
        ax.set_aspect("equal")
    if no_ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
def plot_fitted_map_gromov(
    batch, 
    num_points=None,
    seed=0,
    show=True,
    return_fig=False,
    title_size=20,
    legend_size=25,
    tick_size=20,
    lims=None,
    size_points=150,
    alpha=.8,
    view_init=None,
    set_equal_aspect_source=False,
    set_equal_aspect_target=False,
    plot_sphere_source=False,
    plot_sphere_target=False,
    plot_hypercube_source=False,
    plot_hypercube_target=False,
    plot_source=True,
    *args,
    **kwargs,
):
    """
    Plot the fitted map on a batch of samples from the source measure.
    """    
    
    assert (
        all(
            [x in batch.keys() for x in ['colors_source', 'colors_target']]
        )
    ), (
        "Provide colors for Gromov Wasserstein map plotting."
    )
    if num_points is None:
        subsample = jnp.arange(len(batch['source']))
    else:
        rng = jr.PRNGKey(seed)
        subsample = jr.choice(
            rng, a=len(batch['source']), shape=(num_points,),
            replace=False
        )

    num_plots = (
        3 if plot_source
        else 2
    )
    input_dim = batch['source'].shape[-1]
    output_dim = batch['target'].shape[-1]
    
    dims = (
        (input_dim, output_dim) if plot_source
        else (output_dim,)
    )
    assert all(
        [
            x <= 3 for x in dims
        ]
    ), (
        "Plot is only supported for 2D and 3D data."
    )
    
    fig = plt.figure(figsize=(8, 6*num_plots))
    source_plotter = (
        plot_2d if input_dim <= 2
        else partial(
            plot_3d, 
            view_init=view_init,
            plot_sphere=plot_sphere_source,
            plot_hypercube=plot_hypercube_source,
        )
    )
    target_plotter = (
        plot_2d if output_dim <= 2
        else partial(
            plot_3d, 
            view_init=view_init,
            plot_sphere=plot_sphere_target,
            plot_hypercube=plot_hypercube_target,
        )
    )
    
    num_subplots = (num_plots, 1, 1)
    if plot_source:
        lim_source = lims['source'] if lims is not None else None
        if lim_source is not None:
            assert len(lim_source) == input_dim, (
                "Provide limits along each dimension. "
                f"Here, source dimension is {input_dim}."
            ) 
        source_plotter(
            fig, 
            points=batch["source"][subsample], 
            num_subplots=num_subplots, 
            color=batch["colors_source"][subsample],
            title="Source samples",
            title_size=title_size,
            legend_size=legend_size,
            tick_size=tick_size,
            lims=lim_source,
            alpha=alpha,
            size_points=size_points,
            set_equal_aspect=set_equal_aspect_source
        )
        num_subplots = (num_plots, 1, num_subplots[-1] + 1)
        
    lim_target = lims['target'] if lims is not None else None
    if lim_target is not None:
        assert len(lim_target) == output_dim, (
            "Provide limits along each dimension. "
            f"Here, source dimension is {output_dim}."
        )
    target_plotter(
        fig, 
        points=batch["target"][subsample], 
        num_subplots=num_subplots, 
        color=batch["colors_target"][subsample],
        title="Target samples",
        title_size=title_size,
        legend_size=legend_size,
        tick_size=tick_size,
        lims=lim_target,
        alpha=alpha,
        size_points=size_points,
        set_equal_aspect=set_equal_aspect_target
    )
    
    num_subplots = (num_plots, 1, num_subplots[-1] + 1)
    if "mapped_source" in batch.keys():
        target_plotter(
            fig, 
            points=batch["mapped_source"][subsample], 
            num_subplots=num_subplots, 
            color=batch["colors_source"][subsample],
            title="Predicted target samples",
            title_size=title_size,
            legend_size=legend_size,
            tick_size=tick_size,
            lims=lim_target,
            alpha=alpha,
            size_points=size_points,
            set_equal_aspect=set_equal_aspect_target
        )
    
    fig.tight_layout()
    if show:
        plt.show()
    plt.close()
    if return_fig:
        return fig

def kmeans_plot(prior_samples, num_clusters=10):
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(prior_samples)
    if prior_samples.shape[1] == 2:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(prior_samples[:, 0], prior_samples[:, 1], c=cluster_labels, s=10, cmap='tab10')
        plt.colorbar(scatter)
        plt.title("Prior Samples with Clusters")
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
    elif prior_samples.shape[1] == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(prior_samples[:, 0], prior_samples[:, 1], prior_samples[:, 2], c=cluster_labels, s=10, cmap='tab10')
        plt.colorbar(scatter)
        ax.set_title("Prior Samples with Clusters")
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
    else:
        raise ValueError("Prior samples must have 2 or 3 dimensions for plotting.")

def tsne_kmeans_plot(prior_samples, num_clusters=10):
    tsne = TSNE(n_components=2, random_state=0)
    prior_2d = tsne.fit_transform(prior_samples)
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(prior_samples)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(prior_2d[:, 0], prior_2d[:, 1], c=cluster_labels, s=10, cmap='tab10')
    plt.colorbar(scatter)
    plt.title("t-SNE Visualization of Prior Samples with Clusters")
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

def latent_plot(data, labels, model, params, num_samples, prob_toggle=False, use_tsne=True):
    sampled_data = data[:num_samples]
    sampled_labels = labels[:num_samples]
    if prob_toggle:
        _, _, latent_representations = jax.vmap(lambda x: model.apply({'params': params}, x, jr.PRNGKey(0), prob_toggle))(sampled_data)
    else:
        _, latent_representations = jax.vmap(lambda x: model.apply({'params': params}, x, jr.PRNGKey(0), prob_toggle))(sampled_data)
    latent_dim = latent_representations.shape[-1]
    if latent_dim == 2:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(latent_representations[:, 0], latent_representations[:, 1], c=sampled_labels, cmap='tab10', alpha=0.5)
        plt.title('Latent Space Visualization')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.colorbar(scatter, label='Digit Label')
    if latent_dim == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(latent_representations[:, 0], latent_representations[:, 1], latent_representations[:, 2], 
                                c=sampled_labels, cmap='tab10', alpha=0.6)
        ax.set_title('3D Latent Space Visualization')
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 2')
        ax.set_zlabel('Latent Dimension 3')
        fig.colorbar(scatter, label='Digit Label')
    if use_tsne:
        # t-SNE for dimensionality reduction to 2D
        tsne = TSNE(n_components=2)
        latent_reduced = tsne.fit_transform(latent_representations)
    else:
        # PCA for dimensionality reduction to 2D
        pca = PCA(n_components=2)
        latent_reduced = pca.fit_transform(latent_representations)
    # 2D plotting after dimensionality reduction
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_reduced[:, 0], latent_reduced[:, 1], c=sampled_labels, cmap='tab10', alpha=0.5)
    plt.title('Latent Space Visualization')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.colorbar(scatter, label='Digit Label')
        
def mnist_img_plot(original, reconstructed, num_plots):
    plt.figure(figsize=(20, 4))
    for i in range(num_plots):
        ax = plt.subplot(2, num_plots, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
        
        ax = plt.subplot(2, num_plots, i + 1 + num_plots)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        ax.axis('off')

# find better way of scaling grid size
def mnist_interp_euclidean(side_dim, decoder_model, decoder_params, digit_size = 28):
    figure = np.zeros((digit_size * side_dim, digit_size * side_dim))
    grid_x = norm.ppf(np.linspace(-1, 1, side_dim))
    grid_y = norm.ppf(np.linspace(-1, 1, side_dim))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            z_sample = jnp.array(z_sample)
            x_decoded = decoder_model.apply({'params': decoder_params}, z_sample)
            digit = jnp.squeeze(x_decoded, axis=-1)
            figure[i * digit_size: (i + 1) * digit_size,
                    j * digit_size: (j + 1) * digit_size] = np.array(digit)
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gnuplot2')

def mnist_interp_euclidean_scaled(side_dim, decoder_model, decoder_params, digit_size = 28):
    scale_factor = 1.28  # std deviations
    x_values = np.linspace(-scale_factor, scale_factor, side_dim)
    y_values = np.linspace(-scale_factor, scale_factor, side_dim)
    xv, yv = np.meshgrid(x_values, y_values)
    fig, axes = plt.subplots(side_dim, side_dim, figsize=(side_dim, side_dim))

    for i in range(side_dim):
        for j in range(side_dim):
            z = np.array([xv[i, j], yv[i, j]])
            decoded_z = decoder_model.apply({'params': decoder_params}, z)
            axes[i, j].imshow(decoded_z, cmap='gray')
            axes[i, j].axis('off')
