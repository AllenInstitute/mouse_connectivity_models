import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_loss_surface(surface):
    xs = np.linspace(0, surface.coordinates_normed[:, 0].max(), 100)
    ys = np.linspace(0, surface.coordinates_normed[:, 1].max(), 100)
    preds = np.empty((100, 100))
    for x in range(100):
        for y in range(100):
            preds[x, y] = surface.predict(np.asarray([[xs[x], ys[y]]]))

    mxy = np.asarray(np.meshgrid(xs, ys)).transpose()

    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig)

    ax.scatter(mxy[:, :, 0], mxy[:, :, 1], preds, s=10)

    ax.set_xlabel("Centroid distance", fontsize=30, rotation=150)
    ax.set_ylabel("Cre-distance", fontsize=30, rotation=150)
    ax.set_zlabel("Projection distance predicted", fontsize=30, rotation=150)


def plot_loss_scatter(loss_surface):
    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig)

    ax.scatter(
        loss_surface.coordinates_normed[:, 0],
        loss_surface.coordinates_normed[:, 1],
        loss_surface.projection_distances[:, 0],
        s=1,
    )
    ax.set_xlabel("Centroid distance", fontsize=30, rotation=150)
    ax.set_ylabel("Cre-distance", fontsize=30, rotation=150)
    ax.set_zlabel("Projection distance", fontsize=30, rotation=150)
