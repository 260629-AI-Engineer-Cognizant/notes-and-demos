"""
clustering_helpers.py

Shared helper functions for the K-Means and DBSCAN lesson scripts.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


RANDOM_STATE = 42

def plot_raw_points(X, title, xlabel="Feature 1", ylabel="Feature 2"):
    """
    Plot unlabeled 2D data.

    This is useful before clustering because engineers should first ask:

        "What groups do I see?"
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], s=45, edgecolor="black", alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.show()


def plot_labeled_points(X, labels, title, xlabel="Feature 1", ylabel="Feature 2"):
    """
    Plot 2D data colored by labels.

    Labels may come from:
        • true classes
        • K-Means cluster assignments
        • DBSCAN cluster assignments

    DBSCAN uses label -1 for noise/outliers.
    """
    plt.figure(figsize=(8, 6))

    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label

        if label == -1:
            plt.scatter(
                X[mask, 0],
                X[mask, 1],
                s=70,
                marker="x",
                label="Noise / Outlier (-1)"
            )
        else:
            plt.scatter(
                X[mask, 0],
                X[mask, 1],
                s=45,
                edgecolor="black",
                alpha=0.85,
                label=f"Cluster/Class {label}"
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def run_kmeans(X, k, random_state=RANDOM_STATE, n_init=10):
    """
    Train K-Means and return the fitted model.

    n_init controls how many different random centroid initializations are tried.
    K-Means can land in different solutions depending on initial centroids.
    """
    model = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=n_init
    )
    model.fit(X)
    return model


def plot_kmeans_result(X, model, title, xlabel="Feature 1", ylabel="Feature 2"):
    """
    Plot K-Means cluster assignments and centroids.
    """
    labels = model.labels_
    centers = model.cluster_centers_

    plt.figure(figsize=(8, 6))

    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(
            X[mask, 0],
            X[mask, 1],
            s=45,
            edgecolor="black",
            alpha=0.85,
            label=f"Cluster {label}"
        )

    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        s=260,
        marker="X",
        edgecolor="black",
        linewidth=2,
        label="Centroids"
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def compare_two_clusterings(
    X,
    labels_left,
    labels_right,
    title_left,
    title_right,
    overall_title,
    xlabel="Feature 1",
    ylabel="Feature 2"
):
    """
    Plot two clustering results side by side.

    Useful for:
        • K-Means vs DBSCAN
        • actual labels vs predicted clusters
        • two different parameter choices
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(overall_title, fontsize=16)

    for ax, labels, title in [
        (axes[0], labels_left, title_left),
        (axes[1], labels_right, title_right)
    ]:
        unique_labels = np.unique(labels)

        for label in unique_labels:
            mask = labels == label

            if label == -1:
                ax.scatter(
                    X[mask, 0],
                    X[mask, 1],
                    s=70,
                    marker="x",
                    label="Noise / Outlier (-1)"
                )
            else:
                ax.scatter(
                    X[mask, 0],
                    X[mask, 1],
                    s=45,
                    edgecolor="black",
                    alpha=0.85,
                    label=f"{label}"
                )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_elbow_curve(k_values, inertias):
    """
    Plot K values against inertia.

    The goal is to look for an "elbow" where adding more clusters stops
    improving the score dramatically.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, inertias, marker="o")
    plt.title("Elbow Method: Choosing K")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia")
    plt.xticks(k_values)
    plt.grid(alpha=0.3)
    plt.show()


def plot_silhouette_scores(k_values, scores):
    """
    Plot K values against silhouette scores.

    Silhouette score roughly measures how well-separated clusters are.
    Higher is generally better.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, scores, marker="o")
    plt.title("Silhouette Score by K")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Silhouette score")
    plt.xticks(k_values)
    plt.grid(alpha=0.3)
    plt.show()


def nearest_centroid_labels(X, centroids):
    """
    Assign each point to the nearest centroid.

    This is the assignment step in manual K-Means.
    """
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    return labels


def update_centroids(X, labels, k, old_centroids):
    """
    Move each centroid to the mean of the points assigned to it.

    This is the update step in manual K-Means.

    If a centroid receives no points, keep the old centroid.
    """
    new_centroids = []

    for cluster_id in range(k):
        points_in_cluster = X[labels == cluster_id]

        if len(points_in_cluster) == 0:
            new_centroids.append(old_centroids[cluster_id])
        else:
            new_centroids.append(points_in_cluster.mean(axis=0))

    return np.array(new_centroids)


def plot_manual_kmeans_step(X, labels, centroids, iteration):
    """
    Plot one iteration of manual K-Means.
    """
    plt.figure(figsize=(8, 6))

    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(
            X[mask, 0],
            X[mask, 1],
            s=45,
            edgecolor="black",
            alpha=0.85,
            label=f"Cluster {label}"
        )

    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=300,
        marker="X",
        edgecolor="black",
        linewidth=2,
        label="Centroids"
    )

    plt.title(f"Manual K-Means - Iteration {iteration}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def run_dbscan(X, eps, min_samples):
    """
    Run DBSCAN and return labels.

    Label -1 means noise/outlier.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels


def print_cluster_label_summary(labels):
    """
    Print how many points were assigned to each cluster label.
    """
    unique, counts = np.unique(labels, return_counts=True)

    print("Cluster label summary:")
    for label, count in zip(unique, counts):
        if label == -1:
            print(f"  Noise / Outlier (-1): {count}")
        else:
            print(f"  Cluster {label}: {count}")


def calculate_kmeans_metrics(X, k_values):
    """
    Calculate inertia and silhouette score for a range of K values.

    Silhouette score cannot be calculated for K=1.
    """
    inertias = []
    silhouette_scores = []

    for k in k_values:
        model = run_kmeans(X, k)
        inertias.append(model.inertia_)

        if k >= 2:
            silhouette_scores.append(silhouette_score(X, model.labels_))
        else:
            silhouette_scores.append(None)

    return inertias, silhouette_scores
