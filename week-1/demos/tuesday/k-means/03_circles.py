"""
Kmeans is a great algorithm for unsupervised learning but there is a
caveat, it doesn't work well on clusters that are not circle sized

"""

import numpy as np
from sklearn.datasets import make_circles
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN

from clustering_helpers import (
    RANDOM_STATE, plot_raw_points, run_kmeans, plot_elbow_curve, plot_silhouette_scores, compare_two_clusterings,
    plot_labeled_points
)

np.random.seed(RANDOM_STATE)

X_circles, y_circles = make_circles(
    n_samples=500,
    noise=0.05,
    factor=0.45,
    random_state=RANDOM_STATE
)

# Let's plot these as labelled sections
# plot_labeled_points(
#     X_circles,
#     y_circles,
#     title="Kmeans on complex shapes"
# )

kmeans_circles = run_kmeans(X_circles, k=2)

# compare_two_clusterings(
#     X_circles,
#     y_circles,
#     kmeans_circles.labels_,
#     title_left="Actual",
#     title_right="Predicted",
#     overall_title="Where K Means Fails"
# )

"""
In this case K-Means fails because our clusters are not circle ish
We need a different algorithm to handle this

Introducing DBScan -> Similar to K-Means but it works on points
in the same "neighborhood" to group them together

Computationally slower than K-Means but better for more complex shapes
This can also mark outliers (K-Means does not do this)
"""
dbscan_circles = DBSCAN(
    # Hyperparameters used to decide the "neighborhood"
    eps=0.25,
    min_samples=5,
)

dbscan_labels = dbscan_circles.fit_predict(X_circles)

# Compare K means vs DBScan and we can see how it works

compare_two_clusterings(
    X_circles,
    kmeans_circles.labels_,
    dbscan_labels,
    title_left = "K-Means",
    title_right = "DBSCAN",
    overall_title = "K-Means vs DBSCAN on complex shapes"
)