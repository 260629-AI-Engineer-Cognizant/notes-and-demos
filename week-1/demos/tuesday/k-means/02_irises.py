"""
Let's do a quick review of some K-means clustering with a dataset that we did
not create

"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score

from clustering_helpers import (
    RANDOM_STATE, plot_raw_points, run_kmeans, plot_elbow_curve, plot_silhouette_scores, compare_two_clusterings
)

np.random.seed(RANDOM_STATE)

iris = load_iris()
# This is a data set containing information on Iris flowers
# It's used for classification and it has 4 features
# include sepal length, sepal width, petal length and petal width

# To retrieve the data we call iris.data and to retreive the targets it is
# iris.target

X = iris.data[:, [2, 3]]
# This just selects petal width and height
# print(X[:5])

plot_raw_points(
    X,
    title= "Raw Data Set",
    xlabel = "Petal Length",
    ylabel = "Petal Width",
)

# We want to use K means to evaluate this data so we need to find K
# Elbow method and silhouette methods
k_values = list(range(1,11))
inertias = []

for k in k_values:
    model = run_kmeans(X, k=k)
    inertias.append(model.inertia_)

plot_elbow_curve(k_values, inertias)

k_values = list(range(2,11))
scores = []

for k in k_values:
    model = run_kmeans(X, k=k)
    scores.append(silhouette_score(X, model.labels_))

plot_silhouette_scores(k_values, scores)

# We'll estimate it to be about 3
kmeans_iris = run_kmeans(X, k=3)

# Compare the model to the actual
# Normally not possibly with unsupervised learning (just for demo)
compare_two_clusterings(
    X,
    iris.target,
    kmeans_iris.labels_,
    title_left="Actual Values",
    title_right="K-Means Labels",
    xlabel="Petal Length",
    ylabel="Petal Width",
    overall_title=" Comparison"
)