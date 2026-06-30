"""
Unsupervised Learning

Now we're getting into unsupervised learning. As a reminder, this is
ML on data that does not have a Label or "output". We just examine the inputs
and gather information about the data itself

One of the most popular algorithms for is K-Means Clustering (K is the number
of clusters)
"""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from clustering_helpers import (
    RANDOM_STATE, plot_raw_points, run_kmeans, plot_kmeans_result, plot_elbow_curve, plot_silhouette_scores
)


np.random.seed(RANDOM_STATE)

"""
Motivating Example: We have income and a numeric "Spending Score" for our 
customer data, we want to group our customers together based off of this

We want our computer to determine the groups naturally
"""

# Create some data
X_customers, true_customer_groups = make_blobs(
    n_samples=500,
    centers=5,
    cluster_std=1.2,
    random_state=RANDOM_STATE
)

# Plot the points to get a idea of what we're looking at
# plot_raw_points(
#     X_customers,
#     title="Raw Customer Data",
#     xlabel="Income",
#     ylabel="Spending Score",
# )

# Let's actually run the Kmeans model and then plot what we're looking at
kmeans_customers = run_kmeans(X_customers, k=5)
# Thinking ahead, how do we determine k?

plot_kmeans_result(
    X_customers,
    kmeans_customers,
    title = 'K-Means Analysis',
    xlabel="Income",
    ylabel="Spending Score",
)

"""
So how do we determine K? That's what we need for our clusters but too low is 
an issue and too high is also an issue

How do we determine this? 
There is no hard and fast answer

Elbow Method and Silhouette Score are good indicators but not perfect

Elbow method measures this thing called inertia which is how far apart a group
is spread

Intertia is the sum of the Squared distances from each point to its assigned 
centroid
Lower inertia means points are closer to their centroids, as we add more
K or centroids inertia will go down
"""

k_values = list(range(1, 11))
inertias = []

for k in k_values:
    model = run_kmeans(X_customers, k=k)
    inertias.append(model.inertia_)

# Plot elbow method helper
plot_elbow_curve(k_values, inertias)

"""
Silhouette Scores
A Silhouette score roughly measures how well separated each cluster is
Used to help determine what K Should be
Score range goes from -1 to 1
    -1 means poor clustering
    0 means overlapping
    1 well separated clusters
"""
k_values = list(range(2, 11))
scores = []
for k in k_values:
    model = run_kmeans(X_customers, k=k)
    score = silhouette_score(X_customers, model.labels_)
    scores.append(score)

plot_silhouette_scores(k_values, scores)