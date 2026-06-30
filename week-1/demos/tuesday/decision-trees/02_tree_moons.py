"""
Goal is to create more complex groups to see if our decision tree can still
classify and at what point do we hit overfitting

"""

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from decision_tree_helpers import (
    RANDOM_STATE,
    plot_2d_classification_data,
    plot_decision_boundary
)

np.random.seed(RANDOM_STATE)

X_moons, y_moons = make_moons(
    n_samples=300,
    noise=0.25,
    random_state=RANDOM_STATE
)

# plot_2d_classification_data(
#     X_moons,
#     y_moons,
#     title="Two Moons Dataset",
#     x_label="x",
#     y_label="y",
#     class_names= ["Moon 1", "Moon 2"]
# )


# For times sake we will skip the splitting of the data and just look at
# things manually

moon_tree = DecisionTreeClassifier(
    # None means unlimited depth as needed
    max_depth=4,
    random_state=RANDOM_STATE
)

# Trains the model on the data itself
moon_tree.fit(X_moons, y_moons)

plot_decision_boundary(
    moon_tree,
    X_moons,
    y_moons,
    title="Two Moons Decision Boundary",
    x_label="x",
    y_label="y",
    class_names= ["Moon 1", "Moon 2"]
)