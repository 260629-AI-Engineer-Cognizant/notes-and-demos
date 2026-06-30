"""
We've used decision trees for classification pretty heavily but
we can also use it for regression
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from decision_tree_helpers import (
    RANDOM_STATE,
    plot_2d_classification_data,
    plot_decision_boundary, plot_regression_results
)

np.random.seed(RANDOM_STATE)

# We'll pretend this is a model tracking data that moves up and
# is a cyclic manner

# Create a linear space for our X values
x = np.linspace(0, 10, 200)
noise = np.random.normal(loc=0, scale=0.1, size=x.shape)
y = np.sin(x) + noise

X_regression = x.reshape(-1,1)
y_regression = y

# plt.scatter(X_regression, y_regression, color='black')
# plt.show()

regression_tree = DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE)
regression_tree.fit(X_regression, y_regression)

# Use plot function to showcase this
plot_regression_results(
    X_regression,
    y_regression,
    regression_tree,
    "Regression Tree Predictions"
)
