"""
decision_tree_helpers.py

Shared helper functions for the Decision Tree lessons.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score


RANDOM_STATE = 42


def plot_2d_classification_data(
    X,
    y,
    title,
    x_label="Feature 1",
    y_label="Feature 2",
    class_names=None
):
    """
    Plot a 2D classification dataset.

    X must have exactly 2 columns.
    y is expected to contain class labels like 0, 1, 2, etc.
    """
    plt.figure(figsize=(8, 6))

    unique_classes = np.unique(y)

    for class_value in unique_classes:
        points = X[y == class_value]
        label = f"Class {class_value}"

        if class_names is not None:
            label = class_names[class_value]

        plt.scatter(
            points[:, 0],
            points[:, 1],
            edgecolor="black",
            label=label,
            alpha=0.8
        )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_decision_boundary(
    model,
    X,
    y,
    title,
    x_label="Feature 1",
    y_label="Feature 2",
    class_names=None
):
    """
    Plot a model's decision boundary over a 2D dataset.

    This function creates a grid of points covering the plot area.
    It asks the model to predict every point on that grid.
    Then it colors the background based on those predictions.

    Decision trees often produce rectangular regions because they split
    one feature at a time.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = model.predict(grid_points)
    predictions = predictions.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, predictions, alpha=0.25)

    unique_classes = np.unique(y)

    for class_value in unique_classes:
        points = X[y == class_value]
        label = f"Class {class_value}"

        if class_names is not None:
            label = class_names[class_value]

        plt.scatter(
            points[:, 0],
            points[:, 1],
            edgecolor="black",
            label=label,
            alpha=0.85
        )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_tree_diagram(
    model,
    feature_names,
    class_names=None,
    title="Decision Tree"
):
    """
    Plot the structure of a decision tree.

    This shows the questions the tree asks.
    Example:
        Feature 1 <= 0.42
    """
    plt.figure(figsize=(18, 10))

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=9
    )

    plt.title(title)
    plt.show()


def plot_feature_importance(
    model,
    feature_names,
    title="Feature Importance"
):
    """
    Plot feature importances from a fitted decision tree.

    Importance is a rough measure of how useful each feature was when
    splitting the data.
    """
    importances = model.feature_importances_

    plt.figure(figsize=(8, 5))
    plt.bar(feature_names, importances)
    plt.title(title)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.ylim(0, 1)
    plt.grid(True, axis="y", alpha=0.3)
    plt.show()

    print("\nFeature importances:")
    for name, importance in zip(feature_names, importances):
        print(f"  {name}: {importance:.3f}")


def plot_regression_results(x, y, model, title):
    """
    Plot a regression tree's predictions against noisy data.

    Decision tree regression often looks like stair steps because the tree
    predicts a constant value inside each leaf region.
    """
    x_plot = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)
    y_prediction = model.predict(x_plot)

    plt.figure(figsize=(9, 6))
    plt.scatter(x, y, edgecolor="black", alpha=0.7, label="Noisy data")
    plt.plot(x_plot, y_prediction, linewidth=3, label="Tree prediction")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def quick_depth_comparison(X, y, depths, title_prefix="Decision Tree Depth"):
    """
    Train multiple trees with different max_depth values and plot their
    decision boundaries one at a time.
    """
    for depth in depths:
        model = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
        model.fit(X, y)

        depth_label = "None / unlimited" if depth is None else str(depth)
        train_accuracy = accuracy_score(y, model.predict(X))

        plot_decision_boundary(
            model,
            X,
            y,
            title=f"{title_prefix}: max_depth={depth_label}, training accuracy={train_accuracy:.2f}"
        )
