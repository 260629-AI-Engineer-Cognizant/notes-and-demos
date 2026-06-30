# Goal -> Create a simple classification example and visualize it then analyze
# with a decision tree
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from decision_tree_helpers import (
    RANDOM_STATE,
    plot_2d_classification_data,
    plot_decision_boundary,
    plot_tree_diagram,
    plot_feature_importance
)

np.random.seed(RANDOM_STATE)

"""
Imagine we're building a simple weather app and we want to one thing
Should we go outside and play or not

Features (Inputs):
    Feature 1 : Temperature-like feature
    Feature 2: Humidity-like feature

Label (Output):
    0: Do not play outside
    1: Play outside
"""

# Synthetic Data Creation (Just for practice)
X_weather, y_weather = make_classification(
    n_samples=250,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_clusters_per_class=1,
    class_sep=1.4,
    random_state=RANDOM_STATE
)

class_names_weather= ["Do not play", "Play"]

# print(X_weather[:5])
# print(y_weather[:5])

plot_2d_classification_data(
    X_weather,
    y_weather,
    title="Synthetic Weather Data",
    x_label="Temperature-like feature",
    y_label="Humidity-like feature",
    class_names=class_names_weather
)


"""
Our goal is to use a DecisionTreeClassifier that will ask questions about our
features to divide them into groups 

    Feature 1 <= .37
    Feature 2 > 10
    
Each question will split the data into smaller groups
"""

X_train, X_test, y_train, y_test = train_test_split(
    X_weather,
    y_weather,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_weather
)

weather_tree = DecisionTreeClassifier(
    # Two different hyperparameters
    # Max Depth controls how "deep" the tree can go, or how many questions
    max_depth=3,
    random_state=RANDOM_STATE,
)

# Fit the model to the data
weather_tree.fit(X_train, y_train)

# Reminder, use predict to predict the actual value
train_predictions = weather_tree.predict(X_train)
test_predictions = weather_tree.predict(X_test)

print("Training Accuracy: ", accuracy_score(y_train, train_predictions))
print("Test Accuracy: ", accuracy_score(y_test, test_predictions))

plot_decision_boundary(
    weather_tree,
    X_weather,
    y_weather,
    title="Decision Tree Boundary for Weather data",
    x_label="Temperature-like feature",
    y_label="Humidity-like feature",
    class_names=class_names_weather
)

feature_names_weather = ["Temperature-Like Feature", "Humidity-like Feature"]

# So let's look at the tree to see the questions being asked
plot_tree_diagram(
    weather_tree,
    feature_names=feature_names_weather,
    class_names=class_names_weather,
    title="Weather Decision Tree Structure"
)

plot_feature_importance(
    weather_tree,
    feature_names=feature_names_weather,
    title="Weather Tree Feature Importance"
)