# Machine Learning Practice Assignment Packet

## Overview

This assignment packet gives you hands-on practice with four common machine learning approaches:

1. **Linear Regression**
2. **Logistic Regression**
3. **Decision Trees**
4. **K-Means Clustering**

The goal is not to memorize code. The goal is to practice the workflow:

```text
Understand the problem
        ↓
Create or load data
        ↓
Visualize the data
        ↓
Train a model
        ↓
Evaluate the model
        ↓
Tune something
        ↓
Explain what happened
```

You should be able to run these assignments in a normal Python file or in a Jupyter notebook.

Recommended imports:

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import (
    make_regression,
    make_classification,
    make_blobs,
    make_moons,
    load_diabetes,
    load_breast_cancer,
    load_iris
)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score
)
```

---

# Assignment 1 — Linear Regression

## Goal

Use linear regression to predict a numeric value.

Linear regression is used when the target is continuous.

Examples:

- Predict house price
- Predict test score
- Predict sales
- Predict temperature
- Predict medical progression score

---

## Part A — Synthetic Study Hours Dataset

### Scenario

You are analyzing whether the number of hours a student studies can predict their exam score.

### Tasks

1. Create a synthetic dataset where:
   - `X` = hours studied
   - `y` = exam score
2. Plot the raw data.
3. Train a `LinearRegression` model.
4. Plot the regression line.
5. Print:
   - slope/coefficient
   - intercept
   - mean squared error
6. Change the amount of noise and observe what happens.

### Questions

1. What does the slope mean in this example?
2. What happens to the model when you increase the noise?

<details>
<summary>Starting point code</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Create fake study hours
X = np.linspace(0, 10, 100).reshape(-1, 1)

# Create fake exam scores
# Try changing the noise value from 5 to 15 or 25
noise = np.random.normal(0, 8, size=100)
y = 50 + 4 * X.ravel() + noise

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Print metrics
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("MSE:", mean_squared_error(y_test, predictions))

# Plot data and line
x_line = np.linspace(0, 10, 100).reshape(-1, 1)
y_line = model.predict(x_line)

plt.scatter(X, y, edgecolor="black", label="Actual data")
plt.plot(x_line, y_line, linewidth=3, label="Regression line")
plt.xlabel("Hours studied")
plt.ylabel("Exam score")
plt.title("Linear Regression: Study Hours vs Exam Score")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

</details>

---

## Part B — Built-In Dataset: Diabetes Regression

### Scenario

Scikit-learn includes a diabetes dataset where the goal is to predict disease progression one year after baseline.

### Tasks

1. Load the diabetes dataset using `load_diabetes()`.
2. Use **one feature** at first so you can plot it.
3. Train a `LinearRegression` model.
4. Plot the feature vs target.
5. Plot the regression line.
6. Print MSE.
7. Try using **all features** and compare the score.

### Questions

1. Did using all features improve the model?
2. Why might a one-feature model be easier to visualize but weaker?

<details>
<summary>Starting point code</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

diabetes = load_diabetes()

# Use one feature first
X = diabetes.data[:, [2]]  # BMI-like feature
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("One-feature model")
print("MSE:", mean_squared_error(y_test, predictions))

x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)

plt.scatter(X, y, edgecolor="black", alpha=0.7)
plt.plot(x_line, y_line, linewidth=3)
plt.xlabel("BMI-like feature")
plt.ylabel("Disease progression")
plt.title("Diabetes Regression with One Feature")
plt.grid(alpha=0.3)
plt.show()

# Try all features
X_all = diabetes.data

X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE
)

model_all = LinearRegression()
model_all.fit(X_train, y_train)

predictions_all = model_all.predict(X_test)

print()
print("All-feature model")
print("MSE:", mean_squared_error(y_test, predictions_all))
```

</details>

---

## Part C — Play Around

Try at least **two** of the following:

1. Increase the noise in your synthetic dataset.
2. Decrease the number of samples.
3. Add a second feature.
4. Compare training score vs testing score.
5. Try a dataset where the relationship is curved instead of linear.

---

# Assignment 2 — Logistic Regression

## Goal

Use logistic regression to predict a category.

Logistic regression is used for classification, often binary classification.

Examples:

- Pass/fail
- Spam/not spam
- Sick/not sick
- Churn/not churn
- Fraud/not fraud

---


## Part A — Built-In Dataset: Breast Cancer Classification

### Scenario

Scikit-learn includes a breast cancer dataset. The goal is to classify tumors as malignant or benign based on measurements.

### Tasks

1. Load the dataset with `load_breast_cancer()`.
2. Train a `LogisticRegression` model.
3. Print:
   - accuracy
4. Try changing `max_iter`.
5. Try using only two features and plot the data.

### Questions

1. Why do we split into training and testing data?
2. Which errors seem more serious in this context: false positives or false negatives?
3. Why might scaling matter for logistic regression?

<details>
<summary>Starting point code</summary>

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

RANDOM_STATE = 42

cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y
)

# Pipeline = scaler first, then model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
```

</details>

---

## Part C — Play Around

Try at least **two** of the following:

1. Use only two features and plot them.
2. Change the train/test split size.
3. Change the probability threshold.
4. Compare results with and without scaling.
5. Create your own binary classification dataset with `make_classification()`.

---

# Assignment 3 — Decision Trees

## Goal

Use decision trees for classification and model interpretation.

Decision trees are useful because they are visual and interpretable.

Examples:

- Classify products into categories
- Predict whether a customer accepts an offer
- Sort applicants into decision groups
- Predict whether a player action is risky or safe

---

## Part A — Game Action Classifier

### Scenario

You are working on a simple video game analytics system.

A player can choose whether to **attack** or **defend**. You want to predict whether the action was a **safe choice** or a **risky choice** based on the game state.

Features:

- `enemy_distance`
- `player_health`

Label:

- `0 = risky action`
- `1 = safe action`

### Tasks

1. Create your own synthetic dataset with NumPy.
2. Plot the raw data.
3. Train a `DecisionTreeClassifier`.
4. Plot the decision boundary.
5. Plot the tree using `plot_tree`.
6. Try `max_depth=1`, `max_depth=3`, and `max_depth=None`.

### Suggested Hidden Rule

You can create labels using a rule like:

```text
safe if player_health is high OR enemy_distance is far
risky if player_health is low AND enemy_distance is close
```

### Questions

1. What rule did the tree seem to learn?
2. Why do decision trees often create box-like decision regions?
3. What happens when `max_depth` is too small?
4. What happens when `max_depth` is unlimited?

<details>
<summary>Starting point code</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = model.predict(grid).reshape(xx.shape)

    plt.contourf(xx, yy, predictions, alpha=0.25)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="black")
    plt.title(title)
    plt.xlabel("Enemy distance")
    plt.ylabel("Player health")
    plt.grid(alpha=0.3)
    plt.show()

# Create fake game-state data
n_samples = 300

enemy_distance = np.random.uniform(0, 100, size=n_samples)
player_health = np.random.uniform(0, 100, size=n_samples)

X = np.column_stack([enemy_distance, player_health])

# Hidden rule:
# Safe if the enemy is far away OR the player has high health.
# Risky if the enemy is close and the player has low health.
y = ((enemy_distance > 55) | (player_health > 60)).astype(int)

# Add some randomness/noise so the problem is not perfectly clean
noise_indices = np.random.choice(n_samples, size=25, replace=False)
y[noise_indices] = 1 - y[noise_indices]

plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="black")
plt.xlabel("Enemy distance")
plt.ylabel("Player health")
plt.title("Game Action Dataset")
plt.grid(alpha=0.3)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y
)

for depth in [1, 3, None]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("max_depth:", depth)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print()

    plot_decision_boundary(
        model,
        X,
        y,
        title=f"Game Action Decision Boundary, max_depth={depth}"
    )

# Plot one readable tree
final_tree = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
final_tree.fit(X_train, y_train)

plt.figure(figsize=(16, 8))
plot_tree(
    final_tree,
    filled=True,
    rounded=True,
    feature_names=["enemy_distance", "player_health"],
    class_names=["Risky", "Safe"]
)
plt.title("Game Action Decision Tree")
plt.show()

print(classification_report(y_test, final_tree.predict(X_test)))
```

</details>

---

## Part B — Loan Pre-Screening Classifier

### Scenario

A bank wants a simple, explainable model to pre-screen loan applications.

This is not meant to be a real lending model. It is only a practice dataset.

Features:

- `income`
- `debt`
- `credit_score`

Label:

- `0 = likely deny`
- `1 = likely approve`

### Tasks

1. Generate synthetic applicant data with NumPy.
2. Create labels using a hidden rule.
3. Train a `DecisionTreeClassifier`.
4. Print accuracy and a classification report.
5. Plot the tree.
6. Print feature importances.
7. Try changing:
   - `max_depth`
   - `min_samples_leaf`

### Suggested Hidden Rule

You can create labels using a rule like:

```text
approve if credit_score is high and debt is not too high
or if income is high and credit_score is decent
```

### Questions

1. Which feature did the tree use first?
2. Which feature had the highest importance?
3. How did `min_samples_leaf` affect the tree?
4. Why might interpretability matter for this kind of problem?

<details>
<summary>Starting point code</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

n_samples = 500

income = np.random.normal(75000, 25000, size=n_samples)
debt = np.random.normal(25000, 12000, size=n_samples)
credit_score = np.random.normal(680, 70, size=n_samples)

# Keep values in reasonable ranges
income = np.clip(income, 25000, 180000)
debt = np.clip(debt, 0, 100000)
credit_score = np.clip(credit_score, 300, 850)

X = np.column_stack([income, debt, credit_score])

# Hidden approval rule
y = (
    ((credit_score > 700) & (debt < 35000)) |
    ((income > 95000) & (credit_score > 650) & (debt < 60000))
).astype(int)

# Add label noise
noise_indices = np.random.choice(n_samples, size=40, replace=False)
y[noise_indices] = 1 - y[noise_indices]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y
)

feature_names = ["income", "debt", "credit_score"]
class_names = ["Likely Deny", "Likely Approve"]

for depth in [2, 4, None]:
    model = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_leaf=10,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("max_depth:", depth)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Feature importances:")
    for name, importance in zip(feature_names, model.feature_importances_):
        print(f"  {name}: {importance:.3f}")
    print()

final_tree = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=10,
    random_state=RANDOM_STATE
)

final_tree.fit(X_train, y_train)

print(classification_report(
    y_test,
    final_tree.predict(X_test),
    target_names=class_names
))

plt.figure(figsize=(20, 10))
plot_tree(
    final_tree,
    filled=True,
    rounded=True,
    feature_names=feature_names,
    class_names=class_names,
    fontsize=9
)
plt.title("Loan Pre-Screening Decision Tree")
plt.show()
```

</details>

---

## Part C — Product Return Risk Classifier

### Scenario

An online store wants to predict whether an order is likely to be returned.

Features:

- `item_price`
- `discount_percent`
- `customer_past_returns`

Label:

- `0 = low return risk`
- `1 = high return risk`

### Tasks

1. Generate a synthetic e-commerce dataset.
2. Train a decision tree.
3. Print accuracy and classification report.
4. Plot feature importances.
5. Try different values of:
   - `max_depth`
   - `min_samples_split`
   - `min_samples_leaf`
6. Write a short explanation of the rules your tree learned.

### Suggested Hidden Rule

You can create labels using a rule like:

```text
high return risk if the discount is very high and the customer has many past returns
or if the item is expensive and the customer has at least some past returns
```

### Questions

1. Which feature mattered most?
2. Did the tree seem easy to explain?
3. What tuning value helped prevent the tree from becoming too complicated?
4. What are some ethical or business concerns with using this type of model?

<details>
<summary>Starting point code</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

n_samples = 500

item_price = np.random.uniform(10, 500, size=n_samples)
discount_percent = np.random.uniform(0, 70, size=n_samples)
customer_past_returns = np.random.poisson(lam=2, size=n_samples)

X = np.column_stack([
    item_price,
    discount_percent,
    customer_past_returns
])

# Hidden return-risk rule
y = (
    ((discount_percent > 45) & (customer_past_returns >= 3)) |
    ((item_price > 250) & (customer_past_returns >= 2))
).astype(int)

# Add noise
noise_indices = np.random.choice(n_samples, size=50, replace=False)
y[noise_indices] = 1 - y[noise_indices]

feature_names = [
    "item_price",
    "discount_percent",
    "customer_past_returns"
]

class_names = [
    "Low Return Risk",
    "High Return Risk"
]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y
)

model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=10,
    random_state=RANDOM_STATE
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(
    y_test,
    predictions,
    target_names=class_names
))

plt.figure(figsize=(8, 5))
plt.bar(feature_names, model.feature_importances_)
plt.title("Product Return Risk Feature Importances")
plt.ylabel("Importance")
plt.xticks(rotation=20)
plt.grid(axis="y", alpha=0.3)
plt.show()

plt.figure(figsize=(20, 10))
plot_tree(
    model,
    filled=True,
    rounded=True,
    feature_names=feature_names,
    class_names=class_names,
    fontsize=9
)
plt.title("Product Return Risk Decision Tree")
plt.show()
```

</details>

---

## Part D — Play Around

Try at least **two** of the following:

1. Change `max_depth`.
2. Change `min_samples_leaf`.
3. Change `min_samples_split`.
4. Add more label noise.
5. Remove one feature and see how performance changes.
6. Create your own decision tree scenario with at least two features and one label.



# Assignment 4 — K-Means Clustering

## Goal

Use K-Means to discover groups in unlabeled data.

K-Means is unsupervised learning. That means the model does not get answer labels during training.

Examples:

- Grouping athletes by performance profile
- Grouping neighborhoods by housing patterns
- Grouping products by price and popularity
- Grouping game players by behavior

---

## Part A — Athlete Training Groups

### Scenario

You are helping a coach group athletes based on training data.

Features:

- `speed_score`
- `strength_score`

There are no labels. The coach wants to discover natural training groups.

### Tasks

1. Generate synthetic athlete data using NumPy or `make_blobs()`.
2. Plot the raw data.
3. Run K-Means with `K=3`.
4. Plot the clusters and centroids.
5. Try `K=2`, `K=3`, `K=4`, and `K=5`.
6. Print inertia for each K.
7. Decide which K seems most useful for a coach.

### Questions

1. What might each cluster represent in real life?
2. What does each centroid represent?
3. What happens when K is too small?
4. What happens when K is too large?

<details>
<summary>Starting point code</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

RANDOM_STATE = 42

# Synthetic athlete profiles
# Think of these as groups like:
#   speed-focused, strength-focused, balanced
X, hidden_groups = make_blobs(
    n_samples=300,
    centers=[
        [80, 45],   # high speed, lower strength
        [45, 85],   # lower speed, high strength
        [70, 75]    # balanced/high performers
    ],
    cluster_std=[8, 9, 7],
    random_state=RANDOM_STATE
)

plt.scatter(X[:, 0], X[:, 1], edgecolor="black")
plt.title("Raw Athlete Training Data")
plt.xlabel("Speed score")
plt.ylabel("Strength score")
plt.grid(alpha=0.3)
plt.show()

for k in [2, 3, 4, 5]:
    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    model.fit(X)

    labels = model.labels_
    centers = model.cluster_centers_

    print("K:", k)
    print("Inertia:", model.inertia_)
    print("Centroids:")
    print(centers)
    print()

    plt.scatter(X[:, 0], X[:, 1], c=labels, edgecolor="black")
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="X",
        s=250,
        edgecolor="black"
    )
    plt.title(f"Athlete Training Groups, K={k}")
    plt.xlabel("Speed score")
    plt.ylabel("Strength score")
    plt.grid(alpha=0.3)
    plt.show()
```

</details>

---

## Part B — Product Shelf Groups

### Scenario

A store wants to group products based on their selling behavior.

Features:

- `price`
- `monthly_units_sold`

There are no labels. The goal is to discover product types such as:

- cheap/high-volume
- expensive/low-volume
- mid-price/mid-volume

### Tasks

1. Generate synthetic product data.
2. Plot the raw data.
3. Run K-Means with different K values.
4. Plot the clusters and centroids.
5. Use the elbow method to help choose K.
6. Write a short business interpretation of the clusters.

### Questions

1. Which K value seems most useful?
2. What does a centroid mean in this business context?
3. How could a store use these groups?
4. Why should we be careful about blindly trusting the clusters?

<details>
<summary>Starting point code</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

RANDOM_STATE = 42

X, hidden_groups = make_blobs(
    n_samples=350,
    centers=[
        [15, 900],    # low price, high volume
        [80, 300],    # mid price, mid volume
        [250, 80],    # high price, low volume
        [40, 650]     # lower-mid price, strong volume
    ],
    cluster_std=[
        [5, 120],
        [15, 80],
        [45, 30],
        [8, 100]
    ],
    random_state=RANDOM_STATE
)

plt.scatter(X[:, 0], X[:, 1], edgecolor="black")
plt.title("Raw Product Shelf Data")
plt.xlabel("Price")
plt.ylabel("Monthly units sold")
plt.grid(alpha=0.3)
plt.show()

# Try a few K values
for k in [2, 3, 4, 5]:
    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    model.fit(X)

    plt.scatter(X[:, 0], X[:, 1], c=model.labels_, edgecolor="black")
    plt.scatter(
        model.cluster_centers_[:, 0],
        model.cluster_centers_[:, 1],
        marker="X",
        s=250,
        edgecolor="black"
    )
    plt.title(f"Product Shelf Groups, K={k}")
    plt.xlabel("Price")
    plt.ylabel("Monthly units sold")
    plt.grid(alpha=0.3)
    plt.show()

# Elbow method
k_values = range(1, 11)
inertias = []

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )
    model.fit(X)
    inertias.append(model.inertia_)

plt.plot(list(k_values), inertias, marker="o")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow Method for Product Shelf Groups")
plt.grid(alpha=0.3)
plt.show()
```

</details>

---

## Part C — Neighborhood Housing Profiles

### Scenario

A city analyst wants to group neighborhoods based on housing patterns.

Features:

- `median_home_price`
- `average_commute_minutes`

There are no labels. The analyst wants to discover neighborhood profiles.

### Tasks

1. Generate synthetic neighborhood data.
2. Scale the data using `StandardScaler`.
3. Run K-Means before scaling.
4. Run K-Means after scaling.
5. Compare the plots.
6. Explain why scaling matters.

### Questions

1. Why can different feature scales confuse K-Means?
2. What changed after scaling?
3. In the original units, what might the clusters represent?
4. Should every K-Means problem use scaling?

<details>
<summary>Starting point code</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

# Feature 1 is much larger than Feature 2.
# This makes the scaling issue easier to see.
X, hidden_groups = make_blobs(
    n_samples=300,
    centers=[
        [180000, 22],
        [350000, 35],
        [650000, 50]
    ],
    cluster_std=[
        [35000, 6],
        [55000, 7],
        [90000, 8]
    ],
    random_state=RANDOM_STATE
)

plt.scatter(X[:, 0], X[:, 1], edgecolor="black")
plt.title("Raw Neighborhood Housing Data")
plt.xlabel("Median home price")
plt.ylabel("Average commute minutes")
plt.grid(alpha=0.3)
plt.show()

# K-Means before scaling
model_unscaled = KMeans(
    n_clusters=3,
    random_state=RANDOM_STATE,
    n_init=10
)

model_unscaled.fit(X)

plt.scatter(X[:, 0], X[:, 1], c=model_unscaled.labels_, edgecolor="black")
plt.scatter(
    model_unscaled.cluster_centers_[:, 0],
    model_unscaled.cluster_centers_[:, 1],
    marker="X",
    s=250,
    edgecolor="black"
)
plt.title("K-Means Before Scaling")
plt.xlabel("Median home price")
plt.ylabel("Average commute minutes")
plt.grid(alpha=0.3)
plt.show()

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model_scaled = KMeans(
    n_clusters=3,
    random_state=RANDOM_STATE,
    n_init=10
)

model_scaled.fit(X_scaled)

# Plot scaled clustering labels back on the original data
plt.scatter(X[:, 0], X[:, 1], c=model_scaled.labels_, edgecolor="black")
plt.title("K-Means After Scaling, Shown in Original Units")
plt.xlabel("Median home price")
plt.ylabel("Average commute minutes")
plt.grid(alpha=0.3)
plt.show()
```

</details>

---

## Part D — Play Around

Try at least **two** of the following:

1. Add a fourth athlete group or product group.
2. Increase the cluster overlap by increasing `cluster_std`.
3. Decrease the number of samples.
4. Use `silhouette_score` to compare K values.
5. Add a third feature, such as age, rating, or experience level.
6. Create your own K-Means scenario with at least two features.



# Capstone — Choose the Best Model

## Goal

Practice identifying which type of model fits which type of problem.

For each scenario below, choose one of:

- Linear Regression
- Logistic Regression
- Decision Tree
- K-Means Clustering

Then write a short explanation.

---

## Scenario 1

You have customer data but no labels. You want to discover groups of similar customers based on spending habits.

**Best model:**

**Why:**

---

## Scenario 2

You want to predict the selling price of a used car based on mileage, age, and engine size.

**Best model:**

**Why:**

---

## Scenario 3

You want to predict whether an email is spam or not spam.

**Best model:**

**Why:**

---

## Scenario 4

You want to classify flowers into species, and you also want to explain the rules the model used.

**Best model:**

**Why:**

---

## Scenario 5

You have a dataset of online users and want to discover strange behavior groups, but you do not have labels.

**Best model:**

**Why:**

---