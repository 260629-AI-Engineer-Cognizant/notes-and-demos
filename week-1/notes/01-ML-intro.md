# Machine Learning Overview

## What is Machine Learning?

Machine Learning (ML) is the process of teaching computers to recognize patterns from data instead of writing explicit rules.

**Traditional Programming**

```
Rules + Data -> Answers
```

**Machine Learning**

```
Data + Answers -> Model
New Data -> Predicted Answer
```

Real-world examples include spam detection, fraud detection, house price prediction, recommendation systems, medical diagnosis, customer segmentation, and predictive maintenance.

---

# Common Use Cases

## Regression

Predicts a **number**.

Examples:

- House price
- Temperature tomorrow
- Sales next month
- Delivery time

## Classification

Predicts a **category**.

Examples:

- Spam vs Not Spam
- Fraud vs Legitimate
- Dog vs Cat
- Customer Will Churn vs Won't Churn

---

# Types of Machine Learning

## Supervised Learning

Uses **labeled data**.

```
Features -> Label
```

Example:

```
Square Feet -> House Price
```

Popular algorithms:

- Linear Regression
- Logistic Regression
- Decision Trees
- Random Forests
- Support Vector Machines (SVM)
- Gradient Boosting

---

## Unsupervised Learning

No labels.

Goal is to discover patterns.

Examples:

- Customer segmentation
- Topic modeling
- Finding unusual behavior

Popular algorithms:

- K-Means
- DBSCAN
- Hierarchical Clustering
- PCA

---

## Semi-Supervised Learning

Small amount of labeled data.

Large amount of unlabeled data.

Useful when labels are expensive.

Example:

100 labeled X-rays

100,000 unlabeled X-rays

---

## Reinforcement Learning

An agent learns through rewards and punishments.

Examples:

- Robotics
- Self-driving research
- Game AI
- Recommendation optimization

---

# Common Terminology

| Term | Meaning |
|------|---------|
| Feature | Input variable |
| Label | Correct answer |
| Sample | One row of data |
| Training Set | Data used to learn |
| Test Set | Data used to evaluate |
| Prediction | Model output |
| Epoch | One full pass over training data |
| Model | Learned relationship |
| Cost/Loss Function | Measures prediction error |
| Hyperparameter | Setting chosen before training |

---

# Error / Loss Functions

Regression often uses:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

Classification commonly uses:

- Log Loss (Cross Entropy)

Lower loss generally means a better model.

---

# Supervised Learning

## Common Uses

- Predict prices
- Predict demand
- Detect fraud
- Email spam
- Disease diagnosis
- Loan approval

## Linear Regression

Fits the best straight line.

Example:

```
Price = m * Size + b
```

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict([[1800]])
```

### Multiple Linear Regression

Uses multiple features.

```
Price = Size + Bedrooms + Age
```

### Polynomial Regression

Allows curves by creating higher-order features.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

model = make_pipeline(
    PolynomialFeatures(2),
    LinearRegression()
)
```

---

# Logistic Regression

Despite its name, this is a **classification** algorithm.

Outputs probabilities.

Example:

```
Will customer churn?
```

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

prob = model.predict_proba(X_test)
```

---

# Decision Trees

Makes decisions by asking questions.

```
Income > 60k?

   Yes
    |
Own House?

   No
```

Advantages

- Easy to explain
- Handles nonlinear data
- Little preprocessing

Disadvantages

- Can overfit

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)
```

---

# Random Forests

Many decision trees vote together.

Benefits:

- Better accuracy
- Less overfitting
- Handles noisy data well

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100
)
```

---

# Support Vector Machines (SVM)

Finds the boundary with the **largest margin** between classes.

Works especially well on medium-sized datasets.

```python
from sklearn.svm import SVC

model = SVC(kernel="linear")
```

Common kernels

- Linear
- Polynomial
- RBF

---

# Unsupervised Learning

## Common Uses

- Customer segmentation
- Document grouping
- Image compression
- Anomaly detection

---

# K-Means Clustering

Groups similar data together.

Algorithm:

1. Pick K centers
2. Assign every point
3. Move centers
4. Repeat

```python
from sklearn.cluster import KMeans

model = KMeans(
    n_clusters=3,
    random_state=42
)

model.fit(X)
```

## Choosing K

Common techniques

- Elbow Method
- Silhouette Score

The elbow method looks for where adding more clusters stops providing large improvements.

---

## Distance

Most implementations use Euclidean distance.

```
distance =
sqrt((x2-x1)^2 + (y2-y1)^2)
```

Other possibilities:

- Manhattan
- Cosine Similarity

---

# Other Important Unsupervised Algorithms

## DBSCAN

Great when clusters have irregular shapes.

Can automatically detect outliers.

---

## Hierarchical Clustering

Builds clusters like a family tree.

Useful for visualization.

---

## PCA (Principal Component Analysis)

Reduces the number of features while preserving most information.

Useful for:

- Visualization
- Speeding up models
- Noise reduction

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_new = pca.fit_transform(X)
```

---

# Model Evaluation

Regression

- MAE
- MSE
- RMSE
- R² Score

Classification

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

---

# Overfitting vs Underfitting

## Underfitting

Model is too simple.

- Poor training performance
- Poor testing performance

## Overfitting

Memorizes the training data.

- Excellent training score
- Poor testing score

Solutions

- More data
- Simpler model
- Regularization
- Cross Validation
- Early stopping
- Pruning trees

---

# Hyperparameter Tuning

Examples

- max_depth
- learning_rate
- n_estimators
- C
- K

Common approaches

- Grid Search
- Random Search

```python
from sklearn.model_selection import GridSearchCV
```

---

# Typical ML Workflow

1. Collect data
2. Clean data
3. Split train/test
4. Choose algorithm
5. Train
6. Evaluate
7. Tune
8. Deploy
9. Monitor

---

# Choosing an Algorithm

| Problem | Good Starting Point |
|----------|---------------------|
| Predict a number | Linear Regression |
| Binary decision | Logistic Regression |
| Explainable classification | Decision Tree |
| High accuracy | Random Forest |
| Clear margins | SVM |
| Unknown groups | K-Means |
| Reduce dimensions | PCA |
| Odd shaped clusters | DBSCAN |

Remember: there is no universally best algorithm, choose based on the data, interpretability needs, and performance.
