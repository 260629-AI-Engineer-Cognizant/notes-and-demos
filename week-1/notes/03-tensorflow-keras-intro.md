# TensorFlow Day One

## Big Picture

### What is TensorFlow?

TensorFlow is a library/framework for building and training machine learning models, especially neural networks.

It helps with:

- storing numeric data
- performing fast math operations
- building neural networks
- calculating gradients
- updating model weights
- training models on larger datasets

Simple version:

> TensorFlow is the deep learning engine.

---

## What is Keras?

Keras is the high-level API we use to build neural networks in TensorFlow.

Simple version:

> Keras is the beginner-friendly way to build TensorFlow models.

Most of our TensorFlow code will use this pattern:

```python
model = tf.keras.Sequential([...])

model.compile(
    optimizer=...,
    loss=...,
    metrics=[...]
)

model.fit(X_train, y_train, epochs=...)
model.evaluate(X_test, y_test)
model.predict(new_data)
```

---

# Tensor Basics

## What is a Tensor?

A tensor is a container for numbers.

You can think of tensors like arrays with a shape and datatype.

| Name | Example | Meaning |
|---|---|---|
| Scalar | `5` | one number |
| Vector | `[1, 2, 3]` | a list of numbers |
| Matrix | `[[1, 2], [3, 4]]` | rows and columns |
| Higher-dimensional tensor | image batch | more complex numeric data |

In machine learning, our data usually becomes tensors before being passed into a model.

Example:

```python
import tensorflow as tf

x = tf.constant([
    [1.0, 2.0],
    [3.0, 4.0]
])

print(x)
print(x.shape)
```

This tensor has shape:

```text
(2, 2)
```

That means 2 rows and 2 columns.

---

## Tensor Shape

The shape tells us how the data is arranged.

Examples:

| Tensor | Shape | Meaning |
|---|---|---|
| `tf.constant(7)` | `()` | scalar |
| `[1, 2, 3]` | `(3,)` | vector with 3 values |
| `[[1, 2], [3, 4], [5, 6]]` | `(3, 2)` | 3 rows, 2 columns |
| image batch | `(batch, height, width, channels)` | many images at once |

For tabular data, we usually use:

```text
(number_of_rows, number_of_features)
```

So if we have 500 customers and 3 features per customer, our input shape is:

```text
(500, 3)
```

When building a Keras model, each individual row has 3 features, so we write:

```python
tf.keras.layers.Input(shape=(3,))
```

---

## Tensor Reductions

A reduction takes many values and reduces them into fewer values.

Common reductions:

| Function | Meaning |
|---|---|
| `tf.reduce_sum()` | adds values |
| `tf.reduce_mean()` | averages values |
| `tf.reduce_max()` | finds largest value |
| `tf.reduce_min()` | finds smallest value |

Example:

```python
scores = tf.constant([
    [80.0, 90.0, 100.0],
    [70.0, 75.0, 85.0]
])

tf.reduce_mean(scores)
```

This reduces all values into one average.

You can also reduce across an axis:

```python
tf.reduce_mean(scores, axis=0)
```

This reduces down the rows and gives a mean for each column.

```python
tf.reduce_mean(scores, axis=1)
```

This reduces across columns and gives a mean for each row.

Why this matters:

> Loss functions often calculate an error for each row, then reduce those errors into one overall loss score.

---

# Neural Network Recap

A neural network is made of layers.

A simple network may look like this:

```text
input layer -> hidden layer -> output layer
```

Each neuron does something like this:

```text
weighted sum = input1 * weight1 + input2 * weight2 + bias
output = activation(weighted sum)
```

In code, the weighted sum can look like this:

```python
z = np.dot(inputs, weights) + bias
```

The activation function transforms that value.

---

## Weights and Biases

### Weights

Weights control how important each input is.

Example:

```text
prediction = square_feet * weight + bias
```

If the weight is large, that input has a bigger effect.

### Bias

Bias shifts the result up or down.

A bias lets the model adjust even when the input is zero.

---

# How Training Works

## Forward Pass

The model makes predictions using the current weights.

```text
inputs -> model -> predictions
```

At the beginning, the predictions are usually bad because the weights are mostly random.

---

## Loss Function

The loss function measures how wrong the model is.

Simple version:

> Loss is the model's error score.

Common examples:

| Problem Type | Loss Function |
|---|---|
| Binary classification | `binary_crossentropy` |
| Multiclass classification | `sparse_categorical_crossentropy` |
| Regression | `mean_squared_error` or `mse` |

For regression, mean squared error usually follows this idea:

```text
prediction error = actual - predicted
squared error = error²
mse = average of squared errors
```

That final average is a reduction.

---

## Backpropagation

Backpropagation is how a neural network figures out which weights contributed to the error.

Simple version:

> Backpropagation works backward from the loss and calculates gradients for the weights.

A gradient tells us:

```text
If we change this weight a little bit, how does the loss change?
```

You do not usually write backpropagation manually in Keras.

Keras handles it during:

```python
model.fit(X_train, y_train)
```

---

## Gradient Descent

Gradient descent is the strategy of updating weights to reduce loss.

Simple version:

> Backpropagation calculates the gradients. Gradient descent uses those gradients to move the weights in a better direction.

A tiny update looks conceptually like this:

```text
new_weight = old_weight - learning_rate * gradient
```

The `learning_rate` controls how big the update is.

- Too small: training may be very slow.
- Too large: training may jump around or fail to settle.

---

## Optimizer

The optimizer is the tool that actually updates the model's weights.

Simple version:

> The loss tells the model how wrong it is. Backpropagation calculates gradients. The optimizer updates the weights.

A common optimizer is Adam:

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
```

Strong beginner default:

```python
tf.keras.optimizers.Adam(learning_rate=0.001)
```

For small classroom demos, `0.01` may train faster, but `0.001` is the safer default.

---

## Common Optimizers

| Optimizer | Beginner intuition | When to use |
|---|---|---|
| `SGD` | plain gradient descent updates | useful for learning the concept |
| `SGD(momentum=0.9)` | builds speed in consistent directions | can train smoother than plain SGD |
| `RMSprop` | adapts update sizes based on recent gradients | often used for neural networks |
| `Adam` | adaptive, fast, strong default | best beginner default |
| `AdamW` | Adam with weight decay | useful when adding regularization |

For most beginner Keras projects:

```python
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
```

---

# Activation Functions

Activation functions help neural networks learn nonlinear patterns.

Without activation functions, stacked Dense layers would mostly collapse into one big linear transformation.

## Strong Default

For hidden layers, use:

```python
activation="relu"
```

ReLU is usually the safest beginner default for hidden layers.

---

## Common Activations

| Activation | Common Use |
|---|---|
| `relu` | hidden layers |
| `sigmoid` | binary classification output |
| `softmax` | multiclass classification output |
| `linear` or no activation | regression output |
| `tanh` | sometimes useful in hidden layers, especially centered data |

### ReLU

ReLU returns 0 for negative values and returns the original value for positive values.

```text
negative value -> 0
positive value -> same value
```

### Sigmoid

Sigmoid squishes a number between 0 and 1.

That makes it useful for binary classification.

Example:

```text
0.92 -> probably class 1
0.08 -> probably class 0
0.51 -> uncertain, but slightly class 1
```

### Softmax

Softmax is used when there are more than two possible classes.

Example:

```text
cat: 0.05
dog: 0.90
bird: 0.05
```

### Linear

Linear means no special squishing.

Use it for regression when the model should predict a number like price, temperature, minutes, or sales.

---

# Metrics

## Loss vs Metrics

Loss is what the model optimizes.

Metrics are what we display to understand performance.

They can be similar, but they are not exactly the same thing.

Example:

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

Here, the model optimizes `binary_crossentropy`, but we display `accuracy`.

---

## Common Regression Metrics

| Metric | Meaning |
|---|---|
| `mae` | mean absolute error |
| `mse` | mean squared error |
| `RootMeanSquaredError()` | square root of MSE |
| `R2Score()` | how much variance the model explains |

For beginner regression:

```python
metrics=["mae"]
```

MAE is easy to explain because it is in the same unit as the target.

Example:

```text
MAE = 4.8 minutes
```

means the model is off by about 4.8 minutes on average.

---

## Common Classification Metrics

| Metric | Meaning |
|---|---|
| `accuracy` | percent correct |
| `BinaryAccuracy()` | binary classification accuracy |
| `Precision()` | of predicted positives, how many were actually positive |
| `Recall()` | of actual positives, how many did we catch |
| `AUC()` | ranking/separation quality across thresholds |
| `SparseCategoricalAccuracy()` | multiclass accuracy with integer labels |
| `CategoricalAccuracy()` | multiclass accuracy with one-hot labels |

For balanced beginner binary classification:

```python
metrics=["accuracy"]
```

For imbalanced data, accuracy can lie.

Example:

```text
If only 5% of customers churn, a model that always predicts "no churn" is 95% accurate but useless.
```

In that case, consider:

```python
metrics=[
    "accuracy",
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall(),
    tf.keras.metrics.AUC()
]
```

---

# Scaling Features

## What is Scaling?

Scaling changes numeric features so they are on a more comparable range.

Example:

```text
square_feet: 800 to 3500
bedrooms: 1 to 5
home_age: 0 to 80
```

Without scaling, `square_feet` has much larger raw numbers than `bedrooms`.

That can make training harder.

---

## StandardScaler

`StandardScaler` transforms values so each feature has approximately:

```text
mean = 0
standard deviation = 1
```

Conceptually:

```text
scaled_value = (value - mean) / standard_deviation
```

Example:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Important:

> Fit the scaler on the training data only.

Do not do this:

```python
X_test_scaled = scaler.fit_transform(X_test)
```

That leaks information from the test set.

---

## MinMaxScaler

`MinMaxScaler` transforms values to a range, usually 0 to 1.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

This can be useful when you want bounded values.

---

## Keras Normalization Layer

Keras also has a `Normalization` layer.

```python
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(X_train)
```

This lets preprocessing live inside the model.

For beginner demos, `StandardScaler` is often easier to explain.

---

## Should We Always Scale?

For neural networks with numeric tabular features:

> Usually, yes.

Scaling is also important for models like:

- logistic regression
- SVMs
- K-Means
- KNN
- neural networks

Scaling is less important for tree-based models like:

- decision trees
- random forests
- gradient boosted trees

Do not scale categorical labels like:

```text
0 = stayed
1 = churned
```

For regression targets, scaling the target can sometimes help, but it is optional for beginner demos.

---

# Building a Simple Keras Model

Here is a small binary classification model:

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
```

Line by line:

```python
tf.keras.Sequential([...])
```

This creates a model where layers are stacked one after another.

```python
tf.keras.layers.Input(shape=(3,))
```

Each input example has 3 features.

```python
tf.keras.layers.Dense(8, activation="relu")
```

This adds a hidden layer with 8 neurons.

```python
tf.keras.layers.Dense(4, activation="relu")
```

This adds another hidden layer with 4 neurons.

```python
tf.keras.layers.Dense(1, activation="sigmoid")
```

This adds one output neuron. The sigmoid activation returns a value between 0 and 1.

---

## Regression vs Classification Model Heads

The hidden layers can be the same.

The output layer changes based on the problem.

### Regression

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)
```

Use this when predicting a number.

---

### Binary Classification

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

Use this when predicting yes/no, true/false, 0/1.

---

### Multiclass Classification

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

Use this when each example belongs to one of several possible classes.

---

# Training a Model

```python
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=100,
    validation_split=0.2
)
```

An epoch is one full pass through the training data.

During training, Keras repeatedly does this:

```text
make predictions
calculate loss
calculate gradients with backpropagation
update weights with the optimizer
repeat
```

---

# Reading Training Curves

When training, compare training performance and validation performance.

## Underfitting

Training loss is bad and validation loss is bad.

Possible fixes:

- more neurons
- more layers
- more epochs
- better features
- better preprocessing

## Overfitting

Training loss is good but validation loss is bad.

Possible fixes:

- smaller model
- more data
- dropout
- regularization
- early stopping

## Healthy Training

Training and validation both improve, and validation does not get much worse.

That is usually what we want.

---

# Model Tuning Knobs

You can tune many things in a neural network.

Common beginner knobs:

| Knob | What It Means |
|---|---|
| number of hidden layers | how many transformations the model can learn |
| number of neurons | how much capacity each layer has |
| activation function | how the model handles nonlinear patterns |
| learning rate | how large weight updates are |
| epochs | how many times the model sees the dataset |
| batch size | how many examples are used per update step |
| optimizer | how gradients are converted into weight updates |
| scaler | how numeric inputs are normalized before training |

Important reminder:

> Bigger is not always better. More neurons/layers can help, but they can also make the model slower or more likely to overfit.

---

# Strong Defaults

For beginner tabular neural network problems:

| Question | Strong Default |
|---|---|
| Hidden activation? | `relu` |
| Optimizer? | `Adam(learning_rate=0.001)` |
| Regression loss? | `mse` |
| Regression metric? | `mae` |
| Binary classification loss? | `binary_crossentropy` |
| Binary classification metric? | `accuracy`, then add precision/recall/AUC if needed |
| Multiclass loss with integer labels? | `sparse_categorical_crossentropy` |
| Numeric scaler? | `StandardScaler` |
| Small starting architecture? | `Dense(8) -> Dense(4)` |
| Slightly larger architecture? | `Dense(16) -> Dense(8)` |

---

# TFLearn vs Keras

TFLearn was an older library that made TensorFlow easier to use.

Modern TensorFlow code usually uses Keras instead.

### TFLearn-style code

```python
net = tflearn.input_data(shape=[None, 3])
net = tflearn.fully_connected(net, 8, activation="relu")
net = tflearn.fully_connected(net, 4, activation="relu")
net = tflearn.fully_connected(net, 1, activation="sigmoid")
model = tflearn.DNN(net)
```

### Keras-style code

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
```

The idea is the same:

```text
input layer -> hidden layer -> hidden layer -> output layer
```

But Keras is the better tool for modern TensorFlow practice.

---

# The Main Keras Pattern to Remember

```python
# 1. Build
model = tf.keras.Sequential([...])

# 2. Compile
model.compile(optimizer=..., loss=..., metrics=[...])

# 3. Train
model.fit(X_train, y_train, epochs=...)

# 4. Evaluate
model.evaluate(X_test, y_test)

# 5. Predict
model.predict(new_data)
```

That pattern is the foundation for many TensorFlow/Keras projects.
