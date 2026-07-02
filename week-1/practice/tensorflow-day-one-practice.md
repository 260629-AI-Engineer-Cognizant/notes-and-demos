# Keras Mini-Assignment: Regression and Classification with 3 Features

## Goal

You are going to build two small neural networks with Keras.

By the end, you should be able to:

- Create feature arrays `X` and target arrays `y`
- Split data into training and testing sets
- Scale input features
- Build a small Keras neural network
- Choose the correct output layer for regression vs classification
- Choose the correct loss function for regression vs classification
- Train, evaluate, and make predictions
- Tune the model and explain whether it improved

---

# Part 1: Regression

## Scenario: Predict Delivery Time

A delivery company wants to estimate how many minutes a delivery will take.

You are given three input features:

1. `distance_miles`
2. `num_items`
3. `traffic_level`

The target is:

- `delivery_time_minutes`

This is a **regression** problem because the model predicts a number.

---

## Part 1 Requirements

Build a neural network that predicts delivery time.

Your model should start with this general architecture:

```text
3 input features → hidden layer → hidden layer → 1 output
```

A reasonable first model is:

```python
tf.keras.layers.Dense(8, activation="relu")
tf.keras.layers.Dense(4, activation="relu")
tf.keras.layers.Dense(1)
```

Use:

```python
loss="mse"
metrics=["mae"]
```

---

## Part 1 Tasks

1. Generate the dataset using the starter code.
2. Split the data into training and testing sets.
3. Scale the input features.
4. Build a Keras regression model.
5. Train the model.
6. Evaluate the model using MAE.
7. Make predictions on at least three new deliveries.
8. Change the model in one way and compare results.

Possible changes:

- Increase neurons from `8, 4` to `16, 8`
- Decrease neurons from `8, 4` to `4, 2`
- Add another hidden layer
- Change the learning rate
- Change the number of epochs

---

<details>
<summary>Starter Code for Part 1</summary>

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
tf.random.set_seed(42)

n_samples = 500

distance_miles = np.random.uniform(1, 25, size=n_samples)
num_items = np.random.randint(1, 12, size=n_samples)
traffic_level = np.random.randint(1, 6, size=n_samples)

noise = np.random.normal(0, 5, size=n_samples)

delivery_time_minutes = (
    8
    + distance_miles * 3.2
    + num_items * 1.5
    + traffic_level * 6
    + noise
)

delivery_df = pd.DataFrame({
    "distance_miles": distance_miles,
    "num_items": num_items,
    "traffic_level": traffic_level,
    "delivery_time_minutes": delivery_time_minutes
})

delivery_df.head()
```

</details>

---

# Part 2: Classification

## Scenario: Predict Equipment Maintenance Risk

A company wants to predict whether a machine is at risk of needing maintenance soon.

You are given three input features:

1. `hours_used`
2. `temperature`
3. `vibration_level`

The target is:

- `needs_maintenance`

Where:

- `0` = does not need maintenance soon
- `1` = likely needs maintenance soon

This is a **binary classification** problem because the model predicts one of two categories.

---

## Part 2 Requirements

Build a neural network that predicts maintenance risk.

Your model should start with this general architecture:

```text
3 input features → hidden layer → hidden layer → 1 output
```

A reasonable first model is:

```python
tf.keras.layers.Dense(8, activation="relu")
tf.keras.layers.Dense(4, activation="relu")
tf.keras.layers.Dense(1, activation="sigmoid")
```

Use:

```python
loss="binary_crossentropy"
metrics=["accuracy"]
```

---

## Part 2 Tasks

1. Generate the dataset using the starter code.
2. Split the data into training and testing sets.
3. Scale the input features.
4. Build a Keras classification model.
5. Train the model.
6. Evaluate the model using accuracy.
7. Make predictions on at least three new machines.
8. Convert the predicted probabilities into class labels.
9. Change the model in one way and compare results.

Possible changes:

- Increase neurons from `8, 4` to `16, 8`
- Decrease neurons from `8, 4` to `4, 2`
- Add another hidden layer
- Change the learning rate
- Change the number of epochs
- Try `activation="tanh"` in the hidden layers

---

<details>
<summary>Starter Code for Part 2</summary>

```python
n_machines = 600

hours_used = np.random.randint(50, 5000, size=n_machines)
temperature = np.random.normal(75, 15, size=n_machines)
vibration_level = np.random.normal(4, 1.5, size=n_machines)

temperature = np.clip(temperature, 40, 130)
vibration_level = np.clip(vibration_level, 0.5, 10)

risk_score = (
    -6.0
    + 0.0012 * hours_used
    + 0.045 * temperature
    + 0.65 * vibration_level
)

risk_probability = 1 / (1 + np.exp(-risk_score))

needs_maintenance = np.random.binomial(1, risk_probability)

machine_df = pd.DataFrame({
    "hours_used": hours_used,
    "temperature": temperature,
    "vibration_level": vibration_level,
    "needs_maintenance": needs_maintenance
})

machine_df.head()
```

</details>

---

# Bonus Challenge

Try building a reusable function that creates a model.

Example:

```python
def build_model(hidden_units=[8, 4], output_activation=None, loss="mse", metrics=["mae"]):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(3,)))

    for units in hidden_units:
        model.add(tf.keras.layers.Dense(units, activation="relu"))

    model.add(tf.keras.layers.Dense(1, activation=output_activation))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=loss,
        metrics=metrics
    )

    return model
```

Then try:

```python
regression_model = build_model(
    hidden_units=[8, 4],
    output_activation=None,
    loss="mse",
    metrics=["mae"]
)

classification_model = build_model(
    hidden_units=[8, 4],
    output_activation="sigmoid",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```
