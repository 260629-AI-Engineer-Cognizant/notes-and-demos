# Ticket-Buying Perceptron

## Goal

In this assignment, you will build and experiment with a simple perceptron that predicts whether someone buys a concert ticket.

The model will use three inputs:

1. interest level
2. ticket price
3. distance from the venue

The prediction will be:

```text
1 = buys ticket
0 = does not buy ticket
```

By the end, you should be able to explain:

- how a perceptron makes a prediction
- what weights do
- what bias does
- how the model updates weights when it is wrong
- why a single perceptron is limited

---

## Scenario

A concert venue wants to predict whether a person will buy a ticket.

Each customer has three features:

| Feature | Meaning |
|---|---|
| `interest_level` | how interested they are in the artist, from 1 to 10 |
| `ticket_price` | ticket price in dollars |
| `distance_miles` | how far they live from the venue |

The perceptron calculates a weighted sum:

```text
z = interest_level*w1 + ticket_price*w2 + distance_miles*w3 + bias
```

Then it uses a step activation:

```text
if z >= 0, predict 1
if z < 0, predict 0
```

---

## Part 1: Make One Prediction

Create a function that predicts whether a customer buys a ticket.

Use these starting values:

```python
weights = np.array([1.2, -0.04, -0.08])
bias = -3.0
```

Use this customer:

```python
interest_level = 8
ticket_price = 60
distance_miles = 10
```

### Tasks

1. Create a NumPy array for the customer inputs.
2. Calculate the weighted sum.
3. Apply a step activation.
4. Print the weighted sum and prediction.

<details>
<summary>Starter code</summary>

```python
import numpy as np

def step(z):
    if z >= 0:
        return 1
    else:
        return 0

def predict_ticket_purchase(interest_level, ticket_price, distance_miles, weights, bias):
    x = np.array([interest_level, ticket_price, distance_miles])

    z = np.dot(x, weights) + bias

    prediction = step(z)

    return z, prediction


weights = np.array([1.2, -0.04, -0.08])
bias = -3.0

interest_level = 8
ticket_price = 60
distance_miles = 10

z, prediction = predict_ticket_purchase(
    interest_level,
    ticket_price,
    distance_miles,
    weights,
    bias
)

print("Weighted sum:", z)
print("Prediction:", prediction)

if prediction == 1:
    print("Model says: Buys ticket")
else:
    print("Model says: Does not buy ticket")
```

</details>

### Questions

1. What happens when interest level increases?
2. What happens when ticket price increases?
3. What happens when distance increases?
4. Do the starting weights make sense for this scenario? Why?

---

## Part 2: Predict for Multiple Customers

Use the following customer data:

```python
customers = np.array([
    [10, 40, 5],
    [8, 100, 10],
    [6, 50, 60],
    [3, 30, 5],
    [9, 150, 20],
    [7, 70, 15],
])
```

Each row means:

```text
[interest_level, ticket_price, distance_miles]
```

### Tasks

1. Loop through each customer.
2. Use your prediction function.
3. Print each customer's inputs, weighted sum, and prediction.
4. Label the prediction as `"Buys ticket"` or `"Does not buy ticket"`.

<details>
<summary>Starter code</summary>

```python
customers = np.array([
    [10, 40, 5],
    [8, 100, 10],
    [6, 50, 60],
    [3, 30, 5],
    [9, 150, 20],
    [7, 70, 15],
])

for customer in customers:
    interest_level = customer[0]
    ticket_price = customer[1]
    distance_miles = customer[2]

    z, prediction = predict_ticket_purchase(
        interest_level,
        ticket_price,
        distance_miles,
        weights,
        bias
    )

    if prediction == 1:
        label = "Buys ticket"
    else:
        label = "Does not buy ticket"

    print(
        f"Interest: {interest_level}, "
        f"Price: ${ticket_price}, "
        f"Distance: {distance_miles} miles, "
        f"z={z:.2f}, "
        f"Prediction: {label}"
    )
```

</details>

### Questions

1. Which customer was most likely to buy a ticket?
2. Which customer was least likely to buy a ticket?
3. How can you tell from the weighted sum?
4. Did any prediction surprise you?

---

## Part 3: Modify the Weights and Bias

Now try different versions of the model.

### Version A

```python
weights = np.array([1.2, -0.04, -0.08])
bias = -3.0
```

### Version B

```python
weights = np.array([2.0, -0.02, -0.03])
bias = -5.0
```

### Version C

```python
weights = np.array([0.8, -0.08, -0.12])
bias = -1.0
```

### Tasks

1. Run your customer loop with Version A.
2. Run your customer loop with Version B.
3. Run your customer loop with Version C.
4. Compare the predictions.

### Questions

1. Which version makes interest level matter the most?
2. Which version punishes high ticket prices the most?
3. Which version punishes distance the most?
4. Which version seems most realistic to you?
5. What does the bias seem to control?

---

## Part 4: Update Weights Manually

So far, we have picked weights ourselves.

Now you will practice the basic perceptron update rule.

A perceptron updates its weights when it makes a mistake.

The update rule is:

```text
error = actual - prediction

new_weight = old_weight + learning_rate * error * input_value

new_bias = old_bias + learning_rate * error
```

If the model is correct, the error is `0`, so nothing changes.

If the model predicts too low, the error is positive, so the weights increase.

If the model predicts too high, the error is negative, so the weights decrease.

### Example

Suppose a customer has:

```python
customer = np.array([9, 80, 10])
actual = 1
```

And the model predicts:

```python
prediction = 0
```

Then:

```text
error = actual - prediction
error = 1 - 0
error = 1
```

The model predicted too low, so it needs to adjust upward.

### Tasks

Use this one customer:

```python
customer = np.array([9, 80, 10])
actual = 1
```

Use these starting parameters:

```python
weights = np.array([1.0, -0.05, -0.05])
bias = -5.0
learning_rate = 0.01
```

Then:

1. Make a prediction.
2. Calculate the error.
3. Update the weights.
4. Update the bias.
5. Print the old weights and new weights.
6. Print the old bias and new bias.

<details>
<summary>Starter code</summary>

```python
customer = np.array([9, 80, 10])
actual = 1

weights = np.array([1.0, -0.05, -0.05])
bias = -5.0

learning_rate = 0.01

# Make prediction
z = np.dot(customer, weights) + bias
prediction = step(z)

# Calculate error
error = actual - prediction

print("Before update")
print("Weighted sum:", z)
print("Prediction:", prediction)
print("Actual:", actual)
print("Error:", error)
print("Weights:", weights)
print("Bias:", bias)

# Save old values for comparison
old_weights = weights.copy()
old_bias = bias

# Update weights and bias
weights = weights + learning_rate * error * customer
bias = bias + learning_rate * error

print("\nAfter update")
print("Old weights:", old_weights)
print("New weights:", weights)
print("Old bias:", old_bias)
print("New bias:", bias)
```

</details>

### Questions

1. Was the model correct before the update?
2. What was the error?
3. Which weight changed the most?
4. Why did that weight change the most?
5. What happened to the bias?
6. Why does nothing change when the error is `0`?

---

## Part 5: Train the Perceptron

Now you will train the perceptron on several examples.

Use this training data:

```python
X = np.array([
    [10, 40, 5],
    [9, 60, 10],
    [8, 100, 10],
    [7, 70, 15],
    [6, 50, 60],
    [5, 120, 20],
    [3, 30, 5],
    [2, 80, 40],
])
```

The labels are:

```python
y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
```

Each row in `X` is:

```text
[interest_level, ticket_price, distance_miles]
```

Each value in `y` is:

```text
1 = buys ticket
0 = does not buy ticket
```

### Tasks

1. Start with weights equal to zero.
2. Start with bias equal to zero.
3. Loop through the data.
4. Make a prediction for each customer.
5. Calculate the error.
6. Update the weights and bias.
7. Repeat for multiple epochs.
8. Print the number of errors each epoch.

<details>
<summary>Starter code</summary>

```python
import numpy as np

X = np.array([
    [10, 40, 5],
    [9, 60, 10],
    [8, 100, 10],
    [7, 70, 15],
    [6, 50, 60],
    [5, 120, 20],
    [3, 30, 5],
    [2, 80, 40],
])

y = np.array([1, 1, 1, 1, 0, 0, 0, 0])

weights = np.zeros(3)
bias = 0.0

learning_rate = 0.01
epochs = 20

def step(z):
    if z >= 0:
        return 1
    else:
        return 0

for epoch in range(epochs):
    total_errors = 0

    for xi, actual in zip(X, y):
        z = np.dot(xi, weights) + bias
        prediction = step(z)

        error = actual - prediction

        weights = weights + learning_rate * error * xi
        bias = bias + learning_rate * error

        total_errors += abs(error)

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Errors: {total_errors} | "
        f"Weights: {weights} | "
        f"Bias: {bias:.2f}"
    )

print("\nFinal weights:", weights)
print("Final bias:", bias)
```

</details>

### Questions

1. Did the number of errors decrease over time?
2. What learning rate did you use?
3. What happened when you increased the learning rate?
4. What happened when you decreased the learning rate?
5. What do the final weights suggest about interest, price, and distance?
6. Did the final weights make sense? Why or why not?

---

## Part 6: Test the Trained Model

After training, test your model on new customers.

Use these examples:

```python
new_customers = np.array([
    [10, 100, 5],
    [4, 40, 5],
    [8, 160, 15],
    [7, 50, 90],
    [9, 75, 20],
])
```

### Tasks

1. Use the final trained weights and bias.
2. Predict whether each customer buys a ticket.
3. Print the weighted sum and prediction.
4. Decide whether each prediction seems reasonable.

<details>
<summary>Starter code</summary>

```python
new_customers = np.array([
    [10, 100, 5],
    [4, 40, 5],
    [8, 160, 15],
    [7, 50, 90],
    [9, 75, 20],
])

for customer in new_customers:
    z = np.dot(customer, weights) + bias
    prediction = step(z)

    if prediction == 1:
        label = "Buys ticket"
    else:
        label = "Does not buy ticket"

    print(
        f"Customer: {customer}, "
        f"z={z:.2f}, "
        f"Prediction: {label}"
    )
```

</details>

### Questions

Answer in a few sentences:

1. Which new customer was most likely to buy?
2. Which new customer was least likely to buy?
3. Did the model behave how you expected?
4. Which predictions felt realistic?
5. Which predictions felt questionable?

---

## Part 7: Create Your Own Examples

Create at least three new customers of your own.

Each customer should have:

```text
interest_level, ticket_price, distance_miles
```

Example:

```python
my_customers = np.array([
    [9, 45, 8],
    [4, 120, 10],
    [7, 50, 80],
])
```

### Tasks

1. Create three new customers.
2. Run them through the trained model.
3. Explain whether the predictions make sense.

### Questions

1. Why did you choose those customers?
2. Did the predictions make sense?
3. Which feature seemed to matter most?
4. How could you change the training data to make the model behave differently?

---

## Part 8: Reflection

You should be able to answer all of the following questions

1. What does a perceptron do?
2. What are weights?
3. What is bias?
4. What does the activation function do?
5. What is the purpose of the learning rate?
6. Why do weights update when the model is wrong?
7. Why does the model not update when the prediction is correct?
8. Why might this model be too simple for real ticket-buying behavior?
9. What kinds of hidden patterns might a larger neural network learn?
