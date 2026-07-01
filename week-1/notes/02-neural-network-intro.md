# Neural Networks & Deep Learning Foundations

## 1. Why Are We Moving Beyond Traditional Machine Learning?

So far, many machine learning models we have discussed follow this pattern:

1. We provide input features.
2. The model learns a relationship between the features and the target.
3. The model makes predictions on new data.

Examples:

- Linear regression learns a line or surface.
- Logistic regression learns a decision boundary for classification.
- Decision trees split data into regions.
- K-Means groups similar points together.

These are powerful, but they often depend heavily on **feature engineering**.

Feature engineering means humans manually decide which patterns matter.

Example:

If we want to classify images of cats and dogs, traditional ML might require us to manually create features such as:

- ear shape
- eye spacing
- fur texture
- snout length
- color patterns

That is difficult because images are huge grids of numbers, and the patterns we care about are not always obvious.

Neural networks are useful because they can learn increasingly complex patterns from raw or semi-raw data.

---

## 2. What Is a Neural Network?

A neural network is a machine learning model built from connected layers of small computational units called **neurons**.

A neuron usually does three things:

1. Takes input values.
2. Multiplies those inputs by weights and adds a bias.
3. Passes the result through an activation function.

The core idea looks like this:

```text
inputs -> weighted sum -> activation function -> output
```

A very simple neuron might compute:

```text
z = (x1 * w1) + (x2 * w2) + b
output = activation(z)
```

Where:

- `x1`, `x2` are input features
- `w1`, `w2` are weights
- `b` is the bias
- `z` is the weighted sum before activation
- `activation(z)` is the final neuron output

---

## 3. Why Weights and Biases Matter

### Weights

Weights control how much each input matters.

Example:

Suppose we predict whether a student passes based on:

- hours studied
- hours slept

A model might learn that study hours matter more than sleep hours for the specific dataset.

```text
pass_score = (hours_studied * 2.5) + (hours_slept * 0.7) + bias
```

A larger weight means that feature has a stronger influence on the output.

### Bias

The bias shifts the output up or down.

In linear regression, we often wrote:

```text
y = mx + b
```

The `b` is the bias/intercept.

In neural networks, biases allow neurons to activate even when inputs are small or zero.

---

## 4. From Linear Models to Neurons

A single neuron is very similar to a linear model plus an activation function.

Linear regression:

```text
y = (x * weight) + bias
```

Neuron:

```text
z = (x * weight) + bias
output = activation(z)
```

The activation function is what allows neural networks to model nonlinear patterns.

Without activation functions, stacking layers would still behave like one large linear model.

---

## 5. What Is an Activation Function?

An activation function decides how much signal should pass forward.

It introduces nonlinearity into the network.

Common activation functions:

| Activation | Basic Idea | Common Use |
|---|---|---|
| Step | Outputs 0 or 1 | Early perceptrons, teaching demos |
| Sigmoid | Squashes values between 0 and 1 | Binary probabilities, older networks |
| Tanh | Squashes values between -1 and 1 | Older hidden layers |
| ReLU | Turns negative values into 0 | Common hidden-layer activation |
| Softmax | Converts scores into class probabilities | Multi-class output layers |

### Step Function

```text
if z >= 0: output = 1
else: output = 0
```

Good for simple yes/no decisions, but not smooth.

### Sigmoid

```text
sigmoid(z) = 1 / (1 + e^-z)
```

Outputs values between 0 and 1.

Useful when interpreting output as a probability.

### ReLU

```text
ReLU(z) = max(0, z)
```

Negative values become 0. Positive values pass through.

ReLU is popular because it is simple and works well in many deep networks.

### Softmax

Softmax is used when we have multiple possible classes.

Example:

```text
cat: 0.10
dog: 0.80
rabbit: 0.10
```

The values sum to 1, so they can be interpreted as probabilities.

---

## 6. What Is a Perceptron?

A perceptron is one of the simplest neural network models.

It is usually a single neuron used for binary classification.

It computes a weighted sum and then applies a step function.

```text
z = (x1 * w1) + (x2 * w2) + b
prediction = 1 if z >= 0 else 0
```

### Example: Study Decision

Inputs:

- `x1`: hours studied
- `x2`: hours slept

Output:

- `1`: likely to pass
- `0`: likely to fail

A perceptron can learn a straight-line decision boundary.

That means it can solve simple linearly separable problems.

---

## 7. Limitation of a Single Perceptron

A single perceptron can only draw a straight boundary.

This works for problems like AND or OR logic.

### AND

| x1 | x2 | output |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

A line can separate the `1` point from the `0` points.

### OR

| x1 | x2 | output |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

A line can separate the `0` point from the `1` points.

### XOR

| x1 | x2 | output |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

A single straight line cannot separate the classes.

This is one reason we need hidden layers.

---

## 8. What Is a Layer?

A layer is a group of neurons that process information together.

Common layer types:

- Input layer
- Hidden layer
- Output layer

### Input Layer

The input layer receives the original data.

Example:

For a student dataset:

```text
[hours_studied, hours_slept]
```

For an image:

```text
[pixel_1, pixel_2, pixel_3, ...]
```

### Hidden Layer

A hidden layer learns intermediate patterns.

Example:

In an image network:

- early layers may detect edges
- middle layers may detect shapes
- later layers may detect object parts

### Output Layer

The output layer produces the final prediction.

Examples:

- one number for regression
- one probability for binary classification
- multiple probabilities for multi-class classification

---

## 9. Mini-Preview of a Hidden Layer

Suppose we have two inputs:

```text
x1 = hours studied
x2 = hours slept
```

Instead of sending those inputs directly to one output neuron, we send them to several hidden neurons.

```text
Input Layer          Hidden Layer             Output

x1 --------------->  neuron A  ----
  \                 neuron B  ----> final prediction
x2 --------------->  neuron C  ----
```

Each hidden neuron can learn a different pattern.

Example:

```text
neuron A: detects “studied a lot”
neuron B: detects “slept enough”
neuron C: detects “bad combination”
```

The output neuron then combines those learned patterns.

This is where neural networks become more expressive than a single linear model.

---

## 10. What Is Deep Learning?

Deep learning is a type of machine learning that uses neural networks with multiple layers.

A “deep” network usually means it has more than one hidden layer.

```text
Input -> Hidden Layer 1 -> Hidden Layer 2 -> Hidden Layer 3 -> Output
```

The deeper structure allows the model to learn hierarchical patterns.

### Example: Image Classification

For image classification, layers might learn:

1. edges
2. corners
3. textures
4. shapes
5. object parts
6. full objects

### Example: Natural Language Processing

For text, layers might learn:

1. characters or tokens
2. word patterns
3. phrase meaning
4. sentence meaning
5. document-level meaning

Deep learning is especially useful for complex data such as:

- images
- audio
- text
- video
- sensor data
- large-scale tabular data with complex interactions

---

## 11. Neural Network Types at a High Level

### 1. Feedforward Neural Networks / Multi-Layer Perceptrons

A feedforward neural network sends information in one direction:

```text
input -> hidden layers -> output
```

There are no loops.

Commonly used for:

- tabular classification
- tabular regression
- simple pattern recognition

A multi-layer perceptron, or MLP, is a common feedforward network with one or more hidden layers.

---

### 2. Convolutional Neural Networks / CNNs

CNNs are commonly used for images.

They are designed to detect spatial patterns.

Instead of treating every pixel independently, CNNs look for local patterns like:

- edges
- corners
- curves
- textures
- object parts

Commonly used for:

- image classification
- object detection
- medical imaging
- facial recognition
- quality control in manufacturing

Teaching analogy:

A CNN is like sliding a small pattern detector over an image and asking, “Do I see this feature here?”

---

### 3. Recurrent Neural Networks / RNNs

RNNs are designed for sequences.

They process one step at a time and keep some memory of previous steps.

Commonly used for older approaches to:

- text sequences
- time series
- speech
- sensor streams

Teaching analogy:

An RNN reads information in order, like reading a sentence word by word.

---

### 4. LSTM / GRU Networks

LSTMs and GRUs are improved versions of RNNs.

They are better at remembering useful information over longer sequences.

Commonly used for:

- text classification
- time-series forecasting
- speech processing
- sequence prediction

---

### 5. Transformer Networks

Transformers are modern neural networks heavily used in NLP and generative AI.

They use a mechanism called attention to decide which parts of the input are most relevant.

Commonly used for:

- translation
- summarization
- question answering
- chatbots
- code generation
- large language models

Teaching analogy:

A transformer does not just read left to right. It can compare different parts of the input and decide what matters most.

---

### 6. Autoencoders

Autoencoders learn to compress data and then reconstruct it.

They have two main parts:

```text
encoder -> compressed representation -> decoder
```

Commonly used for:

- dimensionality reduction
- anomaly detection
- denoising
- representation learning

---

### 7. Generative Models

Generative models learn patterns in data so they can create new examples.

Examples include:

- GANs
- diffusion models
- variational autoencoders

Commonly used for:

- image generation
- audio generation
- synthetic data
- creative AI tools

---

## 12. Training a Neural Network

Training means adjusting weights and biases so the model makes better predictions.

The basic training loop is:

```text
1. Make predictions.
2. Measure error with a loss function.
3. Calculate how weights contributed to the error.
4. Update weights to reduce the error.
5. Repeat many times.
```

Key terms:

| Term | Meaning |
|---|---|
| Forward pass | Data moves through the network to make predictions |
| Loss function | Measures how wrong the prediction is |
| Backpropagation | Calculates how each weight affected the error |
| Gradient descent | Updates weights to reduce loss |
| Epoch | One full pass through the training data |
| Learning rate | Controls how big each weight update is |
| Parameter | A learned value, usually a weight or bias |
| Hyperparameter | A setting chosen by the developer, like learning rate or number of layers |

---

## 13. Forward Pass Example

Imagine this neuron:

```text
x1 = 2
x2 = 3
w1 = 0.5
w2 = -1.0
b = 1.0
```

Compute weighted sum:

```text
z = (2 * 0.5) + (3 * -1.0) + 1.0
z = 1.0 - 3.0 + 1.0
z = -1.0
```

Apply ReLU:

```text
ReLU(-1.0) = 0
```

The neuron output is `0`.

---

## 14. Parameters vs Hyperparameters

### Parameters

Parameters are learned by the model.

Examples:

- weights
- biases

### Hyperparameters

Hyperparameters are chosen before or during training by the developer.

Examples:

- number of hidden layers
- number of neurons per layer
- learning rate
- number of epochs
- activation function
- batch size

A major part of deep learning is choosing reasonable hyperparameters and tuning them over time.

---

## 15. Common Neural Network Use Cases

### Classification

Predict a category.

Examples:

- spam or not spam
- cat, dog, or rabbit
- disease present or not present
- customer will churn or stay

### Regression

Predict a number.

Examples:

- house price
- temperature
- demand forecast
- delivery time

### Image Classification

Predict what is in an image.

Examples:

- handwritten digit recognition
- product defect detection
- medical scan classification

### NLP / Text Processing

Work with language.

Examples:

- sentiment analysis
- summarization
- translation
- chatbot responses
- document classification

### Anomaly Detection

Find unusual patterns.

Examples:

- fraud detection
- network intrusion
- machine failure detection

---

## 16. Why Neural Networks Need Data and Compute

Neural networks often have many parameters.

A small model may have hundreds or thousands.

Large deep learning models may have millions, billions, or more.

More parameters can make a model more powerful, but also increase the risk of:

- overfitting
- slow training
- high memory usage
- needing more data
- being harder to interpret

This is why deep learning is not always the best choice.

For small tabular datasets, models like logistic regression, random forests, or gradient boosting may be easier and stronger.

---

## 17. Overfitting in Neural Networks

Overfitting happens when the model memorizes training data instead of learning general patterns.

Signs of overfitting:

- training loss keeps improving
- validation loss gets worse
- training accuracy is high
- test accuracy is much lower

Ways to reduce overfitting:

- get more data
- use a smaller network
- use regularization
- use dropout
- use early stopping
- simplify the feature set

For now, the most important idea is:

> A bigger neural network is not automatically a better neural network.

---
