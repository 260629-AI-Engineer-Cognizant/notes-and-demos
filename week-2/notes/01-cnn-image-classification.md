# CNNs and Image Classification

## 1. Why Start With CNNs?

So far, most beginner neural networks use `Dense` layers.

Dense layers are great for simple tabular data or already-flattened inputs:

```text
input features -> Dense -> Dense -> output
```

But images are different.

An image has structure:

```text
height x width x channels
```

For a grayscale image:

```text
28 x 28 x 1
```

For a color image:

```text
32 x 32 x 3
```

A normal Dense layer does not understand that nearby pixels are related. It just sees a long list of numbers.

A CNN is designed to preserve and learn from spatial structure.

---

## 2. Image Tensors

An image dataset usually has this shape:

```text
(number_of_images, height, width, channels)
```

Examples:

```text
MNIST:      (60000, 28, 28, 1)
CIFAR-10:   (50000, 32, 32, 3)
```

Channels mean:

| Image type | Channels |
|---|---|
| Grayscale | 1 |
| RGB color | 3 |

MNIST starts as:

```text
(60000, 28, 28)
```

But Keras CNN layers expect a channel dimension, so we reshape to:

```text
(60000, 28, 28, 1)
```

---

## 3. Dense Model vs CNN Model

### Dense model

A Dense model normally flattens the image:

```text
28 x 28 image -> 784 values
```

Then the model learns from that long vector.

```python
tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
```

Problem:

> Flattening destroys the 2D layout of the image.

The model can still work on MNIST, but it is not using the image structure very intelligently.

---

## 4. What Is a Convolution?

A convolution uses a small filter/kernel that slides over the image.

Example:

```text
3 x 3 filter
```

The filter looks at small local regions and detects patterns.

Common beginner intuition:

> A convolution is a tiny pattern detector.

Early filters may detect:
- vertical edges
- horizontal edges
- corners
- curves
- blobs

Later filters combine these into more meaningful patterns.

---

## 5. Conv2D Layer

A common convolution layer looks like this:

```python
tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation="relu"
)
```

### `filters=32`

This means the layer learns 32 different pattern detectors.

Each filter produces one feature map.

### `kernel_size=(3, 3)`

This means each filter looks at a 3x3 region at a time.

### `activation="relu"`

ReLU keeps positive signal and turns negative signal into zero.

This helps the network learn nonlinear patterns.

---

## 6. Feature Maps

A feature map shows where a filter activated.

If a filter detects vertical lines, its feature map lights up where vertical lines appear.

```text
original image -> filter asks "where do I see this pattern?" -> feature map
```

CNNs do not need us to manually program the filters. The filters are learned during training.

---

## 7. Max Pooling

Max pooling shrinks the image representation while keeping the strongest signals.

```python
tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
```

A 2x2 max pool takes each 2x2 region and keeps the largest value.

Why this helps:

- reduces computation
- makes feature maps smaller
- keeps strong detected features
- gives some tolerance to small shifts

---

## 8. A Basic CNN

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
```

High-level flow:

```text
image -> detect local patterns -> shrink -> detect bigger patterns -> classify
```

---

## 9. Why Softmax for MNIST?

MNIST has 10 possible classes:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

The output layer needs 10 neurons:

```python
tf.keras.layers.Dense(10, activation="softmax")
```

Softmax converts raw model scores into probabilities that sum to 1.

Example:

```text
0: 0.01
1: 0.02
2: 0.91
3: 0.01
...
```

The model predicts the class with the highest probability.

---

## 10. Loss Function for MNIST

MNIST labels are integers:

```text
0, 1, 2, ...
```

So we use:

```python
loss="sparse_categorical_crossentropy"
```

Use this when:
- there are multiple classes
- each example belongs to one class
- labels are integers, not one-hot encoded

---

## 11. Real-World Bridge: Reading a Zip Code

A real OCR system is harder than classifying MNIST digits.

It usually needs to:
1. Find the text region.
2. Segment the digits or characters.
3. Preprocess the image.
4. Classify each character.
5. Combine predictions into a final string.

We can simulate this:

```text
Train CNN on MNIST digits.
Create a fake envelope image.
Paste five MNIST-like digits onto it.
Crop each known digit region.
Convert each crop back into the format the model expects.
Predict each digit.
Join the digits into a zip code.
```

If the model trained on white digits on a black background, then a white envelope with black ink must be inverted before prediction.

---

## 12. Important CNN Terms

| Term | Meaning |
|---|---|
| Filter/kernel | Small learned pattern detector |
| Feature map | Output showing where a filter activated |
| Stride | How far the filter moves each step |
| Padding | Whether edges are preserved |
| Pooling | Shrinking feature maps while keeping strong signals |
| Flatten | Turning feature maps into a vector before Dense layers |
| Data augmentation | Random transformations that improve generalization |

---


