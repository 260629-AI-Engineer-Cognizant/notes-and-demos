# TensorFlow Final Day Cleanup Notes
# Graphs and Sessions

## Key Idea

A **computational graph** is a plan for how tensors flow through operations.

Example graph:

```text
input features -> matrix multiply -> add bias -> activation -> output
```

In TensorFlow 2, code runs in **eager execution** by default. That means operations happen immediately, which feels more like normal Python.

```python
x = tf.constant([1, 2, 3])
y = x + 10
print(y.numpy())
```

For performance and deployment, TensorFlow can convert Python functions into graphs using `@tf.function`.

```python
@tf.function
def predict_step(x):
    return model(x)
```

## Sessions

Older TensorFlow 1 code often used this pattern:

1. Build the graph.
2. Open a `Session`.
3. Run parts of the graph inside the session.
4. Feed values into placeholders.

Modern TensorFlow 2 usually avoids manual sessions. Keras and eager execution make the workflow much easier:

```text
TensorFlow 1 style: build graph first, run later in a Session
TensorFlow 2 style: write Python naturally, use Keras and @tf.function when needed
```

---

# Functional Composition

## Sequential Composition

A Sequential model is a straight line of layers.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
```

This is great when the model has one input, one output, and a simple layer-by-layer flow.

## Functional Composition

The Functional API treats layers like functions. A layer receives a tensor and returns a tensor.

```python
inputs = tf.keras.Input(shape=(8,))
x = tf.keras.layers.Dense(32, activation="relu")(inputs)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

Functional composition is better when a model needs:

- Multiple inputs
- Multiple outputs
- Branches
- Shared layers
- Skip connections
- More visible architecture graphs

## Simple Mental Model

```text
Sequential API: stack layers in a line
Functional API: connect layers like a graph
```

---

# Saving and Loading Models

Saving a model lets us reuse it without retraining.

## Save in Keras Format

```python
model.save("my_model.keras")
```

## Load the Model

```python
loaded_model = tf.keras.models.load_model("my_model.keras")
```

## Why Save Models?

- Reuse the model later
- Share the model with another app
- Compare versions
- Deploy the model
- Avoid retraining every time

## Exporting for Deployment

A `.keras` file is best for continuing Keras development. A SavedModel-style export is usually better for serving/deployment workflows.

```python
model.export("exported_model")
```

Depending on TensorFlow/Keras version, older examples may use:

```python
tf.saved_model.save(model, "exported_model")
```

---

# 5. TensorBoard

TensorBoard helps visualize training.

Common things to inspect:

- Training loss
- Validation loss
- Accuracy or other metrics
- Model graph
- Histograms of weights

Basic callback:

```python
callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
model.fit(X_train, y_train, callbacks=[callback])
```

In Colab:

```python
%load_ext tensorboard
%tensorboard --logdir logs
```

# RBMs: Brief Overview

## What Is an RBM?

A **Restricted Boltzmann Machine** is an older unsupervised neural network with two main groups of units:

```text
visible units <-> hidden units
```

The visible units represent the input data. The hidden units learn patterns in that data.

## Why They Appear in Recommender Systems

For collaborative filtering, visible units can represent user-item preferences.

Example:

```text
User likes: sci-fi, mystery, action
RBM learns: hidden taste pattern
Model suggests: another item with a similar pattern
```

## What You Should Remember

RBMs are useful historically, but they are less common in modern applied projects than:

- Matrix factorization
- Embeddings
- Autoencoders
- Deep recommender systems
- Transformer-based recommenders

## One-Sentence Version

An RBM learns hidden patterns from visible input data and can be used to reconstruct or recommend related items.

---

# TFlearn Curriculum Bridge

The curriculum mentions TFlearn, but most modern applications are easier to do with Keras.

| Curriculum Term | Keras Equivalent / Bridge |
|---|---|
| TFlearn API | Higher-level neural network API, similar goal to Keras |
| Sequential composition | `tf.keras.Sequential` |
| Functional composition | Keras Functional API |
| Predefined layers | `Dense`, `Conv2D`, `Dropout`, `BatchNormalization`, etc. |
| Saving/loading | `model.save()` and `tf.keras.models.load_model()` |
| TensorBoard with TFlearn | Keras `TensorBoard` callback |

---

# 8. Bite-Sized Review

- A tensor is a multi-dimensional array.
- A computational graph is a map of tensor operations.
- Eager mode runs TensorFlow operations immediately.
- `@tf.function` converts Python-style TensorFlow code into a graph.
- A Session is an older TensorFlow 1 concept for running graphs.
- Sequential models are layer stacks.
- Functional models are connected layer graphs.
- TensorBoard helps inspect training behavior.
- Saving a model preserves architecture, weights, and training configuration.
- RBMs learn hidden patterns from visible data.

---

# Interview-Style Review

## What is a computational graph?

A computational graph is a representation of operations and data flow. In TensorFlow, it describes how tensors move through operations like matrix multiplication, activation functions, and loss calculations.

## Why did TensorFlow 2 move away from sessions?

TensorFlow 2 emphasizes eager execution and Keras, making code easier to write, debug, and teach. Graph execution is still available through `@tf.function`, but most users no longer need to manually open sessions.

## When would you use the Functional API instead of Sequential?

Use the Functional API when the model architecture is not a simple straight line. It is useful for branching models, multiple inputs, multiple outputs, shared layers, or skip connections.

## Why save a model?

Saving a model allows developers to reuse it, deploy it, version it, and make predictions later without retraining.

## What is an RBM?

An RBM is an older unsupervised neural network with visible and hidden units. It learns hidden patterns in data and has historically been used for collaborative filtering and recommendation tasks.