# Autoencoders and Unsupervised Neural Networks

# Big Picture: What Is an Autoencoder?

An **autoencoder** is a neural network trained to recreate its input.

At first, that sounds strange. If the input and output are the same, what is the model actually learning?

The useful part is that the network usually has to pass the input through a smaller compressed layer in the middle.

```text
input -> encoder -> bottleneck -> decoder -> reconstructed input
```

The model learns to compress the important parts of the data and then rebuild the original input from that compressed version.

## Bite-sized version

> An autoencoder learns to compress data and then reconstruct it.

## Slightly deeper version

An autoencoder has two main parts:

```text
Encoder: compresses the input
Decoder: reconstructs the input
```

The middle compressed representation is called the **bottleneck** or **latent representation**.

## Interview version

> An autoencoder is a neural network trained to reconstruct its input. It learns a compressed latent representation through an encoder and uses a decoder to reconstruct the original data. Autoencoders are commonly used for dimensionality reduction, denoising, anomaly detection, and representation learning.

---

# Why Is This Unsupervised Learning?

In supervised learning, we train a model with labels.

Example:

```text
image -> cat/dog label
review -> positive/negative label
network traffic -> normal/attack label
```

With an autoencoder, we usually do not need a separate label.

Instead, the model learns from the input itself.

```python
model.fit(X_train, X_train)
```

The input is `X_train`.

The target is also `X_train`.

The model is trying to reconstruct the same data it received.

## Bite-sized version

> The model trains on the data itself, not external labels.

## Important wording

Autoencoders are often called **unsupervised**, but you may also hear the term **self-supervised**.

Why?

Because the model creates its own training target from the input.

```text
Input:  original data
Target: original data
```

No human-provided label is required.

## Interview version

> Autoencoders are usually considered unsupervised or self-supervised because they do not require labeled outputs. The model uses the input as the target and learns by minimizing reconstruction error.

---

# Autoencoder Architecture

A basic dense autoencoder for tabular data might look like this:

```text
Input Features
      ↓
Dense(64)
      ↓
Dense(32)
      ↓
Dense(8)       <- bottleneck / latent vector
      ↓
Dense(32)
      ↓
Dense(64)
      ↓
Reconstructed Features
```

The first half is the **encoder**.

The middle is the **bottleneck**.

The second half is the **decoder**.

---

# Encoder

The **encoder** compresses the original input into a smaller representation.

Example:

```text
Original input: 100 features
Compressed representation: 8 features
```

The encoder is learning:

> What information is important enough to keep?

## Bite-sized version

> The encoder turns large input data into a smaller learned vector.

## Interview version

> The encoder maps the original input into a lower-dimensional latent representation. This representation should preserve the most important information needed to reconstruct the input.

---

# Bottleneck / Latent Representation

The **bottleneck** is the compressed middle layer.

Example:

```text
Input dimension:      100
Bottleneck dimension: 8
```

The model is forced to summarize the data using fewer values.

This creates pressure on the model to learn useful patterns instead of simply memorizing every input feature.

## Bite-sized version

> The bottleneck is the compressed version of the input.

## Why bottleneck size matters

If the bottleneck is too large:

```text
The model may simply copy the input without learning useful structure.
```

If the bottleneck is too small:

```text
The model may lose too much information and reconstruct poorly.
```

A good bottleneck is small enough to force compression but large enough to preserve meaningful patterns.

## Interview version

> The bottleneck is the compressed latent space of the autoencoder. It forces the model to represent the input using fewer dimensions, which can help with feature learning, dimensionality reduction, and clustering.

---

# Decoder

The **decoder** takes the compressed representation and tries to rebuild the original input.

```text
bottleneck vector -> decoder -> reconstructed input
```

For tabular data, the decoder often uses `Dense` layers.

For image data, the decoder may use layers such as:

- `UpSampling2D`
- `Conv2DTranspose`
- `Conv2D`

## Bite-sized version

> The decoder tries to rebuild the original data from the compressed vector.

## Interview version

> The decoder maps the latent representation back into the original input space. Its goal is to reconstruct the input as accurately as possible.

---

# Reconstruction Loss

Autoencoders usually train by minimizing **reconstruction error**.

The model compares:

```text
original input vs reconstructed output
```

If the reconstruction is close to the original, the loss is low.

If the reconstruction is poor, the loss is high.

Common losses:

```python
loss="mse"
```

```python
loss="binary_crossentropy"
```

## When to use MSE

Mean squared error is common for continuous numeric data.

Examples:

- normalized tabular data
- sensor readings
- network traffic features
- scaled transaction data

## When to use binary crossentropy

Binary crossentropy can be useful when inputs are normalized to values between 0 and 1, especially for simple image examples.

Example:

- MNIST-style images with pixel values from 0 to 1

## Bite-sized version

> Reconstruction loss measures how different the output is from the input.

## Interview version

> Autoencoders optimize reconstruction loss, which measures the difference between the original input and the reconstructed output. A lower reconstruction loss means the model is better at recreating the input.

---

# Dense Autoencoder Example

This is a simple autoencoder for tabular data.

```python
import tensorflow as tf

def build_autoencoder(input_dim, encoding_dim=8):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),

        # Encoder
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),

        # Bottleneck
        tf.keras.layers.Dense(encoding_dim, activation="relu", name="bottleneck"),

        # Decoder
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),

        # Reconstruction
        tf.keras.layers.Dense(input_dim, activation="linear")
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model
```

Training:

```python
autoencoder = build_autoencoder(input_dim=X_train.shape[1])

history = autoencoder.fit(
    X_train,
    X_train,
    validation_data=(X_val, X_val),
    epochs=30,
    batch_size=128
)
```

The important part:

```python
model.fit(X_train, X_train)
```

The model receives the input and tries to recreate that same input.

---

# Denoising Autoencoders

A **denoising autoencoder** learns to remove noise.

Instead of training with:

```text
clean input -> clean target
```

we train with:

```text
noisy input -> clean target
```

Example:

```text
Noisy image  -> autoencoder -> clean image
```

This teaches the model to preserve important structure while ignoring random noise.

## Bite-sized version

> A denoising autoencoder learns to clean corrupted data.

## Example use cases

- removing noise from images
- cleaning sensor signals
- smoothing corrupted measurements
- restoring partially damaged data

## Interview version

> A denoising autoencoder is trained with noisy inputs and clean targets. It learns to reconstruct the original clean data, which makes it useful for noise reduction and robust representation learning.

---

# Autoencoders for Anomaly Detection

Autoencoders can be used for **anomaly detection**.

The typical idea:

```text
Train on normal data.
Learn to reconstruct normal patterns.
Unusual examples reconstruct poorly.
High reconstruction error may indicate an anomaly.
```

Example domains:

- fraud detection
- manufacturing defects
- network intrusion detection
- unusual medical scans
- sensor failure detection
- abnormal system behavior

## Bite-sized version

> If the autoencoder cannot reconstruct something well, that example may be unusual.

## Workflow

1. Train the autoencoder mostly on normal data.
2. Use the model to reconstruct new examples.
3. Calculate reconstruction error for each example.
4. Choose a threshold.
5. Flag examples above the threshold as possible anomalies.

Example:

```python
import numpy as np

reconstructions = autoencoder.predict(X_test)

reconstruction_errors = np.mean(
    np.square(X_test - reconstructions),
    axis=1
)

threshold = np.percentile(reconstruction_errors, 95)

anomaly_flags = reconstruction_errors > threshold
```

## Important caution

A high reconstruction error does not automatically prove something is malicious or defective.

It means:

```text
This example does not look like the patterns the autoencoder learned well.
```

Human review, domain knowledge, or additional models may still be needed.

## Interview version

> Autoencoders can detect anomalies by learning to reconstruct normal data. When a new example has high reconstruction error, it may be considered unusual because the model cannot recreate it well based on the normal patterns it learned.

---

# Autoencoders for Clustering

Autoencoders can also help with **clustering**.

The autoencoder itself is not usually the clustering algorithm.

Instead, it learns a better representation of the data first.

Then we cluster the compressed vectors.

```text
raw data -> encoder -> bottleneck vectors -> clustering algorithm
```

Example:

```text
network traffic records -> autoencoder encoder -> 8-dimensional vectors -> K-Means
```

## Bite-sized version

> Use the autoencoder to create better features, then cluster those features.

## Why this can help

Clustering directly on raw data can be difficult when the data is:

- high-dimensional
- noisy
- sparse
- correlated
- hard to compare with simple distance metrics

The autoencoder can learn a compressed representation that captures important patterns.

Then a clustering algorithm such as K-Means may find cleaner groups.

## Example workflow

1. Scale the original data.
2. Train an autoencoder to reconstruct the data.
3. Create a separate encoder model.
4. Use the encoder to generate bottleneck vectors.
5. Run K-Means, DBSCAN, or another clustering algorithm on those vectors.
6. Inspect the clusters and assign meaning.

```python
from sklearn.cluster import KMeans
import tensorflow as tf

input_dim = X_train.shape[1]
encoding_dim = 8

inputs = tf.keras.Input(shape=(input_dim,))

# Encoder
x = tf.keras.layers.Dense(64, activation="relu")(inputs)
x = tf.keras.layers.Dense(32, activation="relu")(x)
encoded = tf.keras.layers.Dense(encoding_dim, activation="relu", name="bottleneck")(x)

# Decoder
x = tf.keras.layers.Dense(32, activation="relu")(encoded)
x = tf.keras.layers.Dense(64, activation="relu")(x)
outputs = tf.keras.layers.Dense(input_dim, activation="linear")(x)

autoencoder = tf.keras.Model(inputs, outputs, name="autoencoder")
encoder = tf.keras.Model(inputs, encoded, name="encoder")

autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.fit(
    X_train,
    X_train,
    validation_data=(X_val, X_val),
    epochs=30,
    batch_size=128
)

X_encoded = encoder.predict(X_train)

kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
cluster_labels = kmeans.fit_predict(X_encoded)
```

## What the clusters might mean

For network traffic, clusters might represent:

```text
Cluster 0: normal browsing traffic
Cluster 1: file transfer traffic
Cluster 2: repeated authentication attempts
Cluster 3: high-volume unusual traffic
```

The model does not automatically name the clusters.

You still need to inspect examples from each cluster and interpret what the clusters represent.

## Interview version

> Autoencoders can support clustering by learning compressed latent representations of data. After training the autoencoder, we can extract the bottleneck vectors and run clustering algorithms like K-Means or DBSCAN on those learned features. This can improve clustering when the original data is high-dimensional or noisy.

---

# Autoencoders and Embeddings

The bottleneck vector of an autoencoder can be thought of as an **embedding** of the input.

Example:

```text
network traffic record -> encoder -> 8-number vector
```

That 8-number vector is a learned representation of the original traffic record.

This connects directly to vector databases.

In Week 3, you may see a workflow like:

```text
text chunk -> embedding model -> vector -> vector database
```

An autoencoder gives us a similar idea for non-text data:

```text
tabular record -> encoder -> vector -> similarity search or clustering
```

## Bite-sized version

> The encoder turns each input into a learned vector. That vector can be used for clustering, similarity, or downstream models.

## Interview version

> The latent representation from an autoencoder can be treated as an embedding of the input. These embeddings can be used for clustering, visualization, similarity search, or as features for another model.

---

# RBMs vs Autoencoders

An **RBM**, or Restricted Boltzmann Machine, is an older unsupervised neural network that learns hidden patterns in data.

RBMs were historically used for:

- feature learning
- dimensionality reduction
- recommender systems
- collaborative filtering
- pretraining deeper networks

Autoencoders are usually easier to teach and implement with modern Keras.

## Bite-sized version

> RBMs are historically important, but autoencoders are usually more practical for a modern TensorFlow lesson.

## Interview version

> RBMs are older energy-based unsupervised models that learn relationships between visible and hidden units. They were historically used in recommender systems and feature learning. Autoencoders are more common in modern neural network instruction because they are easier to implement and connect well to reconstruction, anomaly detection, and representation learning.


## Additional Resource
[Geeks For Geeks article on creating an RBM](https://www.geeksforgeeks.org/machine-learning/restricted-boltzmann-machine/)
---

# Common Autoencoder Types

## Dense Autoencoder

Uses `Dense` layers.

Best for:

- tabular data
- numeric features
- simple demos

## Convolutional Autoencoder

Uses convolutional layers.

Best for:

- images
- spatial patterns
- denoising images

## Denoising Autoencoder

Learns to reconstruct clean data from noisy data.

Best for:

- noise reduction
- robust features

## Sparse Autoencoder

Encourages only some neurons to activate.

Best for:

- learning compact features
- representation learning

## Variational Autoencoder

Learns a structured latent space and can generate new examples.

Best for:

- generative modeling
- advanced unsupervised learning

For this course, focus mostly on:

```text
basic dense autoencoder
denoising autoencoder
autoencoder for anomaly detection
autoencoder for clustering support
```

---

# Choosing a Bottleneck Size

There is no perfect rule, but here are practical starting points.

## For tabular data

If the input has 50-100 features, try:

```text
bottleneck size: 4, 8, 16, or 32
```

## For images

The bottleneck may be a flattened vector or a smaller feature map.

Example:

```text
28 x 28 x 1 image
-> smaller convolutional feature maps
-> reconstructed image
```

## General guidance

A smaller bottleneck forces more compression.

A larger bottleneck preserves more information.

Try a few sizes and compare validation reconstruction loss.

## Bite-sized version

> The bottleneck should be small enough to force learning, but not so small that reconstruction becomes impossible.

---

# Preprocessing Matters

Autoencoders are sensitive to feature scale.

For tabular data, scale the features before training.

Common choices:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Why?

If one feature is much larger than the others, it can dominate the reconstruction loss.

Example:

```text
bytes_sent: 900000
failed_logins: 3
```

Without scaling, the model may focus heavily on large numeric features and ignore smaller but important ones.

## Bite-sized version

> Scale your data or large features can dominate reconstruction error.

---

# How to Evaluate an Autoencoder

Evaluation depends on the use case.

## For reconstruction

Look at:

- training reconstruction loss
- validation reconstruction loss
- example reconstructions

## For denoising

Compare:

```text
noisy input
clean target
model output
```

## For anomaly detection

Look at:

- reconstruction error distribution
- chosen threshold
- false positives
- false negatives, if labels are available
- examples with highest reconstruction error

## For clustering

Look at:

- cluster sizes
- sample records from each cluster
- distance between clusters
- silhouette score
- whether clusters make domain sense

## Bite-sized version

> Do not only look at loss. Inspect what the model is reconstructing or what the clusters actually contain.

# Mini Glossary

## Autoencoder

A neural network trained to reconstruct its input.

## Encoder

The part of the model that compresses the input.

## Decoder

The part of the model that reconstructs the input.

## Bottleneck

The compressed middle representation.

## Latent representation

Another name for the learned compressed representation.

## Reconstruction

The model's attempt to recreate the original input.

## Reconstruction error

The difference between the original input and the reconstructed output.

## Denoising

Removing noise or corruption from data.

## Anomaly detection

Finding examples that do not match normal patterns.

## Clustering

Grouping similar data points without labels.

## Embedding

A learned vector representation of data.
