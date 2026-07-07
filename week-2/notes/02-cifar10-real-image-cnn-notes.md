# TensorFlow Week 2 Notes: CNNs on Real Color Images with CIFAR-10

## 1. Why CIFAR-10 After MNIST?

MNIST is useful for learning, but it is very clean:
- grayscale
- centered digits
- simple background
- low resolution
- only 10 digit classes

CIFAR-10 is harder:
- color images
- real-world objects
- noisier backgrounds
- more variation within each class

CIFAR-10 images are small, but they are closer to real image classification.

---

## 2. CIFAR-10 Classes

CIFAR-10 contains 10 classes:

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

Each image has shape:

```text
32 x 32 x 3
```

The `3` means RGB color channels.

---

## 3. Why Real Images Are Harder

A handwritten digit is usually a single clear object.

A real image can vary by:
- lighting
- angle
- background
- object size
- object position
- color
- occlusion
- similar classes

For example, cats and dogs may be confused because they share shapes, textures, and backgrounds.

---

## 4. Data Augmentation

Data augmentation creates slightly changed versions of training images.

Examples:
- flip left/right
- rotate slightly
- zoom slightly
- shift slightly

In Keras:

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])
```

This helps the model learn that small changes should not change the label.

Important:

> Data augmentation is usually applied during training, not evaluation.

---

## 5. A CNN for CIFAR-10

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),
    data_augmentation,

    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax")
])
```

---

## 6. Overfitting in Image Models

Signs of overfitting:
- training accuracy rises
- validation accuracy stalls or drops
- training loss improves
- validation loss gets worse

Ways to reduce overfitting:
- data augmentation
- dropout
- smaller model
- early stopping
- more data
- transfer learning

---

## 7. Error Analysis

A confusion matrix helps answer:

```text
Which classes does the model confuse?
```

This is more useful than only saying:

```text
accuracy = 70%
```

Example:
- cats confused with dogs
- trucks confused with automobiles
- airplanes confused with ships if background is mostly sky/water

The goal is not only to train a model. The goal is to understand model behavior.

