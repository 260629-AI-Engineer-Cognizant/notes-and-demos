# Project 1 - Neural Network Families

## Project overview

You will build a TensorFlow/Keras project using a real-world dataset.

Your project should demonstrate the full machine learning workflow:

```text
problem -> dataset -> preprocessing -> model -> training -> evaluation -> improvement -> presentation
```

This project should connect back to Week 1 ML fundamentals and Week 2 TensorFlow neural network architectures.

---

## Acceptable project types

Choose one:

### Option 1: Image classification with CNNs

Examples:
- classify handwritten digits
- classify clothing images
- classify animals/plants
- classify product defects
- classify food images

Expected model family:
- CNN
- optional transfer learning

Required evaluation:
- accuracy
- confusion matrix
- misclassified examples

---

### Option 2: Text classification with NLP models

Examples:
- review sentiment
- spam detection
- news topic classification
- support ticket classification

Expected model family:
- Embedding + Dense
- Embedding + GRU/LSTM
- optional pretrained model/transfer learning

Required evaluation:
- accuracy
- precision/recall/F1 for binary or multiclass classification
- confusion matrix
- false positive/false negative examples

---

### Option 3: Tabular classification or regression with a Dense network

Examples:
- customer churn
- loan default
- house price prediction
- medical risk prediction
- student performance prediction

Expected model family:
- Dense / MLP

Required evaluation:
- classification: accuracy, precision, recall, F1, confusion matrix
- regression: MAE, RMSE or MSE, prediction-vs-actual plot

---

### Option 4: Autoencoder / anomaly detection

Examples:
- unusual transactions
- network anomaly data
- manufacturing defect data
- image denoising

Expected model family:
- autoencoder

Required evaluation:
- reconstruction examples
- reconstruction error
- explanation of anomaly threshold if applicable

---

## Dataset requirements

Your dataset must be real-world or realistic.

Good sources:
- Kaggle
- TensorFlow Datasets
- Keras built-in datasets
- UCI Machine Learning Repository
- government/open-data portals
- public CSV datasets

The dataset must have:
- a clear problem
- enough rows/images/text samples to train a model
- a target label, unless using an autoencoder/unsupervised setup
- understandable features or examples

---

## Required project components

### 1. Problem statement

Explain:
- What are you trying to predict or classify?
- Why does this problem matter?
- Who would use this model?

### 2. Dataset explanation

Include:
- source/link
- number of examples
- feature/column explanation, if tabular
- class labels, if classification
- example records/images/text samples

### 3. Preprocessing

Depending on project type, include:

For images:
- resizing
- normalization
- channel shape
- augmentation, if used

For text:
- tokenization/vectorization
- padding
- vocabulary size
- sequence length

For tabular:
- missing values
- categorical encoding
- scaling
- train/test split

### 4. Model architecture

Explain:
- what neural network type you used
- why that architecture fits the data
- number/type of layers
- activation functions
- output layer
- loss function
- optimizer

### 5. Training

Include:
- epochs
- batch size
- validation split or validation set
- callbacks, if used
- training curves

### 6. Evaluation

Classification:
- accuracy
- precision/recall/F1
- confusion matrix
- misclassified examples

Regression:
- MAE
- MSE/RMSE
- prediction-vs-actual plot
- examples of large errors

Autoencoder:
- reconstruction visuals
- reconstruction error
- anomaly threshold or denoising comparison

### 7. Improvement attempt

Every group must try at least one improvement:
- add/remove layers
- change learning rate
- add dropout
- add regularization
- use data augmentation
- change preprocessing
- change threshold
- compare architectures

### 8. Limitations and next steps

Discuss:
- what the model does well
- what it struggles with
- whether the dataset is biased/limited
- what you would do with more time/data

---

## Presentation requirements

Presentation date: **Friday, July 17, 2026**

Length: **5-7 minutes**

Every team member must speak.

Recommended slide flow:

1. Project title and team
2. Problem statement
3. Dataset overview
4. Preprocessing
5. Model architecture
6. Results and metrics
7. Error analysis
8. Improvement attempt
9. Limitations and next steps
10. Questions
