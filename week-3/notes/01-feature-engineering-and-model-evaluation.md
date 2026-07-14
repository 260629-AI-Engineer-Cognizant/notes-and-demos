# Feature Engineering and Model Evaluation

## Big Idea

Feature engineering is the process of turning raw data into useful model inputs. A machine learning model does not automatically know what to do with missing values, messy categories, inconsistent scales, or irrelevant columns. Good feature engineering makes the model's job easier.

Model evaluation answers a different question: after we train a model, how do we know whether it is useful?

These two topics belong together because every preprocessing decision should eventually be tested with model performance.

---

## What Is a Feature?

A feature is an input column used by a model to make predictions.

Example customer churn features:

| Feature | Example Value | Type |
|---|---:|---|
| `age` | 34 | numeric |
| `monthly_spend` | 79.99 | numeric |
| `contract_type` | Month-to-month | categorical |
| `region` | Southeast | categorical |
| `support_tickets` | 3 | numeric |

The target/label is the thing we want to predict.

Example target:

| Target | Meaning |
|---|---|
| `churned = 1` | customer left |
| `churned = 0` | customer stayed |

---

## Common Raw Data Problems

### Missing Values

Missing values occur when a column has no value for a row.

Common strategies:

| Strategy | When It Might Make Sense |
|---|---|
| Drop rows | Only a small number of rows are missing |
| Drop column | Column is mostly missing or not useful |
| Fill with mean | Numeric data without extreme outliers |
| Fill with median | Numeric data with outliers |
| Fill with most frequent | Categorical data |
| Fill with `Unknown` | Missing category may itself be meaningful |

Important: the imputer should be fit on the training data only. If we use information from the test set during preprocessing, we leak future information into training.

---

### Outliers

An outlier is an extreme value compared to the rest of the data.

Example:

```text
Monthly spend: 55, 60, 62, 70, 75, 10000
```

The value `10000` might be a real VIP customer, a data entry error, or a system bug. Do not automatically delete outliers without thinking about the business meaning.

Common strategies:

| Strategy | Meaning |
|---|---|
| Leave alone | The outlier is real and important |
| Cap/winsorize | Replace extreme values with a high/low threshold |
| Log transform | Compress large values |
| Remove row | Value is clearly invalid |
| Add an indicator column | Tell the model a value was extreme |

---

### Categorical Data

Most ML models need numbers, not strings.

Example:

```text
contract_type = "monthly"
```

The model needs this converted into numeric form.

#### One-Hot Encoding

One-hot encoding creates a separate column for each category.

| contract_type | month_to_month | one_year | two_year |
|---|---:|---:|---:|
| month_to_month | 1 | 0 | 0 |
| one_year | 0 | 1 | 0 |
| two_year | 0 | 0 | 1 |

Use this when categories do not have a natural order.

#### Ordinal Encoding

Ordinal encoding maps categories to ordered numbers.

Example:

```text
low = 0, medium = 1, high = 2
```

Use this only when the order is meaningful.

Bad example:

```text
red = 0, green = 1, blue = 2
```

That accidentally tells the model blue is "larger" than red.

---

### Scaling

Scaling puts numeric features on similar ranges.

Example:

| Feature | Range |
|---|---|
| age | 18 to 80 |
| annual_income | 20,000 to 250,000 |
| support_tickets | 0 to 20 |

Some models are sensitive to scale, especially models based on distance or gradient optimization.

Scaling matters a lot for:

- Logistic Regression
- Linear Regression with regularization
- K-Nearest Neighbors
- SVMs
- Neural networks
- K-Means

Scaling matters less for:

- Decision Trees
- Random Forests
- Gradient-boosted trees

---

## Data Leakage

Data leakage happens when training accidentally uses information it would not have in the real world.

Common leakage mistakes:

- Scaling the full dataset before train/test split.
- Imputing missing values using the full dataset before train/test split.
- Encoding categories using the full dataset before train/test split.
- Using future information as a feature.
- Including a feature that directly reveals the target.

Correct pattern:

```text
Split data first
Fit preprocessing on training data only
Transform training and test data using the fitted preprocessing steps
Train model on training data
Evaluate on test data
```

This is why `Pipeline` and `ColumnTransformer` are so useful.

---

## Pipeline and ColumnTransformer

A `Pipeline` chains steps together.

Example:

```text
impute missing values → scale numeric features → train model
```

A `ColumnTransformer` applies different preprocessing to different columns.

Example:

```text
numeric columns → median imputation + scaling
categorical columns → most frequent imputation + one-hot encoding
```

Using these tools makes your workflow:

- safer against leakage
- easier to reuse
- easier to cross-validate
- easier to deploy later

---

## Feature Selection

Feature selection means choosing which features should be used by the model.

Reasons to select features:

- Reduce noise.
- Improve model speed.
- Improve interpretability.
- Reduce overfitting.

Common techniques:

| Technique | Idea |
|---|---|
| Domain knowledge | Use features that make sense for the problem |
| Correlation | Remove highly duplicate or irrelevant numeric features |
| Model coefficients | For linear models, larger coefficients may indicate stronger features |
| Tree feature importance | Tree models can rank feature usefulness |
| Permutation importance | Shuffle one feature and see how much performance drops |
| SelectKBest | Statistical feature scoring method |

Feature selection should also happen inside the training workflow to avoid leakage.

---

## Binary Classification Terms

For binary classification, we usually define one class as the **positive class**.

Example: in a churn model, we may define:

```text
Positive class = customer churned
Negative class = customer stayed
```

| Term | Meaning |
|---|---|
| True Positive, TP | Predicted positive, and the actual value was positive |
| True Negative, TN | Predicted negative, and the actual value was negative |
| False Positive, FP | Predicted positive, but the actual value was negative |
| False Negative, FN | Predicted negative, but the actual value was positive |

For a churn model:

| Term | Example |
|---|---|
| True Positive | Predicted churn, and the customer actually churned |
| True Negative | Predicted stay, and the customer actually stayed |
| False Positive | Predicted churn, but the customer actually stayed |
| False Negative | Predicted stay, but the customer actually churned |

---

## Confusion Matrix

A confusion matrix shows how many predictions fell into each category.

Typical binary classification layout:

|  | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | True Negative | False Positive |
| Actual 1 | False Negative | True Positive |

The confusion matrix is often more useful than accuracy because it shows the specific kinds of mistakes the model is making.

---

## Accuracy

Accuracy answers:

```text
Of all predictions, how many were correct?
```

Formula:

```text
accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Use accuracy when:

- The classes are balanced.
- False positives and false negatives have similar costs.
- You want a quick overall correctness score.

Accuracy can be misleading when classes are imbalanced.

Example:

If only 5% of customers churn, a model that always predicts "not churn" would be 95% accurate, but it would fail to identify any actual churners.

---

## Precision

Precision answers:

```text
Of everything we predicted as positive, how many were actually positive?
```

Formula:

```text
precision = TP / (TP + FP)
```

Precision focuses on **false positives**.

High precision means:

```text
When the model says "positive," it is usually right.
```

Use precision when false positives are expensive.

Examples:

- Spam detection: avoid marking important emails as spam.
- Fraud detection: avoid incorrectly blocking legitimate customers.
- Churn outreach: avoid spending retention resources on customers who were not actually at risk.

---

## Recall

Recall answers:

```text
Of all actual positives, how many did we catch?
```

Formula:

```text
recall = TP / (TP + FN)
```

Recall focuses on **false negatives**.

High recall means:

```text
The model misses fewer actual positives.
```

Use recall when false negatives are expensive.

Examples:

- Cancer screening: avoid missing patients who may have cancer.
- Security threat detection: avoid missing real threats.
- Churn prediction: avoid missing customers who are likely to leave.

---

## F1-Score

F1-score balances precision and recall.

Formula:

```text
F1 = 2 * (precision * recall) / (precision + recall)
```

Use F1-score when:

- Both false positives and false negatives matter.
- The classes are imbalanced.
- Accuracy does not tell enough of the story.

F1-score will be low if either precision or recall is low.

---

## Multiclass Classification

For multiclass classification, the same ideas apply, but they are calculated per class.

Example classes:

```text
cat
dog
bird
fish
```

To calculate precision or recall for `dog`, we temporarily treat the problem like this:

```text
Positive class = dog
Negative class = everything else
```

Then we repeat that process for each class.

Common averaging methods:

| Average Type | Meaning |
|---|---|
| Macro average | Calculates the metric for each class equally, then averages them |
| Weighted average | Averages class scores based on how many examples each class has |
| Micro average | Counts total TP, FP, and FN across all classes first |

Simple rule:

- Use **macro average** when every class matters equally.
- Use **weighted average** when class size should influence the score.
- Use **micro average** when you care about total performance across all predictions.

---

## Regression Metrics

Regression metrics are used when the model predicts a **number**.

Examples:

- House price
- Monthly sales
- Temperature
- Delivery time
- Customer lifetime value

Classification metrics use TP, FP, TN, and FN.

Regression metrics do not. Instead, they measure the size of the model's prediction errors.

For each row:

```text
error = actual value - predicted value
```

or:

```text
residual = y - ŷ
```

Where:

```text
y = actual value
ŷ = predicted value
```

---

## MAE: Mean Absolute Error

MAE stands for **Mean Absolute Error**.

It answers:

```text
On average, how far off are our predictions?
```

Formula:

```text
MAE = average of |actual - predicted|
```

More formally:

```text
MAE = (1/n) * Σ |yᵢ - ŷᵢ|
```

MAE is easy to explain because it uses the same units as the target.

Example:

If we are predicting house prices and the MAE is `$20,000`, then the model is off by about `$20,000` on average.

---

## MSE: Mean Squared Error

MSE stands for **Mean Squared Error**.

Formula:

```text
MSE = average of (actual - predicted)²
```

More formally:

```text
MSE = (1/n) * Σ (yᵢ - ŷᵢ)²
```

MSE penalizes large errors more heavily because the errors are squared.

Use MSE when large mistakes should be punished more strongly.

---

## RMSE: Root Mean Squared Error

RMSE stands for **Root Mean Squared Error**.

Formula:

```text
RMSE = sqrt(MSE)
```

RMSE is useful because it brings the error back into the same unit as the original target.

Example:

If we are predicting delivery time in minutes and RMSE is `12`, then the model's typical error is around 12 minutes, with larger mistakes penalized more heavily.

---

## R² Score

R² is called the **coefficient of determination**.

It answers:

```text
How much of the variation in the target can the model explain?
```

General interpretation:

| R² Value | Meaning |
|---:|---|
| 1.0 | Perfect predictions |
| 0.0 | About as useful as predicting the average |
| Negative | Worse than predicting the average |

Example:

```text
R² = 0.82
```

Interpretation:

```text
The model explains about 82% of the variation in the target.
```

R² is useful, but it should not be the only regression metric. It is usually best to pair it with an error metric like MAE or RMSE.

---

## Choosing Regression Metrics

| Metric | What It Tells Us | Best Use |
|---|---|---|
| MAE | Average absolute error | Easy explanation in real units |
| MSE | Average squared error | Penalizes large errors strongly |
| RMSE | Square root of MSE | Penalizes large errors but returns to original units |
| R² | Variation explained by the model | Good secondary model quality score |

A good default:

```text
Use MAE for explainability.
Use RMSE when large errors are especially bad.
Use R² as a secondary score.
```

---

# Improving Precision or Recall

Most models do not directly optimize precision or recall during training.

For example:

- Logistic Regression often optimizes log loss.
- Decision Trees optimize splits using Gini impurity or entropy.
- Neural networks usually optimize a differentiable loss like cross-entropy.

Precision and recall usually depend on the final class predictions, and those predictions often depend on a **decision threshold**.

For binary classification, a model may output a probability:

```text
Churn probability = 0.72
Cancer probability = 0.31
Spam probability = 0.88
```

Then we convert that probability into a class:

```text
If probability >= 0.50, predict positive.
If probability < 0.50, predict negative.
```

The value `0.50` is the threshold.

---

## Threshold Tuning

Changing the threshold changes the precision/recall tradeoff.

If we lower the threshold:

```text
Predict positive if probability >= 0.25
```

The model predicts positive more often.

This usually:

- Increases recall.
- Decreases precision.
- Creates more false positives.
- Creates fewer false negatives.

This can be useful when false negatives are more dangerous.

Example:

For cancer screening, we may prefer higher recall because missing a real cancer case is worse than flagging someone for follow-up testing.

---

## Threshold Tuning Code Example

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Predicted probabilities for the positive class
y_proba = model.predict_proba(X_valid)[:, 1]

thresholds = [0.50, 0.40, 0.30, 0.20]

for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)

    precision = precision_score(y_valid, y_pred)
    recall = recall_score(y_valid, y_pred)
    f1 = f1_score(y_valid, y_pred)

    print(f"Threshold: {threshold}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print()
```

Important:

```text
Tune the threshold on a validation set, not the final test set.
```

The test set should be saved for final evaluation.

---

## Overfitting

Overfitting happens when a model learns the training data too closely and fails to generalize to new data.

Signs of overfitting:

- Very high training score.
- Much lower validation/test score.
- Model performs poorly on new examples.

Ways to reduce overfitting:

| Method | Idea |
|---|---|
| Simpler model | Reduce complexity |
| Regularization | Penalize overly complex models |
| Cross-validation | Test the model across multiple splits |
| More data | Give the model more examples |
| Feature selection | Remove noisy features |
| Early stopping | Stop training before memorization, common in neural networks |

---

## Cross-Validation

Cross-validation splits the data into multiple train/test folds.

Example: 5-fold cross-validation

```text
Train on 4 folds, validate on 1 fold
Repeat 5 times
Average the scores
```

Why use it?

- More reliable than one train/test split.
- Helps catch models that perform well by luck on one split.
- Useful for comparing model choices.

---

## Bite-Sized Version

Feature engineering means preparing raw columns so a model can use them. This includes handling missing values, outliers, categorical variables, scaling, and selecting useful features. We should split the data before fitting preprocessing steps to avoid leakage. Model evaluation tells us whether the model is useful, using metrics like accuracy, precision, recall, F1-score, confusion matrix, and cross-validation.



