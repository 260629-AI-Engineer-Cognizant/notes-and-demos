# We've covered regression (Linear Regression)
# The other basic group of ML problems is Classification
# Classification is the process of putting something in a Class based off its
# features (inputs)
# Today we can use this for supervised learning but it's also used in unsupervised
# Spam, Not Spam | Feline, Canine, Fish |

# Now we have a discrete relationship (only finite number of values, in this case 0 or 1
# We need a new function to classify this
# For this we'll use Logistic Regression

import math
import matplotlib.pyplot as plt

hours_studied = [1,2,3,4,5,6,7,8]
grade = [0,0,0,0,1,1,1,1] # 0 is fail and one is pass

w = 3
b = -5

def sigmoid(x):
    # The sigmoid function is used for our Logistic Regression (Classification)
    # It is the signature squiggle of binary classification
    return 1 / (1 + math.exp(-x))

def binary_cross_entropy(grade_actual, predicted_value):
    return -(grade_actual * math.log(predicted_value) + (1 - grade_actual) * math.log(1 - predicted_value))


predictions = []
for entry in hours_studied:
    # Let's apply the sigmoid function
    # Apply the weights and balances before calculating the sigmoid function

    linear = w * entry + b

    probability = sigmoid(linear)

    predictions.append(probability)

# This SUCKS

plt.scatter(
    hours_studied,
    grade,
    color='blue',
)

plt.plot(
    hours_studied,
    predictions,
    linewidth=2,
    color='red',
)

plt.show()