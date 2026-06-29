# Defining everything by hand, while a worthwhile endeavor is kinda a pain
# Luckily there are plenty of ML libraries we can use, one of the most popular
# is Scikit-Learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# When using Sklearn we need to use numpy arrays
# I'm also going to give it some more data

hours_studied = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 5.3, 9]
grades = [52, 57, 65, 70, 76, 82, 87, 93, 46, 55, 71, 85, 95]

# Create numpy arrays for this
X = np.array(hours_studied).reshape(-1,1)
y = np.array(grades).reshape(-1,1)

# Split the data into test and train sets
# If we train over and over on the same data, we run the risk of overfitting
# We split the data into a train section (~80%) and the remainder is the test section
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Doing all of the linear regression work is pretty simple
model = LinearRegression()
model.fit(X_train, y_train) # This is common for all sklearn, this is the process used

predictions = model.predict(X_test)

# We can print our weight and bias values
print(f"Weight (Coefficient): {model.coef_}")
print(f"Bias (Intercept): {model.intercept_}")
# We can also calculate the MSE
print(f"MSE: {mean_squared_error(y_test, predictions)}")

plt.scatter(hours_studied, grades, color="blue")
plt.plot(X_test, predictions, color="red")
plt.show()
