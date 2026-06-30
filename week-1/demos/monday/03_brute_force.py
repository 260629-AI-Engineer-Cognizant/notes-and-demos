# We'll start by import matplotlib which is a library used for creating plots
import matplotlib.pyplot as plt

# Data
hours_studied = [1, 2, 3, 4, 5, 6, 7, 8]
grades = [52, 57, 65, 70, 76, 82, 87, 93]

w = 6
b = 46

predictions = []

for entry in hours_studied:
    predictions.append(w * entry + b)

# Goal: measure how good or bad our model is
# Here we're calculating the Error of the model
# Specifically this is called Absolute Error
# Absolute error is simply the difference between the expected and the
# predicted

# absolute_error = float("inf")
# temp = 0.0
# for predicted, actual in zip(predictions, grades):
#     temp += predicted - actual
# absolute_error =  temp / len(predictions)


# We're actually going to use a different error function
# MSE: Mean Squared Error
# Taking the difference and then squaring it
# Why? One, always positive and two, heavily punishes outliers

def mse(w, b):

    total = 0
    for x,y in zip(hours_studied, grades):
        prediction = w * x + b
        total += (y-prediction)**2

    return total/len(hours_studied)


# print(f"MSE: {round(mse(w, b), 2)}")

# Brute force a solution to find the LOWEST ERROR
lowest_error = float('inf')
best_w = 0
best_b = 0

for i in range(-10,11):
    for j in range(20, 61):
        error = mse(i, j)
        if error < lowest_error:
            lowest_error = error
            best_w = i
            best_b = j

print(f"Optimal w: {best_w} | Optimal b: {best_b} | MSE: {lowest_error}")

plt.figure(figsize = (8,5))

# Scatter the points on the plot (graph them)
plt.scatter(
    hours_studied,
    grades,
    color = 'blue',
    s=80
)

# Plot our line
plt.plot(
    hours_studied,
     predictions,
    color = 'red',
    linewidth=3
     )

plt.title("Hours Studied vs Test Score")

plt.xlabel("Hours Studied")

plt.ylabel("Grades")

plt.grid(True)

plt.show()