# We'll start by import matplotlib which is a library used for creating plots
import matplotlib.pyplot as plt

# Data
hours_studied = [1, 2, 3, 4, 5, 6, 7, 8]
grades = [52, 57, 65, 70, 76, 82, 87, 93]

# We are going to attempt to create our first "Model"
# This model is for Supervised Learning and is called Linear Regression
# Linear indicates a line
# What is regression? In ML regression is the ability to predict a
# continuous outcome for a specific set of features (numerical value)
# Algebra one => y = mx+b
# Pretty similar here=> y = wx+b
# Why w and b? They stand for Weights and Biases

w = -7
b = 40

predictions = []

for entry in hours_studied:
    predictions.append(w * entry + b)


print("Hours Studied | Test Score")
print('-'*20)

for hour, score in zip(hours_studied, grades):
    print(f"{hour} | {score}")

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