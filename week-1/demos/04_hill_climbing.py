import matplotlib.pyplot as plt

# Data
hours_studied = [1, 2, 3, 4, 5, 6, 7, 8]
grades = [52, 57, 65, 70, 76, 82, 87, 93]

def mse(w, b):

    total = 0
    for x,y in zip(hours_studied, grades):
        prediction = w * x + b
        total += (y-prediction)**2

    return total/len(hours_studied)

# We want to find the lowest possible error, there are a lot of ways to do this,
# we just saw one which was brute forcing but this scales horribly
# Motivating concept: Imagine you're blindfolded and at the top of a mountain. If I
# asked you to find the lowest point or get to the bottom how would you do it?
# Intuitively, you start by moving in the direction the ground seems to be downhill
# and from there you keep stepping in the lowest direction till you get to the bottom

# Initial guess
# Typically randomized
w = 1.0
b = 20.0

# We need a step size
# This controls how far we go we take that step down the mountain
step_size = 0.25

# Let's consider how many rounds of optimization I want
# We'll do this 100 times for right now
for iteration in range(100):

    current_error = mse(w, b)

    # Let's try taking a step in a direction
    if mse(w+step_size, b) < current_error:
        w = w + step_size
    elif mse(w - step_size, b) < current_error:
        w = w - step_size

    # Try updating our bias after our weight
    current_error = mse(w, b)
    if mse(w, b+step_size)< current_error:
        b = b + step_size
    elif mse(w, b-step_size)< current_error:
        b = b - step_size

predictions = []
for entry in hours_studied:
    predictions.append(w * entry + b)

print(f"Optimized w = {w}, b = {b}")
print(f"Current MSE: {mse(w, b)}")

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