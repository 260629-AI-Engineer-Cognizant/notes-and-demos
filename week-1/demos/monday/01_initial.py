# We'll start by import matplotlib which is a library used for creating plots
import matplotlib.pyplot as plt

# Data
hours_studied = [1, 2, 3, 4, 5, 6, 7, 8]
grades = [52, 57, 65, 70, 76, 82, 87, 93]

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

plt.title("Hours Studied vs Test Score")

plt.xlabel("Hours Studied")

plt.ylabel("Grades")

plt.grid(True)

plt.show()