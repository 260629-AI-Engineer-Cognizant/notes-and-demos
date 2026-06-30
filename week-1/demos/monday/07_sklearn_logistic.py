# We did all of this Sklearn stuff since it's difficult to write all of these
# functions by ourselves and we should rely on these libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

hours_studied = [1,2,3,4,5,6,7,8]
grade = [0,0,0,0,1,1,1,1] # 0 is fail and one is pass

# X, y = make_classification(
#     n_features=1,
#     n_redundant=0,
#     n_informative=1,
#     n_clusters_per_class=1,
#     random_state=42
# )

# We need the numpy arrays
X = np.array(hours_studied).reshape(-1,1)
y = np.array(grade)

# We'll skip the splitting for now since we have almost no data, it's not really
# needed

# Create our model and fit
model = LogisticRegression()
model.fit(X, y)

new_student = np.array([7.5]).reshape(-1,1)
# Let's predict if they will pass
print(model.predict(new_student))

# To make this visual we need a slightly different approach
# Gives us 200 evenly spaced points between 0 and 9
study_hours = np.linspace(0,9, 200).reshape(-1,1)

# Calculate the probabilities at each point
probabilities = model.predict_proba(study_hours)[:, 1]

predictions = model.predict(study_hours)

# print(probabilities)

plt.figure(figsize = (8,5))

plt.scatter(
    hours_studied,
    grade
)

# Plot our curve
plt.plot(
    study_hours,
    probabilities,
    color = 'green'
)

plt.show()