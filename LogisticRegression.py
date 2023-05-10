import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('Data/ExampleData.csv')

# Print out the column names of the dataset
print(data.columns)

# Preprocess the data
data = data.dropna()  # remove missing values
X = data[['Experience', 'Age', 'Awareness']]
y = data['Leak']  # whether the employee was involved in a data leak incident

X = pd.get_dummies(X)  # one-hot encode categorical variables
scaler = StandardScaler()
X = scaler.fit_transform(X)  # scale numerical variables

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model's performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Visualize the decision boundary for the logistic regression model
X_vis = scaler.inverse_transform(X)[:, :2]  # Transform X back to its original scale (only first 2 columns)
y_vis = y.values

# Set up the scatter plot with the decision boundary
sns.scatterplot(x=X_vis[:, 0], y=X_vis[:, 1], hue=y_vis, style=y_vis, palette="viridis")
ax = plt.gca()

# Calculate the decision boundary
xx, yy = np.meshgrid(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100),
                     np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 100))
Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel(), np.zeros(10000)]))
Z = Z.reshape(xx.shape)

# Plot the decision boundary
ax.contour(xx, yy, Z, [0.5], colors='k')

plt.xlabel('Experience')
plt.ylabel('Age')
plt.title('Logistic Regression Decision Boundary')

plt.show()

# Fine-tune the model's hyperparameters (e.g. regularization parameter)
# ...

# Make a prediction for a new employee
new_employee = [[3, 27, 1]]  # experience, age, and low Awareness
new_employee = pd.DataFrame(new_employee, columns=['Experience', 'Age', 'Awareness'])
new_employee = pd.get_dummies(new_employee)
new_employee = scaler.transform(new_employee)
prediction = model.predict(new_employee)
if prediction[0] == 1:
    print('Teach the employee about data leaks.')
else:
    print('No need to teach the employee about data leaks.')
