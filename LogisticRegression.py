import pandas as pd
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
