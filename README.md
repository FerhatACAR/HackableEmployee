Employee Data Leak Predictor

This Python project utilizes several Machine Learning and Data Science libraries to create a predictive model which aims to predict whether an employee might be involved in a data leak incident.
Libraries Used

    pandas
    numpy
    seaborn
    matplotlib
    scikit-learn

How It Works

    The code first imports the necessary Python libraries and loads an example dataset using pandas.

    The dataset is preprocessed to handle missing values and categorical variables. The numerical variables are then scaled using StandardScaler.

    The data is split into a training set and a test set.

    A logistic regression model is trained on the training data.

    The model's performance is evaluated by making predictions on the test set and comparing these predictions to the actual values.

    A decision boundary for the logistic regression model is visualized using a scatterplot and seaborn.

    The model can be fine-tuned to improve its performance.

    The trained model can then be used to make a prediction for a new employee.