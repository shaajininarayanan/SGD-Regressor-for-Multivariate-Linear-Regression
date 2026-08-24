# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required packages and print the present data. 
2.Print the placement data and salary data. 
3.Find the null and duplicate values. 
4.Using logistic regression find the predicted values of accuracy , confusion matrices. 
5.Display the results.
```

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: shaajini n
RegisterNumber: 212225040395
*/
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load California Housing dataset
housing = fetch_california_housing()

X = housing.data
y = housing.target

# Display dataset
data = pd.DataFrame(X, columns=housing.feature_names)
data["Target"] = y

print(data.head())

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create SGD Regression model
model = SGDRegressor(
    learning_rate="constant",
    eta0=0.01,
    max_iter=1000,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Display coefficients and intercept
print("\nCoefficients:")
print(model.coef_)

print("\nIntercept:")
print(model.intercept_)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R2 Score:", r2)

# Actual vs Predicted values
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print("\nActual vs Predicted:")
print(results.head(10))
```

## Output:
![multivariate linear regression model for predicting the price of the house and number of occupants in the house](sam.png)
<img width="922" height="905" alt="image" src="https://github.com/user-attachments/assets/477b5289-cb9f-46be-9d53-a08626b3f817" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
