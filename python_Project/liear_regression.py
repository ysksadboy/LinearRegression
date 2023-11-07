#Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Splitting the dataset into the training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#Training the simple linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test set results
y_pred = regressor.predict(X_test)

#Visualizing the Training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('salaries vs experience(Training Set)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('salaries vs experience(test set)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()

#Making a single prediction
print(regressor.predict([[5]]))

#Getting the final Linear Regession Equation
print(regressor.coef_)
print(regressor.intercept_)