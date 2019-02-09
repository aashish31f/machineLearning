#SIMPLE LINEAR REGRESSION

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#Splitting the dataset into training and test sets
from sklearn.cross_validation import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 0)

#Fitting the simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train , y_train )

#Predicting the test set results
y_pred = regressor.predict(x_test)

#Visualising the training set results
plt.scatter(x_train , y_train)
plt.plot(x_train , regressor.predict(x_train) , color = 'red')
plt.title('salary vs experience (Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#Visualising the test set results
plt.scatter(x_test , y_test)
plt.plot(x_train , regressor.predict(x_train) , color = 'red')
plt.title('salary vs experience (Test Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()