#POLYNOMIAL REGRESSION

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Fitting linear regression to model
from sklearn.linear_model import LinearRegression
lin_reg =  LinearRegression()
lin_reg.fit(x , y)

#Fitting Polynomial regression to model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly , y)

#Visualising the Linear regression results
plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg.predict(x))
plt.title('BLUFF DETECTOR (LINEAR REGRESSION)')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()

#Visualising the polynomial regression results
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color = 'red')
plt.plot(x_grid,lin_reg2.predict(poly_reg.fit_transform(x_grid)))
plt.title('BLUFF DETECTOR (POLYNOMIAL REGRESSION)')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()

#Predicting a new result with linear regression
lin_reg.predict(6.5)


#Predicting a new result with polynomial regression
lin_reg2.predict(poly_reg.fit_transform(6.5))


