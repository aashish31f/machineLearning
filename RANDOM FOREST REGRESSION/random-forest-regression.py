#RANDOM FOREST REGRESSION

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Fitting random forest regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 277 , random_state = 0)
regressor.fit(x,y)

#Predicting a new result
y_pred = regressor.predict(6.5)

#Plotting our random forest regression results
x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color = 'red')
plt.plot(x_grid,regressor.predict(x_grid))
plt.title('BLUFF DETECTOR (DECISION TREE)')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()
