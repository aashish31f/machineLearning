#SVR

#Importing Datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Datasets
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)
sc_y = StandardScaler()
y = sc_y.fit_transform(y.reshape((len(y),1)))

#Fitting SVR to dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x,y)

#Predicting a new result
y_pred = sc_y.inverse_transform([[regressor.predict(sc_x.transform(np.array([[6.5]])))]])

#Visualising the SVR results 
plt.scatter(x,y,color = 'red')
plt.plot(x , regressor.predict(x))
plt.title('BLUFF DETECTOR (LINEAR REGRESSION)')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()

#Visualising the SVR results with a higher resolution
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color = 'red')
plt.plot(x_grid , regressor.predict(x_grid))
plt.title('BLUFF DETECTOR (LINEAR REGRESSION)')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()

