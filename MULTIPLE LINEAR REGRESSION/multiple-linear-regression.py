#MULTIPLE LINEAR REGRESSION

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelEncoder = LabelEncoder()
x[:,3] = labelEncoder.fit_transform(x[:,3])
oneHotEncoder = OneHotEncoder(categorical_features = [3])
x = oneHotEncoder.fit_transform(x).toarray()

#Avoiding Dummy variable trap
x = x[:,1:]

#Splitting dataset into test set and training set
from sklearn.cross_validation import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 0)

#Fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train , y_train)

#Predicting the test results
y_pred = regressor.predict(x_test)

#Building optimal model using backward elimination
import statsmodels.formula.api as ap
x = np.append(arr = np.ones((50,1)).astype(int) , values = x , axis = 1)
x_opt = x[:,[0,1,2,3,4,5]]
regressor_ols = ap.OLS(endog = y ,exog = x_opt).fit()
regressor_ols.summary()
x_opt = x[:,[0,1,3,4,5]]
regressor_ols = ap.OLS(endog = y ,exog = x_opt).fit()
regressor_ols.summary()
x_opt = x[:,[0,3,4,5]]
regressor_ols = ap.OLS(endog = y ,exog = x_opt).fit()
regressor_ols.summary()
x_opt = x[:,[0,3,5]]
regressor_ols = ap.OLS(endog = y ,exog = x_opt).fit()
regressor_ols.summary()

#Dividing optimal dataset into training and test set
x_opt_train , x_opt_test , y_train ,y_test = train_test_split(x_opt , y , test_size = 0.2 ,  random_state = 0 )


#Fitting multiple linear regression optimally
regressor_org = LinearRegression()
regressor_org.fit(x_opt_train , y_train)

#Predicting optimal y values 
y_opt_pred = regressor_org.predict(x_opt_test)
