import numpy as np
from sklearn import datasets,linear_model,metrics

diabetes = datasets.load_diabetes()

diabetes_x = diabetes.data #matrix of dimension 442 x 10


 #split the data into training testing sets
diabetes_x_train = diabetes_x[:-20]
diabetes_x_test = diabetes_x[-20:]

diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

#create linear regression object
reg = linear_model.LinearRegression()

#train the model using the training sets
reg.fit(diabetes_x_train,diabetes_y_train)

#make predictions using training set
diabetes_y_pred = reg.predict(diabetes_x_test)

#the coefficients

print('coefficients : \n', reg.coef_)


#the mean squared error

meanSquaredError = metrics.mean_squared_error(diabetes_y_test,diabetes_y_pred)

print("mean squared error : %.2f" % meanSquaredError)

