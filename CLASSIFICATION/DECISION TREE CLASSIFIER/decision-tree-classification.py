#DECISION TREE CLASSIFICATION

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset 
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,2:4].values
y = dataset.iloc[:,-1].values

#Splitting dataset into training and test sets
from sklearn.cross_validation import train_test_split
x_train ,  x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state = 0)

#Training our classifier 
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy',random_state = 0)
classifier.fit(x_train,y_train)

#Predicting the results
y_pred = classifier.predict(x_test)

#The confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#Visualising the training set results
from matplotlib.colors import ListedColormap
x_set , y_set = x_train , y_train
x1 , x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1,stop = x_set[:,0].max()+1,step = 1),
                      np.arange(start = x_set[:,1].min()-1,stop = x_set[:,1].max()+1,step = 1))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha = 0.75 ,
             cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i , j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier decision tree(Training set)') 
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()