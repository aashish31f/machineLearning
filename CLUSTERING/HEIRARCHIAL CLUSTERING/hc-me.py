# HEIRARCHIAL CLUSTERING

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:,[3,4]].values

# Using dendogram to find optimal no. of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(x,method = 'ward'))
plt.title('DENDOGRAM')
plt.xlabel('Customers')
plt.ylabel('Euclidean distance')
plt.show()     

# Fitting hierarchial clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5)
y_hc = hc.fit_predict(x)

# Visualising the clusters
plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=50,c='red')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=50,c='blue')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=50,c='green')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=50,c='orange')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=50,c='magenta')
plt.title('MALL CUSTOMERS CLUSTERS')
plt.xlabel('Annual income')
plt.ylabel('Spending Score')
plt.show()
