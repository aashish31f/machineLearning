# K MEANS CLUSTERING 

#Importing llibraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset
dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:,[3,4]].values

#Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,random_state =0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title("ELBOW METHOD")   
plt.xlabel('NO. OF CLUSTERS')
plt.ylabel('WCSS')
plt.show()

#Applying k-means to the dataset
kmeans = KMeans(n_clusters=5,random_state = 0)
y_kmeans = kmeans.fit_predict(x)

#Visualising the clusters
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=50,c='red',label='cluster 1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=50,c='blue',label='cluster 2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=50,c='yellow',label='cluster 3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=50,c='orange',label='cluster 4')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=50,c='black',label='cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='green',label='CENTROIDS')
plt.title('mall customers clusters')
plt.xlabel('annual income')
plt.ylabel('spending score')
plt.show()