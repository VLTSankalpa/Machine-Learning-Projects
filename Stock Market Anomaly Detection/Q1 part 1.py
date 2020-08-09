###################################################################################################
# Q1 part 1 
# Identifing outlier trades based on Executed Price & Executed Qty using Hierarchical Clustering
####################################################################################################
# Importing the libraries

import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

####################################################################################################
# Data preprocessing with pandas

# Importing the trades data as pandas DataFrame
dataset = pd.read_csv('Trades.csv')
# Filtering the trades of stock ES0158252033 and create a new DataFrame
Trades = dataset.loc[dataset.Stock=='ES0158252033', :]
# Add indexing column to Dataframe (Lets consider this index represent the Trade Date)
Trades['Index'] = range(1, len(Trades) + 1)
# Create NumPy array with Execute Qty and Execute Price of Trades
X = Trades.iloc[:, [ 1, 2]].values

#####################################################################################################
# Using the dendrogram to find the optimal number of clusters suitable to identify outliners

# change the font siXe on a matplotlib plot
font = {'family' : 'normal','weight' : 'bold','size'   : 15}
plt.rc('font', **font)
# change the graph siXe of a matplotlib plot
plt.rcParams['figure.figsize'] = (30, 40)

# plot dendrogram for X array
dendrogram_X = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram of array X')
plt.xlabel('Trades')
plt.ylabel('Euclidean distances')
plt.show()

#####################################################################################################
# Fitting Hierarchical Clustering to the array X
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
# Create Array with clusters
X_hc = hc.fit_predict(X)
# Add a X_hc array as Cluster column to a Trades dataframe
Trades['Cluster'] = X_hc

####################################################################################################
# change the graph size of a matplotlib plot
plt.rcParams['figure.figsize'] = (20, 10)
# Visualising the clusters
plt.scatter(X[X_hc == 0, 0], X[X_hc == 0, 1], s = 10, c = 'black', label = 'Cluster 0')
plt.scatter(X[X_hc == 1, 0], X[X_hc == 1, 1], s = 10, c = 'blue', label = 'Cluster 1')
plt.scatter(X[X_hc == 2, 0], X[X_hc == 2, 1], s = 10, c = 'green', label = 'Cluster 2')
plt.scatter(X[X_hc == 3, 0], X[X_hc == 3, 1], s = 10, c = 'red', label = 'Cluster 3')
plt.scatter(X[X_hc == 4, 0], X[X_hc == 4, 1], s = 10, c = 'orange', label = 'Cluster 4')
plt.title('Clusters of trades')
plt.xlabel('Executed Qty')
plt.ylabel('Executed Price')
plt.legend()
plt.show()

###############################################################################################
# Calculating statistical information of clusters
#
Cluster_Statistics_Qty = Trades.groupby("Cluster")['Executed Qty'].describe()
#
Cluster_Statistics_Price = Trades.groupby("Cluster")['Executed Price'].describe()
#
Cluster_Statistics_index = Trades.groupby("Cluster")['Index'].describe()

############################################################################################
# Create outlier Trdaes Dataframe

Cluster = [3]
Outlier_Trades = Trades[Trades.Cluster.isin(Cluster)]