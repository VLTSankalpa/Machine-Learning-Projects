#############################################################################################################
# Q1 part 2 
# Identifing outlier traders based on sum of Executed Qty using Hierarchical Clustering
########################################################################################################
# Importing the libraries

import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

####################################################################################################
# Data preprocessing with pandas

# Importing the trades data as pandas DataFrame
dataset = pd.read_csv('Trades.csv')
# Create a new DataFrame contain sum of Executed Qty for each traders who brought stock ES0158252033 
Traders = dataset[dataset.Stock=="ES0158252033"].groupby("Buy Broker ID")['Executed Qty'].sum().reset_index()
# Add indexing column to Traders Dataframe
Traders['Index'] = range(1, len(Traders) + 1)
# Create NumPy array with sum of Executed Qty of Traders
X = Traders.iloc[:, [2, 1]].values

#####################################################################################################
# change the font size on a matplotlib plot
font = {'family' : 'normal','weight' : 'bold','size'   : 25}
plt.rc('font', **font)
# change the graph size of a matplotlib plot
plt.rcParams['figure.figsize'] = (30, 40)

#####################################################################################################
# Using the dendrogram to find the optimal number of clusters suitable to identify outliners

# plot dendrogram for X array
dendrogram_X = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram of array X')
plt.xlabel('Trades')
plt.ylabel('Euclidean distances')
plt.show()

#####################################################################################################

# Fitting Hierarchical Clustering to the array X
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
# Create Array with clusters
X_hc = hc.fit_predict(X)
# Add a X_hc array as Cluster column to a Trades dataframe
Traders['Cluster'] = X_hc

##################################################################################################
# change the graph size of a matplotlib plot
plt.rcParams['figure.figsize'] = (20, 10)
# Visualising the clusters
plt.scatter(X[X_hc == 0, 0], X[X_hc == 0, 1], s = 30, c = 'red', label = 'Cluster 0')
plt.scatter(X[X_hc == 1, 0], X[X_hc == 1, 1], s = 30, c = 'orange', label = 'Cluster 1')
plt.scatter(X[X_hc == 2, 0], X[X_hc == 2, 1], s = 30, c = 'green', label = 'Cluster 2')
plt.title('Clusters of traders')
plt.xlabel('Trader Index')
plt.ylabel('Sum of Executed Qty')
plt.legend()
plt.show()

###############################################################################################
# Calculating statistical information of clusters
Cluster_Statistics = Traders.groupby("Cluster")['Executed Qty'].describe()

############################################################################################
# Create outlier Trdaes Dataframe
Cluster = [0, 1]
Outlier_Traders= Traders[Traders.Cluster.isin(Cluster)]