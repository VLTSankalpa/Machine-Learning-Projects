###################################################################################################
# Q2
# Identifing identify collusive trader group using Apriori Algorithm
###################################################################################################
# Importing the libraries

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from apyori import apriori

###################################################################################################
# Data preprocessing with pandas

# Importing the trades data as pandas DataFrame
dataset = pd.read_csv('Trades.csv')
# Filtering the trades of stock ES0158252033 and create a new DataFrame
Trades = dataset.loc[dataset.Stock=='ES0158252033', :]
# Add indexing column to Dataframe (Lets consider this index represent the Trade Date)
Trades['Index'] = range(1, len(Trades) + 1)
# Create NumPy array with Execute Qty and Execute Price of Trades
X = Trades.iloc[:, [ 8, 2]].values

####################################################################################################

# change the font size on a matplotlib plot
font = {'family' : 'normal','weight' : 'bold','size'   : 15}
plt.rc('font', **font)
# change the graph size of a matplotlib plot
plt.rcParams['figure.figsize'] = (30, 10)
# Plotting the graph of Executed Price of Stock ES0158252033
plt.plot(X[:,0], X[:,1])
plt.title('Executed Price of Stock ES0158252033')
plt.xlabel('Trade Index')
plt.ylabel('Executed Price')
plt.show()

###################################################################################################

# Fitting Hierarchical Clustering to the array X
hc = AgglomerativeClustering(n_clusters = 200, affinity = 'euclidean', linkage = 'ward')
# Create Array with clusters
X_hc = hc.fit_predict(X)
# Add a X_hc array as Cluster column to a Trades dataframe
Trades['Cluster'] = X_hc

###################################################################################################
# compute a summary statistic for each clusters using groupby aggregation
Cluster_Statistics_Price = Trades.groupby("Cluster")['Executed Price'].describe()
# Add a index clounm to Cluster Statistics Price dataframe
Cluster_Statistics_Price['Index'] = range(0, len(Cluster_Statistics_Price)+0)
# Sort all clusters according to ascending order by standard deviation
Cluster_Statistics_Price = Cluster_Statistics_Price.sort_values(by ='std', ascending=False).reset_index()
# Drop all clusters has standard deviation less than 0.1
T_Clusters = Cluster_Statistics_Price.iloc[0:52, [9]].values.tolist()
# Create a list of lists (Apriori library I am going to use requires our dataset to be in the form of a list of lists)
transactions = (
     [Trades.loc[Trades.Cluster==0, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==0, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==2, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==2, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==93, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==93, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==38, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==38, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==5, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==5, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==66, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==66, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==8, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==8, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==31, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==31, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==65, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==65, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==27, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==27, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==3, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==3, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==108, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==108, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==194, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==194, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==172, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==172, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==52, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==52, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==109, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==109, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==28, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==28, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==35, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==35, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==39, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==39, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==47, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==47, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==123, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==123, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==158, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==158, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==173, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==173, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==81, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==81, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==147, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==147, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==77, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==77, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==49, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==49, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==167, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==167, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==95, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==95, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==16, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==16, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==134, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==134, :]['Sell Broker ID'].unique().tolist()] + 
     [Trades.loc[Trades.Cluster==6, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==6, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==13, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==13, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==60, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==60, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==146, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==146, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==64, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==64, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==191, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==191, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==132, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==132, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==155, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==155, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==83, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==83, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==50, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==50, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==128, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==128, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==40, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==40, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==186, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==186, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==137, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==137, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==17, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==17, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==29, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==29, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==97, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==97, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==91, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==91, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==68, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==68, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==85, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==85, :]['Sell Broker ID'].unique().tolist()] +
     [Trades.loc[Trades.Cluster==10, :]['Buy Broker ID'].unique().tolist() + Trades.loc[Trades.Cluster==10, :]['Sell Broker ID'].unique().tolist()])

##################################################################################################
# Training Apriori on the dataset to identify potential collusive trader groups
rules_test = apriori(transactions, min_support = 0.1,min_confidence = 0.3, min_lift = 1.0001)
# Create a list with all potential collusive trader groups
results_test = list(rules_test)
# Visualising the number of potential collusive trader groups  
print(len(results_test))
# Visualising the potential collusive trader groups   
the_rules = [] 
for result in results_test: 
    the_rules.append({'Collusive Trader Groups Test': ','.join(result.items),
                      'Support':result.support, 
                      'Confidence':result.ordered_statistics[0].confidence,
                      'Lift':result.ordered_statistics[0].lift})
# Create a dataframe with all potential collusive trader groups
collusive_trader_groups_test = (pd.DataFrame(the_rules, columns = ['Collusive Trader Groups Test', 'Support', 'Confidence', 'Lift'])).sort_values(by ='Support', ascending=False)
# Sort all potential collusive trader groups according to ascending order by support
collusive_trader_groups_test = collusive_trader_groups_test.sort_values(by ='Support', ascending=False)

####################################################################################################
# Training Apriori on the dataset to filter collusive trader groups 
rules = apriori(transactions, min_support = 0.15, min_confidence = 0.75, min_lift = 2, min_length = 2)
# Create a list with all collusive trader groups
results = list(rules)
# Visualising the number of collusive trader groups
print(len(results))
# Visualising the collusive trader groups     
the_rules = [] 
for result in results: 
    the_rules.append({'Collusive Trader Groups': ','.join(result.items),
                      'Support':result.support, 
                      'Confidence':result.ordered_statistics[0].confidence,
                      'Lift':result.ordered_statistics[0].lift})
# Create a dataframe with all collusive trader groups 
collusive_trader_groups = pd.DataFrame(the_rules, columns = ['Collusive Trader Groups', 'Support', 'Confidence', 'Lift'])
