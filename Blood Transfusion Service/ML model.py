# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#################################################################################################################
# Data pre-processing

# Importing the blood transfusion service centre dataset
dataset = pd.read_csv('Dataset.csv')

# Create a list of feature names (independent variables names)
feat_labels = ['Recency','Frequency','Monetary','Time']

# selecting the matrix of independent variables and dependent variables
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling (visualising the results at the end)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

##################################################################################################################

# Create a random forest classifier for fully Featured (4 Features) dataset
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Print the name and gini importance of each feature
from sklearn.feature_selection import SelectFromModel
feature_importances = SelectFromModel(feat_labels, classifier.feature_importances_)
print(feature_importances)

# Create a selector object that will use the random forest classifier to identify features that have an importance of more than 0.25
Select_From_Model = SelectFromModel(classifier, threshold=0.25)
Select_From_Model.fit(X_train, y_train)

# Print the names of the most important features
for feature_list_index in Select_From_Model.get_support(indices=True):
    print(feat_labels[feature_list_index])

# Transform the data to create a new dataset containing only the most important features
# Note: Applying the transform to both the training X and test X data.
X_important_train = Select_From_Model.transform(X_train)
X_important_test = Select_From_Model.transform(X_test)

#################################################################################################################

# Create a new random forest classifier for the most important features
from sklearn.ensemble import RandomForestClassifier
classifier_important = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)
classifier_important.fit(X_important_train, y_train)

# Apply The Full Featured Classifier To The Test Data
y_pred = classifier.predict(X_test)

# View The Accuracy Of Our Full Feature (4 Features) Model
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

# Apply The Limited Featured (2 Features) Classifier To The Test Data
y_important_pred = classifier_important.predict(X_important_test)

# View The Accuracy Of Our Limited Feature (2 Features) Model
accuracy_score(y_test, y_important_pred)

#################################################################################################################
# Visualising the Training set results for limited Featured model
from matplotlib.colors import ListedColormap
X_set, y_set = X_important_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier_important.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('limited Featured model (Training set)')
plt.xlabel('Recency')
plt.ylabel('Time')
plt.legend()
plt.show()

#################################################################################################################
# Visualising the Test set results for limited Featured model
from matplotlib.colors import ListedColormap
X_set, y_set = X_important_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier_important.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('limited Featured model (Test set)')
plt.xlabel('Recency')
plt.ylabel('Time')
plt.legend()
plt.show()
