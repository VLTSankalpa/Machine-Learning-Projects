# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#####################################################################################################
# Data preprocessing

# Importing the dataset
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.01, random_state = 0)


# Model performance optimization with Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

####################################################################################################
# Implementing Simple Linear Regression (SLR) model

# Fitting Simple Linear Regression (SLR) to the Training set
from sklearn.linear_model import LinearRegression
SL_regressor = LinearRegression()
SL_regressor.fit(X_train, y_train)

# Predicting the Test set results for SLR
y_pred_SLR = SL_regressor.predict(X_test)

# Visualising the Training set results of SLR
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, SL_regressor.predict(X_train), color = 'blue')
plt.title('Age vs CO2 level-Simple Linear Regression (Training set)')
plt.xlabel('Age of the glaciers and ice caps')
plt.ylabel('CO2 level')
plt.show()

# Visualising the Test set results of SLR
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, SL_regressor.predict(X_train), color = 'blue')
plt.title('Age vs CO2 level-Simple Linear Regression (Test set)')
plt.xlabel('Age of the glaciers and ice caps')
plt.ylabel('CO2 level')
plt.show()

###################################################################################################

# Fitting Random Forest Regression (RFR) to the dataset
from sklearn.ensemble import RandomForestRegressor
RF_regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
RF_regressor.fit(X_train, y_train)

# Predicting the Test set results of the RFR
y_pred_RFR = RF_regressor.predict(X_test)

# Visualising the Training set results of RFR (higher resolution)
X_grid = np.arange(min(X_train), max(X_train), 0.5)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_grid, RF_regressor.predict(X_grid), color = 'blue')
plt.title('Age vs CO2 level-Random Forest Regression (Training set)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results of RFR (higher resolution)
X_grid = np.arange(min(X_test), max(X_test), 0.5)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_grid, RF_regressor.predict(X_grid), color = 'blue')
plt.title('Age vs CO2 level-Random Forest Regression (Test set)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
