# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 23:16:37 2019

@author: anmol
"""

# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset=pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values # Cause X should be a matrix not a vector like y
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""

# Fitting Regression model to the dataset
# Create your regressor

# Predicting a new result
y_pred = regressor.predict(6.5)

# Visualising the Polynomial Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x)), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel("Position Level")
plt.ylabel('Salary')
plt.plot()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid)), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel("Position Level")
plt.ylabel('Salary')
plt.plot()