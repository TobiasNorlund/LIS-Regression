# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 14:36:10 2015

@author: Borislav
"""
# Load and normalize the data
import load_data

# Train linear regressor
import sklearn.linear_model as sklin
from sklearn import svm
import numpy as np
import sklearn.cross_validation as skcv
# Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

X = load_data.X_norm
Y = load_data.Y

X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
print 
print X_new.shape

# Create and train linear regressor
e = -1
g = 3
c = 2
d = 15

# Train and predict the whole training set
regressor = svm.SVR(kernel = 'rbf', gamma = np.power(10,g), degree = d, C = np.power(10,c), epsilon = np.power(10,e)) #evtl change kernel #sklin.LinearRegression(normalize=True)
regressor.fit(X,Y)
Y_pred = regressor.predict(X)
print('Score (full training set): ', load_data.logscore(Y, Y_pred))

# Perform 5 fold cross validation_curve
scores = skcv.cross_val_score(regressor, load_data.X_norm, load_data.Y, scoring=load_data.scorefun, cv=4)
print('C-V score =', np.mean(scores), '+/-', np.std(scores))