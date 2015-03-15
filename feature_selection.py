# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 14:36:10 2015

@author: Borislav
"""
# Load and normalize the data
import load_data
import datetime
import time
# Train linear regressor
import sklearn.linear_model as sklin
from sklearn import svm
import numpy as np
import sklearn.cross_validation as skcv
# Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
# Provides grid search functionality
import sklearn.grid_search as skgs 


# Get the features to use in this linear regressor
def get_features(row):
    t = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
    t2 = time.strptime(str(row[0]), "%Y-%m-%d %H:%M:%S") # t is timestamp of train_x
    weekTime = (t2.tm_wday*24 + t2.tm_hour + float(t2.tm_min)/60. + float(t2.tm_sec)/3600.) #in hours
    weekDayOrEnd = (-1. if t2.tm_wday < 5 else 1.)

    return np.array([[weekTime,float(t.month),weekDayOrEnd]]) #, float(t.hour), float(row[1]), float(row[3]), float(row[4]), float(row[5])]]) #np.array([[weekTime]]) #

(X, Y, X_test) = load_data.load(get_features)


#X = SelectKBest(f_regression, k=6).fit_transform(X, Y)
#print X_new.shape

# Create and train regressor
e = np.array([-1])
c = np.array([2])
g = np.array([3])
d = np.array([0])

# Train and predict the whole training set
regressor = svm.SVR(kernel = 'rbf')

# Perform gris search - it has CV built in
param_grid = { 'epsilon': np.power(10, e),
               'C': np.power(10, c),
               'gamma': np.power(10, g),
               'degree': d
               }
grid_search = skgs.GridSearchCV(regressor, param_grid, scoring=load_data.scorefun,cv=3)
grid_search.fit(X, Y)
print('Best Parameters = ', grid_search.best_params_)
print('Best score =', grid_search.best_score_)