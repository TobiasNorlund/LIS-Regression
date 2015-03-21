
# Load and normalize the data
import load_data
import datetime
import time
import numpy as np
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
enc.n_values = 24
enc.n_values_ = 24
enc.feature_indices_ = np.array([0, 24], dtype='int32')

""" [weektime, month] """
def get_features2(row):
    t = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
    t2 = time.strptime(str(row[0]), "%Y-%m-%d %H:%M:%S") # t is timestamp of train_x
    weekTime = (t2.tm_wday*24 + t2.tm_hour + t2.tm_min/60 + t2.tm_sec/3600) #in hours

    hours = enc.transform([[t2.tm_hour]]).toarray().tolist()[0]

    c = [0., 0., 0., 0.]
    c[int(row[2])] = 1.

    feature_vec = [float(weekTime), float(t.month), float(row[1]), float(row[3]), c[0], c[1], c[2], c[3]]
    feature_vec.extend(hours)

    return np.array([feature_vec]) #, float(t.hour), float(row[1]), float(row[3]), float(row[4]), float(row[5])]]) #np.array([[weekTime]]) #


(X, Y, X_val, X_test) = load_data.load(get_features2)

# ------- SVR REGRESSOR -----------------

import sklearn.svm as svm
import sklearn.grid_search as skgs # Provides grid search functionality

# Create SVR regressor
regressor = svm.SVR(kernel = 'rbf')

# Perform gris search
param_grid = { 'epsilon': [0.5],
               'C': [10],
               'gamma': [1e-3, 1e-2, 1e-1, 1],
               'degree': [1]
               }

grid_search = skgs.GridSearchCV(regressor, param_grid, scoring=load_data.scorefun, cv=4)
grid_search.fit(X, Y)

best = grid_search.best_estimator_
print(best)
print('best score =', grid_search.best_score_)
print('SVs: ', best.support_.shape)

# Predict validation set
Y_val = best.predict(X_val)
np.savetxt('result_validation.txt', Y_val)

# Predict test set
Y_test = best.predict(X_test)
np.savetxt('result_test.txt', Y_test)
