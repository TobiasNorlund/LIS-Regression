# For reading and writing csv files
import csv
# For parsing date/time strings
import datetime
import time
# Contains linear models, e.g., linear regression, ridge regression, LASSO, etc.
import sklearn.linear_model as sklin
# Allows us to create custom scoring functions
import sklearn.metrics as skmet
# Provides train-test split, cross-validation, etc.
import sklearn.cross_validation as skcv

import numpy as np


MAX_TRAIN_SAMPLES = 5000 # 10427

def get_features(row):
    t = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
    t2 = time.strptime(str(row[0]), "%Y-%m-%d %H:%M:%S") # t is timestamp of train_x
    weekTime = (t2.tm_wday*24 + t2.tm_hour + t2.tm_min/60 + t2.tm_sec/3600) #in hours

    c = [0., 0., 0., 0.]
    c[int(row[2])] = 1.

    return np.array([[float(weekTime), float(t.month), c[0], c[1], c[2], c[3]]]) #, float(t.hour), float(row[1]), float(row[3]), float(row[4]), float(row[5])]]) #np.array([[weekTime]]) #

def read_data(inpath, get_features_fun):
    X = None
    with open(inpath, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        i = 0
        for row in reader:
            i = i+1
            if(i > MAX_TRAIN_SAMPLES):
                break
            
            features = get_features_fun(row)
            
            if(X is None):
                X = np.empty((0,features.shape[1]))
            
            X = np.append(X, features, axis=0)
    return np.atleast_2d(X)

# Define score function
def logscore(gtruth, pred):
    pred = np.clip(pred, 0, np.inf)
    logdif = gtruth -pred
    
    score = np.sqrt(np.mean(np.square(logdif)))
    print('logscore: ', score, ' mean error:', np.mean(np.abs(gtruth - pred)))
    return score
scorefun = skmet.make_scorer(logscore, greater_is_better=False)

def load(get_features_fun = get_features):

    X = read_data('project_data/train.csv', get_features_fun)
    X_val = read_data('project_data/validate.csv', get_features_fun)
    X_test = read_data('project_data/test.csv', get_features_fun)
    
    Y = np.genfromtxt('project_data/train_y.csv', delimiter=',')
    Y = Y[0:MAX_TRAIN_SAMPLES]
    Y = np.log(1+Y)
    print('Shape of X:', X.shape)
    print('Shape of Y:', Y.shape)

    # Normalize Data
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    means[4:] = 0
    stds[4:] = 1
    stds[stds == 0] = 1
    
    X_norm = (X-means)/stds
    X_val_norm = (X_val-means)/stds
    X_test_norm = (X_test - means)/stds

    print('Data loaded sucessfully')
    
    return (X_norm, Y, X_val_norm, X_test_norm)