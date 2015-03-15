
# Load and normalize the data
import load_data
import datetime
import time
import numpy as np
from load_data import logscore

""" [weektime, month] """
def get_features(row):
    t = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
    t2 = time.strptime(str(row[0]), "%Y-%m-%d %H:%M:%S") # t is timestamp of train_x
    weekTime = (t2.tm_wday*24 + t2.tm_hour + t2.tm_min/60 + t2.tm_sec/3600) #in hours

    return np.array([[float(weekTime), float(t.month), float(row[1]), float(row[3])]]) #, float(t.hour), float(row[1]), float(row[3]), float(row[4]), float(row[5])]]) #np.array([[weekTime]]) #

(X_, Y_, X_test) = load_data.load(get_features)

# ---- NN ---

X = X_[1:5000,]

def find_nearest(input):
    
    dst = np.sqrt(np.sum(np.square(X-input), axis=1))
    
    min = np.argmin(dst)
    
    return Y_[min]

def validate(X_test, Y_test):
    pred = np.zeros(len(X_test))
    
    for i in range(0, len(X_test)):
        pred[i] = find_nearest(X_test[i,])
        
    return load_data.logscore(Y_test, pred)

print(validate(X_[10000:,], Y_[10000:]))