
# Load and normalize the data
import load_data
import datetime
import time
import numpy as np

# Get the features to use in this linear regressor
def get_features(row):
    t = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
    t2 = time.strptime(str(row[0]), "%Y-%m-%d %H:%M:%S") # t is timestamp of train_x
    weekTime = (t2.tm_wday*24 + t2.tm_hour + t2.tm_min/60 + t2.tm_sec/3600) #in hours

    c = [0., 0., 0., 0.]
    c[int(row[2])] = 1.

    return np.array([[float(weekTime), float(t.month),  float(row[1]), c[0], c[1], c[2], c[3], float(row[3]), float(row[4]), float(row[5]), float(row[6])]]) #, float(t.hour), float(row[1]), float(row[3]), float(row[4]), float(row[5])]]) #np.array([[weekTime]]) #

(X, Y, X_val, X_test) = load_data.load(get_features)

# Train linear regressor
import sklearn.linear_model as sklin
import sklearn.cross_validation as skcv

regressor = sklin.LinearRegression()

# Train and predict the whole training set
regressor.fit(X, Y)
Y_pred = regressor.predict(X)
print('Score (full training set): ', load_data.logscore(Y, Y_pred))

# Perform 5 fold cross validation
scores = skcv.cross_val_score(regressor, X, Y, scoring=load_data.scorefun, cv=4)
print('C-V score =', np.mean(scores), '+/-', np.std(scores))