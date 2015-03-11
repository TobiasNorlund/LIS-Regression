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
# Provides grid search functionality
import sklearn.grid_search as skgs
import numpy as np
import sklearn.svm as svm

def get_features(row):
    t = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
    t2 = time.strptime(str(row[0]), "%Y-%m-%d %H:%M:%S") # t is timestamp of train_x
    weekTime = (t2.tm_wday*24 + t2.tm_hour + t2.tm_min/60 + t2.tm_sec/3600) #in hours
    return np.array([[float(weekTime), float(t.month)]]) #, float(t.hour), float(t.day % 7), float(row[1]), float(row[3]), float(row[4]), float(row[5])]]) #np.array([[weekTime]]) #

def read_data(inpath):
    X = np.empty((0,2))
    with open(inpath, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
            X = np.append(X, get_features(row), axis=0)
    return np.atleast_2d(X)


X = read_data('project_data/train.csv')
X_test = read_data('project_data/validate.csv')
Y = np.genfromtxt('project_data/train_y.csv', delimiter=',')
print('Shape of X:', X.shape)
print('Shape of Y:', Y.shape)

# Define score function
def logscore(gtruth, pred):
    pred = np.clip(pred, 0, np.inf)
    logdif = np.log(1 + gtruth) - np.log(1 + pred)
    return np.sqrt(np.mean(np.square(logdif)))

# Normalize Data
means = np.mean(X, axis=0)
stds = np.std(X, axis=0)

X_norm = (X-means)/stds
X_test_norm = (X_test - means)/stds

# Create and train linear regressor
e = -1
g = 3
c = 2
d = 15
regressor = svm.SVR(kernel = 'rbf', gamma = np.power(10,g), degree = d, C = np.power(10,c), epsilon = np.power(10,e)) #evtl change kernel #sklin.LinearRegression(normalize=True)

# Cross Validate
scorefun = skmet.make_scorer(logscore)
regressor.fit(X_norm, Y)
#scores = skcv.cross_val_score(regressor, X_norm, Y, scoring=scorefun, cv=5)
#print('C-V score =', np.mean(scores), '+/-', np.std(scores))

# Predict test set
test_y = regressor.predict(X_test_norm)
np.savetxt('result_validate.txt', test_y)
