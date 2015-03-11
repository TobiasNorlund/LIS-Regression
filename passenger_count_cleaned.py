import numpy as np
import matplotlib.pylab as plt
import time
import sklearn.cross_validation as skcv

from numpy import genfromtxt
from random import randint
from sklearn.preprocessing import normalize
from sklearn.svm import SVR

""" Read Data """
inTypes=[('timestamp','a19'),('w1',float),('w2',int),('w3',float),('w4',float),('w5',float),('w6',float)]

#print("start reading data")
train_data = np.asarray(genfromtxt('project_data/train.csv', delimiter=',', dtype = inTypes))
train_data_y = np.asarray(genfromtxt('project_data/train_y.csv'))

validate_data = np.asarray(genfromtxt('project_data/validate.csv', delimiter=',', dtype = inTypes))
official_test_data = np.asarray(genfromtxt('project_data/test.csv', delimiter=',', dtype = inTypes))

#print("end reading data")

# maybe shuffle train data with according y

""" Prepare Data """

def parseDataX (data):
    data_size = data.shape[0]
    weekTime = np.zeros(data_size) 
    yearDay = np.zeros(data_size)
    weatherFeatures = np.empty([data_size, 6])
    final_train_data_x = np.empty([data_size, 1])


    for i in range (0, data_size):

        train_x = data[i]
        #print(train_x)
        weatherFeatures[i] = [train_x['w1'], train_x['w2'], train_x['w3'], train_x['w4'], train_x['w5'], train_x['w6']] #train_x[['w1','w2','w3','w4','w5','w6']]
        t = time.strptime(str(train_x[0]), "b'%Y-%m-%d %H:%M:%S'") # t is timestamp of train_x
        weekTime[i] = (t.tm_wday*24 + t.tm_hour + t.tm_min/60 + t.tm_sec/3600) #in hours
        yearDay[i] = t.tm_yday

        final_train_data_x[i] = [weekTime[i]] #np.concatenate(([weekTime[i],yearDay[i]], weatherFeatures[i]))

    return final_train_data_x

final_train_data_x = parseDataX(train_data)
final_validate_data_x = parseDataX(validate_data)
final_official_test_data = parseDataX(official_test_data)

""" Normalize Data """

def get_feature(data, index): #get the array of feature "index" from a data set with multiple features
    res = [None]*data.shape[0]
    for i in range (0, data.shape[0]):
        res[i] = data[i][index]
    return res

means = np.empty(final_train_data_x.shape[1])
variances = np.empty(final_train_data_x.shape[1])

for j in range (0, final_train_data_x.shape[1]):
    m = np.mean(get_feature(final_train_data_x, j))
    var = 0.0
    n = final_train_data_x.shape[0]
    for i in range (0, n):
        var = var + np.square(final_train_data_x[i][j]-m)
    var = (var/n)

    means[j] = m
    variances[j] = var
        

""" normalize training data """
for i in range (0, final_train_data_x.shape[0]):
    for j in range (0, final_train_data_x.shape[1]):

        final_train_data_x[i][j] = (final_train_data_x[i][j] - means[j])/np.sqrt(variances[j])


""" normalize validation data """

for i in range (0, final_validate_data_x.shape[0]):
    for j in range (0, final_validate_data_x.shape[1]):

        final_validate_data_x[i][j] = (final_validate_data_x[i][j] - means[j])/np.sqrt(variances[j])


""" normalize test data """
#TODO




""" Split Data, k-Fold """

def scorefun (y_test, y_predict): #the function which will evaluate our prediction
        sum = 0.0
        n = y_test.shape[0]        
        for i in range(0, n):
            sum = sum + np.square(np.log(1+y_test[i]) - np.log(1 + np.maximum(0, y_predict[i])))
        return np.sqrt((1/n)*sum)
    
                           

for e in [-1]:   
    for c in range (2, 3):
        for d in [15]:
            for g in range (3,4):

                svr_model = SVR(kernel = 'rbf', gamma = np.power(10,g), degree = d, C = np.power(10,c), epsilon = np.power(10,e)) #evtl change kernel

                kf = skcv.KFold(train_data.size, n_folds=2, shuffle = True)
                scores = []

                for train, test in kf:
                    print (train.shape)
                    X_train, X_test = final_train_data_x[train], final_train_data_x[test]
                    Y_train, Y_test = train_data_y[train], train_data_y[test]
                    svr_model.fit(X_train, Y_train)
                    
                    svr_score = svr_model.score(X_test, Y_test)
                    ETH_score = scorefun(Y_test, svr_model.predict(X_test))                   
                    scores.append([svr_score, ETH_score])
                    print('d=', d, 'g=', g, 'c=', c, 'e=', e, 'score=', svr_score)
                    print('scorefun =', ETH_score) 
                    
                    
                    plt.plot(get_feature(X_test, 0), Y_test, 'rx', get_feature(X_test, 0), svr_model.predict(X_test), 'bo', markersize = 2)
                    plt.xlim(-2,2)
                    plt.show()
                    
                    

                print('mean svr_score:', np.mean(np.asarray(scores)[:,[0]]), 'mean ETH_score:', np.mean(np.asarray(scores)[:,[1]]))


print("end k-Fold")

""" Predict & Print """ #use all the train data with the decided model

final_model = SVR(kernel = 'rbf', gamma = np.power(10,3), degree = 15, C = np.power(10,2), epsilon = np.power(10,-1))
final_model.fit(final_train_data_x, train_data_y)

val_y = final_model.predict(final_validate_data_x)
np.savetxt('result_validate.txt', val_y)


print("done")
              
                




