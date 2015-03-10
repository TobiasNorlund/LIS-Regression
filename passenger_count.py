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
#print("end reading data")
#print(train_data.shape)

# maybe shuffle train data with according y

""" Prepare Data """
train_data_size = train_data.shape[0]
print(train_data_size)

weekTime = np.zeros(train_data_size) 
yearDay = np.zeros(train_data_size)
weatherFeatures = np.empty([train_data_size, 6])
final_train_data_x = np.empty([train_data_size, 8])


for i in range (0, train_data_size):

    train_x = train_data[i]
    #print(train_x)
    weatherFeatures[i] = [train_x['w1'], train_x['w2'], train_x['w3'], train_x['w4'], train_x['w5'], train_x['w6']] #train_x[['w1','w2','w3','w4','w5','w6']]
    t = time.strptime(str(train_x[0]), "b'%Y-%m-%d %H:%M:%S'") # t is timestamp of train_x
    weekTime[i] = (t.tm_wday*24 + t.tm_hour + t.tm_min/60 + t.tm_sec/3600) #in hours
    yearDay[i] = t.tm_yday

    final_train_data_x[i] = weekTime[i] #np.concatenate(([weekTime[i],yearDay[i]], weatherFeatures[i]))


""" Normalize Data """

final_train_data_x = normalize(final_train_data_x, axis = 0)


""" Split Data, k-Fold """


def get_feature(data, index):
    res = [None]*data.shape[0]
    for i in range (0, data.shape[0]):
        res[i] = data[i][index]
    return res
    

for d in [10]:
    for g in np.arange(6.0, 7.0, 0.2):

        svr_poly = SVR(kernel = 'rbf', gamma = np.power(10,g), degree = d) #evtl change kernel,  C=1e3, 

        kf = skcv.KFold(train_data_size, n_folds=2, shuffle = False)

        for train, test in kf:
            print (train.shape)
            X_train, X_test = final_train_data_x[train], final_train_data_x[test]
            Y_train, Y_test = train_data_y[train], train_data_y[test]
            svr_poly.fit(X_train, Y_train)
            print(d, g, svr_poly.score(X_test, Y_test))

        
            plt.plot(get_feature(X_test, 0), Y_test, 'rx', get_feature(X_test, 0), svr_poly.predict(X_test), 'bo', markersize = 2)
            plt.xlim(-0.001,0.02)
            plt.show()
        
    


print("end k-Fold")


"""

for i in range (0, 3):

    print(final_train_data_x[i])

"""

""" Plot subset of all data """
"""

numberOfExamples = 3 

weekday_sub = [None]*numberOfExamples 
dayTime_sub = [None]*numberOfExamples
weekTime_sub = [None]*numberOfExamples
yearDay_sub = [None]*numberOfExamples
train_y_sub = [None]*numberOfExamples


for i in range (0, numberOfExamples):

    j = randint(0, train_data_size -1)

    train_y_sub[i] = train_data_y[j]
    train_x = train_data[j]
    
    
    t = time.strptime(str(train_x[0]), "b'%Y-%m-%d %H:%M:%S'") # t is timestamp of train_x
    #print(t)

    weekday_sub[i] = t.tm_wday
    dayTime_sub[i] = (t.tm_hour + t.tm_min/60 + t.tm_sec/3600) #in hours
    weekTime_sub[i] = t.tm_wday + dayTime_sub[i]/24 #in days
    

print("end of loop")


plt.plot(weekTime_sub, train_y_sub, 'ro', markersize=3)
plt.xlim(-1,8)

#plt.show()
"""

