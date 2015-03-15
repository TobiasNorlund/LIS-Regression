

# Create and train linear regressor
e = -1
g = 3
c = 2
d = 15
regressor = svm.SVR(kernel = 'rbf', gamma = np.power(10,g), degree = d, C = np.power(10,c), epsilon = np.power(10,e)) #evtl change kernel #
regressor = sklin.LinearRegression()

param_grid = { 'epsilon': np.power(10, np.array([-1.])),
               'C': np.power(10, np.array([-1., 0., 1., 2., ])),
               'gamma': np.power(10, np.array([-5., -2., 0., 3.])),
               'degree': [1, 3, 5]
               }
grid_search = skgs.GridSearchCV(regressor, param_grid, scoring=scorefun, cv=4)
grid_search.fit(X_norm, Y)

best = grid_search.best_estimator_
print(best)
print('best score =', grid_search.best_score_)

#scores = skcv.cross_val_score(regressor, X_norm, Y, scoring=scorefun, cv=4)
#print('C-V score =', np.mean(scores), '+/-', np.std(scores))

# Predict training set
#regressor.fit(X_norm, Y)
#Y_pred = regressor.predict(X_norm)
#print('Score on training data: ', logscore(Y, Y_pred))

