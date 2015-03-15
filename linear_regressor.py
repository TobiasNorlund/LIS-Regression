
# Load and normalize the data
import load_data

# Train linear regressor
import sklearn.linear_model as sklin
import numpy as np
import sklearn.cross_validation as skcv

regressor = sklin.LinearRegression()

# Train and predict the whole training set
regressor.fit(load_data.X_norm, load_data.Y)
Y_pred = regressor.predict(load_data.X_norm)
print('Score (full training set): ', load_data.logscore(load_data.Y, Y_pred))

# Perform 5 fold cross validation_curve
scores = skcv.cross_val_score(regressor, load_data.X_norm, load_data.Y, scoring=load_data.scorefun, cv=4)
print('C-V score =', np.mean(scores), '+/-', np.std(scores))