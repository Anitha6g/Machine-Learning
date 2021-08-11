import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error 
from math import sqrt
# Reading the CSV file using pandas
data = pd.read_csv(r"C:\Users\anith\OneDrive\Desktop\Machine_Learning\Assignment3\steel.csv")

#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler(feature_range=(0, 1))
#data_scaled = scaler.fit_transform(data)

#data = pd.DataFrame(data_scaled, columns = ["normalising_temperature","tempering_temperature","sample","percent_silicon","percent_chromium","manufacture_year","percent_copper","percent_nickel","percent_sulphur","percent_carbon","percent_manganese","tensile_strength"])

from sklearn.model_selection import train_test_split
X = data.drop(['tensile_strength'], axis = 1)
y = data['tensile_strength']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state = 1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(X_test)
X_test = pd.DataFrame(x_test_scaled)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 40)
# Train the model on training data
rf.fit(X_train, y_train);

# Use the forest's predict method on the test data
predictions = rf.predict(X_test)
# Calculate the absolute errors
error = sqrt(mean_squared_error(y_test,predictions))
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(error), 2), 'degrees.')

# Calculate mean absolute percentage error
err_per = 100 * (error / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(err_per)
print('Accuracy:', round(accuracy, 2), '%.')


from sklearn.model_selection import RandomizedSearchCV
random_grid = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000]}
gf = RandomForestRegressor()
gf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, 
                               cv = 10, verbose=2, random_state=42, n_jobs = -1)


# Fit the random search model
gf_random.fit(X_train, y_train)
predictions_gf = gf_random.predict(X_test)
best_random = gf_random.best_estimator_
error_gf = sqrt(mean_squared_error(y_test,predictions_gf))
err_per_gf = 100 * (error_gf / y_test)
random_accuracy = 100 - np.mean(err_per_gf)
print(gf_random.best_params_)
print('Mean Absolute Error for the model with optimal parameters obtained after Grid search CV :', 
      round(np.mean(error_gf), 2), 'degrees.')
print("Accuracy of the model with optimal parameters obtained after Grid search CV: ",random_accuracy)

print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - accuracy) / accuracy))

