import pandas as pd 
# Reading the CSV file using pandas
data = pd.read_csv(r"C:\Users\anith\OneDrive\Desktop\Machine_Learning\Assignment3\steel.csv")
print(data.head())
#data_cus = pd.DataFrame(data, columns = ["normalising_temperature","tempering_temperature","sample","percent_silicon","percent_chromium","manufacture_year","percent_copper","percent_nickel","percent_sulphur","percent_carbon","percent_manganese","tensile_strength"])

#Splitting data into Input X and output y
X = data.drop(['tensile_strength','sample_id'], axis = 1)
print(X.head())
y = data['tensile_strength']
print(y.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state = 1)

#Normalizing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(X_test)
X_test = pd.DataFrame(x_test_scaled)

#import required packages
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
rmse_val = [] 
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)
    model.fit(X_train, y_train)  
    pred=model.predict(X_test) 
    error = sqrt(mean_squared_error(y_test,pred)) 
    rmse_val.append(error) 
    print('RMSE value for k=' , K , 'is:', error)
    
#plotting the rmse values against k values
import matplotlib.pyplot as plt 
plt.plot(rmse_val)  
plt.xlabel('K Values') 
plt.ylabel('RMSE Values')   
# giving a title to my graph 
plt.title('Optimal K value graph')  
plt.show() 
#curve = pd.DataFrame(rmse_val) 
#curve.plot()

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
scorer = make_scorer(mean_squared_error, greater_is_better=False)

parameters = [{'n_neighbors': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], 
               'weights': ['uniform', 'distance'],
               'metric': ['euclidean', 'manhattan', 'minkowski']}]
gs = GridSearchCV(neighbors.KNeighborsRegressor(), parameters, cv = 10, scoring = scorer)
gs.fit(X_train, y_train)
pred=gs.predict(X_test) 
error = sqrt(mean_squared_error(y_test,pred))
print('The best parameters selected by Grid search CV is: ',gs.best_params_)
print('RMSE value for optimal parameters obtained from Grid search CV is: ',error)

from sklearn.model_selection import cross_val_score
import numpy as np
knn = neighbors.KNeighborsRegressor(n_neighbors=5)
cv_scores = cross_val_score(knn, X, y, cv=10)
#print accuracy for each classified data
print(cv_scores)
print("cv_scores mean:{}".format(np.mean(cv_scores)))
