import numpy as np

import pandas as pd 
data = pd.read_csv(r"C:\Users\anith\Hazelnut.csv")

y = data['variety']

X = data.drop(['variety','sample_id'], axis=1)

import seaborn as sns; sns.set()
#import matplotlib.pyplot as plt
ax = sns.scatterplot(x="length", y="width",hue="variety", data=data)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

nb.fit(X,y)
y_pred = nb.predict(X)

print("Prediction: {}".format(y_pred)) 
print("Untrained Data Score:",nb.score(X, y))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state = 1, stratify = y)
#Fit the training data into classifier
nb.fit(X_train,y_train)
y_pred = nb.predict(X_test)
print("Actual Value: {}".format(y_test))
print("Prediction: {}".format(y_pred)) 
print("Trained Data score:",nb.score(X_test, y_test))

from sklearn.model_selection import cross_val_score
cv_scores1 = cross_val_score(nb, X, y, cv=10)
#print each cv score (accuracy) and average them
print(cv_scores1)
print("cv_scores1 mean:{}".format(np.mean(cv_scores1)))