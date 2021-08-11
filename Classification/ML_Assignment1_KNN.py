import pandas as pd 
# Reading the CSV file using pandas
data = pd.read_csv(r"C:\Users\anith\Hazelnut.csv")
#print(data.head())

#Dropped samlple_id(to increase the prediction accuracy as attribute is not relevant) and variety(target)
X = data.drop(['variety','sample_id'], axis=1)
#print(X.head())

#Splitting the data into Input data(X) and target(y)
y = data['variety']
#print(y.head())

'''import seaborn as sns; sns.set()
#import matplotlib.pyplot as plt
ax = sns.scatterplot(x="length", y="width",hue="variety", data=data)'''

from sklearn.model_selection import train_test_split
#Splitting data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state = 1, stratify = y)

from sklearn.neighbors import KNeighborsClassifier
#Creting the classifier and setting the K value to optimal value 6
knn = KNeighborsClassifier(n_neighbors=6)
#Classifier fit to data
knn.fit(X_train,y_train)

#Predicting the training dataset
y_pred = knn.predict(X_test)
#print("Actual value:",knn.predict(X_test)[0:21])
print("Actual Value: {}".format(y_test))
print("Prediction: {}".format(y_pred))
#Calculating the accuracy of the model built
print(knn.score(X_test, y_test))


from sklearn.model_selection import cross_val_score
import numpy as np
knn = KNeighborsClassifier(n_neighbors=6)
#10-fold cross validation
cv_scores = cross_val_score(knn, X, y, cv=10)
#print accuracy for each classified data
print(cv_scores)
print("cv_scores mean:{}".format(np.mean(cv_scores)))



'''import matplotlib.pyplot as plt
colors = ("red", "green", "blue")
groups = ("Avellana", "Americana", "Cornuta")
for i in cv_scores:
    for color, group in zip(colors, groups):      
        plt.scatter('length','width',c=color, data=data)
        plt.show()
for data, color, group in zip(data, colors, groups):
    
        plt.scatter('length', 'width', c=color, data=data)

        plt.show()
'''
import seaborn as sns; sns.set()
#import matplotlib.pyplot as plt
ax = sns.scatterplot(x="length", y="width",hue="variety", data=data)  

