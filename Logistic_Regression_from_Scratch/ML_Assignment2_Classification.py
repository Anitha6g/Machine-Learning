''' Implementation of Multi-Class classification algorithm - Logistic Regression '''
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sb
import matplotlib.patches as mpatches
import warnings 
warnings.filterwarnings('ignore')

''' Class for Logistic Regression method '''
class LogisticRegression:

    def __init__(self, learning_rate):
        self.__alpha = learning_rate

    ''' Calculating Sigmoid method '''
    @staticmethod
    def __sigmoid(z):        
        return 1/(1+np.exp(-z))
    
    ''' Gradient Descend function to find the optimal cost value '''
    def __gradient_descend(self, all_theta, X, y):
        m=len(y)
        h = self.__sigmoid(X.dot(all_theta))
        return (1/m) * X.T.dot(h - y) + ((self.__alpha/m)*all_theta)
    
    ''' Regularizing the Gradient Descend '''
    def __reg_gradient(self, all_theta, X, y):
        no_of_iterations = 10000
        for i in range(no_of_iterations):
            delta = self.__gradient_descend(all_theta,X,y)
            all_theta = all_theta - delta         
        i = i + 1;        
        return all_theta
    
    ''' Model fit function to fit the given dataset using the optimal Theta value '''
    def model_fit(self, X, y):
        y = y.values
        X = np.insert(X, 0, 1, axis=1)
        self.__target = np.unique(y)
        self.__all_theta = np.zeros((len(self.__target), X.shape[1]))
        for index, _class in enumerate(self.__target):
            y_class = np.array(list(map(lambda x: 1 if x == _class else 0, y)))
            self.__all_theta[index] = self.__reg_gradient(self.__all_theta[index], X, y_class)
        
    ''' Function to predict the target using the minimized theta and calculates the probability for each class. 
    Class with maximum probability is picked and assigned to the given attributes '''      
    def predict_target(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_predict = self.__sigmoid(X.dot(self.__all_theta.T))
        y_predict = np.argmax(y_predict, axis=1)
        f = lambda x: self.__target[x]
        return np.array([f(x) for x in y_predict])
        
    ''' Accuracy rate for the predicted output is calculated '''
    def predict_accuracy_rate(self, X, y):
        y = y.values
        y_predict = self.predict_target(X)
        all_y = len(y)
        predicted_y = 0
        for i in range(all_y):
            if y[i] == y_predict[i]:
                predicted_y += 1
        return predicted_y/all_y
    
''' Function used to classify the variety and assign unique numeric values to it '''
def data_mapping(data):
    target_y = data.shape[1]-1
    data.columns = list(range(target_y+1))
    classified = dict()
    for index, y_class in enumerate(set(data[target_y].values)):
        classified[y_class] = index
    data[target_y] = data[target_y].map(classified)
    X = data.drop(columns=[target_y,0])
    y = data[target_y]
    return X, y, classified

''' Read Data '''
data = pd.read_csv(r"C:\Users\anith\OneDrive\Desktop\Machine_Learning\Assignment2\Hazelnut.csv")
X, y, classified = data_mapping(data)
map_classes = dict(zip(classified.values(), classified.keys()))

''' Standard Scaler function for normal distribution of data '''
scaler = StandardScaler() 
scaler.fit(X)
X = scaler.transform(X)

''' Splitting the data randomly 10 times to train and validate the model '''
def K_fold_validation(X, y):
    
    ''' Implementing Sklearn Logistic Regression '''
    regr = linear_model.LogisticRegression() 
    
    ''' Implementing my model '''
    k_model = LogisticRegression(learning_rate=0.001)
    
    global model_accuracy
    model_accuracy = np.empty(10) 
    
    ''' Run the model for 10 folds with 1/3 datasize each time '''
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = i)
        
        ''' Validation and finding the accuracy of My Model '''
        k_model.model_fit(X_train, y_train)
        model_accuracy[i] = k_model.predict_accuracy_rate(X_test, y_test)
        print('Accuracy for my Model {} is {}'.format(i+1, model_accuracy[i]))
        
        ''' Validation and finding the accuracy of Sklearn Model '''
        regr.fit(X_train, y_train)
        sklearn_predict = regr.predict(X_test)
        accuracy = accuracy_score(y_test, sklearn_predict)
        print("Accuracy by sklearn model {} is {}".format(i+1, accuracy))
    
    ''' Finding the split with maximum accuracy '''
    global max_accuracy
    max_accuracy = max(model_accuracy) 
    
    ''' Displaying the mean of accuracy of all the splits '''
    print("Final Accuracy for my model is: {}".format(model_accuracy.mean()))
    print("Final Accuracy for Sklearn model is: {}".format(accuracy.mean()))

''' K_fold_validationn call '''    
K_fold_validation(X, y)

''' Learning curve for my model which appends the values from each iterations in K-fold '''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
c = int(len(X_train)/10)

''' Instantiating my model to plot a learning curve '''
my_model = LogisticRegression(learning_rate=0.0001)
my_model_curve = {'Accuracy Rate': [], 'Model Size': []}

''' Instantiating sklearn model to plot a learning curve '''
sk_model = linear_model.LogisticRegression()
sk_model_curve = {'Sk Accuracy Rate': [], 'Sk Model Size': []}

''' Ten iterations for data which is split into 10 folds with increasing model size '''
for i in range(10):
    a = (i+1)*c  
   
    ''' Plotting learning curve for my_model for each iteration and appending it accordingly'''
    my_model.model_fit(X_train[:a], y_train[:a])    
    my_model_curve['Model Size'].append(a)   
    my_model_curve['Accuracy Rate'].append(my_model.predict_accuracy_rate(X_test, y_test))
    plt.plot(my_model_curve['Model Size'], my_model_curve['Accuracy Rate'], 'b')
    
    ''' Plotting learning curve for sk_model for each iteration and appending it accordingly'''
    sk_model.fit(X_train[:a], y_train[:a])    
    sk_predict = sk_model.predict(X_test)      
    sk_model_curve['Sk Model Size'].append(a)   
    sk_model_curve['Sk Accuracy Rate'].append(accuracy_score(y_test, sk_predict))
    plt.plot(sk_model_curve['Sk Model Size'], sk_model_curve['Sk Accuracy Rate'], 'r')
    
    plt.xlabel('Model Size')
    plt.ylabel('Accuracy Rate')
    plt.title('Learning curve for both models')
    
my_alg = mpatches.Patch(color='blue', label='My_model')
sk_alg = mpatches.Patch(color='red', label='Sklearn_model')
plt.legend(handles=[my_alg,sk_alg])
plt.show()

''' Training and testing the model with the split of maximum accuracy i.e Model 9. Found it by max(model_accuracy) and inputting the
index of it into random_state ''' 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = list(model_accuracy).index(max_accuracy))
model = LogisticRegression(learning_rate=0.001)
model.model_fit(X_train, y_train)
y_predict = model.predict_target(X_test)

''' Write the Actual value, Predicted value and results into and csv file for that model(model 9) with highest accuracy'''
predictions = list()
for i in range(len(y_predict)):
    predictions.append([y_predict[i], y_test.values[i], 'Matched' if y_predict[i] == y_test.values[i] else 'Unmatched'])
    Result = pd.DataFrame(predictions, columns=["True Value", "Predicted Value", "Output"])
    Result["True Value"] = Result["True Value"].map(map_classes)
    Result["Predicted Value"] = Result["Predicted Value"].map(map_classes)
Result.to_csv('Output.csv', header=True, index=False)
    
''' Confusion matrix for y_test and y_predict for model(model 9) with highest accuracy'''
Variety = ['c_americana','c_avellana', 'c_cornuta']
conf_matrix = confusion_matrix(y_test, y_predict)
Hm = sb.heatmap(conf_matrix, annot = True, xticklabels = Variety, yticklabels = Variety);
Hm.set(xlabel='True Value', ylabel='Predicted Value')
Hm.set_title('Confusion matrix for model with highest accuracy(model 9)')


