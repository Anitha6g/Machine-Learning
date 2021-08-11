import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def reLu(x):
  return np.max(0,x)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def derivative_of_sigmoid(x):
  return x * (1-x)

def initvalues(input_x, hidden_nodes, output_node_y = 1):
  weights_hidden = np.random.rand(input_x.shape[1], hidden_nodes)
  weights_output = np.random.rand(hidden_nodes, output_node_y)
  bias_hidden = np.random.rand(hidden_nodes)
  bias_output = np.random.rand(output_node_y)
  return (weights_hidden,weights_output, bias_hidden, bias_output)

def feedforwardnet(input_x, hidden_nodes, output_node_y, weights_hidden, weights_output, bias_hidden, bias_output):
  #print("feed forward weights hidden:",weights_hidden)
  hidden_layer = sigmoid(np.dot(input_x, weights_hidden)+bias_hidden)
  output_layer = sigmoid(np.dot(hidden_layer,weights_output)+bias_output)
  y_pred = output_layer 
  return y_pred, hidden_layer, output_layer, weights_hidden,weights_output, bias_hidden, bias_output

def backpropogation(X_train, y_pred, y_train, weights_hidden, weights_output, hidden_layer, output_layer,learning_rate = 0.01):
    error_rate = np.mean((y_train - y_pred))
    
    der_hidden_layer = derivative_of_sigmoid(hidden_layer)
    der_output_layer = derivative_of_sigmoid(output_layer)
    
    new_weights_output = np.dot(hidden_layer.T,der_output_layer) * error_rate
    new_weights_hidden = np.dot(X_train.T, np.dot(der_hidden_layer, new_weights_output)) * weights_output * error_rate
    
    weights_output = weights_output + learning_rate * new_weights_output 
    weights_hidden = weights_hidden + learning_rate * new_weights_hidden 
    #print("back propogation weights hidden:",weights_hidden)
    
    return (weights_hidden,weights_output, error_rate)
    
    
data = pd.read_csv(r"C:\Users\anith\OneDrive\Desktop\Semester_2\Deep_learning\Assignment1\circles500.csv")
input_x = data.drop(['Class'], axis=1).values
output_y = data['Class'].values 
X_train, X_test, y_train, y_test = train_test_split(input_x, output_y, test_size = 0.10)
y_train = np.reshape(y_train, (len(y_train)))
hidden_z = 2
weights_hidden,weights_output, bias_hidden, bias_output = initvalues(X_train, hidden_z)
epoch = 1000
for i in range(epoch):
    y_pred, hidden_layer, output_layer, weights_hidden, weights_output, bias_hidden, bias_output = feedforwardnet(X_train,hidden_z,y_train,weights_hidden,weights_output, bias_hidden, bias_output)
    weights_hidden,weights_output, error_rate = backpropogation(X_train, y_pred, y_train, weights_hidden, weights_output, hidden_layer, output_layer,learning_rate = 0.01)
    if i % 10 == 0:
        print("error_rate:",error_rate)
print("Actual Value: {}".format(y_test))
print("Prediction: {}".format(y_pred))
#Calculating the accuracy of the model   
#def main(input_x, hidden_z, output_y, lr, epoch):
##  output_y = 1
##  hidden_z = 3
#  weights_input, weights_hidden, bias_input, bias_hidden = initvalues(input_x, hidden_z, output_y)
#  y_pred, input_layer, hidden_layer, weights_input, weights_hidden, bias_input, bias_hidden = feedforwardnet(input_x, hidden_z, output_y, lr,weights_input, weights_hidden, bias_input, bias_hidden)
#  for i in range(epoch):
#      if i % 100 ==0: 
#        print ("Actual Output: \n" + str(output_y))        
#        print ("Predicted Output: \n" + str(y_pred))
#        print ("Loss: \n" + str(np.mean(np.square(output_y - y_pred))))
#      y_pred, input_layer, hidden_layer, weights_input, weights_hidden, bias_input, bias_hidden = feedforwardnet(input_x, hidden_z, output_y, lr,weights_input, weights_hidden, bias_input, bias_hidden)
#  bias_input = np.zeros((hidden_nodes,1))
#  bias_hidden = np.zeros((len(output_node_y),1))
#X_train = X_train.T
#X_test = X_test.T
#y_train = np.reshape(y_train, (1, len(y_train)))
#y_test = np.reshape(y_test, (1, len(y_test)))
#main(X_train, hidden_z, y_train, 0.01, 550)