import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

#define neural network architecture
input_nodes = 4  # Assuming 4 input features for iris dataset
hidden_nodes = 3
output_nodes = 3  # We have 3 classes in iris dataset
learning_rate = 0.1

#Define weights and biases
def init_weights_biases(input, hidden, output):
    hidden_weights = np.random.uniform(-1,1,(input, hidden))
    output_weights = np.random.uniform(-1,1, (hidden, output))
    
    hidden_bias = np.random.uniform(-1, 1, hidden)
    output_bias = np.random.uniform(-1,1,output)
    
    return [hidden_weights, output_weights, hidden_bias, output_bias]

#Define the activation function
def sigmoid(activation):
    return 1.0/(1.0 + np.exp(-activation))

#Feed forward function 
def feed_forward(input_values, weights_biases):
    hidden_layer = sigmoid(np.dot(input_values, weights_biases[0]) + weights_biases[2])
    output_layer = sigmoid(np.dot(hidden_layer, weights_biases[1]) + weights_biases[3])
    
    return hidden_layer, output_layer

#Loss function - sum of squares 
def sum_of_squares(y_true, y_pred):
    return 1/2*(np.sum((y_true - y_pred)**2))

weights_biases = init_weights_biases(input_nodes, hidden_nodes, output_nodes)

#Backwards prop - training the NN
def trainNetwork(input, target):
    epoch = 10000
    for epoch in range(epoch):
        for x,t in zip(input, target):
            #feedforward input x to get output y
            hidden_a, output_a = feed_forward(x, weights_biases)
            
            error = sum_of_squares(output_a, t)
            #for each output node n, where an is the output value, compute the delta
            output_deltas = (output_a - t)* output_a * (1- output_a)
            
            #for each every node m in the hidden layer, compute the delta
            hidden_deltas = np.dot(output_deltas, weights_biases[1].T)*hidden_a*(1-hidden_a)
            
            #for every node m in the hidden layer and node n in the output layer - update the weights
            weights_biases[1] = weights_biases[1] - learning_rate* np.outer(hidden_a, output_deltas)
            weights_biases[3] = weights_biases[3] - learning_rate*output_deltas
            
            #for every node l in input layer and every node m in the hidden layer
            weights_biases[0] = weights_biases[0] - learning_rate*np.outer(x, hidden_deltas)
            weights_biases[2] = weights_biases[2] - learning_rate*hidden_deltas
            
        updated_hidden_val, updated_output_val = feed_forward(input, weights_biases)
        updated_error = sum_of_squares(updated_output_val, target)
        
            
        print("%.4f"%updated_error)
        
data = pd.read_csv('./lab 3/iris_dataset.csv')

X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
T = data[['class']]

# One-hot encode the target labels
ohe = OneHotEncoder()
transformed = ohe.fit_transform(data[['class']])
T = transformed.toarray()
X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.2, random_state=42)
trainNetwork(X, T)
