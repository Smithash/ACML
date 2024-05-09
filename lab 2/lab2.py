import numpy as np

#define neural network architecture
input_nodes = 2
hidden_nodes = 3
output_nodes = 2
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

input_values = [2,1]

weights_biases = init_weights_biases(2, 3, 2)

forward = feed_forward(input_values, weights_biases)

    
    