import numpy as np

#a) Implementing the neural network with 3 layers 
input_nodes = 4
hidden_nodes = 8
output_nodes = 3
hidden_weights = np.ones((input_nodes, hidden_nodes))
output_weights = np.ones((hidden_nodes, output_nodes))
hidden_biases = np.ones(hidden_nodes)
output_biases = np.ones(output_nodes)

#Activation function is sigmoid
def sigmoid(activation):
    return 1.0/(1.0 + np.exp(-activation))

#b) implement the feedforward step 

def feedForward(input_values):
    hidden_layer = sigmoid(np.dot(input_values, hidden_weights) + hidden_biases)
    output_layer = sigmoid(np.dot(hidden_layer, output_weights) + output_biases)
    
    return np.array(hidden_layer), np.array(output_layer)

#c) implement sum-of-squares loss computation 
def sum_of_squares(y, t):
    return 1/2*(np.sum((y-t)**2))

#d) implement backprop 
learning_rate = 0.1

def backpropagation(input_values, target):
    global output_weights, output_biases, hidden_weights, hidden_biases
    
    for i in range(1):
        hidden_a, output_a = feedForward(input_values)
        
        error = sum_of_squares(output_a, target)
        
        output_delta = (output_a - target)*output_a*(1-output_a)
        
        hidden_delta = np.dot(output_delta, output_weights.T)*hidden_a*(1-hidden_a)
        
        output_weights = output_weights - learning_rate*np.dot(hidden_a[:, np.newaxis], output_delta[:, np.newaxis].T)
        output_biases = output_biases - learning_rate*(output_delta)
        
        hidden_weights = hidden_weights - learning_rate*(np.dot(input_values[:, np.newaxis],hidden_delta[:, np.newaxis].T))
        hidden_biases = hidden_biases - learning_rate*(hidden_delta)
    updated_hidden_a , updated_output_a = feedForward(input_values)
    new_error = sum_of_squares(updated_output_a, target)
    
    print("%.4f"%error,"\n%.4f"% new_error)
        
def main():
    
    #1. Read from standard input a list of 7 numbers
    input_values =[]
    target_values =[]
    
    for i in range(4):
        user_input = float(input())
        input_values.append(user_input)
    input_values = np.array(input_values)
    
    for i in range(3):
        target = float(input())
        target_values.append(target)
    target_values = np.array(target_values)
    
    backpropagation(input_values, target_values)
        
    
if __name__ == "__main__":
    main()   
      

    

