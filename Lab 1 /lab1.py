import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, N, learning_rate = 0.01, epochs =100):
        self.num_inputs =N
        self.weights = np.random.rand(N)
        self.threshold = np.random.rand()
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def activations(self, x):
        return 1 if x > self.threshold else 0
    
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.threshold
        return self.activations(summation)
    
    def train(self, X,T):
        #threshold is 0
        #weights are random 
        for _ in range(self.epochs):
            N = len(X)
            # Randomly shuffle the inputs   
            I = np.arange(N)
            np.random.shuffle(I)
            
            for i in I:
                y = self.predict(X[i]) #feed x into the perceptron to get output y
                #calculate the error
                loss = T[i] - y
                #update the weights
                self.weights = self.weights + self.learning_rate*(loss)*X[i]
                #update the threshold
                self.threshold = self.threshold - self.learning_rate*loss
    
    #computing the loss
    def compute_loss(self, X,T):
        loss =0
        for i in range(len(X)):
            y = self.predict(X[i])
            loss += np.abs(T[i] - y)
        return loss

def main():
    #using provided dataset
    X = np.array([[0,0], [1,1], [1,0], [1,1]])
    T = np.array([1,1,1,0])
    
    #creating and training the perceptron
    perceptron = Perceptron(N =2)
    perceptron.train(X,T)
    
    #computing and printing the loss
    loss = perceptron.compute_loss(X,T)
    print("Final Loss:", loss)
    
    #Plotting the dataset and linear discriminant
    plt.scatter(X[:, 0], X[:, 1], c=T)
    plt.xlabel('X1')
    plt.ylabel('X2')
    
    # x_vals = X[:,0]
    # y_vals = X[:,1]
    # plt.plot(x_vals, y_vals, '-r')
    plt.title('Linear Discriminant of the Perceptron')
    plt.show()

if __name__ == "__main__":
    main()