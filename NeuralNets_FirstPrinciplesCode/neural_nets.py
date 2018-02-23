import numpy as np
import math

"""
 ==================================
 Problem 3: Neural Network Basics
 ==================================
    Generates a neural network with the following architecture:
        Fully connected neural network.
        Input vector takes in two features.
        One hidden layer with three neurons whose activation function is ReLU.
        One output neuron whose activation function is the identity function.
"""


def rectified_linear_unit(x):
    """ Returns the ReLU of x, or the maximum between 0 and x."""
    return np.maximum(0,x)
    # TODO

def rectified_linear_unit_derivative(x):
    """ Returns the derivative of ReLU."""
    # TODO
    a=x
    for i in range(0,len(x)):
        if x[i]>0:
            a[i]=1
        else:
            a[i]=0
    return a
    
def output_layer_activation(x):
    """ Linear function, returns input as is. """
    return x

def output_layer_activation_derivative(x):
    """ Returns the derivative of a linear function: 1. """
    return 1
        
class Neural_Network():
    """
        Contains the following functions:
            -train: tunes parameters of the neural network based on error obtained from forward propagation.
            -predict: predicts the label of a feature vector based on the class's parameters.
            -train_neural_network: trains a neural network over all the data points for the specified number of epochs during initialization of the class.
            -test_neural_network: uses the parameters specified at the time in order to test that the neural network classifies the points given in testing_points within a margin of error.
    """

    def __init__(self):
        
        # DO NOT CHANGE PARAMETERS
        self.input_to_hidden_weights = np.matrix('1 1; 1 1; 1 1')
        self.hidden_to_output_weights = np.matrix('1 1 1')
        self.biases = np.matrix('0; 0; 0')
        self.learning_rate = .001
        self.epochs_to_train = 10
        self.training_points = [((2,1), 10), ((3,3), 21), ((4,5), 32), ((6, 6), 42)]
        self.testing_points = [(1,1), (2,2), (3,3), (5,5), (10,10)]
        
    def train(self, x1, x2, y):

        ### Forward propagation ###
        input_values = np.matrix([[x1],[x2]]) # 2 by 1

        # Calculate the input and activation of the hidden layer
        w=self.input_to_hidden_weights
        #print('w =', w)
        b=self.biases
        hidden_layer_weighted_input = np.dot(w,input_values)+b # TODO (3 by 1 matrix)
        z=hidden_layer_weighted_input
        hidden_layer_activation = rectified_linear_unit(z) # TODO (3 by 1 matrix)
        
        v=self.hidden_to_output_weights
        #print('v =', v)
        #print('hidden_layer_activation is ',hidden_layer_activation)
        output =  np.dot(v,hidden_layer_activation)  # TODO 
        activated_output = output_layer_activation(output) # TODO
        
        #print('y is ',y)
        #print('output is ', output)                                          
        print("Point:", x1, x2, " Error: ", (0.5)*pow((y - output),2))

        ### Backpropagation ###
        
        # Compute gradients
        
        dc_do=y-activated_output
        do_du=1
        du_d_hidden_to_output_weights=hidden_layer_activation #3 by 1 matrix
        du_d_hidden_layer_activation = v
        dfz_d_hidden_layer_weighted_input =  rectified_linear_unit_derivative(hidden_layer_activation)
        
        a=w*1
        #print('w first',a)
        tempt=input_values.squeeze()
        tempa=w*1
        tempa[0]=tempt
        tempa[1]=tempt
        tempa[2]=tempt    
        #print('w first', w)
        output_layer_error =  activated_output-y# TODO
        hidden_layer_error = output_layer_error* du_d_hidden_layer_activation * dfz_d_hidden_layer_weighted_input# TODO (3 by 1 matrix)

        bias_gradients = output_layer_error * do_du* du_d_hidden_layer_activation * dfz_d_hidden_layer_weighted_input# TODO
        hidden_to_output_weight_gradients =np.float64(output_layer_error) * do_du* du_d_hidden_to_output_weights  # TODO

        
        weight_from_x1=tempa[:,0]
        weight_from_x2=tempa[:,1]
        f=np.multiply(weight_from_x1,hidden_layer_error)
        g=np.multiply(weight_from_x2,hidden_layer_error)
        input_to_hidden_weight_gradients = tempa  #initialize 
        input_to_hidden_weight_gradients[:,0] = f# TODO
        input_to_hidden_weight_gradients[:,1] = g
        #print('w before substract', w)                               
        # Use gradients to adjust weights and biases using gradient descent
        self.biases = self.biases - (self.learning_rate*bias_gradients) # TODO
        self.input_to_hidden_weights = self.input_to_hidden_weights - (self.learning_rate*input_to_hidden_weight_gradients)# TODO
        #print('w grad is ', input_to_hidden_weight_gradients)
        #print('subtract', 0.001*input_to_hidden_weight_gradients)
        #rint('w after train is', self.input_to_hidden_weights)
        self.hidden_to_output_weights = self.hidden_to_output_weights - (self.learning_rate*hidden_to_output_weight_gradients.T)# TODO
        #print('v after train is',self.hidden_to_output_weights )
    def predict(self, x1, x2):
        
        input_values = np.matrix([[x1],[x2]])
    
        # Compute output for a single input(should be same as the forward propagation in training)
        w=self.input_to_hidden_weights
        b=self.biases
    
        hidden_layer_weighted_input = np.dot(w,input_values)+b  # TODO 
        z=hidden_layer_weighted_input
        
        hidden_layer_activation = rectified_linear_unit(z)# TODO
        v=self.hidden_to_output_weights
        output = np.dot(v,hidden_layer_activation) # TODO
        activated_output = output_layer_activation(output)# TODO
        
        return activated_output.item()
    
    
    # Run this to train your neural network once you complete the train method
    def train_neural_network(self):
        
        for epoch in range(self.epochs_to_train):
            for x,y in self.training_points:                
                self.train(x[0], x[1], y)
    
    # Run this to test neural network implementation for correctness after it is trained
    def test_neural_network(self):
        
        for point in self.testing_points:
            print("Point,", point, "Prediction,", self.predict(point[0], point[1]))
            if abs(self.predict(point[0], point[1]) - 7*point[0]) < 0.1:
                print("Test Passed")
            else:
                print("Point ", point[0], point[1], " failed to be predicted correctly.")
                return
        

x = Neural_Network()

x.train_neural_network()

#TEST NEURAL NETWORK
x.test_neural_network()  
