#Created by Tristan Bester
import numpy as np
import random

#Training data
#[input vector, output activation]
training_data = np.array(([[0,0],0],[[1,1],0],[[0,1],1],[[1,0],1]))

class Neural_Network(object):

    def __init__(self):
        #Initialise the network.
        self.input_layer_size = 2
        self.hidden_layer_size = 3
        self.output_layer_size = 1
        
        #Fill the weight matrices with random float values as per standard normal distribution
        self.W1 = np.random.randn(self.hidden_layer_size,self.input_layer_size)
        self.W2 = np.random.randn(self.output_layer_size, self.hidden_layer_size)

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self,z):
        return np.exp(-z)/((1+np.exp(-z))**2)

    def feedForward(self, training_example):
        #The values calculated during forward propagation are stored to be used during backpropagation
        
        #Extract activations for first layer from training_example list
        self.a1 = np.array((training_example[0][0],training_example[0][1]))
        
        #Calculate the weighted inputs and activations for all other layers in the network
        self.z2 = np.dot(self.W1, self.a1.reshape(-1,1))
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.W2, self.a2)
        self.a3 = self.sigmoid(self.z3)
        #return the activations of the neuron in the output layer
        return self.a3

    def costFunction(self, training_example, answer):
        #Calculate output activation
        output = self.feedForward(training_example)
        #return cost function value
        return 0.5*((answer - output)**2)

    def costFunctionPrime(self, training_example):
        #Extract label from training_example list
        y = training_example[1]
        
        #Perform forward propagation in order to calculate all of the 
        #weighted-input and activation values in the network.
        self.yHat = self.feedForward(training_example)
        
        #Calculate error in output neurons
        delta3 = np.multiply((-(y - self.yHat)), self.sigmoidPrime(self.z3))
        #Calculate rate of change of cost function with respect to each weight in last set of weights
        dCdW2 = np.multiply(delta3,self.a2)

        #Calculate error in neurons in second layer
        delta2 = np.multiply(delta3,self.W2.T)
        delta2 = delta2*self.sigmoidPrime(self.z2)
        #Calculate rate of change of the cost function with respect to each weight in the first
        #set of weights
        dCdW1 = np.dot(delta2,self.a1.reshape(1,-1))
        #return the partial derivative
        return dCdW1, dCdW2

    def getWeightVector(self):
        vec_one = self.W1.ravel()
        vec_two = self.W2.ravel()
        weight_vector = np.concatenate((vec_one,vec_two))
        return weight_vector

    def setWeights(self, weight_vector):
        W1_vec = weight_vector[:self.input_layer_size*self.hidden_layer_size]
        self.W1 = W1_vec.reshape(self.hidden_layer_size, self.input_layer_size)
        W2_vec = weight_vector[self.input_layer_size*self.hidden_layer_size:]
        self.W2 = W2_vec.reshape(self.output_layer_size,self.hidden_layer_size)

    def train(self, training_data, eta, iterations):
        #online training/incremental learning
        for i in range(iterations):
            #choose a random training example
            example = random.choice(training_data)
            #Calculate partial derivative for training example
            dCdW1, dCdW2 = self.costFunctionPrime(example)
            #Adjust the weights of the network based on result of backpropagation
            self.W1 = self.W1 - eta*dCdW1
            self.W2 = self.W2 - (eta*dCdW2).reshape(1,-1)



def calculateNumericalGradients(net, training_example):
    #Get the weights of the network in a coloumn vector
    weight_vector = net.getWeightVector()
    epsilon = 1e-4
    #Create vector used to alter weight vector
    perturb_vector = np.zeros(weight_vector.shape)
    #Create vector to store the numerical gradients
    numerical_gradients_vector = np.zeros(weight_vector.shape)
    #Extract the label from the training example list
    answer = training_example[1]

    #Loop through each wehigh in the network calculating the numerical gradient for that weight
    for i in range(len(weight_vector)):
        perturb_vector[i] = epsilon

        temp_weight_vector =  weight_vector - perturb_vector
        net.setWeights(temp_weight_vector)
        costOne = net.costFunction(training_example,answer)

        temp_weight_vector =  weight_vector + perturb_vector
        net.setWeights(temp_weight_vector)
        costTwo = net.costFunction(training_example, answer)

        numerical_gradients_vector[i] = (costTwo - costOne)/(2*epsilon)
        perturb_vector[i] = 0

    #Reset the weights to their initial values so the network will be set a back to it's inital state
    net.setWeights(weight_vector)
    #Return the numerical weight vector
    return numerical_gradients_vector

def test(net):
    print("Answer: 1\tGuess:", int(net.feedForward([[0,1],1]) > 0.5))
    print("Answer: 1\tGuess:", int(net.feedForward([[1,0],1]) > 0.5))
    print("Answer: 0\tGuess:", int(net.feedForward([[0,0],0]) > 0.5))
    print("Answer: 0\tGuess:", int(net.feedForward([[1,1],0]) > 0.5))
    

net = Neural_Network()
#Simple test of the networks performance.
print("Before training:")
test(net)
net.train(training_data,0.1,100000)
print("\nAfter training:")
test(net)
