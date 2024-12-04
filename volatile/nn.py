import numpy as np
import matplotlib.pyplot as plt

def generateWeights(prevLayerSize, currLayerSize, init_epsilon=1.0):
    '''
    Generate array of weights to map previous layer size(including the bias unit) to current layer size(excluding its bias unit)
    '''
    return np.random.rand(currLayerSize, prevLayerSize) * (2 * init_epsilon) - init_epsilon

def addBiasTerm(array):
  '''
  Stack a 1.0 on top an input vector
  '''
  return np.vstack([np.array([[1.0]]), array])

def sigmoid(n):
    return (1 / (1 + (np.e ** -n)))

def sigmoidGradient(n):
    k = sigmoid(n)
    return k * (1 - k)

def cost_function(A, Y):
    '''
    Log loss function - multi class classification
    '''
    cost = np.sum(np.dot(-np.transpose(Y), np.log(A)) - np.dot(np.transpose(1-Y), np.log(1-A)))
    return cost


class neural_net:
    def __init__(self, layerSizes):
        self.layer_sizes = layerSizes
        self.inputSize = layerSizes[0]
        self.outputSize = layerSizes[-1]
        self.num_layers = len(layerSizes)
        self.num_hidden_layers = len(layerSizes) - 2

        self.weights = []
        for i in range(0, self.num_layers - 1):
                self.weights.append(generateWeights(self.layer_sizes[i] + 1, self.layer_sizes[i+1]))

    def forward_prop(self, X):
        '''
        Args
            X (np.array): must be a column vector of inputs
        Returns
            Layer Activations
        '''
        inputX = addBiasTerm(X)
        layers = [[inputX, 0]]
        for i in range(0, self.num_layers - 1):
            if i == self.num_layers - 2: # Last layer to calculate 
                currLayer = sigmoid(np.dot(self.weights[i], layers[-1][0]))
                layers.append([currLayer, currLayer]) # Copy the layer twice because pre activation layer is not needed
            else:
                if type(layers[-1]) == list: # If we have an activation for the previous layer(input layer doesnt have an activation)
                    z_val = np.dot(self.weights[i], layers[-1][0]) # pre activation values
                else:
                    z_val = np.dot(self.weights[i], layers[-1])
                currLayer = addBiasTerm(sigmoid(z_val))
                layers.append([currLayer, z_val]) # Represent each layer as [activationValues, pre Activation values]
        return layers
    
    def back_prop(self, layers, y):
        '''
        Use the layer activations and correct input to calculate gradients
        '''
        errors = []
        gradientAccum = []
        for i in range(0, len(self.weights)):
            gradientAccum.append(np.zeros(self.weights[i].shape))
        for i in range(self.num_layers-1, 0, -1):
            if i == self.num_layers-1:
                startError = layers[-1][0] - y
                prevLayer = layers[i-1][0]
                k = np.dot(startError, np.transpose(prevLayer))
                gradientAccum[i-1] += k
                errors.append(startError)
            else:
                currWeights = self.weights[i]
                currLayerZVal = layers[i][1]
                currError = np.dot(np.transpose(currWeights), errors[0])
                currError = np.multiply(currError, sigmoidGradient(addBiasTerm(currLayerZVal)))
                currError = currError[1:, :]
                errors.insert(0, currError)
                k = np.dot(currError, np.transpose(layers[i-1][0]))
                gradientAccum[i-1] += k
        return gradientAccum
    

    def update_params(self, params, gradient_accum, learning_rate):
        '''
        Gradient Descent applied after the entire training set is forward propagated
        '''
        for i in range(0, len(params)):
            params[i] = params[i] - (learning_rate * gradient_accum[i])

        return params
    
    def train(self, X, y, epochs, learning_rate):
        '''
        Returns costs after training completion
        '''
        costs = []
        numTrainingExamples = len(X)
        for j in range(0, epochs):
            cost = 0
            gradientAccum = []
            for i in range(0, len(self.weights)):
                gradientAccum.append(np.zeros(self.weights[i].shape))


            for i in range(0, numTrainingExamples):
                layerActivations = self.forward_prop(X[i])
                cost += cost_function(layerActivations[-1][0], y[i])
                grads = self.back_prop(layerActivations, y[i])
                for k in range(0, len(gradientAccum)):
                    gradientAccum[k] += grads[k]

            for i in range(0, len(gradientAccum)):
                gradientAccum[i] /= numTrainingExamples
            self.weights = self.update_params(self.weights, gradientAccum, learning_rate)
            cost = cost * (1 / numTrainingExamples)
            #print(cost)
            costs.append(cost)
        return costs



    
