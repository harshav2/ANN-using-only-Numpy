import numpy as np
import os

class NeuralNetwork:
    #network is to have 1 input variable x
    #network is to have 1 output variable z

    def __init__(self,no_of_neurons, no_of_layers):
        self.no_of_neurons=no_of_neurons
        self.no_of_layers=no_of_layers

    def init_params(self):
        #input layer parameters=1*neurons
        #intermediary layer paramaters=neurons*neurons (for "layers" number of layers)
        #output layer parameter=1*neurons
        self.input_weights=np.random.randn(self.no_of_neurons)
        self.hidden_weights=np.random.randn(self.no_of_layers,self.no_of_neurons)
        self.output_weights=np.random.randn(self.no_of_neurons)
        
    def load_params(self):
        #loading stored weights from npy files
        self.input_weights = np.load(os.path.join('/data', "input_weights.npy"))
        self.hidden_weights = np.load(os.path.join('/data', "hidden_weights.npy"))
        self.output_weights = np.load(os.path.join('/data', "output_weights.npy"))

    