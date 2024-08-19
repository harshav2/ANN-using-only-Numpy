import numpy as np
import os

class NeuralNetwork:
    #network is to have 1 input variable x
    #network is to have 1 output variable z

    def __init__(self):
        self.input_weights=None
        self.hidden_weights=None
        self.output_weights=None

    def init_params(self,no_of_neurons,no_of_layers):
        #input layer parameters=1*neurons
        #intermediary layer paramaters=neurons*neurons (for "layers" number of layers)
        #output layer parameter=1*neurons
        self.input_weights=np.random.randn(no_of_neurons)
        self.hidden_weights=np.random.randn(no_of_layers,no_of_neurons)
        self.output_weights=np.random.randn(no_of_neurons)
        
    def load_params(self):
        #loading stored weights from npy files
        self.input_weights = np.load(os.path.join('/data', "input_weights.npy"))
        self.hidden_weights = np.load(os.path.join('/data', "hidden_weights.npy"))
        self.output_weights = np.load(os.path.join('/data', "output_weights.npy"))

from forward_prop import forward_prop
nn=NeuralNetwork()
nn.load_params()

o,_=forward_prop(nn,1)
print("Output:\n",o)